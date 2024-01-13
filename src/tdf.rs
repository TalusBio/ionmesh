use log::{error, info, trace};
use rusqlite::{Connection, Result};
use std::path::Path;
use timsrust::{ConvertableIndex, Frame};

use crate::ms::frames::{DenseFrame, DenseFrameWindow, FrameWindow};

// Diaframemsmsinfo = vec of frame_id -> windowgroup_id
// diaframemsmswindows = vec[(windowgroup_id, scanstart, scanend, iso_mz, iso_with, nce)]

#[derive(Debug)]
pub struct ScanRange {
    pub scan_start: usize,
    pub scan_end: usize,
    pub iso_mz: f32,
    pub iso_width: f32,
    pub nce: f32,
    pub ims_start: f32,
    pub ims_end: f32,
    pub iso_low: f32,
    pub iso_high: f32,
}

impl ScanRange {
    pub fn new(
        scan_start: usize,
        scan_end: usize,
        iso_mz: f32,
        iso_width: f32,
        nce: f32,
        scan_converter: &timsrust::Scan2ImConverter,
    ) -> Self {
        // Note that here I swap the start and end,
        // because lower scan numbers are actually
        // higher 1/k0 values. ... i think...
        let ims_end = scan_converter.convert(scan_start as u32);
        let ims_start = scan_converter.convert(scan_end as u32);

        debug_assert!(ims_start < ims_end);
        let iso_low = iso_mz - iso_width / 2.0;
        let iso_high = iso_mz + iso_width / 2.0;

        Self {
            scan_start,
            scan_end,
            iso_mz,
            iso_width,
            nce,
            ims_start: ims_start as f32,
            ims_end: ims_end as f32,
            iso_low,
            iso_high,
        }
    }
}

#[derive(Debug)]
pub struct DIAWindowGroup {
    pub id: usize,
    pub scan_ranges: Vec<ScanRange>,
}

pub struct DIAFrameInfo {
    pub groups: Vec<Option<DIAWindowGroup>>,
    pub frame_groups: Vec<Option<usize>>,
}

// TODO rename or split this ... since it is becoming more
// of a splitter than a frame info reader.
// Maybe a builder -> splitter pattern?
impl DIAFrameInfo {
    pub fn get_group(&self, frame_id: usize) -> Option<&DIAWindowGroup> {
        let group_id = self.frame_groups[frame_id];

        match group_id {
            None => return None,
            Some(group_id) => self.groups[group_id].as_ref(),
        }
    }

    pub fn split_frame(&self, frame: Frame) -> Result<Vec<FrameWindow>, &'static str> {
        let group = self.get_group(frame.index);
        if group.is_none() {
            return Err("Frame not in DIA group");
        }
        let group = group.unwrap();

        let mut out_frames = Vec::new();
        for (i, scan_range) in group.scan_ranges.iter().enumerate() {
            scan_range.scan_start;
            scan_range.scan_end;

            let scan_offsets_use =
                &frame.scan_offsets[scan_range.scan_start..(scan_range.scan_end - 1)];
            let scan_start = scan_offsets_use[0];
            let mz_indptr_start = scan_offsets_use[0] as usize;
            let mz_indptr_end = *scan_offsets_use.last().unwrap() as usize;

            let tof_indices_keep = frame.tof_indices[mz_indptr_start..mz_indptr_end].to_vec();
            let intensities_keep = frame.intensities[mz_indptr_start..mz_indptr_end].to_vec();

            let frame_window = FrameWindow {
                scan_offsets: scan_offsets_use
                    .iter()
                    .map(|x| x - scan_start)
                    .collect::<Vec<_>>(),
                tof_indices: tof_indices_keep,
                intensities: intensities_keep,
                index: frame.index,
                rt: frame.rt,
                frame_type: frame.frame_type,
                scan_start: scan_range.scan_start as usize,
                group_id: group.id,
                quad_group_id: i,
            };

            out_frames.push(frame_window);
        }

        Ok(out_frames)
    }

    pub fn split_dense_frame(&self, mut denseframe: DenseFrame) -> Result<Vec<DenseFrameWindow>> {
        let group = self.get_group(denseframe.index);
        // if group.is_none() {
        //     return Err("Frame not in DIA group".into());
        // }
        let group = group.unwrap();

        // Steps
        // 1. Sort by ims
        // 2. Get ims bounds
        // 3. Binary search for start and end
        denseframe.sort_by_mobility();
        let mut frames = Vec::new();
        let imss = denseframe
            .raw_peaks
            .iter()
            .map(|peak| peak.mobility)
            .collect::<Vec<_>>();
        for (i, scan_range) in group.scan_ranges.iter().enumerate() {
            let start = imss.binary_search_by(|v| {
                v.partial_cmp(&scan_range.ims_start)
                    .expect("Couldn't compare values")
            });

            let start = match start {
                Ok(x) => x,
                Err(x) => x,
            };

            let end = imss.binary_search_by(|v| {
                v.partial_cmp(&scan_range.ims_end)
                    .expect("Couldn't compare values")
            });

            // i might need to add 1 here to make the range [closed, open)
            let end = match end {
                Ok(x) => x,
                Err(x) => x,
            };

            let frame = DenseFrame {
                raw_peaks: denseframe.raw_peaks[start..end].to_vec(),
                index: denseframe.index.clone(),
                rt: denseframe.rt.clone(),
                frame_type: denseframe.frame_type.clone(),
                sorted: denseframe.sorted.clone(),
            };

            let frame_window = DenseFrameWindow {
                frame,
                ims_start: scan_range.ims_start,
                ims_end: scan_range.ims_end,
                mz_start: scan_range.iso_low.into(),
                mz_end: scan_range.iso_high.into(),
                group_id: group.id,
                quad_group_id: i,
            };
            frames.push(frame_window);
        }

        Ok(frames)
    }

    /// Returns a vector of length equal to the number of groups.
    /// Each element is a vector of frames that belong to that group.
    fn bundle_by_group(&self, frames: Vec<DenseFrame>) -> Vec<Vec<DenseFrame>> {
        let mut frame_groups = Vec::new();
        for frame in frames {
            let group = self.get_group(frame.index);
            if group.is_none() {
                continue;
            }
            let group = group.unwrap();
            let group_id = group.id;
            if frame_groups.len() <= group_id {
                frame_groups.resize(group_id + 1, Vec::new());
            }
            frame_groups[group_id].push(frame);
        }
        frame_groups
    }

    pub fn split_dense_frames(&self, frames: Vec<DenseFrame>) -> Vec<Vec<Vec<DenseFrameWindow>>> {
        info!("Splitting {} frames", frames.len());

        // Returns a vector of length equal to the number of groups.
        // Each element is a vector with length equal to the number of quad groups within
        // that group.
        // Each element of that vector is a vector of frames that belong to that quad group.
        let max_num_quad_groups = self
            .groups
            .iter()
            .map(|group| {
                if group.is_none() {
                    0
                } else {
                    group.as_ref().unwrap().scan_ranges.len()
                }
            })
            .max()
            .unwrap();

        let num_groups = self.groups.len();

        let mut out = Vec::new();
        for _ in 0..num_groups {
            let mut group_vec = Vec::new();
            for _ in 0..max_num_quad_groups {
                group_vec.push(Vec::new());
            }
            out.push(group_vec);
        }

        let bundled_split_frames = self.bundle_by_group(frames);
        for (i, frame_bundle) in bundled_split_frames.into_iter().enumerate() {
            info!("Processing group {}", i);
            for frame in frame_bundle {
                let frame_windows = self.split_dense_frame(frame);
                match frame_windows {
                    Ok(frame_windows) => {
                        for frame_window in frame_windows {
                            out[i][frame_window.quad_group_id].push(frame_window);
                        }
                    }
                    Err(e) => {
                        error!("Error splitting frame: {}", e);
                    }
                }
            }
        }

        let counts = out
            .iter()
            .map(|group| {
                group
                    .iter()
                    .map(|quad_group| quad_group.len())
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();

        trace!("Counts: {:?}", counts);

        for (i, group) in counts.iter().enumerate() {
            trace!("Group {}", i);
            for (j, quad_group) in group.iter().enumerate() {
                trace!("  Quad group {}: {}", j, quad_group);
            }
        }

        out
    }

    pub fn get_quad_windows(
        &self,
        scan_group_id: usize,
        quad_group_id: usize,
    ) -> Option<&ScanRange> {
        let group = self.groups[scan_group_id].as_ref()?;
        let quad_group = group.scan_ranges.get(quad_group_id)?;
        Some(quad_group)
    }
}

// TODO implement splitting frames into dia group+quad groups.
// [usize, math::round::floor(quad_mz_center)]

// Reference for the tables:

// CREATE TABLE DiaFrameMsMsInfo (
//     Frame INTEGER PRIMARY KEY,
//     WindowGroup INTEGER NOT NULL,
//     FOREIGN KEY (Frame) REFERENCES Frames (Id),
//     FOREIGN KEY (WindowGroup) REFERENCES DiaFrameMsMsWindowGroups (Id)
//  )

// CREATE TABLE DiaFrameMsMsWindows (
//     WindowGroup INTEGER NOT NULL,
//     ScanNumBegin INTEGER NOT NULL,
//     ScanNumEnd INTEGER NOT NULL,
//     IsolationMz REAL NOT NULL,
//     IsolationWidth REAL NOT NULL,
//     CollisionEnergy REAL NOT NULL,
//     PRIMARY KEY(WindowGroup, ScanNumBegin),
//     FOREIGN KEY (WindowGroup) REFERENCES DiaFrameMsMsWindowGroups (Id)
//  ) WITHOUT ROWID

pub fn read_dia_frame_info(dotd_file: String) -> Result<DIAFrameInfo> {
    let reader = timsrust::FileReader::new(dotd_file.clone()).unwrap();
    let scan_converter = reader.get_scan_converter().unwrap();

    // Find an 'analysis.tdf' file inside the dotd file (directory).
    let tdf_path = Path::new(dotd_file.as_str()).join("analysis.tdf");

    info!("tdf_path: {:?}", tdf_path);
    let conn = Connection::open(tdf_path)?;

    let mut stmt_ids = conn.prepare("SELECT Frame, WindowGroup FROM DiaFrameMsMsInfo")?;

    let mut ids_vec: Vec<(usize, usize)> = Vec::new();
    let res = stmt_ids.query_map([], |row| {
        let id: usize = row.get(0)?;
        let group: usize = row.get(1)?;
        Ok((id, group))
    });

    match res {
        Ok(x) => {
            for id_group in x {
                ids_vec.push(id_group.unwrap());
            }
        }
        Err(e) => {
            error!("Error reading DiaFrameMsMsInfo: {}", e);
        }
    }

    let max_id = ids_vec.iter().map(|(id, _)| id).max().unwrap();
    let mut ids_map_vec = vec![None; max_id + 1];
    for (id, group) in ids_vec {
        ids_map_vec[id] = Some(group);
    }

    let mut stmt_groups = conn.prepare(
        "SELECT
            WindowGroup,
            ScanNumBegin,
            ScanNumEnd,
            IsolationMz,
            IsolationWidth,
            CollisionEnergy
        FROM DiaFrameMsMsWindows",
    )?;
    let mut groups_vec: Vec<(usize, usize, usize, f32, f32, f32)> = Vec::new();
    let res = stmt_groups.query_map([], |row| {
        Ok((
            row.get(0)?,
            row.get(1)?,
            row.get(2)?,
            row.get(3)?,
            row.get(4)?,
            row.get(5)?,
        ))
    });

    match res {
        Ok(x) => {
            for group in x {
                groups_vec.push(group.unwrap());
            }
        }
        Err(e) => {
            error!("Error reading DiaFrameMsMsWindows: {}", e);
        }
    }

    let max_window_id = groups_vec
        .iter()
        .map(|(id, _, _, _, _, _)| id.clone())
        .max()
        .unwrap();

    let mut groups_map_vec: Vec<Option<Vec<ScanRange>>> =
        (0..(max_window_id + 1)).map(|_| None).collect();

    for (group, scan_start, scan_end, iso_mz, iso_width, nce) in groups_vec {
        let scan_range = ScanRange::new(
            scan_start,
            scan_end,
            iso_mz,
            iso_width,
            nce,
            &scan_converter,
        );

        if groups_map_vec[group].is_none() {
            groups_map_vec[group] = Some(Vec::new());
        }

        match &mut groups_map_vec[group] {
            None => continue,
            Some(scan_ranges) => {
                scan_ranges.push(scan_range);
            }
        }
    }

    let mut groups_vec_o = (0..(max_window_id + 1)).map(|_| None).collect::<Vec<_>>();
    for (i, scan_ranges) in groups_map_vec.into_iter().enumerate() {
        let scan_ranges = match scan_ranges {
            None => continue,
            Some(scan_ranges) => scan_ranges,
        };
        if scan_ranges.len() == 0 {
            continue;
        } else {
            groups_vec_o[i] = Some(DIAWindowGroup { id: i, scan_ranges });
        }
    }

    let frame_info = DIAFrameInfo {
        groups: groups_vec_o,
        frame_groups: ids_map_vec,
    };
    Ok(frame_info)
}
