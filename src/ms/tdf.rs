use log::{debug, error, info, trace};

use sqlx::Pool;
use std::path::{Path, PathBuf};
use timsrust::{ConvertableIndex, Frame};
use sqlx::{Row, Sqlite, SqlitePool,FromRow};
use tokio;
use tokio::runtime::Runtime;


use crate::ms::frames::{DenseFrame, DenseFrameWindow, FrameQuadWindow};

// Diaframemsmsinfo = vec of frame_id -> windowgroup_id
// diaframemsmswindows = vec[(windowgroup_id, scanstart, scanend, iso_mz, iso_with, nce)]

#[derive(Debug, Clone)]
pub struct ScanRange {
    pub id: usize,
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
        id: usize,
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
            id,
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

#[derive(Debug, Clone)]
pub struct DIAWindowGroup {
    pub id: usize,
    pub scan_ranges: Vec<ScanRange>,
}

#[derive(Debug, Clone)]
pub enum GroupingLevel {
    WindowGroup, // Technically this is the same as the frame level ...
    QuadWindowGroup,
}

#[derive(Debug, Clone)]
pub struct DIAFrameInfo {
    pub groups: Vec<Option<DIAWindowGroup>>,
    /// Frame Groups is a vec of length equal to the number of frames.
    /// Each element is an Option<usize> that is the index of the group
    /// that the frame belongs to.
    pub frame_groups: Vec<Option<usize>>,
    pub retention_times: Vec<Option<f32>>,
    pub grouping_level: GroupingLevel,
}

// TODO rename or split this ... since it is becoming more
// of a splitter than a frame info reader.
// Maybe a builder -> splitter pattern?
impl DIAFrameInfo {
    pub fn get_dia_frame_window_group(&self, frame_id: usize) -> Option<&DIAWindowGroup> {
        let group_id = self.frame_groups[frame_id];
        if group_id.is_none() {
            return None;
        }
        self.groups[group_id.unwrap()].as_ref()
    }


    async fn rts_from_tdf_connection(conn: &Pool<Sqlite>) -> Result<Vec<Option<f32>>, sqlx::Error> {
        // To calculate cycle time ->
        // DiaFrameMsMsInfo -> Get the frames that match a specific id (one for each ...)
        // Frames -> SELECT id, time FROM Frames -> make a Vec<Option<f32>>, map the former
        // framer id list (no value should be None).
        // Scan diff the new vec!
        let results:Vec<(i32, f32)> = sqlx::query_as("SELECT Id, Time FROM Frames").fetch_all(conn).await?;
        let mut retention_times = Vec::new();
        for row in results.iter() {
            let id: usize = row.0 as usize;
            let time: f32 = row.1;
            retention_times.resize(id + 1, None);
            retention_times[id] = Some(time);
        }
        Ok(retention_times)
    }

    pub fn calculate_cycle_time(&self) -> f32 {
        let mut group_cycle_times = Vec::new();

        for (i, group) in self.groups.iter().enumerate() {
            if group.is_none() {
                continue;
            }

            let mapping_frames: Vec<usize> = self
                .frame_groups
                .iter()
                .enumerate()
                .filter(|(_, group_id)| {
                    if group_id.is_none() {
                        return false;
                    }
                    let group_id = group_id.unwrap();
                    group_id == i
                })
                .map(|(frame_id, _group_id)| frame_id)
                .collect();

            let local_times = mapping_frames
                .iter()
                .map(|frame_id| self.retention_times[*frame_id].unwrap())
                .scan(0.0, |acc, x| {
                    let out = x - *acc;
                    *acc = x;
                    Some(out)
                })
                .collect::<Vec<_>>();

            let cycle_time = local_times.iter().sum::<f32>() / local_times.len() as f32;
            group_cycle_times.push(cycle_time);
        }

        debug!("Group cycle times: {:?}", group_cycle_times);
        let avg_cycle_time = group_cycle_times.iter().sum::<f32>() / group_cycle_times.len() as f32;
        avg_cycle_time
    }

    pub fn split_frame(&self, frame: Frame) -> Result<Vec<FrameQuadWindow>, &'static str> {
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
            let mz_indptr_start = scan_offsets_use[0];
            let mz_indptr_end = *scan_offsets_use.last().unwrap();

            let tof_indices_keep = frame.tof_indices[mz_indptr_start..mz_indptr_end].to_vec();
            let intensities_keep = frame.intensities[mz_indptr_start..mz_indptr_end].to_vec();

            let frame_window = FrameQuadWindow {
                scan_offsets: scan_offsets_use
                    .iter()
                    .map(|x| (x - scan_start) as u64)
                    .collect::<Vec<_>>(),
                tof_indices: tof_indices_keep,
                intensities: intensities_keep,
                index: frame.index,
                rt: frame.rt,
                frame_type: frame.frame_type,
                scan_start: scan_range.scan_start,
                group_id: group.id,
                quad_group_id: i,
            };

            out_frames.push(frame_window);
        }

        Ok(out_frames)
    }


    pub fn split_frames() {

    }

    pub fn split_dense_frame(&self, mut denseframe: DenseFrame) -> Vec<DenseFrameWindow> {
        let group = self.get_dia_frame_window_group(denseframe.index).expect("Frame not in DIA group");

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
                index: denseframe.index,
                rt: denseframe.rt,
                frame_type: denseframe.frame_type,
                sorted: denseframe.sorted,
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

        frames
    }

    /// Returns a vector of length equal to the number of groups.
    /// Each element is a vector of frames that belong to that group.
    fn bundle_by_group(&self, frames: Vec<DenseFrame>) -> Vec<Vec<DenseFrame>> {
        let mut frame_groups = Vec::new();
        for frame in frames {
            let group = self.get_dia_frame_window_group(frame.index);
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
                for frame_window in frame_windows {
                    out[i][frame_window.quad_group_id].push(frame_window);
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


#[derive(Clone, FromRow, Debug)]
pub struct DiaFrameMsMsWindowInfo {
    pub window_group: i32,
    pub scan_num_begin: i32,
    pub scan_num_end: i32,
    pub isolation_mz: f32,
    pub isolation_width: f32,
    pub collision_energy: f32,
}

impl DiaFrameMsMsWindowInfo {
    fn into_scan_range(&self, id: usize, scan_converter: &timsrust::Scan2ImConverter) -> ScanRange {
        ScanRange::new(
            id,
            self.scan_num_begin as usize,
            self.scan_num_end as usize,
            self.isolation_mz,
            self.isolation_width,
            self.collision_energy,
            scan_converter,
        )
    }
}

struct FrameInfoBuilder {
    pub tdf_path: String,
    pub scan_converter: timsrust::Scan2ImConverter,
}

impl FrameInfoBuilder {
    pub fn from_dotd_path(dotd_path: String) -> Self {
        let reader = timsrust::FileReader::new(dotd_path.clone()).unwrap();
        let scan_converter = reader.get_scan_converter().unwrap();

        // Find an 'analysis.tdf' file inside the dotd file (directory).
        let tdf_path = Path::new(dotd_path.as_str()).join("analysis.tdf").into_os_string().into_string().unwrap();
        info!("tdf_path: {:?}", tdf_path);
        Self { tdf_path, scan_converter }
    }

    pub fn build(&self) -> Result<DIAFrameInfo, sqlx::Error> {
        let mut rt = Runtime::new().unwrap();

        rt.block_on(async {
            self.build_async().await
        })
    }

    async fn build_async(&self) -> Result<DIAFrameInfo, sqlx::Error> {
        let db =  SqlitePool::connect(&self.tdf_path).await?;

        // This vec maps frame_id -> window_group_id
        let frame_info = self.get_frame_mapping(&db).await?;

        // This vec maps window_group_id -> Vec<ScanRange>
        // And also returns the grouping level.
        let (group_mapping, grouping_level) = self.get_frame_windows(&db).await?;

        let max_window_id = group_mapping.len() - 1;

        let mut groups_vec_o = (0..(max_window_id + 1)).map(|_| None).collect::<Vec<_>>();
        for (i, scan_ranges) in group_mapping.into_iter().enumerate() {
            let scan_ranges = match scan_ranges {
                None => continue,
                Some(scan_ranges) => scan_ranges,
            };
            if scan_ranges.is_empty() {
                continue;
            } else {
                groups_vec_o[i] = Some(DIAWindowGroup { id: i, scan_ranges });
            }
        }

        let frame_info = DIAFrameInfo {
            groups: groups_vec_o,
            frame_groups: frame_info,
            retention_times: DIAFrameInfo::rts_from_tdf_connection(&db).await?,
            grouping_level,
        };

        Ok(frame_info)

    }

    async fn get_frame_mapping(&self, db: &Pool<Sqlite>) -> Result<Vec<Option<usize>>, sqlx::Error>{
        let result: Vec<(i32, i32)> = sqlx::query_as(
            "SELECT Frame, WindowGroup FROM DiaFrameMsMsInfo;",
        )
        .fetch_all(db).await?;

        let frame_info = result.iter().map(|(id, group)| (*id as usize, *group as usize)).collect::<Vec<(usize, usize)>>();

        let max_id = frame_info.iter().map(|(id, _)| id).max().unwrap();
        let mut ids_map_vec = vec![None; max_id + 1];
        for (id, group) in frame_info {
            ids_map_vec[id] = Some(group);
        }

        Ok(ids_map_vec)
    }

    async fn get_frame_windows(&self, db: &Pool<Sqlite>) -> Result<(Vec<Option<Vec<ScanRange>>>, GroupingLevel), sqlx::Error> {
        let result: Vec<DiaFrameMsMsWindowInfo> = sqlx::query_as::<_, DiaFrameMsMsWindowInfo>(
            "SELECT
                WindowGroup,
                ScanNumBegin,
                ScanNumEnd,
                IsolationMz,
                IsolationWidth,
                CollisionEnergy
            FROM DiaFrameMsMsWindows",
        )
        .fetch_all(db).await.unwrap();

        let grouping_level = if result.len() > 200 {
            log::info!("More than 200 scan ranges, using WindowGroup grouping level. (diagonal PASEF?)");
            GroupingLevel::WindowGroup
        } else {
            log::info!("More than 200 scan ranges, using WindowGroup grouping level. (diaPASEF?)");
            GroupingLevel::QuadWindowGroup
        };

        let max_window_id: usize = result
            .iter()
            .map(|window| window.window_group)
            .max()
            .unwrap() as usize;

        let mut group_map_vec: Vec<Option<Vec<ScanRange>>> = vec![None; max_window_id + 1];

        let mut scangroup_id = 0;
        for window in result {
            // TODO this is maybe a good place to make the trouping ...
            // If its diapasef, the groups are quad+window groups.
            // If its diagonal, the groups are only window groups.
            let usize_wg = window.window_group as usize;
            if group_map_vec[usize_wg].is_none() {
                group_map_vec[usize_wg] = Some(Vec::new());
            }

            match &mut group_map_vec[usize_wg] {
                None => continue,
                Some(scan_ranges) => {
                    scan_ranges.push(window.into_scan_range(scangroup_id.clone(), &self.scan_converter));
                    scangroup_id += 1;
                }
            }
        }
        Ok((group_map_vec, grouping_level))
    }

}

// TODO refactor this to make it a constructor method ...
pub fn read_dia_frame_info(dotd_file: String) -> Result<DIAFrameInfo, sqlx::Error> {
    let builder = FrameInfoBuilder::from_dotd_path(dotd_file);
    builder.build()
}
