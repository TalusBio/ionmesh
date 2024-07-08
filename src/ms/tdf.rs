use log::{debug, info};

use sqlx::Pool;
use sqlx::{FromRow, Sqlite, SqlitePool};
use std::path::Path;
use timsrust::{ConvertableIndex, Frame};
use tokio;
use tokio::runtime::Runtime;

use crate::ms::frames::{FrameMsMsWindowInfo, FrameSlice, MsMsFrameSliceWindowInfo};

// Diaframemsmsinfo = vec of frame_id -> windowgroup_id
// diaframemsmswindows = vec[(windowgroup_id, scanstart, scanend, iso_mz, iso_with, nce)]

#[derive(Debug, Clone)]
pub struct ScanRange {
    pub row_id: usize,
    pub scan_start: usize,
    pub scan_end: usize,
    pub iso_mz: f32,
    pub iso_width: f32,
    pub nce: f32,
    pub ims_start: f32,
    pub ims_end: f32,
    pub iso_low: f32,
    pub iso_high: f32,
    pub window_group_id: usize,
    pub within_window_quad_group_id: usize,
}

impl ScanRange {
    pub fn new(
        row_id: usize,
        window_group_id: usize,
        within_window_quad_group_id: usize,
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
            row_id,
            scan_start,
            scan_end,
            iso_mz,
            iso_width,
            nce,
            ims_start: ims_start as f32,
            ims_end: ims_end as f32,
            iso_low,
            iso_high,
            window_group_id,
            within_window_quad_group_id,
        }
    }
}

impl From<ScanRange> for FrameMsMsWindowInfo {
    fn from(val: ScanRange) -> Self {
        FrameMsMsWindowInfo {
            mz_start: val.iso_low,
            mz_end: val.iso_high,
            window_group_id: val.window_group_id,
            within_window_quad_group_id: val.within_window_quad_group_id,
            global_quad_row_id: val.row_id,
        }
    }
}

#[derive(Debug, Clone)]
pub struct DIAWindowGroup {
    pub window_group_id: usize,
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
    pub number_of_groups: usize,

    /// The row to group is meant to map the `Isolation window row id`
    /// to the grouping level it will have... for diaPASEF, since every
    /// scan range has a different quand window, the number of distinct
    /// groups is the number of scan ranges (window groups+scan range
    /// combinations). For the case of diagonal PASEF, the number of
    /// groups is the number of window groups, since the scan ranges
    /// are not independent from each other.
    pub row_to_group: Vec<usize>,
}

// TODO rename or split this ... since it is becoming more
// of a splitter than a frame info reader.
// Maybe a builder -> splitter pattern?
impl DIAFrameInfo {
    pub fn get_dia_frame_window_group(
        &self,
        frame_id: usize,
    ) -> Option<&DIAWindowGroup> {
        let group_id = self.frame_groups[frame_id];
        match group_id {
            None => None,
            Some(group_id) => self.groups[group_id].as_ref(),
        }
    }

    async fn rts_from_tdf_connection(conn: &Pool<Sqlite>) -> Result<Vec<Option<f32>>, sqlx::Error> {
        // To calculate cycle time ->
        // DiaFrameMsMsInfo -> Get the frames that match a specific id (one for each ...)
        // Frames -> SELECT id, time FROM Frames -> make a Vec<Option<f32>>, map the former
        // framer id list (no value should be None).
        // Scan diff the new vec!
        let results: Vec<(i32, f32)> = sqlx::query_as("SELECT Id, Time FROM Frames")
            .fetch_all(conn)
            .await?;
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

    pub fn split_frame<'a, 'b>(
        &'b self,
        frame: &'a Frame,
        window_group: &DIAWindowGroup,
    ) -> Result<Vec<FrameSlice>, &'static str>
    where
        'a: 'b,
    {
        let mut out_frames = Vec::new();
        for scan_range in window_group.scan_ranges.iter() {
            let slice_w_info: MsMsFrameSliceWindowInfo =
                MsMsFrameSliceWindowInfo::SingleWindow(scan_range.clone().into());
            let frame_slice = FrameSlice::slice_frame(
                frame,
                scan_range.scan_start,
                scan_range.scan_end,
                Some(slice_w_info),
            );
            out_frames.push(frame_slice);
        }

        Ok(out_frames)
    }

    pub fn split_frame_windows<'a>(
        &'a self,
        frames: &'a [Frame],
    ) -> Vec<Vec<FrameSlice>> {
        let mut out = Vec::new();

        match self.grouping_level {
            GroupingLevel::WindowGroup => {
                for _ in 0..(self.groups.len() + 1) {
                    out.push(Vec::new());
                }
            },
            GroupingLevel::QuadWindowGroup => {
                for _ in 0..(self.row_to_group.len() + 1) {
                    out.push(Vec::new());
                }
            },
        }

        for frame in frames {
            let group = self
                .get_dia_frame_window_group(frame.index)
                .expect("Frame is not in MS2 frames");

            match self.grouping_level {
                GroupingLevel::WindowGroup => {
                    panic!("WindowGroup grouping level not implemented for splitting frames")
                    //out[group.id].push(frame_window);
                },
                GroupingLevel::QuadWindowGroup => {
                    let frame_windows = self
                        .split_frame(frame, group)
                        .expect("Error splitting frame");
                    for frame_window in frame_windows {
                        match &frame_window.slice_window_info {
                            None => {
                                panic!("Frame window has no slice window info")
                            },
                            Some(MsMsFrameSliceWindowInfo::SingleWindow(scan_range)) => {
                                out[scan_range.global_quad_row_id].push(frame_window);
                            },
                            Some(MsMsFrameSliceWindowInfo::WindowGroup(group)) => {
                                out[*group].push(frame_window);
                            },
                        }
                    }
                },
            }
        }

        // Sort by ascending rt
        for group in out.iter_mut() {
            group.sort_by(|a, b| a.rt.partial_cmp(&b.rt).unwrap());
        }

        // Debug assert that the frames are sorted by rt
        if cfg!(debug_assertions) {
            for group in out.iter() {
                for i in 0..(group.len() - 1) {
                    debug_assert!(group[i].rt <= group[i + 1].rt);
                }
            }
        }

        out
    }

    pub fn get_quad_windows(
        &self,
        scan_group_id: usize,
        quad_group_id: usize,
    ) -> Option<&ScanRange> {
        let group = self.groups[scan_group_id].as_ref();
        let group = match group {
            None => {
                panic!(
                    "Group not found for scan group id: {}, in groups n={}",
                    scan_group_id,
                    self.groups.len()
                )
            },
            Some(group) => group,
        };

        let quad_group = group.scan_ranges.get(quad_group_id);
        let quad_group = match quad_group {
            None => {
                panic!(
                    "Quad group not found for quad group id: {}, in scan_ranges {:?}",
                    quad_group_id, group.scan_ranges
                )
            },
            Some(quad_group) => quad_group,
        };

        Some(quad_group)
    }
}

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
    #[sqlx(rename = "WindowGroup")]
    pub window_group: i32,
    #[sqlx(rename = "ScanNumBegin")]
    pub scan_num_begin: i32,
    #[sqlx(rename = "ScanNumEnd")]
    pub scan_num_end: i32,
    #[sqlx(rename = "IsolationMz")]
    pub isolation_mz: f32,
    #[sqlx(rename = "IsolationWidth")]
    pub isolation_width: f32,
    #[sqlx(rename = "CollisionEnergy")]
    pub collision_energy: f32,
}

impl DiaFrameMsMsWindowInfo {
    fn into_scan_range(
        &self,
        id: usize,
        quad_id: usize,
        scan_converter: &timsrust::Scan2ImConverter,
    ) -> ScanRange {
        ScanRange::new(
            id,
            self.window_group as usize,
            quad_id,
            self.scan_num_begin as usize,
            self.scan_num_end as usize,
            self.isolation_mz,
            self.isolation_width,
            self.collision_energy,
            scan_converter,
        )
    }
}

#[derive(Debug)]
pub struct FrameInfoBuilder {
    pub tdf_path: String,
    pub scan_converter: timsrust::Scan2ImConverter,
}

impl FrameInfoBuilder {
    pub fn from_dotd_path(dotd_path: String) -> Self {
        let reader = timsrust::FileReader::new(dotd_path.clone()).unwrap();
        let scan_converter = reader.get_scan_converter().unwrap();

        // Find an 'analysis.tdf' file inside the dotd file (directory).
        let tdf_path = Path::new(dotd_path.as_str())
            .join("analysis.tdf")
            .into_os_string()
            .into_string()
            .unwrap();
        info!("tdf_path: {:?}", tdf_path);
        Self {
            tdf_path,
            scan_converter,
        }
    }

    pub fn build(&self) -> Result<DIAFrameInfo, sqlx::Error> {
        let rt = Runtime::new().unwrap();

        rt.block_on(async { self.build_async().await })
    }

    async fn build_async(&self) -> Result<DIAFrameInfo, sqlx::Error> {
        let db = SqlitePool::connect(&self.tdf_path).await?;

        // This vec maps frame_id -> window_group_id
        let frame_info = self.get_frame_mapping(&db).await?;

        // This vec maps window_group_id -> Vec<ScanRange>
        // And also returns the grouping level.
        let (group_mapping, grouping_level, row_to_group) = self.get_frame_windows(&db).await?;
        let number_of_groups = row_to_group.iter().max().unwrap() + 1;

        debug!("Number of groups: {}", number_of_groups);

        let max_window_id = group_mapping.len() - 1;

        let mut groups_vec_o = (0..(max_window_id + 1)).map(|_| None).collect::<Vec<_>>();
        for (i, scan_ranges) in group_mapping.into_iter().enumerate() {
            let scan_ranges = match scan_ranges {
                None => continue,
                Some(scan_ranges) => scan_ranges,
            };
            debug!("Scan ranges i={}: {:?}", i, scan_ranges);
            if cfg!(debug_assertions) {
                for scan_range in scan_ranges.iter() {
                    debug_assert!(scan_range.window_group_id == i)
                }
            };
            if scan_ranges.is_empty() {
                continue;
            } else {
                groups_vec_o[i] = Some(DIAWindowGroup {
                    window_group_id: i,
                    scan_ranges,
                });
            }
        }

        let frame_info = DIAFrameInfo {
            groups: groups_vec_o,
            frame_groups: frame_info,
            retention_times: DIAFrameInfo::rts_from_tdf_connection(&db).await?,
            grouping_level,
            number_of_groups,
            row_to_group,
        };

        Ok(frame_info)
    }

    async fn get_frame_mapping(
        &self,
        db: &Pool<Sqlite>,
    ) -> Result<Vec<Option<usize>>, sqlx::Error> {
        let result: Vec<(i32, i32)> =
            sqlx::query_as("SELECT Frame, WindowGroup FROM DiaFrameMsMsInfo;")
                .fetch_all(db)
                .await?;

        let frame_info = result
            .iter()
            .map(|(id, group)| (*id as usize, *group as usize))
            .collect::<Vec<(usize, usize)>>();

        let max_id = frame_info.iter().map(|(id, _)| id).max().unwrap();
        let mut ids_map_vec = vec![None; max_id + 1];
        for (id, group) in frame_info {
            ids_map_vec[id] = Some(group);
        }

        Ok(ids_map_vec)
    }

    async fn get_frame_windows(
        &self,
        db: &Pool<Sqlite>,
    ) -> Result<(Vec<Option<Vec<ScanRange>>>, GroupingLevel, Vec<usize>), sqlx::Error> {
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
        .fetch_all(db)
        .await
        .unwrap();

        let grouping_level = if result.len() > 200 {
            log::info!(
                "More than 200 scan ranges, using WindowGroup grouping level. (diagonal PASEF?)"
            );
            GroupingLevel::WindowGroup
        } else {
            log::info!("Less than 200 scan ranges detected, using QuadWindowGroup grouping level. (diaPASEF?)");
            GroupingLevel::QuadWindowGroup
        };

        let max_window_id: usize = result
            .iter()
            .map(|window| window.window_group)
            .max()
            .unwrap() as usize;

        let mut group_map_vec: Vec<Option<Vec<ScanRange>>> = vec![None; max_window_id + 1];

        let mut scangroup_id = 0;
        let mut row_to_group = Vec::new();
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
                    let quad_id = scan_ranges.len();
                    scan_ranges.push(window.into_scan_range(
                        scangroup_id,
                        quad_id,
                        &self.scan_converter,
                    ));
                    scangroup_id += 1;
                },
            }

            match grouping_level {
                GroupingLevel::WindowGroup => {
                    row_to_group.push(usize_wg);
                },
                GroupingLevel::QuadWindowGroup => {
                    row_to_group.push(scangroup_id);
                },
            }
        }
        Ok((group_map_vec, grouping_level, row_to_group))
    }
}

// TODO refactor this to make it a constructor method ...
pub fn read_dia_frame_info(dotd_file: String) -> Result<DIAFrameInfo, sqlx::Error> {
    let builder = FrameInfoBuilder::from_dotd_path(dotd_file);
    builder.build()
}
