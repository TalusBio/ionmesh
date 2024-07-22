use log::info;
use serde::Serialize;
use std::fmt;
use timsrust::{Frame, FrameType};

use crate::{
    space::space_generics::{
        convert_to_bounds_query, AsNDPointsAtIndex, IntenseAtIndex, NDBoundary, NDPoint,
        QueriableIndexedPoints,
    },
    utils::binary_search_slice,
};

use super::FrameMsMsWindowInfo;

#[derive(Debug, Clone, Copy)]
pub enum ScanNumberType {
    Global(usize),
    Local(usize),
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MaxBounds {
    requested: usize,
    limit: usize,
    local_limit: usize,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ScanOutOfBoundsError {
    Global(MaxBounds),
    Local(MaxBounds),
}

impl ScanOutOfBoundsError {
    pub fn local_limit(&self) -> usize {
        match self {
            ScanOutOfBoundsError::Global(x) => x.local_limit,
            ScanOutOfBoundsError::Local(x) => x.local_limit,
        }
    }
}

impl fmt::Display for ScanOutOfBoundsError {
    fn fmt(
        &self,
        f: &mut std::fmt::Formatter,
    ) -> std::fmt::Result {
        match self {
            ScanOutOfBoundsError::Global(x) => {
                write!(
                    f,
                    "Global scan number out of bounds. Requested: {}, Limit: {}",
                    x.requested, x.limit
                )
            },
            ScanOutOfBoundsError::Local(x) => {
                write!(
                    f,
                    "Local scan number out of bounds. Requested: {}, Limit: {}",
                    x.requested, x.limit
                )
            },
        }
    }
}

/// Unprocessed data from a 'Frame' after breaking by quad isolation_window + ims window.
///
/// 1. every tof-index + intensity represents a peak.
/// 2. Scan offsets are monotonically increasing.
/// 3. Peaks are arranged in increasing m/z order WITHIN a scan.
/// 4. Getting the peaks for scan #x in the frame_slice is done by subsetting
///    the tof indices and intensities, and subtracting the offset of the first
///    scan.
///     - scan_1_intensities = intensities[scan_offsets[1]-scan_offsets[0]:scan_offsets[2]-scan_offsets[0]]
///     - scan_x_intensities = intensities[scan_offsets[x]-scan_offsets[0]:scan_offsets[x+1]-scan_offsets[0]]
///     - NOTE: to get the peaks in the scan #y IN THE FRAME (not the frame slice)
///       you need to add to subtract the scan_start from the scan number.
///         - scan_y_intensities = intensities[scan_offsets[y-scan_start]-scan_offsets[0]:scan_offsets[y-scan_start+1]-scan_offsets[0]]
///         - Then obviously, scans < scan_start are not in the frame slice.
/// 5. The m/z values are a function of the tof indices (the measured m/z
///    of tof index `x` will be the same within the same run/instrument
///    calibration)
///
/// Frame                  Example values
///    - Scan offsets.    `[0,0,0,0,0,3,5,6 ...]` n=number of scans
///    - tof indices.     `[100, 101, 102, 10, 20, 30 ...]` len = len(intensities)
///    - intensities.     `[123, 111, 12 ,  3,  4,  1 ...]` len = len(tof indices)
///    - rt               65.34
///
/// Renamed from the frame:
///    - parent_frame_index   34 // renamed from Frame.index for clarity.
///
/// Additions for FrameSlice:
///    - scan_start       123  // The scan number of the first scan offset in the current window.
///    - slice_window_info Some(MsMsFrameSliceWindowInfo::SingleWindow(FrameMsMsWindow))
#[derive(Debug, Clone, Serialize)]
pub struct FrameSlice<'a> {
    // pub scan_offsets: &'a [usize], // Timsrust changed this later ...
    pub scan_offsets: &'a [u64],
    pub tof_indices: &'a [u32],
    pub intensities: &'a [u32],
    pub parent_frame_index: usize,
    pub rt: f64,

    #[serde(skip)]
    pub frame_type: FrameType,

    // From this point on they are local implementations
    // Before they are used from the timsrust crate.
    pub scan_start: usize,
    pub slice_window_info: Option<MsMsFrameSliceWindowInfo>,
}

#[derive(Debug, Clone, Serialize)]
pub struct ExpandedFrameSlice {
    pub scan_numbers: Vec<usize>,
    pub tof_indices: Vec<u32>,
    pub intensities: Vec<u32>,
    pub parent_frame_index: usize,
    pub rt: f64,
    pub slice_window_info: Option<MsMsFrameSliceWindowInfo>,

    #[serde(skip)]
    pub frame_type: FrameType,
}

impl<'a> FrameSlice<'a> {
    pub fn slice_frame(
        frame: &'a Frame,
        scan_start: usize,
        scan_end: usize,
        slice_window_info: Option<MsMsFrameSliceWindowInfo>,
    ) -> FrameSlice<'a> {
        let scan_offsets = &frame.scan_offsets[scan_start..=scan_end];

        let indprt_start = scan_offsets[0] as usize;
        let indptr_end = *scan_offsets.last().expect("Scan range is empty") as usize;

        let tof_indices = &frame.tof_indices[indprt_start..indptr_end];
        let intensities = &frame.intensities[indprt_start..indptr_end];
        debug_assert!(tof_indices.len() == intensities.len());
        debug_assert!(indptr_end - indprt_start == tof_indices.len());
        #[cfg(debug_assertions)]
        {
            let init_offset = scan_offsets[0];
            for i in 1..(scan_offsets.len() - 1) {
                debug_assert!(scan_offsets[i] <= scan_offsets[i + 1]);
                debug_assert!(
                    (scan_offsets[i + 1] - init_offset) <= tof_indices.len() as u64,
                    "scan_offsets[i+1]: {}, init_offset: {}, tof_indices.len(): {}",
                    scan_offsets[i + 1],
                    init_offset,
                    tof_indices.len()
                );
            }
        }

        FrameSlice {
            scan_offsets,
            tof_indices,
            intensities,
            parent_frame_index: frame.index,
            rt: frame.rt,
            frame_type: frame.frame_type,
            scan_start,
            slice_window_info,
        }
    }

    /// Get the global scan number at the local index.
    ///
    /// This means that ... provided the index of a tof index in the frame slice,
    /// this function will return the global scan number that tof index would belong
    /// to... in other words, "what is the scan number in the parent frame where peak
    /// number `x` in the frame slice would be found in the parent frame?"
    pub fn global_scan_at_index(
        &self,
        local_index: usize,
    ) -> usize {
        debug_assert!(local_index < self.tof_indices.len());
        let search_val = self.scan_offsets[0] + local_index as u64;
        let loc = self
            .scan_offsets
            .binary_search_by(|x| x.partial_cmp(&search_val).unwrap());

        let local_scan_index = match loc {
            Ok(mut x) => {
                while self.scan_offsets[x] == search_val {
                    x += 1;
                }
                x - 1
            },
            Err(x) => x - 1,
        };
        self.scan_start + local_scan_index
    }

    pub fn explode_scan_numbers(&self) -> Vec<usize> {
        let mut scan_numbers = Vec::with_capacity(self.tof_indices.len());
        let curr_scan = self.scan_start;

        for (scan_index, index_offsets) in self.scan_offsets.windows(2).enumerate() {
            let num_tofs = index_offsets[1] - index_offsets[0];
            scan_numbers.extend(vec![curr_scan + scan_index; num_tofs as usize]);
        }

        if cfg!(debug_assertions) {
            // Check that all are monotonically increasing with min == scan_start
            let mut last_scan = 0;
            for scan in scan_numbers.iter() {
                debug_assert!(*scan >= last_scan);
                last_scan = *scan;
            }

            // debug_assert_eq!(scan_numbers[0], self.scan_start);
            debug_assert!(scan_numbers[0] >= self.scan_start);
            debug_assert!(scan_numbers.len() == self.tof_indices.len());
            debug_assert!(
                scan_numbers.last().unwrap() <= &(self.scan_offsets.len() - 1 + self.scan_start)
            );
        }
        scan_numbers
    }

    /// Get the tof indices and intensities at a scan number.
    ///
    /// Returns a tuple.
    /// The first element is another tuple of the tof indices and intensities at the scan number.
    /// The second element is the offset of the first local tof index in the scan.
    ///
    pub fn tof_intensities_at_scan(
        &self,
        scan_number: ScanNumberType,
    ) -> Result<((&[u32], &[u32]), usize), ScanOutOfBoundsError> {
        let local_scan_number = self.scan_number_to_local(scan_number)?;
        Ok(self.tof_intensities_at_local_scan(local_scan_number))
    }

    pub fn scan_number_to_local(
        &self,
        scan_number: ScanNumberType,
    ) -> Result<usize, ScanOutOfBoundsError> {
        match scan_number {
            ScanNumberType::Global(x) => {
                if x < self.scan_start {
                    Err(ScanOutOfBoundsError::Global(MaxBounds {
                        requested: x,
                        limit: self.scan_start,
                        local_limit: 0,
                    }))
                } else if x >= self.scan_start + self.scan_offsets.len() {
                    Err(ScanOutOfBoundsError::Global(MaxBounds {
                        requested: x,
                        limit: self.scan_start + self.scan_offsets.len(),
                        local_limit: self.scan_offsets.len() - 1,
                    }))
                } else {
                    Ok(x - self.scan_start)
                }
            },
            ScanNumberType::Local(x) => {
                if x >= self.scan_offsets.len() {
                    Err(ScanOutOfBoundsError::Local(MaxBounds {
                        requested: x,
                        limit: self.scan_offsets.len(),
                        local_limit: self.scan_offsets.len() - 1,
                    }))
                } else {
                    Ok(x)
                }
            },
        }
    }

    fn tof_intensities_at_local_scan(
        &self,
        scan_index: usize,
    ) -> ((&[u32], &[u32]), usize) {
        let offset_offset = self.scan_offsets[0];
        let scan_start = (self.scan_offsets[scan_index] - offset_offset) as usize;
        let scan_end = (self.scan_offsets[scan_index + 1] - offset_offset) as usize;
        let tof_indices = &self.tof_indices[scan_start..scan_end];
        let intensities = &self.intensities[scan_start..scan_end];
        ((tof_indices, intensities), scan_start)
    }

    pub fn tof_range_in_tolerance_at_scan<T>(
        &self,
        tof_index: i32,
        scan_number: ScanNumberType,
        tolerance: &T,
    ) -> Result<Option<Range>, ScanOutOfBoundsError>
    where
        T: FramePointTolerance,
    {
        // TODO implement later a two pointer approach for sorted slices of tof indices.
        let ((tof_indices, _), local_tof_index_start) =
            self.tof_intensities_at_scan(scan_number)?;
        let tof_len = tof_indices.len();
        let (start, end) = tolerance.tof_index_range(tof_index as u32);
        let tof_index_start = tof_indices.binary_search_by(|x| x.partial_cmp(&start).unwrap());
        let tof_index_end = tof_indices.binary_search_by(|x| x.partial_cmp(&end).unwrap());
        let tof_index_start = match tof_index_start {
            Ok(mut x) => {
                while x > 0 && tof_indices[x - 1] >= start {
                    x -= 1;
                }
                x
            },
            Err(x) => x,
        };

        if tof_index_start >= tof_len {
            return Ok(None);
        };

        let tof_index_end = match tof_index_end {
            Ok(x) => x, // On this branch we dont add more bc tof indices are unique.
            Err(mut x) => {
                while x < tof_len && tof_indices[x] < end {
                    println!("tof_indices[x]: {}, x: {}", tof_indices[x], x);
                    x += 1;
                }
                x
            },
        };

        if tof_index_end > tof_index_start {
            Ok(Some((
                tof_index_start + local_tof_index_start,
                tof_index_end + local_tof_index_start,
            )))
        } else {
            Ok(None)
        }
    }

    pub fn matching_rangeset<T>(
        &self,
        tof_index: i32,
        scan_number: ScanNumberType,
        tolerance: &T,
    ) -> Option<RangeSet>
    where
        T: FramePointTolerance,
    {
        let mut ranges = RangeSet {
            ranges: Vec::new(),
            offset: 0,
        };

        let scan_range = tolerance.scan_range(scan_number);
        let local_start = match self.scan_number_to_local(scan_range.0) {
            Ok(x) => x,
            Err(x) => x.local_limit(),
        };
        let local_end = match self.scan_number_to_local(scan_range.1) {
            Ok(x) => x,
            Err(x) => x.local_limit(),
        };

        for scan_number in local_start..local_end {
            let tmp = self.tof_range_in_tolerance_at_scan(
                tof_index,
                ScanNumberType::Local(scan_number),
                tolerance,
            );

            match tmp {
                Ok(Some(range_offset)) => {
                    ranges.ranges.push(range_offset);
                },
                _ => (),
            }
        }

        if cfg!(debug_assertions) {
            debug_assert!(!ranges.any_overlap());
        }

        if ranges.ranges.len() == 0 {
            None
        } else {
            Some(ranges)
        }
    }

    pub fn tof_int_at_index(
        &self,
        index: usize,
    ) -> (u32, u32) {
        (self.tof_indices[index], self.intensities[index])
    }
}

impl ExpandedFrameSlice {
    pub fn from_frame_slice(frame_slice: &FrameSlice) -> ExpandedFrameSlice {
        let parent_frame_index = frame_slice.parent_frame_index;
        let rt = frame_slice.rt;
        let slice_window_info = frame_slice.slice_window_info.clone();
        let frame_type = frame_slice.frame_type;
        let scan_numbers = frame_slice.explode_scan_numbers();

        // Sort all arrays on the tof indices.
        let mut zipped = frame_slice
            .tof_indices
            .iter()
            .zip(frame_slice.intensities.iter())
            .zip(scan_numbers.iter())
            .collect::<Vec<_>>();

        zipped.sort_unstable_by(|a, b| a.0 .0.cmp(&b.0 .0));

        let (tof_indices, intensities, scan_numbers) = zipped.into_iter().fold(
            (Vec::new(), Vec::new(), Vec::new()),
            |(mut tof_indices, mut intensities, mut scan_numbers),
             ((tof_index, intensity), scan_number)| {
                tof_indices.push(*tof_index);
                intensities.push(*intensity);
                scan_numbers.push(*scan_number);
                (tof_indices, intensities, scan_numbers)
            },
        );

        ExpandedFrameSlice {
            scan_numbers,
            tof_indices,
            intensities,
            parent_frame_index,
            rt,
            slice_window_info,
            frame_type,
        }
    }
}

// Tests for the FrameSlice
#[cfg(test)]
mod tests {
    use super::*;

    fn sample_frame() -> Frame {
        Frame {
            index: 0,
            scan_offsets: vec![0, 0, 0, 0, 0, 3, 5, 6],
            tof_indices: vec![100, 101, 102, 10, 20, 30],
            intensities: vec![123, 111, 12, 3, 4, 1],
            rt: 65.34,
            frame_type: FrameType::MS1,
        }
    }

    #[test]
    fn test_frame_slice() {
        let frame = sample_frame();
        let frame_slice = FrameSlice::slice_frame(&frame, 3, 5, None);

        assert_eq!(frame_slice.scan_offsets, &[0, 0, 3]);
        assert_eq!(frame_slice.tof_indices, &[100, 101, 102]);
        assert_eq!(frame_slice.intensities, &[123, 111, 12]);
        assert_eq!(frame_slice.parent_frame_index, 0);
        assert_eq!(frame_slice.rt, 65.34);
        assert_eq!(frame_slice.frame_type, FrameType::MS1);
        assert_eq!(frame_slice.scan_start, 3);
    }

    #[test]
    fn test_global_scan_at_index() {
        let frame = sample_frame();
        let frame_slice = FrameSlice::slice_frame(&frame, 3, 5, None);

        assert_eq!(frame_slice.tof_indices, &[100, 101, 102]);
        assert_eq!(frame_slice.global_scan_at_index(0), 4);
        assert_eq!(frame_slice.global_scan_at_index(1), 4);
        assert_eq!(frame_slice.global_scan_at_index(2), 4);
    }

    #[test]
    #[should_panic]
    fn test_global_scan_at_index_oob_fails() {
        // these should fail ... test that it fails.
        let frame = sample_frame();
        let frame_slice = FrameSlice::slice_frame(&frame, 3, 5, None);
        assert_eq!(frame_slice.tof_indices, &[100, 101, 102]);
        frame_slice.global_scan_at_index(3);
    }

    #[test]
    fn test_explode_scan_numbers() {
        let frame = sample_frame();
        let frame_slice = FrameSlice::slice_frame(&frame, 3, 5, None);
        assert_eq!(frame_slice.tof_indices, &[100, 101, 102]);
        assert_eq!(frame_slice.scan_offsets, &[0, 0, 3]);
        assert_eq!(frame_slice.explode_scan_numbers(), vec![4, 4, 4]);
    }

    #[test]
    fn test_tof_intensities_at_scan() {
        let frame = sample_frame();
        let frame_slice = FrameSlice::slice_frame(&frame, 3, 5, None);

        assert_eq!(frame_slice.tof_indices, &[100, 101, 102]);
        assert_eq!(frame_slice.scan_offsets, &[0, 0, 3]);
        assert_eq!(frame_slice.intensities, &[123, 111, 12]);

        let arg_expects = vec![
            (
                ScanNumberType::Global(4),
                Ok(((vec![100, 101, 102], vec![123, 111, 12]), 0)),
            ),
            (
                ScanNumberType::Local(1),
                Ok(((vec![100, 101, 102], vec![123, 111, 12]), 0)),
            ),
            (
                ScanNumberType::Global(2),
                Err(ScanOutOfBoundsError::Global(MaxBounds {
                    requested: 2,
                    limit: 3,
                    local_limit: 0,
                })),
            ),
            (ScanNumberType::Global(3), Ok(((Vec::new(), Vec::new()), 0))),
            (ScanNumberType::Local(0), Ok(((Vec::new(), Vec::new()), 0))),
        ];

        for (arg, expect) in arg_expects {
            println!("arg: {:?}", arg);
            let val = frame_slice.tof_intensities_at_scan(arg);
            match (val, expect) {
                (Ok(x), Ok(y)) => {
                    assert_eq!(x.0 .0, y.0 .0);
                    assert_eq!(x.0 .1, y.0 .1);
                    assert_eq!(x.1, y.1);
                },
                (Err(x), Err(y)) => {
                    assert_eq!(x, y);
                },
                (Ok(x), Err(y)) => panic!("Mismatch {:?} vs {:?}", x, y),
                (Err(x), Ok(y)) => panic!("Mismatch {:?} vs {:?}", x, y),
            }
        }
    }

    #[test]
    fn test_tof_range_in_tolerance_at_scan() {
        let frame = sample_frame();
        let frame_slice = FrameSlice::slice_frame(&frame, 3, 7, None);

        assert_eq!(frame_slice.tof_indices, &[100, 101, 102, 10, 20, 30]);
        assert_eq!(frame_slice.scan_offsets, &[0, 0, 3, 5, 6]);
        assert_eq!(frame_slice.intensities, &[123, 111, 12, 3, 4, 1]);

        let tolerance = AbsoluteFramePointTolerance {
            tof_index_tolerance: 1,
            scan_tolerance: 1,
        };

        let param_expect_vec = vec![
            (10, ScanNumberType::Global(5), Ok(Some((3, 4)))),
            (1, ScanNumberType::Global(5), Ok(None)),
            (10, ScanNumberType::Global(4), Ok(None)),
            (100, ScanNumberType::Global(4), Ok(Some((0, 1)))),
            (101, ScanNumberType::Global(4), Ok(Some((0, 2)))),
            (102, ScanNumberType::Global(4), Ok(Some((1, 3)))),
            (100, ScanNumberType::Global(3), Ok(None)),
            (100, ScanNumberType::Global(5), Ok(None)),
            (
                100,
                ScanNumberType::Global(2),
                Err(ScanOutOfBoundsError::Global(MaxBounds {
                    requested: 2,
                    limit: 3,
                    local_limit: 0,
                })),
            ),
            (
                100,
                ScanNumberType::Global(1),
                Err(ScanOutOfBoundsError::Global(MaxBounds {
                    requested: 1,
                    limit: 3,
                    local_limit: 0,
                })),
            ),
            (
                100,
                ScanNumberType::Global(0),
                Err(ScanOutOfBoundsError::Global(MaxBounds {
                    requested: 0,
                    limit: 3,
                    local_limit: 0,
                })),
            ),
        ];
        for (tof_index, scan_number, expect) in param_expect_vec {
            println!("tof_index: {}, scan_number: {:?}", tof_index, scan_number);
            let val =
                frame_slice.tof_range_in_tolerance_at_scan(tof_index, scan_number, &tolerance);

            match (val, expect) {
                (Ok(Some(x)), Ok(Some(y))) => {
                    assert_eq!(x, y);
                },
                (Ok(None), Ok(None)) => (),
                (Err(x), Err(y)) => {
                    assert_eq!(x, y);
                },
                (Ok(x), Ok(None)) => panic!("Mismatch {:?} vs {:?}", x, expect),
                (Ok(None), Ok(x)) => panic!("Mismatch {:?} vs {:?}", x, expect),
                (Err(x), Ok(y)) => panic!("Mismatch {:?} vs {:?}", x, y),
                (Ok(x), Err(y)) => panic!("Mismatch {:?} vs {:?}", x, y),
            }
        }
    }

    fn sample_ms2_frame() -> Frame {
        Frame {
            index: 0,
            scan_offsets: vec![0, 0, 3, 5, 6],
            tof_indices: vec![100, 101, 102, 10, 20, 30],
            intensities: vec![123, 111, 12, 3, 4, 1],
            rt: 65.34,
            frame_type: FrameType::MS2(timsrust::AcquisitionType::DIAPASEF),
        }
    }
}

impl IntenseAtIndex for ExpandedFrameSlice {
    fn intensity_at_index(
        &self,
        index: usize,
    ) -> u64 {
        self.intensities[index] as u64
    }
    fn weight_at_index(
        &self,
        index: usize,
    ) -> u64 {
        self.intensities[index] as u64
    }

    fn intensity_index_length(&self) -> usize {
        self.intensities.len()
    }
}

impl<'a> IntenseAtIndex for FrameSlice<'a> {
    fn intensity_at_index(
        &self,
        index: usize,
    ) -> u64 {
        self.intensities[index] as u64
    }
    fn weight_at_index(
        &self,
        index: usize,
    ) -> u64 {
        self.intensities[index] as u64
    }

    fn intensity_index_length(&self) -> usize {
        self.intensities.len()
    }

    // fn get_intense_at_index(
    //     &self,
    //     index: usize,
    // ) -> RawTimsPeak {
    //     let intensity = self.intensities[index];
    //     let tof_index = self.tof_indices[index];
    //     let scan_index = self.global_scan_at_index(index);

    //     let out = RawTimsPeak {
    //         intensity,
    //         tof_index,
    //         scan_index,
    //     };

    //     out
    // }
}

impl AsNDPointsAtIndex<2> for ExpandedFrameSlice {
    fn get_ndpoint(
        &self,
        index: usize,
    ) -> NDPoint<2> {
        let scan_index = self.scan_numbers[index];
        let tof_index = self.tof_indices[index];

        NDPoint {
            values: [tof_index as f32, scan_index as f32],
        }
    }

    fn num_ndpoints(&self) -> usize {
        self.intensities.len()
    }
}

impl<'a> AsNDPointsAtIndex<2> for FrameSlice<'a> {
    fn get_ndpoint(
        &self,
        index: usize,
    ) -> NDPoint<2> {
        let tof_index = self.tof_indices[index];
        let scan_index = self.global_scan_at_index(index);

        NDPoint {
            values: [tof_index as f32, scan_index as f32],
        }
    }

    fn num_ndpoints(&self) -> usize {
        self.intensities.len()
    }
}

impl QueriableIndexedPoints<2> for ExpandedFrameSlice {
    fn query_ndpoint(
        &self,
        point: &NDPoint<2>,
    ) -> Vec<usize> {
        let query = convert_to_bounds_query(point);
        self.query_ndrange(&query.0, query.1)
    }

    fn query_ndrange(
        &self,
        boundary: &NDBoundary<2>,
        reference_point: Option<&NDPoint<2>>,
    ) -> Vec<usize> {
        // TODO implement passing information on the mz tolerance ...
        // info!("Querying frame slice with boundary: {:?}", boundary);
        // let tol = AbsoluteFramePointTolerance {
        //     tof_index_tolerance: (boundary.widths[0] / 2.) as u32,
        //     scan_tolerance: (boundary.widths[1] / 2.) as usize,
        // };
        const SCAN_NUMBER_TOLERANCE: usize = 10;
        let scan_left = (boundary.starts[1] as usize).saturating_sub(SCAN_NUMBER_TOLERANCE);
        let scan_right = (boundary.ends[1] as usize).saturating_add(SCAN_NUMBER_TOLERANCE);

        let (left, right) = binary_search_slice(
            &self.tof_indices,
            |a, b| a.cmp(b),
            boundary.starts[0] as u32,
            boundary.ends[0] as u32,
        );
        let mut out = Vec::new();
        for i in left..right {
            let scan_i = self.scan_numbers[i];
            if scan_i >= scan_left && scan_i <= scan_right {
                out.push(i);
            }
        }
        out
    }
}

impl<'a> QueriableIndexedPoints<2> for FrameSlice<'a> {
    fn query_ndpoint(
        &self,
        point: &NDPoint<2>,
    ) -> Vec<usize> {
        let tof_index = point.values[0] as i32;
        let scan_index = point.values[1] as usize;
        let rangesets = self.matching_rangeset(
            tof_index,
            ScanNumberType::Global(scan_index),
            &AbsoluteFramePointTolerance {
                tof_index_tolerance: 2,
                scan_tolerance: 5,
            },
        );

        let mut out = Vec::new();
        if let Some(rangesets) = rangesets {
            for range in rangesets.ranges.iter() {
                for i in range.0..range.1 {
                    out.push(i);
                }
            }
        }
        out
    }

    fn query_ndrange(
        &self,
        boundary: &NDBoundary<2>,
        reference_point: Option<&NDPoint<2>>,
    ) -> Vec<usize> {
        // TODO implement passing information on the mz tolerance ...
        // info!("Querying frame slice with boundary: {:?}", boundary);
        // let tol = AbsoluteFramePointTolerance {
        //     tof_index_tolerance: (boundary.widths[0] / 2.) as u32,
        //     scan_tolerance: (boundary.widths[1] / 2.) as usize,
        // };
        let tol = AbsoluteFramePointTolerance {
            tof_index_tolerance: (boundary.widths[0] * 2.) as u32,
            scan_tolerance: (boundary.widths[1] * 10.) as usize,
        };
        let rangesets = self.matching_rangeset(
            boundary.centers[0] as i32,
            ScanNumberType::Global(boundary.centers[1] as usize),
            &tol,
        );

        let mut out = Vec::new();
        if let Some(rangesets) = rangesets {
            for range in rangesets.ranges.iter() {
                for i in range.0..range.1 {
                    out.push(i);
                }
            }
        }
        out
    }
}

pub trait FramePointTolerance {
    fn tof_index_range(
        &self,
        tof_index: u32,
    ) -> (u32, u32);
    fn scan_range(
        &self,
        scan_index: ScanNumberType,
    ) -> (ScanNumberType, ScanNumberType);
}

struct AbsoluteFramePointTolerance {
    tof_index_tolerance: u32,
    scan_tolerance: usize,
}

impl FramePointTolerance for AbsoluteFramePointTolerance {
    fn tof_index_range(
        &self,
        tof_index: u32,
    ) -> (u32, u32) {
        let tof_index_tolerance = self.tof_index_tolerance;
        (
            tof_index.saturating_sub(tof_index_tolerance),
            tof_index.saturating_add(tof_index_tolerance),
        )
    }

    fn scan_range(
        &self,
        scan_index: ScanNumberType,
    ) -> (ScanNumberType, ScanNumberType) {
        match scan_index {
            ScanNumberType::Global(x) => {
                let scan_tolerance = self.scan_tolerance;
                (
                    ScanNumberType::Global(x.saturating_sub(scan_tolerance)),
                    ScanNumberType::Global(x + scan_tolerance),
                )
            },
            ScanNumberType::Local(x) => {
                let scan_tolerance = self.scan_tolerance;
                (
                    ScanNumberType::Local(x.saturating_sub(scan_tolerance)),
                    ScanNumberType::Local(x + scan_tolerance),
                )
            },
        }
    }
}

type Range = (usize, usize);

pub struct RangeSet {
    ranges: Vec<Range>,
    offset: usize,
}

impl RangeSet {
    fn extend(
        &mut self,
        other: RangeSet,
    ) {
        let new_offset = self.offset.min(other.offset);
        let vs_self_offset = self.offset - new_offset;
        let vs_other_offset = other.offset - new_offset;

        for item in self.ranges.iter_mut() {
            item.0 += vs_self_offset;
            item.1 += vs_self_offset;
        }

        for item in other.ranges.iter() {
            self.ranges
                .push((item.0 + vs_other_offset, item.1 + vs_other_offset));
        }

        self.ranges.sort_unstable_by(|a, b| a.0.cmp(&b.0));
    }

    fn any_overlap(&self) -> bool {
        let mut last_end = 0;

        for range in self.ranges.iter() {
            if range.0 < last_end {
                return true;
            }
            last_end = range.1;
        }
        false
    }
}

#[derive(Debug, Clone, Serialize)]
pub enum MsMsFrameSliceWindowInfo {
    WindowGroup(usize),
    SingleWindow(FrameMsMsWindowInfo),
}
