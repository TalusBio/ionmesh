pub use timsrust::Frame;
pub use timsrust::FrameType;
pub use timsrust::{
    ConvertableIndex, FileReader, Frame2RtConverter, Scan2ImConverter, Tof2MzConverter,
};

use crate::ms::tdf::{DIAFrameInfo, ScanRange};
use crate::space::space_generics::HasIntensity;

use log::info;

#[derive(Debug, Clone, Copy)]
pub struct TimsPeak {
    pub intensity: u32,
    pub mz: f64,
    pub mobility: f32,
    pub npeaks: u32,
}

impl HasIntensity for TimsPeak {
    fn intensity(&self) -> u64 {
        self.intensity as u64
    }
}

#[derive(Debug, Clone, Copy)]
pub struct RawTimsPeak {
    pub intensity: u32,
    pub tof_index: u32,
    pub scan_index: usize,
}

fn _check_peak_sanity(peak: &TimsPeak) {
    debug_assert!(peak.intensity > 0);
    debug_assert!(peak.mz > 0.);
    debug_assert!(peak.mobility > 0.);
    debug_assert!(peak.npeaks > 0);
}

#[derive(Debug, Clone, Copy)]
pub enum SortingOrder {
    Mz,
    Mobility,
    Intensity,
}

/// Information on the context of a window in a frame.
///
/// This adds to a frame slice the context of the what isolation was used
/// to generate the frame slice.
#[derive(Debug, Clone)]
pub struct FrameMsMsWindowInfo {
    pub mz_start: f32,
    pub mz_end: f32,
    pub window_group_id: usize,
    pub within_window_quad_group_id: usize,
    pub global_quad_row_id: usize,
}

pub trait FramePointTolerance {
    fn tof_index_range(&self, tof_index: u32) -> (u32, u32);
    fn scan_range(&self, scan_index: usize) -> (usize, usize);
}

struct AbsoluteFramePointTolerance {
    tof_index_tolerance: u32,
    scan_tolerance: usize,
}

impl FramePointTolerance for AbsoluteFramePointTolerance {
    fn tof_index_range(&self, tof_index: u32) -> (u32, u32) {
        let tof_index_tolerance = self.tof_index_tolerance;
        (
            tof_index.saturating_sub(tof_index_tolerance),
            tof_index.saturating_add(tof_index_tolerance),
        )
    }

    fn scan_range(&self, scan_index: usize) -> (usize, usize) {
        let scan_tolerance = self.scan_tolerance;
        (
            scan_index.saturating_sub(scan_tolerance),
            scan_index + scan_tolerance,
        )
    }
}

type Range = (usize, usize);

pub struct RangeSet {
    ranges: Vec<Range>,
    offset: usize,
}

impl RangeSet {
    fn extend(&mut self, other: RangeSet) {
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

#[derive(Debug, Clone)]
pub enum MsMsFrameSliceWindowInfo {
    WindowGroup(usize),
    SingleWindow(FrameMsMsWindowInfo),
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
#[derive(Debug, Clone)]
pub struct FrameSlice<'a> {
    pub scan_offsets: &'a [usize],
    pub tof_indices: &'a [u32],
    pub intensities: &'a [u32],
    pub parent_frame_index: usize,
    pub rt: f64,
    pub frame_type: FrameType,

    // From this point on they are local implementations
    // Before they are used from the timsrust crate.
    pub scan_start: usize,
    pub slice_window_info: Option<MsMsFrameSliceWindowInfo>,
}

impl<'a> FrameSlice<'a> {
    pub fn slice_frame(
        frame: &'a Frame,
        scan_start: usize,
        scan_end: usize,
        slice_window_info: Option<MsMsFrameSliceWindowInfo>,
    ) -> FrameSlice<'a> {
        let scan_offsets = &frame.scan_offsets[scan_start..=scan_end];
        let scan_start = scan_offsets[0];

        let indprt_start = scan_offsets[0];
        let indptr_end = *scan_offsets.last().expect("Scan range is empty");

        let tof_indices = &frame.tof_indices[indprt_start..indptr_end];
        let intensities = &frame.intensities[indprt_start..indptr_end];
        debug_assert!(tof_indices.len() == intensities.len());
        debug_assert!(indptr_end - indprt_start == tof_indices.len());
        #[cfg(debug_assertions)]
        {
            for i in 1..(scan_offsets.len() - 1) {
                debug_assert!(scan_offsets[i] <= scan_offsets[i + 1]);
                debug_assert!((scan_offsets[i + 1] - scan_start) <= tof_indices.len());
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
    pub fn global_scan_at_index(&self, local_index: usize) -> usize {
        let search_val = self.scan_offsets[0] + local_index;
        let loc = self
            .scan_offsets
            .binary_search_by(|x| x.partial_cmp(&search_val).unwrap());
        let local_scan_index = match loc {
            Ok(mut x) => {
                while x > 0 && self.scan_offsets[x - 1] >= search_val {
                    x -= 1;
                }
                x
            }
            Err(x) => x - 1,
        };
        self.scan_start + local_scan_index
    }

    pub fn explode_scan_numbers(&self) -> Vec<usize> {
        let mut scan_numbers = Vec::with_capacity(self.tof_indices.len());
        let curr_scan = self.scan_start;

        for (scan_index, index_offset) in self.scan_offsets[1..].iter().enumerate() {
            let num_tofs = index_offset - self.scan_offsets[scan_index];
            scan_numbers.extend(vec![curr_scan + scan_index; num_tofs]);
        }

        if cfg!(debug_assertions) {
            // Check that all are monotonically increasing with min == scan_start
            let mut last_scan = 0;
            for scan in scan_numbers.iter() {
                debug_assert!(*scan >= last_scan);
                last_scan = *scan;
            }

            debug_assert!(scan_numbers[0] == self.scan_start);
            debug_assert!(scan_numbers.len() == self.tof_indices.len());
            debug_assert_eq!(
                scan_numbers.last().unwrap(),
                &(self.scan_offsets.len() - 1 + self.scan_start)
            );
        }
        scan_numbers
    }

    pub fn tof_intensities_at_scan(&self, scan_number: usize) -> ((&[u32], &[u32]), usize) {
        let scan_index = scan_number - self.scan_start;
        let offset_offset = self.scan_offsets[0];
        let scan_start = self.scan_offsets[scan_index] - offset_offset;
        let scan_end = self.scan_offsets[scan_index + 1] - offset_offset;
        let tof_indices = &self.tof_indices[scan_start..scan_end];
        let intensities = &self.intensities[scan_start..scan_end];
        ((tof_indices, intensities), scan_start)
    }

    pub fn matching_range_at_scan<T>(
        &self,
        tof_index: i32,
        scan_number: usize,
        tolerance: &T,
    ) -> Option<(Range, usize)>
    where
        T: FramePointTolerance,
    {
        // TODO implement later a two pointer approach for sorted slices of tof indices.
        let ((tof_indices, _), start_indptr) = self.tof_intensities_at_scan(scan_number);
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
            }
            Err(x) => x,
        };

        if tof_index_start >= tof_len {
            return None;
        };

        let tof_index_end = match tof_index_end {
            Ok(x) => x,
            Err(mut x) => {
                while x < tof_len && tof_indices[x] < end {
                    x += 1;
                }
                x
            }
        };

        if tof_index_end > tof_index_start {
            Some(((tof_index_start, tof_index_end), start_indptr))
        } else {
            None
        }
    }

    pub fn matching_rangeset<T>(
        &self,
        tof_index: i32,
        scan_number: usize,
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
        for scan_number in scan_range.0..scan_range.1 {
            if let Some(range_offset) =
                self.matching_range_at_scan(tof_index, scan_number, tolerance)
            {
                ranges.ranges.push((
                    range_offset.0 .0 - range_offset.1,
                    range_offset.0 .1 - range_offset.1,
                ));
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
}

#[derive(Debug, Clone)]
pub struct DenseFrame {
    pub raw_peaks: Vec<TimsPeak>,
    pub index: usize,
    pub rt: f64,
    pub frame_type: FrameType,
    pub sorted: Option<SortingOrder>,
}

#[derive(Debug, Clone)]
pub struct DenseFrameWindow {
    pub frame: DenseFrame,
    pub ims_min: f32,
    pub ims_max: f32,
    pub mz_start: f64,
    pub mz_end: f64,
    pub group_id: usize,
    pub quad_group_id: usize,
}

impl DenseFrameWindow {
    pub fn from_frame_window(
        frame_window: &FrameSlice,
        ims_converter: &Scan2ImConverter,
        mz_converter: &Tof2MzConverter,
        dia_info: &DIAFrameInfo,
    ) -> DenseFrameWindow {
        let (window_group_id, ww_quad_group_id, scan_start) = match frame_window.slice_window_info {
            None => {
                panic!("No window info")
                // This branch points to an error in logic ...
                // The window info should always be present in this context.
            }
            Some(MsMsFrameSliceWindowInfo::WindowGroup(_)) => {
                // This branch should be easy to implement for things like synchro pasef...
                // Some details to iron out though ...
                panic!("Not implemented")
            }
            Some(MsMsFrameSliceWindowInfo::SingleWindow(ref x)) => {
                let window_group_id = x.window_group_id;
                let ww_quad_group_id = x.within_window_quad_group_id;
                let scan_start = frame_window.scan_start;
                (window_group_id, ww_quad_group_id, scan_start)
            }
        };

        // NOTE: I am swapping here the 'scan start' to be the `ims_end` because
        // the first scans have lower 1/k0 values.
        let ims_max = ims_converter.convert(scan_start as u32) as f32;
        let ims_min =
            ims_converter.convert((frame_window.scan_offsets.len() + scan_start) as u32) as f32;

        debug_assert!(ims_max <= ims_min);

        let scan_range: Option<&ScanRange> =
            dia_info.get_quad_windows(window_group_id, ww_quad_group_id);
        let scan_range = match scan_range {
            Some(x) => x,
            None => {
                panic!(
                    "No scan range for window_group_id: {}, within_window_quad_group_id: {}",
                    window_group_id, ww_quad_group_id
                );
            }
        };

        let frame = DenseFrame::from_frame_window(frame_window, ims_converter, mz_converter);

        DenseFrameWindow {
            frame,
            ims_min,
            ims_max,
            mz_start: scan_range.iso_low as f64,
            mz_end: scan_range.iso_high as f64,
            group_id: window_group_id,
            quad_group_id: ww_quad_group_id,
        }
    }
}

impl DenseFrame {
    pub fn from_frame(
        frame: &Frame,
        ims_converter: &Scan2ImConverter,
        mz_converter: &Tof2MzConverter,
    ) -> DenseFrame {
        let mut expanded_scan_indices = Vec::with_capacity(frame.tof_indices.len());
        let mut last_scan_offset = frame.scan_offsets[0];
        for (scan_index, index_offset) in frame.scan_offsets[1..].iter().enumerate() {
            let num_tofs = index_offset - last_scan_offset;

            let ims = ims_converter.convert(scan_index as u32) as f32;
            expanded_scan_indices.extend(vec![ims; num_tofs]);
            last_scan_offset = *index_offset;
        }

        let peaks = expanded_scan_indices
            .iter()
            .zip(frame.tof_indices.iter())
            .zip(frame.intensities.iter())
            .map(|((scan_index, tof_index), intensity)| TimsPeak {
                intensity: *intensity,
                mz: mz_converter.convert(*tof_index),
                mobility: *scan_index,
                npeaks: 1,
            })
            .collect::<Vec<_>>();

        if cfg!(debug_assertions) {
            for peak in peaks.iter() {
                _check_peak_sanity(peak);
            }
        }

        let index = frame.index;
        let rt = frame.rt;
        let frame_type = frame.frame_type;

        DenseFrame {
            raw_peaks: peaks,
            index,
            rt,
            frame_type,
            sorted: None,
        }
    }

    pub fn from_frame_window(
        frame_window: &FrameSlice,
        ims_converter: &Scan2ImConverter,
        mz_converter: &Tof2MzConverter,
    ) -> DenseFrame {
        let mut expanded_scan_indices = Vec::with_capacity(frame_window.tof_indices.len());
        let mut last_scan_offset = frame_window.scan_offsets[0];
        for (scan_index, index_offset) in frame_window.scan_offsets[1..].iter().enumerate() {
            let num_tofs = index_offset - last_scan_offset;
            let scan_index_use = (scan_index + frame_window.scan_start) as u32;

            let ims = ims_converter.convert(scan_index as f64) as f32;
            if ims < 0.0 {
                info!("Negative IMS value: {}", ims);
                info!("scan_index_use: {}", scan_index_use);
                info!("scan_index: {}", scan_index);
                info!("frame_window.scan_start: {}", frame_window.scan_start);
            }
            debug_assert!(ims >= 0.0);
            expanded_scan_indices.extend(vec![ims; num_tofs]);
            last_scan_offset = *index_offset;
        }
        debug_assert!(last_scan_offset == frame_window.tof_indices.len());

        let peaks = expanded_scan_indices
            .iter()
            .zip(frame_window.tof_indices.iter())
            .zip(frame_window.intensities.iter())
            .map(|((scan_index, tof_index), intensity)| TimsPeak {
                intensity: *intensity,
                mz: mz_converter.convert(*tof_index),
                mobility: *scan_index,
                npeaks: 1,
            })
            .collect::<Vec<_>>();

        if cfg!(debug_assertions) {
            for peak in peaks.iter() {
                _check_peak_sanity(peak);
            }
        }

        let index = frame_window.parent_frame_index;
        let rt = frame_window.rt;
        let frame_type = frame_window.frame_type;

        DenseFrame {
            raw_peaks: peaks,
            index,
            rt,
            frame_type,
            sorted: None,
        }
    }

    pub fn sort_by_mz(&mut self) {
        match self.sorted {
            Some(SortingOrder::Mz) => (),
            _ => {
                self.raw_peaks
                    .sort_unstable_by(|a, b| a.mz.partial_cmp(&b.mz).unwrap());
                self.sorted = Some(SortingOrder::Mz);
            }
        }
    }

    pub fn sort_by_mobility(&mut self) {
        match self.sorted {
            Some(SortingOrder::Mobility) => (),
            _ => {
                self.raw_peaks
                    .sort_unstable_by(|a, b| a.mobility.partial_cmp(&b.mobility).unwrap());
                self.sorted = Some(SortingOrder::Mobility);
            }
        }
    }
}

pub type Converters = (timsrust::Scan2ImConverter, timsrust::Tof2MzConverter);
