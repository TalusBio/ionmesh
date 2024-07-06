pub use timsrust::Frame;
pub use timsrust::FrameType;
pub use timsrust::{
    ConvertableIndex, FileReader, Frame2RtConverter, Scan2ImConverter, Tof2MzConverter,
};

use crate::ms::tdf::{DIAFrameInfo, ScanRange};

use log::info;

#[derive(Debug, Clone, Copy)]
pub struct TimsPeak {
    pub intensity: u32,
    pub mz: f64,
    pub mobility: f32,
    pub npeaks: u32,
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
/// 4. Getting the peaks for scan #x in the frame is done by subsetting
///    the tof indices and intensities.
///     - scan_1_intensities = intensities[scan_offsets[1]:scan_offsets[2]]
///     - scan_x_intensities = intensities[scan_offsets[x]:scan_offsets[x+1]]
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
    pub fn new(
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
