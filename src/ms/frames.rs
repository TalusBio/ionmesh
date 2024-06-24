
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
///    - Scan offsets.    [0,0,0,0,0,3,5,6 ...] n=number of scans
///    - tof indices.     [100, 101, 102, 10, 20, 30 ...] len = len(intensities)
///    - intensities.     [123, 111, 12 ,  3,  4,  1 ...] len = len(tof indices)
///    - index            34
///    - rt               65.34
/// Additions for FrameQuadWindow:
///    - scan_start       123  // The scan number of the first scan offset in the current window.
///    - group_id         1    // The group id of the current window.
///    - quad_group_id    2    // The quad group id of the current window within the current group.
///    - quad_row_id      3    // The quad row id of the current window within all quad windows.
#[derive(Debug, Clone)]
pub struct FrameQuadWindow {
    pub scan_offsets: Vec<u64>,
    pub tof_indices: Vec<u32>,
    pub intensities: Vec<u32>,
    pub index: usize,
    pub rt: f64,
    pub frame_type: FrameType,

    // From this point on they are local implementations
    // Before they are used from the timsrust crate.
    pub scan_start: usize,
    pub group_id: usize,
    pub quad_group_id: usize,
    pub quad_row_id: usize,
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
    pub ims_start: f32,
    pub ims_end: f32,
    pub mz_start: f64,
    pub mz_end: f64,
    pub group_id: usize,
    pub quad_group_id: usize,
}

impl DenseFrameWindow {
    pub fn from_frame_window(
        frame_window: FrameQuadWindow,
        ims_converter: &Scan2ImConverter,
        mz_converter: &Tof2MzConverter,
        dia_info: &DIAFrameInfo,
    ) -> DenseFrameWindow {
        let group_id = frame_window.group_id;
        let quad_group_id = frame_window.quad_group_id;
        let scan_start = frame_window.scan_start;

        // NOTE: I am swapping here the 'scan start' to be the `ims_end` because
        // the first scans have lower 1/k0 values.
        let ims_end = ims_converter.convert(scan_start as u32) as f32;
        let ims_start =
            ims_converter.convert((frame_window.scan_offsets.len() + scan_start) as u32) as f32;
        let scan_range: &ScanRange = dia_info
            .get_quad_windows(group_id, quad_group_id)
            .expect("Quad group id should be valid");

        let frame = DenseFrame::from_frame_window(frame_window, ims_converter, mz_converter);

        debug_assert!(ims_start <= ims_end);

        DenseFrameWindow {
            frame,
            ims_start,
            ims_end,
            mz_start: scan_range.iso_low as f64,
            mz_end: scan_range.iso_high as f64,
            group_id,
            quad_group_id,
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
        frame_window: FrameQuadWindow,
        ims_converter: &Scan2ImConverter,
        mz_converter: &Tof2MzConverter,
    ) -> DenseFrame {
        let mut expanded_scan_indices = Vec::with_capacity(frame_window.tof_indices.len());
        let mut last_scan_offset = frame_window.scan_offsets[0];
        for (scan_index, index_offset) in frame_window.scan_offsets[1..].iter().enumerate() {
            let num_tofs = index_offset - last_scan_offset;
            let scan_index_use = (scan_index + frame_window.scan_start) as u32;

            let ims = ims_converter.convert(scan_index_use) as f32;
            if ims < 0.0 {
                info!("Negative IMS value: {}", ims);
                info!("scan_index_use: {}", scan_index_use);
                info!("scan_index: {}", scan_index);
                info!("frame_window.scan_start: {}", frame_window.scan_start);
            }
            debug_assert!(ims >= 0.0);
            expanded_scan_indices.extend(vec![ims; num_tofs as usize]);
            last_scan_offset = *index_offset;
        }
        debug_assert!(last_scan_offset == frame_window.tof_indices.len() as u64);

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

        let index = frame_window.index;
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

    fn concatenate(mut self, other: DenseFrame) -> DenseFrame {
        self.raw_peaks.extend(other.raw_peaks);
        self.sorted = None;
        self
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
