use timsrust::{ConvertableIndex, Frame, Scan2ImConverter, Tof2MzConverter};

use crate::ms::{
    frames::MsMsFrameSliceWindowInfo,
    tdf::{DIAFrameInfo, ScanRange},
};

use super::{frames::SortingOrder, DenseFrame, FrameSlice, TimsPeak};
use log::info;

pub type Converters = (timsrust::Scan2ImConverter, timsrust::Tof2MzConverter);
fn check_peak_sanity(peak: &TimsPeak) {
    debug_assert!(peak.intensity > 0);
    debug_assert!(peak.mz > 0.);
    debug_assert!(peak.mobility > 0.);
    debug_assert!(peak.npeaks > 0);
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
            },
            Some(MsMsFrameSliceWindowInfo::WindowGroup(_)) => {
                // This branch should be easy to implement for things like synchro pasef...
                // Some details to iron out though ...
                panic!("Not implemented")
            },
            Some(MsMsFrameSliceWindowInfo::SingleWindow(ref x)) => {
                let window_group_id = x.window_group_id;
                let ww_quad_group_id = x.within_window_quad_group_id;
                let scan_start = frame_window.scan_start;
                (window_group_id, ww_quad_group_id, scan_start)
            },
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
            },
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
                check_peak_sanity(peak);
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
                check_peak_sanity(peak);
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
            },
        }
    }

    pub fn sort_by_mobility(&mut self) {
        match self.sorted {
            Some(SortingOrder::Mobility) => (),
            _ => {
                self.raw_peaks
                    .sort_unstable_by(|a, b| a.mobility.partial_cmp(&b.mobility).unwrap());
                self.sorted = Some(SortingOrder::Mobility);
            },
        }
    }
}
