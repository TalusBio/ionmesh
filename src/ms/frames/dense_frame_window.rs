use std::sync::Arc;

use log::info;
use serde::Serialize;
use timsrust::converters::{
    ConvertableDomain,
    Scan2ImConverter,
    Tof2MzConverter,
};
use timsrust::{
    AcquisitionType,
    Frame,
};

use super::frames::SortingOrder;
use super::{
    DenseFrame,
    FrameSlice,
    SingleQuadrupoleSettings,
    TimsPeak,
};

pub type Converters = (Scan2ImConverter, Tof2MzConverter);
fn check_peak_sanity(peak: &TimsPeak) {
    debug_assert!(peak.intensity > 0);
    debug_assert!(peak.mz > 0.);
    debug_assert!(peak.mobility > 0.);
    debug_assert!(peak.npeaks > 0);
}

#[derive(Debug, Clone, Serialize)]
pub struct DenseFrameWindow {
    pub frame: DenseFrame,
    pub ims_min: f32,
    pub ims_max: f32,
    pub group_id: usize,
    pub quadrupole_setting: SingleQuadrupoleSettings,
}

impl DenseFrameWindow {
    pub fn from_frame_window(
        frame_window: &FrameSlice,
        ims_converter: &Scan2ImConverter,
        mz_converter: &Tof2MzConverter,
    ) -> DenseFrameWindow {
        let window_group_id = frame_window.window_group_id;
        let foo = match frame_window.acquisition_type {
            AcquisitionType::DiagonalDIAPASEF => {
                // This branch should be easy to implement for things like synchro pasef...
                // Some details to iron out though ...
                panic!("Not implemented")
            },
            AcquisitionType::DIAPASEF => {
                frame_window.quadrupole_settings.clone()

                // let window_group_id = frame_window.window_group_id;
                // let ww_quad_group_id = x.within_window_quad_group_id;
                // let scan_start = frame_window.scan_start;
                // (window_group_id, ww_quad_group_id, scan_start)
            },
            _ => panic!("Not implemented"),
        };

        // NOTE: I am swapping here the 'scan start' to be the `ims_end` because
        // the first scans have lower 1/k0 values.
        let ims_max = ims_converter.convert(foo.scan_start as u32) as f32;
        let ims_min =
            ims_converter.convert((frame_window.scan_offsets.len() + foo.scan_start) as u32) as f32;

        debug_assert!(ims_max <= ims_min);

        let frame = DenseFrame::from_frame_window(frame_window, ims_converter, mz_converter);

        DenseFrameWindow {
            frame,
            ims_min,
            ims_max,
            group_id: window_group_id.into(),
            quadrupole_setting: foo,
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
            expanded_scan_indices.extend(vec![ims; num_tofs as usize]);
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
        let acquisition_type = frame.acquisition_type;
        let ms_level = frame.ms_level;

        DenseFrame {
            raw_peaks: peaks,
            index,
            rt,
            acquisition_type,
            ms_level,
            sorted: None,
            intensity_correction_factor: frame.intensity_correction_factor,
            window_group_id: frame.window_group,
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
            expanded_scan_indices.extend(vec![ims; num_tofs as usize]);
            last_scan_offset = *index_offset;
        }
        debug_assert!(last_scan_offset as usize == frame_window.tof_indices.len());

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
        let acquisition_type = frame_window.acquisition_type;
        let ms_level = frame_window.ms_level;

        DenseFrame {
            raw_peaks: peaks,
            index,
            rt,
            acquisition_type,
            ms_level,
            sorted: None,
            intensity_correction_factor: frame_window.intensity_correction_factor,
            window_group_id: frame_window.window_group_id,
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
