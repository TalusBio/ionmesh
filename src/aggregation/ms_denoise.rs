use core::fmt::Debug;
use core::panic;
use std::path::Path;

use indicatif::ParallelProgressIterator;
use log::{
    debug,
    info,
    trace,
    warn,
};
use rayon::prelude::*;
use serde::{
    Deserialize,
    Serialize,
};
use timsrust::converters::{
    Scan2ImConverter,
    Tof2MzConverter,
};
use timsrust::{
    AcquisitionType,
    Frame,
    MSLevel,
    QuadrupoleSettings,
    TimsRustError,
};

use super::aggregators::aggregate_clusters;
use super::dbscan::runner::dbscan_label_clusters;
use crate::aggregation::dbscan::denseframe_dbscan::dbscan_denseframe;
use crate::ms::frames::frame_slice_rt_window::{
    FrameSliceWindow,
    RawWeightedTimsPeakAggregator,
};
use crate::ms::frames::{
    Converters,
    DenseFrame,
    DenseFrameWindow,
    ExpandedFrameSlice,
    FrameSlice,
    SingleQuadrupoleSettings,
    TimsPeak,
};
use crate::space::space_generics::{
    AsNDPointsAtIndex,
    IntenseAtIndex,
};
use crate::utils;
use crate::utils::maybe_save_json_if_debugging;

// TODO I can probably split the ms1 and ms2 ...
#[derive(Debug, Serialize, Deserialize, Clone, Copy)]
pub struct DenoiseConfig {
    pub mz_scaling: f32,
    pub ims_scaling: f32,
    pub max_mz_expansion_ratio: f32,
    pub max_ims_expansion_ratio: f32,
    pub ms2_min_n: u8,
    pub ms1_min_n: u8,
    pub ms1_min_cluster_intensity: u32,
    pub ms2_min_cluster_intensity: u32,
}

impl Default for DenoiseConfig {
    fn default() -> Self {
        DenoiseConfig {
            mz_scaling: 0.015,
            ims_scaling: 0.015,
            max_mz_expansion_ratio: 1.,
            max_ims_expansion_ratio: 4.,
            ms2_min_n: 5,
            ms1_min_n: 10,
            ms1_min_cluster_intensity: 100,
            ms2_min_cluster_intensity: 100,
        }
    }
}

#[derive(Debug, Clone, Copy)]
struct FrameStats {
    max_intensity: u32,
    num_peaks: usize,
    tot_intensity: f64,
}

impl FrameStats {
    fn new(frame: &DenseFrame) -> FrameStats {
        let max_intensity = frame
            .raw_peaks
            .iter()
            .map(|peak| peak.intensity)
            .max()
            .unwrap_or(0);
        let num_peaks = frame.raw_peaks.len();
        let tot_intensity = frame
            .raw_peaks
            .iter()
            .map(|peak| peak.intensity as f64)
            .sum::<f64>();

        FrameStats {
            max_intensity,
            num_peaks,
            tot_intensity,
        }
    }
}

// This is meant to run only in debug mode
fn _sanity_check_framestats(
    frame_stats_start: FrameStats,
    frame_stats_end: FrameStats,
    frame_index: usize,
) {
    let intensity_ratio = frame_stats_start.tot_intensity / frame_stats_end.tot_intensity;
    let peak_ratio = frame_stats_end.num_peaks as f64 / frame_stats_start.num_peaks as f64;

    trace!(
        "Denoising frame {} with intensity ratio {:.2}, peak_ratio {:.2}, prior_max {}, curr_max \
         {}",
        frame_index,
        intensity_ratio,
        peak_ratio,
        frame_stats_start.max_intensity,
        frame_stats_end.max_intensity,
    );
    if frame_stats_end.max_intensity < frame_stats_start.max_intensity {
        trace!(
            "End max intensity is greater than start max intensity for frame {}!",
            frame_index
        );
        if frame_stats_end.max_intensity < 1000 {
            trace!("This is probably fine, since the max intensity is very low.");
        } else {
            warn!("Before: {:?}", frame_stats_start);
            warn!("After: {:?}", frame_stats_end);
            panic!("There is an error somewhere!");
        }
    }

    assert!(peak_ratio <= 1.);
}

fn _denoise_denseframe(
    frame: DenseFrame,
    min_n: usize,
    min_intensity: u64,
    mz_scaling: f64,
    max_mz_extension: f64,
    ims_scaling: f32,
    max_ims_extension: f32,
) -> DenseFrame {
    // I am 99% sure the compiler will remove this section when optimizing ... but I still need to test it.
    let frame_stats_start: FrameStats = FrameStats::new(&frame);
    let index = frame.index;

    // this is the line that matters
    let denoised_frame = dbscan_denseframe(
        frame,
        mz_scaling,
        max_mz_extension,
        ims_scaling,
        max_ims_extension,
        min_n,
        min_intensity,
    );

    let frame_stats_end = FrameStats::new(&denoised_frame);
    if cfg!(debug_assertions) {
        _sanity_check_framestats(frame_stats_start, frame_stats_end, index);
    }

    denoised_frame
}

fn denoise_frame_slice_window(
    frameslice_window: &[ExpandedFrameSlice],
    ims_converter: &Scan2ImConverter,
    mz_converter: &Tof2MzConverter,
    min_n: usize,
    min_intensity: u64,
    _mz_scaling: f64,
    _max_mz_extension: f64,
    _ims_scaling: f32,
    _max_ims_extension: f32,
) -> DenseFrameWindow {
    let timer = utils::ContextTimer::new("dbscan_dfs", true, utils::LogLevel::TRACE);
    let fsw = FrameSliceWindow::new(frameslice_window);
    let ref_frame_parent_index = fsw.window[fsw.reference_index].parent_frame_index;
    let saved_first =
        maybe_save_json_if_debugging(&fsw, &format!("fsw_{}", ref_frame_parent_index), false);

    let mut intensity_sorted_indices = Vec::with_capacity(fsw.num_ndpoints());
    for i in 0..fsw.num_ndpoints() {
        // Should I only add the points in the reference frame??
        let intensity = fsw.intensity_at_index(i);
        intensity_sorted_indices.push((i, intensity));
    }
    intensity_sorted_indices.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    if cfg!(debug_assertions) {
        // I know this should be obviously always true, but I dont trust myself
        // and thinking about orderings.
        let mut last_intensity = u64::MAX;
        for (_i, intensity) in intensity_sorted_indices.iter() {
            assert!(*intensity <= last_intensity);
            last_intensity = *intensity;
        }
    }

    let mut i_timer = timer.start_sub_timer("dbscan");
    let cluster_labels = dbscan_label_clusters(
        &fsw,
        &fsw,
        &fsw,
        min_n,
        min_intensity,
        intensity_sorted_indices,
        None::<&(dyn Fn(&f32) -> bool + Send + Sync)>,
        false,
        &[10., 100.],
    );
    i_timer.stop(true);

    let centroids = aggregate_clusters(
        cluster_labels.num_clusters,
        cluster_labels.cluster_labels,
        &fsw,
        &RawWeightedTimsPeakAggregator::default,
        utils::LogLevel::TRACE,
        false,
    );

    let ref_frame = &frameslice_window[frameslice_window.len() / 2];
    let quad_settings = ref_frame.quadrupole_settings.clone();

    let mut raw_peaks: Vec<TimsPeak> = centroids
        .into_iter()
        .map(|x| x.to_timspeak(mz_converter, ims_converter))
        .collect();

    raw_peaks.retain(|x| x.intensity > min_intensity as u32);

    let mut min_ims = f32::INFINITY;
    let mut max_ims = f32::NEG_INFINITY;

    for peak in raw_peaks.iter() {
        if peak.mobility < min_ims {
            min_ims = peak.mobility;
        }
        if peak.mobility > max_ims {
            max_ims = peak.mobility;
        }
    }

    let out = DenseFrameWindow {
        frame: DenseFrame {
            raw_peaks,
            index: ref_frame.parent_frame_index,
            rt: ref_frame.rt,
            acquisition_type: ref_frame.acquisition_type,
            ms_level: ref_frame.ms_level,
            sorted: None,
            window_group_id: ref_frame.window_group_id,
            intensity_correction_factor: ref_frame.intensity_correction_factor,
        },
        ims_max: max_ims,
        ims_min: min_ims,
        group_id: ref_frame.window_group_id.into(),
        quadrupole_setting: quad_settings,
    };
    maybe_save_json_if_debugging(
        &out,
        &format!("dfw_out_{}", ref_frame_parent_index),
        saved_first,
    );

    out
}

fn denoise_frame_slice(
    frame_window: &FrameSlice,
    ims_converter: &Scan2ImConverter,
    mz_converter: &Tof2MzConverter,
    min_n: usize,
    min_intensity: u64,
    mz_scaling: f64,
    max_mz_extension: f64,
    ims_scaling: f32,
    max_ims_extension: f32,
) -> DenseFrameWindow {
    let denseframe_window =
        DenseFrameWindow::from_frame_window(frame_window, ims_converter, mz_converter);
    let denoised_frame = _denoise_denseframe(
        denseframe_window.frame,
        min_n,
        min_intensity,
        mz_scaling,
        max_mz_extension,
        ims_scaling,
        max_ims_extension,
    );

    DenseFrameWindow {
        frame: denoised_frame,
        ims_min: denseframe_window.ims_min,
        ims_max: denseframe_window.ims_max,
        group_id: denseframe_window.group_id,
        quadrupole_setting: denseframe_window.quadrupole_setting,
    }
}

trait Denoiser<'a, T, W, X, Z>
where
    T: std::marker::Send,
    W: Clone + std::marker::Send,
    X: Clone,
    Z: Clone,
    Vec<T>: IntoParallelIterator<Item = T>,
{
    fn denoise(
        &self,
        elem: T,
    ) -> W;
    fn par_denoise_slice(
        &self,
        elems: Vec<T>,
    ) -> Vec<W>
    where
        Self: Sync,
    {
        debug!("Denoising {} frames", elems.len());
        // randomly viz 1/200 frames
        // Selecting a slice of 1/200 frames

        let progbar = indicatif::ProgressBar::new(elems.len() as u64);
        let denoised_elements: Vec<W> = elems
            .into_par_iter()
            .progress_with(progbar)
            .map(|x| self.denoise(x))
            .collect::<Vec<_>>();

        denoised_elements
    }
}

struct FrameDenoiser {
    min_n: usize,
    min_intensity: u64,
    mz_scaling: f64,
    ims_scaling: f32,
    max_mz_extension: f64,
    max_ims_extension: f32,
    ims_converter: Scan2ImConverter,
    mz_converter: Tof2MzConverter,
}

impl<'a> Denoiser<'a, Frame, DenseFrame, Converters, Option<usize>> for FrameDenoiser {
    fn denoise(
        &self,
        frame: Frame,
    ) -> DenseFrame {
        let denseframe = DenseFrame::from_frame(&frame, &self.ims_converter, &self.mz_converter);
        _denoise_denseframe(
            denseframe,
            self.min_n,
            self.min_intensity,
            self.mz_scaling,
            self.max_mz_extension,
            self.ims_scaling,
            self.max_ims_extension,
        )
    }
}

struct DIAFrameDenoiser {
    min_n: usize,
    min_intensity: u64,
    mz_scaling: f64,
    max_mz_extension: f64,
    ims_scaling: f32,
    max_ims_extension: f32,
    ims_converter: Scan2ImConverter,
    mz_converter: Tof2MzConverter,
}

// impl DIAFrameDenoiser {
//     fn denoise_framewindow_slice(self, elems: Vec<FrameQuadWindow>) -> Vec<DenseFrameWindow> {}
// }

impl<'a> Denoiser<'a, Frame, Vec<DenseFrameWindow>, Converters, Option<usize>>
    for DIAFrameDenoiser
{
    fn denoise(
        &self,
        _frame: Frame,
    ) -> Vec<DenseFrameWindow> {
        panic!("This should not be called")
        // _denoise_dia_frame(
        //     frame,
        //     self.min_n,
        //     self.min_intensity,
        //     &self.dia_frame_info,
        //     &self.ims_converter,
        //     &self.mz_converter,
        //     self.mz_scaling,
        //     self.max_mz_extension,
        //     self.ims_scaling,
        //     self.max_ims_extension,
        // )
    }
    fn par_denoise_slice(
        &self,
        elems: Vec<Frame>,
    ) -> Vec<Vec<DenseFrameWindow>>
    where
        Self: Sync,
    {
        info!("Denoising (centroiding) {} frames", elems.len());

        let mut flat_windows: Vec<FrameSlice<'_>> = elems
            .par_iter()
            .flat_map(|x: &Frame| FrameSlice::from_frame(x))
            .collect();

        flat_windows.par_sort_unstable_by(|a, b| {
            a.quadrupole_settings
                .partial_cmp(&b.quadrupole_settings)
                .unwrap()
        });

        let mut break_points = vec![0];
        for i in 1..flat_windows.len() {
            if flat_windows[i].quadrupole_settings != flat_windows[i - 1].quadrupole_settings {
                break_points.push(i);
            }
        }

        // If profiling and having the "IONMESH_PROFILE_NUM_WINDOWS" env variable set
        // then only process the first N slices of windows.
        // This is useful for profiling the code.
        if let Ok(num_windows) = std::env::var("IONMESH_PROFILE_NUM_WINDOWS") {
            let num_windows: usize = num_windows.parse().unwrap();
            log::warn!("Profiling: Only processing {} windows", num_windows);
            flat_windows.truncate(break_points[num_windows]);
            break_points.truncate(num_windows);
        }

        // This warning reders to denoise_frame_slice_window.
        // to have them be not hard-coded I need a way to convert
        // m/z space ranges to tof indices ... which is not exposed
        // by timsrust ...
        warn!("Using prototype function for denoising, scalings are hard-coded");

        let num_windows = break_points.len() - 1;
        let mut out = Vec::with_capacity(num_windows);
        let frame_window_slices: Vec<(usize, &[FrameSlice])> = (0..num_windows)
            .map(|i| (i, &flat_windows[break_points[i]..break_points[i + 1]]))
            .collect();
        for (i, sv) in frame_window_slices.iter() {
            info!("Denoising window {}/{}", i + 1, num_windows);
            let start_tot_peaks = sv.iter().map(|x| x.num_ndpoints() as u64).sum::<u64>();
            let progbar = indicatif::ProgressBar::new(sv.len() as u64);

            let lambda_denoise = |x: &[ExpandedFrameSlice]| {
                denoise_frame_slice_window(
                    x,
                    &self.ims_converter,
                    &self.mz_converter,
                    self.min_n,
                    self.min_intensity,
                    self.mz_scaling,
                    self.max_mz_extension,
                    self.ims_scaling,
                    self.max_ims_extension,
                )
            };

            let mut denoised_elements: Vec<DenseFrameWindow> = if cfg!(feature = "less_parallel") {
                warn!("Running in less parallel mode");
                sv.iter()
                    .map(|x| ExpandedFrameSlice::from_frame_slice(x))
                    .collect::<Vec<ExpandedFrameSlice>>()
                    .windows(3)
                    .map(lambda_denoise)
                    .collect::<Vec<_>>()
            } else {
                sv.par_iter()
                    .map(|x| ExpandedFrameSlice::from_frame_slice(x))
                    .collect::<Vec<ExpandedFrameSlice>>()
                    .par_windows(3)
                    .progress_with(progbar)
                    .map(lambda_denoise)
                    .collect::<Vec<_>>()
            };

            debug!("Denoised {} frames", denoised_elements.len());
            denoised_elements
                .retain(|x| x.frame.raw_peaks.iter().map(|y| y.intensity).sum::<u32>() > 20);
            debug!("Retained {} frames", denoised_elements.len());
            let end_tot_peaks = denoised_elements
                .iter()
                .map(|x| x.frame.raw_peaks.len() as u64)
                .sum::<u64>();
            let ratio = end_tot_peaks as f64 / start_tot_peaks as f64;
            debug!(
                "Start peaks: {}, End peaks: {} -> ratio: {:.2}",
                start_tot_peaks, end_tot_peaks, ratio
            );
            out.push(denoised_elements);
        }
        out
    }
}

// RN this is dead but will be resurrected soon ...
pub fn read_all_ms1_denoising(
    path: String,
    min_intensity: u64,
    min_n: usize,
    mz_scaling: f64,
    max_mz_extension: f64,
    ims_scaling: f32,
    max_ims_extension: f32,
) -> Result<Vec<DenseFrame>, TimsRustError> {
    let metadata_path = Path::new(&path.clone()).join("analysis.tdf");
    let reader = timsrust::readers::FrameReader::new(path).unwrap();
    let metadata = timsrust::readers::MetadataReader::new(metadata_path).unwrap();

    let mut timer = utils::ContextTimer::new("Reading all MS1 frames", true, utils::LogLevel::INFO);

    let frames: Result<Vec<Frame>, _> = reader.get_all_ms1().into_iter().collect();
    timer.stop(true);

    let mut frames = frames?;

    let ims_converter = metadata.im_converter;
    let mz_converter = metadata.mz_converter;

    frames.retain(|frame| matches!(frame.ms_level, MSLevel::MS1));

    // let min_intensity = 100u64;
    // let min_n: usize = 3;
    let ms1_denoiser = FrameDenoiser {
        min_n,
        mz_scaling,
        max_mz_extension,
        ims_scaling,
        max_ims_extension,
        min_intensity,
        ims_converter,
        mz_converter,
    };

    let mut timer =
        utils::ContextTimer::new("Denoising all MS1 frames", true, utils::LogLevel::INFO);
    let out = ms1_denoiser.par_denoise_slice(frames);
    timer.stop(true);
    Ok(out)
}

// This could probably be a macro ...
pub fn read_all_dia_denoising(
    path: String,
    config: DenoiseConfig,
) -> Result<(Vec<Vec<DenseFrameWindow>>, Vec<QuadrupoleSettings>), TimsRustError> {
    let mut timer = utils::ContextTimer::new("Reading all DIA frames", true, utils::LogLevel::INFO);
    let metadata_path = Path::new(&path.clone()).join("analysis.tdf");
    let reader = timsrust::readers::FrameReader::new(path)?;
    let metadata = timsrust::readers::MetadataReader::new(metadata_path.clone())?;
    let quad_settings = timsrust::readers::QuadrupoleSettingsReader::new(metadata_path)?;

    let frames: Result<Vec<Frame>, _> = reader.get_all_ms2().into_iter().collect();

    let ims_converter = metadata.im_converter;
    let mz_converter = metadata.mz_converter;
    timer.stop(true);

    let mut frames = frames?;

    frames.retain(|frame| match (frame.ms_level, frame.acquisition_type) {
        (MSLevel::MS2, AcquisitionType::DIAPASEF) => true,
        (MSLevel::MS2, AcquisitionType::DiagonalDIAPASEF) => true,
        _ => false,
    });

    let denoiser = DIAFrameDenoiser {
        min_n: config.ms2_min_n.into(),
        min_intensity: config.ms2_min_cluster_intensity.into(),
        mz_scaling: config.mz_scaling.into(),
        max_mz_extension: config.max_mz_expansion_ratio.into(),
        ims_scaling: config.ims_scaling,
        max_ims_extension: config.max_ims_expansion_ratio,
        ims_converter,
        mz_converter,
    };
    let mut timer =
        utils::ContextTimer::new("Denoising all MS2 frames", true, utils::LogLevel::INFO);
    let split_frames = denoiser.par_denoise_slice(frames);
    timer.stop(true);

    Ok((split_frames, quad_settings))
}
