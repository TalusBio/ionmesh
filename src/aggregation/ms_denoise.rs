use core::panic;

use crate::aggregation::dbscan::denseframe_dbscan::dbscan_denseframe;
use crate::ms::frames::Converters;
use crate::ms::frames::DenseFrame;
use crate::ms::frames::DenseFrameWindow;
use crate::ms::frames::FrameSlice;
use crate::ms::tdf;
use crate::ms::tdf::DIAFrameInfo;
use crate::utils;

use indicatif::ParallelProgressIterator;
use log::{info, trace, warn};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use timsrust::Frame;

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
        "Denoising frame {} with intensity ratio {:.2}, peak_ratio {:.2}, prior_max {}, curr_max {}",
        frame_index, intensity_ratio, peak_ratio, frame_stats_start.max_intensity, frame_stats_end.max_intensity,
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

fn _denoise_dia_frame(
    frame: Frame,
    min_n: usize,
    min_intensity: u64,
    dia_frame_info: &DIAFrameInfo,
    ims_converter: &timsrust::Scan2ImConverter,
    mz_converter: &timsrust::Tof2MzConverter,
    mz_scaling: f64,
    max_mz_extension: f64,
    ims_scaling: f32,
    max_ims_extension: f32,
) -> Vec<DenseFrameWindow> {
    let window_group = dia_frame_info
        .get_dia_frame_window_group(frame.index)
        .unwrap();
    let frame_windows = dia_frame_info
        .split_frame(&frame, window_group)
        .expect("Only DIA frames should be passed to this function");

    frame_windows
        .into_iter()
        .map(|frame_window| {
            denoise_frame_slice(
                &frame_window,
                ims_converter,
                mz_converter,
                dia_frame_info,
                min_n,
                min_intensity,
                mz_scaling,
                max_mz_extension,
                ims_scaling,
                max_ims_extension,
            )
        })
        .collect::<Vec<_>>()
}

fn denoise_frame_slice(
    frame_window: &FrameSlice,
    ims_converter: &timsrust::Scan2ImConverter,
    mz_converter: &timsrust::Tof2MzConverter,
    dia_frame_info: &DIAFrameInfo,
    min_n: usize,
    min_intensity: u64,
    mz_scaling: f64,
    max_mz_extension: f64,
    ims_scaling: f32,
    max_ims_extension: f32,
) -> DenseFrameWindow {
    let denseframe_window = DenseFrameWindow::from_frame_window(
        frame_window,
        ims_converter,
        mz_converter,
        dia_frame_info,
    );
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
        mz_start: denseframe_window.mz_start,
        mz_end: denseframe_window.mz_end,
        group_id: denseframe_window.group_id,
        quad_group_id: denseframe_window.quad_group_id,
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
        info!("Denoising {} frames", elems.len());
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
    ims_converter: timsrust::Scan2ImConverter,
    mz_converter: timsrust::Tof2MzConverter,
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
    dia_frame_info: DIAFrameInfo,
    ims_converter: timsrust::Scan2ImConverter,
    mz_converter: timsrust::Tof2MzConverter,
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
        info!("Denoising {} frames", elems.len());

        let frame_window_slices = self.dia_frame_info.split_frame_windows(&elems);
        let mut out = Vec::with_capacity(frame_window_slices.len());
        for sv in frame_window_slices {
            let progbar = indicatif::ProgressBar::new(sv.len() as u64);
            let denoised_elements: Vec<DenseFrameWindow> = sv
                .into_par_iter()
                .progress_with(progbar)
                .map(|x| {
                    denoise_frame_slice(
                        &x,
                        &self.ims_converter,
                        &self.mz_converter,
                        &self.dia_frame_info,
                        self.min_n,
                        self.min_intensity,
                        self.mz_scaling,
                        self.max_mz_extension,
                        self.ims_scaling,
                        self.max_ims_extension,
                    )
                })
                .collect::<Vec<_>>();
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
) -> Vec<DenseFrame> {
    let reader = timsrust::FileReader::new(path).unwrap();

    let mut timer = utils::ContextTimer::new("Reading all MS1 frames", true, utils::LogLevel::INFO);

    let mut frames = reader.read_all_ms1_frames();
    timer.stop(true);

    let ims_converter = reader.get_scan_converter().unwrap();
    let mz_converter = reader.get_tof_converter().unwrap();

    frames.retain(|frame| matches!(frame.frame_type, timsrust::FrameType::MS1));

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
    out
}

// This could probably be a macro ...
pub fn read_all_dia_denoising(
    path: String,
    config: DenoiseConfig,
) -> (Vec<DenseFrameWindow>, DIAFrameInfo) {
    let mut timer = utils::ContextTimer::new("Reading all DIA frames", true, utils::LogLevel::INFO);
    let reader = timsrust::FileReader::new(path.clone()).unwrap();

    let dia_info = tdf::read_dia_frame_info(path.clone()).unwrap();
    let mut frames = reader.read_all_ms2_frames();

    let ims_converter = reader.get_scan_converter().unwrap();
    let mz_converter = reader.get_tof_converter().unwrap();
    timer.stop(true);

    frames.retain(|frame| {
        matches!(
            frame.frame_type,
            timsrust::FrameType::MS2(timsrust::AcquisitionType::DIAPASEF)
        )
    });

    let denoiser = DIAFrameDenoiser {
        min_n: config.ms2_min_n.into(),
        min_intensity: config.ms2_min_cluster_intensity.into(),
        mz_scaling: config.mz_scaling.into(),
        max_mz_extension: config.max_mz_expansion_ratio.into(),
        ims_scaling: config.ims_scaling,
        max_ims_extension: config.max_ims_expansion_ratio,
        dia_frame_info: dia_info.clone(),
        ims_converter,
        mz_converter,
    };
    let mut timer =
        utils::ContextTimer::new("Denoising all MS2 frames", true, utils::LogLevel::INFO);
    let split_frames = denoiser.par_denoise_slice(frames);
    let out: Vec<DenseFrameWindow> = split_frames.into_iter().flatten().collect();
    timer.stop(true);

    (out, dia_info)
}
