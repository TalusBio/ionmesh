use core::panic;
use std::ops::Index;

use crate::aggregation::dbscan::dbscan::dbscan_aggregate;
use crate::aggregation::dbscan::denseframe_dbscan::dbscan_denseframe;
use crate::ms::frames::frames::RawTimsPeak;
use crate::ms::frames::Converters;
use crate::ms::frames::DenseFrame;
use crate::ms::frames::DenseFrameWindow;
use crate::ms::frames::FrameSlice;
use crate::ms::frames::TimsPeak;
use crate::ms::tdf;
use crate::ms::tdf::DIAFrameInfo;
use crate::space::space_generics::AsNDPointsAtIndex;
use crate::space::space_generics::DistantAtIndex;
use crate::space::space_generics::IntenseAtIndex;
use crate::space::space_generics::NDPoint;
use crate::space::space_generics::QueriableIndexedPoints;
use crate::utils;
use timsrust::ConvertableIndex;

use indicatif::ParallelProgressIterator;
use log::{info, trace, warn};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use timsrust::Frame;

use super::aggregators::aggregate_clusters;
use super::aggregators::ClusterAggregator;
use super::aggregators::TimsPeakAggregator;
use super::dbscan::runner::dbscan_label_clusters;

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

#[derive(Debug)]
struct FrameSliceWindow<'a> {
    window: &'a [FrameSlice<'a>],
    reference_index: usize,
    cum_lengths: Vec<usize>,
}

#[derive(Debug, Clone, Copy)]
struct MaybeIntenseRawPeak {
    intensity: u32,
    tof_index: u32,
    scan_index: usize,
    weight_only: bool,
}

impl FrameSliceWindow<'_> {
    fn new<'a>(window: &'a [FrameSlice<'a>]) -> FrameSliceWindow<'a> {
        let cum_lengths = window
            .iter()
            .map(|x| x.num_ndpoints())
            .scan(0, |acc, x| {
                *acc += x;
                Some(*acc)
            })
            .collect();
        FrameSliceWindow {
            window,
            reference_index: window.len() / 2,
            cum_lengths,
        }
    }
    fn get_window_index(
        &self,
        index: usize,
    ) -> (usize, usize) {
        let mut pos = 0;
        for (i, cum_length) in self.cum_lengths.iter().enumerate() {
            if index < *cum_length {
                pos = i;
                break;
            }
        }
        let within_window_index = index - self.cum_lengths[pos];
        (pos, within_window_index)
    }
}

impl Index<usize> for FrameSliceWindow<'_> {
    type Output = MaybeIntenseRawPeak;

    fn index(
        &self,
        index: usize,
    ) -> &Self::Output {
        let (pos, within_window_index) = self.get_window_index(index);
        let tmp = self.window[pos];
        let (tof, int) = tmp.tof_int_at_index(within_window_index);
        let foo = MaybeIntenseRawPeak {
            intensity: int,
            tof_index: tof,
            scan_index: tmp.global_scan_at_index(within_window_index),
            weight_only: pos != self.reference_index,
        };
        &foo
    }
}

impl IntenseAtIndex for FrameSliceWindow<'_> {
    fn intensity_at_index(
        &self,
        index: usize,
    ) -> u64 {
        let (pos, within_window_index) = self.get_window_index(index);
        if pos == self.reference_index {
            self.window[self.reference_index].intensity_at_index(within_window_index)
        } else {
            0
        }
    }

    fn weight_at_index(
        &self,
        index: usize,
    ) -> u64 {
        let (pos, within_window_index) = self.get_window_index(index);
        self.window[pos].weight_at_index(within_window_index)
    }
}

impl<'a> QueriableIndexedPoints<'a, 2, usize> for FrameSliceWindow<'a> {
    fn query_ndpoint(
        &'a self,
        point: &NDPoint<2>,
    ) -> Vec<&'a usize> {
        let mut out = Vec::new();
        for (i, (frame, cum_length)) in self.window.iter().zip(self.cum_lengths).enumerate() {
            let local_outs = frame.query_ndpoint(point);
            for ii in local_outs {
                out.push(&(ii + cum_length));
            }
        }
        out
    }

    fn query_ndrange(
        &'a self,
        boundary: &crate::space::space_generics::NDBoundary<2>,
        reference_point: Option<&NDPoint<2>>,
    ) -> Vec<&'a usize> {
        let mut out = Vec::new();
        for (i, (frame, cum_length)) in self.window.iter().zip(self.cum_lengths).enumerate() {
            let local_outs = frame.query_ndrange(boundary, reference_point);
            for ii in local_outs {
                out.push(&(ii + cum_length));
            }
        }
        out
    }
}

impl DistantAtIndex<f32> for FrameSliceWindow<'_> {
    fn distance_at_indices(
        &self,
        index: usize,
        other: usize,
    ) -> f32 {
        let (pos, within_window_index) = self.get_window_index(index);
        let (pos_other, within_window_index_other) = self.get_window_index(other);
        panic!("unimplemented");
        0.
    }
}

impl AsNDPointsAtIndex<2> for FrameSliceWindow<'_> {
    fn get_ndpoint(
        &self,
        index: usize,
    ) -> NDPoint<2> {
        let (pos, within_window_index) = self.get_window_index(index);
        self.window[pos].get_ndpoint(within_window_index)
    }

    fn num_ndpoints(&self) -> usize {
        self.cum_lengths.last().unwrap().clone()
    }
}

#[derive(Default, Debug, Clone, Copy)]
pub struct RawWeightedTimsPeakAggregator {
    pub cumulative_weighted_cluster_tof: u64,
    pub cumulative_weighted_cluster_scan: u64,
    pub cumulative_cluster_weight: u64,
    pub cumulative_cluster_intensity: u64,
    pub num_peaks: u64,
    pub num_intense_peaks: u64,
}

#[derive(Debug, Clone, Copy)]
struct RawScaleTimsPeak {
    intensity: f64,
    tof_index: f64,
    scan_index: f64,
    npeaks: u64,
}

impl RawScaleTimsPeak {
    fn to_timspeak(
        &self,
        mz_converter: &timsrust::Tof2MzConverter,
        ims_converter: &timsrust::Scan2ImConverter,
    ) -> TimsPeak {
        TimsPeak {
            intensity: self.intensity as u32,
            mz: mz_converter.convert(self.tof_index),
            mobility: ims_converter.convert(self.scan_index) as f32,
            npeaks: self.npeaks as u32,
        }
    }
}

impl ClusterAggregator<MaybeIntenseRawPeak, RawScaleTimsPeak> for RawWeightedTimsPeakAggregator {
    // Calculate the weight-weighted average of the cluster
    // for mz and ims. The intensity is kept as is.
    fn add(
        &mut self,
        elem: &MaybeIntenseRawPeak,
    ) {
        self.cumulative_cluster_intensity +=
            if elem.weight_only { 0 } else { elem.intensity } as u64;
        self.cumulative_cluster_weight += elem.intensity as u64;
        self.cumulative_weighted_cluster_tof += elem.tof_index as u64 * elem.intensity as u64;
        self.cumulative_weighted_cluster_scan += elem.scan_index as u64 * elem.intensity as u64;
        self.num_peaks += 1;
        if !elem.weight_only {
            self.num_intense_peaks += 1;
        };
    }

    fn aggregate(&self) -> RawScaleTimsPeak {
        // Use raw
        RawScaleTimsPeak {
            intensity: self.cumulative_cluster_intensity as f64,
            tof_index: self.cumulative_weighted_cluster_tof as f64
                / self.cumulative_cluster_weight as f64,
            scan_index: self.cumulative_weighted_cluster_scan as f64
                / self.cumulative_cluster_weight as f64,
            npeaks: self.num_intense_peaks,
        }
    }

    fn combine(
        self,
        other: Self,
    ) -> Self {
        Self {
            cumulative_weighted_cluster_tof: self.cumulative_weighted_cluster_tof
                + other.cumulative_weighted_cluster_tof,
            cumulative_weighted_cluster_scan: self.cumulative_weighted_cluster_scan
                + other.cumulative_weighted_cluster_scan,
            cumulative_cluster_weight: self.cumulative_cluster_weight
                + other.cumulative_cluster_weight,
            cumulative_cluster_intensity: self.cumulative_cluster_intensity
                + other.cumulative_cluster_intensity,
            num_peaks: self.num_peaks + other.num_peaks,
            num_intense_peaks: self.num_intense_peaks + other.num_intense_peaks,
        }
    }
}

fn denoise_frame_slice_window(
    frameslice_window: &[FrameSlice],
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
    let timer = utils::ContextTimer::new("dbscan_dfs", true, utils::LogLevel::TRACE);
    let fsw = FrameSliceWindow::new(frameslice_window);
    // dbscan_aggregate(
    //     &fsw,
    //     &fsw,
    //     &fsw,
    //     timer,
    //     min_n,
    //     min_intensity,
    //     TimsPeakAggregator::default,
    //     None::<&(dyn Fn(&f32) -> bool + Send + Sync)>,
    //     utils::LogLevel::TRACE,
    //     false,
    //     &[max_mz_extension as f32, max_ims_extension],
    //     false,
    // );

    let mut intensity_sorted_indices = frameslice_window
        .iter()
        .map(|x| x.intensities)
        .flat_map(|x| x)
        .enumerate()
        .map(|(i, x)| (i, *x as u64))
        .collect::<Vec<_>>();

    intensity_sorted_indices.par_sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

    let mut i_timer = timer.start_sub_timer("dbscan");
    let cluster_labels = dbscan_label_clusters(
        &fsw,
        &fsw,
        &fsw,
        min_n,
        min_intensity,
        &intensity_sorted_indices,
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

    let out = DenseFrameWindow {
        frame: DenseFrame {
            raw_peaks: centroids
                .into_iter()
                .map(|x| x.to_timspeak(mz_converter, ims_converter))
                .collect(),
            index: 0,
            rt: 0.,
            frame_type: timsrust::FrameType::MS2(timsrust::AcquisitionType::DIAPASEF),
            sorted: None,
        },
        ims_min: 0.,
        ims_max: 0.,
        mz_start: 0.,
        mz_end: 0.,
        group_id: 0,
        quad_group_id: 0,
    };

    out
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
        let num_windows = frame_window_slices.len();
        for (i, sv) in frame_window_slices.iter().enumerate() {
            info!("Denoising window {}/{}", i + 1, num_windows);
            let progbar = indicatif::ProgressBar::new(sv.len() as u64);
            let denoised_elements: Vec<DenseFrameWindow> = sv
                .as_slice()
                .par_windows(3)
                .progress_with(progbar)
                .map(|rt_window_of_slices| {
                    denoise_frame_slice_window(
                        rt_window_of_slices,
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
