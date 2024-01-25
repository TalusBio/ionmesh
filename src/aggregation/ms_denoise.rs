use core::panic;

use crate::aggregation::dbscan;
use crate::ms::frames::Converters;
use crate::ms::frames::DenseFrame;
use crate::ms::frames::DenseFrameWindow;
use crate::ms::tdf;
use crate::ms::tdf::DIAFrameInfo;
use crate::utils;
use crate::visualization::RerunPlottable;

use indicatif::ParallelProgressIterator;
use log::{info, trace, warn};
use rayon::prelude::*;
use timsrust::Frame;

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
    let intensity_ratio = frame_stats_end.tot_intensity / frame_stats_end.tot_intensity;
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
    mz_scaling: &f64,
    ims_scaling: &f32,
) -> DenseFrame {
    // I am 99% sure the compiler will remove this section when optimizing ... but I still need to test it.
    let frame_stats_start = FrameStats::new(&frame);
    let index = frame.index;
    // this is the line that matters
    // TODO move the scalings to parameters
    let denoised_frame =
        dbscan::dbscan_denseframes(frame, mz_scaling, ims_scaling, min_n, min_intensity);

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
    mz_scaling: &f64,
    ims_scaling: &f32,
) -> Vec<DenseFrameWindow> {
    let frame_windows = dia_frame_info
        .split_frame(frame)
        .expect("Only DIA frames should be passed to this function");

    

    frame_windows
        .into_iter()
        .map(|frame_window| {
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
                ims_scaling,
            );

            DenseFrameWindow {
                frame: denoised_frame,
                ims_start: denseframe_window.ims_start,
                ims_end: denseframe_window.ims_end,
                mz_start: denseframe_window.mz_start,
                mz_end: denseframe_window.mz_end,
                group_id: denseframe_window.group_id,
                quad_group_id: denseframe_window.quad_group_id,
            }
        })
        .collect::<Vec<_>>()
}

trait Denoiser<'a, T, W, X, Z>
where
    T: RerunPlottable<X> + std::marker::Send,
    W: Clone + RerunPlottable<Z> + std::marker::Send,
    X: Clone,
    Z: Clone,
    Vec<T>: IntoParallelIterator<Item = T>,
{
    fn denoise(&self, frame: T) -> W;
    // TODO maybe add a par_denoise_slice method
    // with the implementation ...
    fn par_denoise_slice(
        &self,
        mut frames: Vec<T>,
        record_stream: &mut Option<rerun::RecordingStream>,
        plotting_extras: (X, Z),
    ) -> Vec<W>
    where
        Self: Sync,
    {
        info!("Denoising {} frames", frames.len());
        // randomly viz 1/200 frames
        // Selecting a slice of 1/200 frames

        if let Some(stream) = record_stream.as_mut() {
            warn!("Viz is enabled, randomly subsetting 1/200 frames");
            let len_keep = frames.len() / 200;
            let start = rand::random::<usize>() % (frames.len() - len_keep);
            // let keep = [false, true, true, false, true];
            let mut keep = vec![false; start]
                .into_iter()
                .chain(vec![true; len_keep])
                .collect::<Vec<_>>();

            keep.append(&mut vec![false; frames.len() - start - len_keep]);
            let mut iter_keep = keep.iter();

            frames.retain(|_| *iter_keep.next().unwrap());

            for (i, frame) in frames.iter().enumerate() {
                info!("Logging frame {}", i);
                frame
                    .plot(
                        stream,
                        String::from("points/Original"),
                        None,
                        plotting_extras.0.clone(),
                    )
                    .unwrap();
            }
        }

        let progbar = indicatif::ProgressBar::new(frames.len() as u64);
        let denoised_frames: Vec<W> = frames
            .into_par_iter()
            .progress_with(progbar)
            .map(|x| self.denoise(x))
            .collect::<Vec<_>>();

        if let Some(stream) = record_stream.as_mut() {
            for (i, frame) in denoised_frames.iter().enumerate() {
                trace!("Logging frame {}", i);
                frame
                    .plot(
                        stream,
                        String::from("points/denoised"),
                        None,
                        plotting_extras.1.clone(),
                    )
                    .unwrap();
            }
        }

        denoised_frames
    }
}

struct FrameDenoiser {
    min_n: usize,
    min_intensity: u64,
    mz_scaling: f64,
    ims_scaling: f32,
    ims_converter: timsrust::Scan2ImConverter,
    mz_converter: timsrust::Tof2MzConverter,
}

impl<'a> Denoiser<'a, Frame, DenseFrame, Converters, Option<usize>> for FrameDenoiser {
    fn denoise(&self, frame: Frame) -> DenseFrame {
        let denseframe = DenseFrame::new(&frame, &self.ims_converter, &self.mz_converter);
        _denoise_denseframe(
            denseframe,
            self.min_n,
            self.min_intensity,
            &self.mz_scaling,
            &self.ims_scaling,
        )
    }
}

struct DIAFrameDenoiser {
    min_n: usize,
    min_intensity: u64,
    mz_scaling: f64,
    ims_scaling: f32,
    dia_frame_info: DIAFrameInfo,
    ims_converter: timsrust::Scan2ImConverter,
    mz_converter: timsrust::Tof2MzConverter,
}

impl<'a> Denoiser<'a, Frame, Vec<DenseFrameWindow>, Converters, Option<usize>>
    for DIAFrameDenoiser
{
    fn denoise(&self, frame: Frame) -> Vec<DenseFrameWindow> {
        _denoise_dia_frame(
            frame,
            self.min_n,
            self.min_intensity,
            &self.dia_frame_info,
            &self.ims_converter,
            &self.mz_converter,
            &self.mz_scaling,
            &self.ims_scaling,
        )
    }
}

pub fn read_all_ms1_denoising(
    path: String,
    min_intensity: u64,
    min_n: usize,
    mz_scaling: f64,
    ims_scaling: f32,
    record_stream: &mut Option<rerun::RecordingStream>,
) -> Vec<DenseFrame> {
    let reader = timsrust::FileReader::new(path).unwrap();

    let timer = utils::ContextTimer::new("Reading all MS1 frames", true, utils::LogLevel::INFO);

    let mut frames = reader.read_all_ms1_frames();
    timer.stop();

    let ims_converter = reader.get_scan_converter().unwrap();
    let mz_converter = reader.get_tof_converter().unwrap();

    frames.retain(|frame| match frame.frame_type {
            timsrust::FrameType::MS1 => true,
            _ => false,
        });

    // let min_intensity = 100u64;
    // let min_n: usize = 3;
    let ms1_denoiser = FrameDenoiser {
        min_n,
        mz_scaling,
        ims_scaling,
        min_intensity,
        ims_converter,
        mz_converter,
    };

    let converters = (ims_converter, mz_converter);
    let timer = utils::ContextTimer::new("Denoising all MS1 frames", true, utils::LogLevel::INFO);
    let out = ms1_denoiser.par_denoise_slice(frames, record_stream, (converters, None));
    timer.stop();
    out
}

// This could probably be a macro ...
pub fn read_all_dia_denoising(
    path: String,
    min_n: usize,
    min_intensity: u64,
    mz_scaling: f64,
    ims_scaling: f32,
    record_stream: &mut Option<rerun::RecordingStream>,
) -> Vec<DenseFrameWindow> {
    let timer = utils::ContextTimer::new("Reading all DIA frames", true, utils::LogLevel::INFO);
    let reader = timsrust::FileReader::new(path.clone()).unwrap();

    let dia_info = tdf::read_dia_frame_info(path.clone()).unwrap();
    let mut frames = reader.read_all_ms2_frames();

    let ims_converter = reader.get_scan_converter().unwrap();
    let mz_converter = reader.get_tof_converter().unwrap();
    timer.stop();

    // frames = frames
    //     .into_iter()
    //     .filter(|frame| match frame.frame_type {
    //         timsrust::FrameType::MS2(timsrust::AcquisitionType::DIAPASEF) => true,
    //         _ => false,
    //     })
    //     .collect();

    frames.retain(|frame| match frame.frame_type {
        timsrust::FrameType::MS2(timsrust::AcquisitionType::DIAPASEF) => true,
        _ => false,
    });

    // let min_intensity = 50u64;
    // let min_n: usize = 2;

    let denoiser = DIAFrameDenoiser {
        min_n,
        min_intensity,
        mz_scaling,
        ims_scaling,
        dia_frame_info: dia_info,
        ims_converter,
        mz_converter,
    };
    let converters = (ims_converter, mz_converter);

    let timer = utils::ContextTimer::new("Denoising all MS2 frames", true, utils::LogLevel::INFO);
    let split_frames = denoiser.par_denoise_slice(frames, record_stream, (converters, None));
    let out: Vec<DenseFrameWindow> = split_frames.into_iter().flatten().collect();
    timer.stop();
    out
}
