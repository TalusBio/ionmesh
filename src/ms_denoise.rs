use crate::dbscan;
use crate::ms::frames::DenseFrame;
use crate::tdf;
use crate::visualization::RerunPlottable;

use indicatif::{ParallelProgressIterator, ProgressStyle};
use log::{info, trace, warn};
use rayon::prelude::*;

fn _denoise_denseframe(frame: &mut DenseFrame, min_n: usize, min_intensity: u64) -> DenseFrame {
    let max_intensity_start = frame
        .raw_peaks
        .iter()
        .map(|peak| peak.intensity)
        .max()
        .unwrap_or(0);
    let num_peaks_start = frame.raw_peaks.len();
    let tot_intensity_start = frame
        .raw_peaks
        .iter()
        .map(|peak| peak.intensity as f64)
        .sum::<f64>();

    // this is the line that matters
    let denoised_frame = dbscan::dbscan(frame, 0.02, 0.03, min_n, min_intensity);
    let tot_intensity_end = denoised_frame
        .raw_peaks
        .iter()
        .map(|peak| peak.intensity as f64)
        .sum::<f64>();
    let num_peaks_end = denoised_frame.raw_peaks.len();
    let max_intensity_end = denoised_frame
        .raw_peaks
        .iter()
        .map(|peak| peak.intensity)
        .max()
        .unwrap_or(0);
    let intensity_ratio = tot_intensity_end / tot_intensity_start;
    let peak_ratio = num_peaks_end as f64 / num_peaks_start as f64;

    if cfg!(debug_assertions) {
        trace!(
            "Denoising frame {} with intensity ratio {:.2}, peak_ratio {:.2}, prior_max {}, curr_max {}",
            frame.index, intensity_ratio, peak_ratio, max_intensity_start, max_intensity_end
        );
        if max_intensity_end < max_intensity_start {
            println!(
                "End max intensity is greater than start max intensity for frame {}!",
                frame.index
            );
            println!("Before: {}", max_intensity_start);
            println!("After: {}", max_intensity_end);
        }

        // Allow the next one to fail if there is very low intensity to start with.
        assert!((max_intensity_end >= max_intensity_start) || (max_intensity_start < 1000));
        assert!(peak_ratio <= 1.);
    };

    denoised_frame
}

fn denoise_denseframe_vec(
    mut frames: Vec<DenseFrame>,
    min_intensity: u64,
    min_n: usize,
    record_stream: &mut Option<rerun::RecordingStream>,
) -> Vec<DenseFrame> {
    info!("Denoising {} frames", frames.len());
    // randomly viz 1/200 frames
    if let Some(stream) = record_stream.as_mut() {
        warn!("Viz is enabled, randomly subsetting 1/200 frames");
        let frames_keep: Vec<DenseFrame> = frames
            .into_iter()
            .filter_map(|x| {
                if rand::random::<f64>() < (1. / 200.) {
                    Some(x)
                } else {
                    None
                }
            })
            .collect::<Vec<_>>();

        frames = frames_keep;

        for frame in frames.iter() {
            info!("Logging frame {}", frame.index);
            frame
                .plot(stream, String::from("points/Original"), frame.rt as f32)
                .unwrap();
        }
    }

    let style = ProgressStyle::default_bar();

    let denoised_frames: Vec<DenseFrame> = frames
        .par_iter_mut()
        .progress_with_style(style)
        .map(|x| _denoise_denseframe(x, min_n, min_intensity))
        .collect::<Vec<_>>();

    if let Some(stream) = record_stream.as_mut() {
        for frame in denoised_frames.iter() {
            trace!("Logging frame {}", frame.index);
            frame
                .plot(stream, String::from("points/denoised"), frame.rt as f32)
                .unwrap();
        }
    }

    denoised_frames
}

pub fn read_all_ms1_denoising(
    path: String,
    record_stream: &mut Option<rerun::RecordingStream>,
) -> Vec<DenseFrame> {
    let reader = timsrust::FileReader::new(path).unwrap();
    info!("Reading all MS1 frames");
    let mut frames = reader.read_all_ms1_frames();

    let ims_converter = reader.get_scan_converter().unwrap();
    let mz_converter = reader.get_tof_converter().unwrap();

    frames = frames
        .into_iter()
        .filter(|frame| match frame.frame_type {
            timsrust::FrameType::MS1 => true,
            _ => false,
        })
        .collect();

    let denseframes = frames
        .into_par_iter()
        .map(|frame| {
            let dense = DenseFrame::new(&frame, &ims_converter, &mz_converter);
            dense
        })
        .collect();

    let min_intensity = 100u64;
    let min_n: usize = 3;

    denoise_denseframe_vec(denseframes, min_intensity, min_n, record_stream)
}

// This could probably be a macro ...
pub fn read_all_dia_denoising(
    path: String,
    record_stream: &mut Option<rerun::RecordingStream>,
) -> Vec<DenseFrame> {
    info!("Reading all DIA frames");
    let reader = timsrust::FileReader::new(path.clone()).unwrap();
    let mut frames = reader.read_all_ms2_frames();

    let ims_converter = reader.get_scan_converter().unwrap();
    let mz_converter = reader.get_tof_converter().unwrap();

    frames = frames
        .into_iter()
        .filter(|frame| match frame.frame_type {
            timsrust::FrameType::MS2(timsrust::AcquisitionType::DIAPASEF) => true,
            _ => false,
        })
        .collect();

    let denseframes = frames
        .into_par_iter()
        .map(|frame| {
            let dense = DenseFrame::new(&frame, &ims_converter, &mz_converter);
            dense
        })
        .collect();

    let dia_info = tdf::read_dia_frame_info(path.clone()).unwrap();
    let split_frames = dia_info.split_dense_frames(denseframes);

    let min_intensity = 50u64;
    let min_n: usize = 2;

    let mut out = Vec::new();
    for dia_group in split_frames {
        for quad_group in dia_group {
            let denoised_frames = denoise_denseframe_vec(
                quad_group.into_iter().map(|x| x.frame).collect(),
                min_intensity,
                min_n,
                record_stream,
            );
            out.extend(denoised_frames);
        }
    }

    out
}
