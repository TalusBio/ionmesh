use crate::dbscan;
use crate::ms::DenseFrame;
use crate::ms::TimsPeak;
use crate::quad;
use crate::quad::{denseframe_to_quadtree_points, Boundary, RadiusQuadTree};
use crate::{ms, tdf};

use indicatif::{ParallelProgressIterator, ProgressBar, ProgressStyle};
use log::{info, trace, warn};
use rayon::prelude::*;

trait DenoisableFrame {
    // Drops peaks that dont have at least one neighbor within a given mz/mobility tolerance
    fn min_neighbor_denoise(&mut self, mz_scaling: f64, ims_scaling: f64, min_n: usize) -> Self;
}

impl DenoisableFrame for ms::DenseFrame {
    /// Drops peaks that dont have at least one neighbor within a given mz/mobility tolerance
    fn min_neighbor_denoise(
        &mut self,
        mz_scaling: f64,
        ims_scaling: f64,
        min_n: usize,
    ) -> ms::DenseFrame {
        // AKA could filter out the points with no mz neighbors sorting
        // and the use the tree to filter points with no mz+mobility neighbors.

        // NOTE: the multiple quad isolation windows in DIA are not being handled just yet.
        let out_frame_type = self.frame_type.clone();
        let out_rt = self.rt.clone();
        let out_index = self.index.clone();

        let (quad_points, prefiltered_peaks, boundary) =
            denseframe_to_quadtree_points(self, mz_scaling, ims_scaling, min_n);

        let mut prefiltered_peaks = prefiltered_peaks.to_owned();

        let mut tree: RadiusQuadTree<'_, Option<usize>> = RadiusQuadTree::new(boundary, 20, 1.);

        let num_peaks = prefiltered_peaks.len();
        for point in quad_points.iter() {
            tree.insert(point.clone(), &None);
        }

        let mut denoised_peaks = Vec::with_capacity(num_peaks);

        let min_n = min_n as u64;
        for (point, peaks) in quad_points.iter().zip(prefiltered_peaks.iter_mut()) {
            let mut result_counts = 0;

            // TODO: implement an 'any neighbor' method.
            tree.count_query(*point, &mut result_counts);

            if result_counts >= min_n {
                denoised_peaks.push(*peaks);
            }
        }

        ms::DenseFrame {
            raw_peaks: denoised_peaks,
            index: out_index,
            rt: out_rt,
            frame_type: out_frame_type,
            sorted: None,
        }
    }
}

fn log_denseframe_points(
    frame: &ms::DenseFrame,
    rec: &mut rerun::RecordingStream,
    entry_path: String,
) -> Result<(), Box<dyn std::error::Error>> {
    let quad_points = frame
        .raw_peaks
        .iter()
        .map(|peak| quad::Point {
            x: (peak.mz / 10.),
            y: (100. * peak.mobility as f64),
        })
        .collect::<Vec<_>>();

    let max_intensity = frame
        .raw_peaks
        .iter()
        .map(|peak| peak.intensity)
        .max()
        .unwrap_or(0) as f32;

    let radii = frame
        .raw_peaks
        .iter()
        .map(|peak| (peak.intensity as f32) / max_intensity)
        .collect::<Vec<_>>();

    rec.log(
        entry_path,
        &rerun::Points2D::new(
            quad_points
                .iter()
                .map(|point| (point.x as f32, point.y as f32)),
        )
        .with_radii(radii),
    )?;

    Ok(())
}

// #[cfg(feature='viz')]
fn setup_recorder() -> rerun::RecordingStream {
    let rec = rerun::RecordingStreamBuilder::new("rerun_jspp_denoiser").connect();

    return rec.unwrap();
}

fn denoise_denseframe_vec(
    mut frames: Vec<DenseFrame>,
    min_intensity: u64,
    min_n: usize,
) -> Vec<ms::DenseFrame> {
    info!("Denoising {} frames", frames.len());
    let mut rec = Option::None;
    if cfg!(feature = "viz") {
        rec = Some(setup_recorder());
    }

    // randomly viz 1/200 frames
    if cfg!(feature = "viz") {
        let rec: &mut rerun::RecordingStream = rec.as_mut().unwrap();

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
            rec.set_time_sequence("frame_idx", frame.index as i64);
            log_denseframe_points(&frame, rec, String::from("points/Original")).unwrap();
        }
    }

    // let mut denoised_frames = Vec::new();

    // // TODO: parallelize this
    // for frame in frames.iter_mut() {
    //     match frame.frame_type {
    //         timsrust::FrameType::MS1 => {}
    //         _ => continue,
    //     }
    //     println!("Denoising frame {}", frame.index);
    //     let dense = ms::DenseFrame::new(frame, &ims_converter, &mz_converter, &rt_converter);
    //     let denoised_frame = dense.min_neighbor_denoise(0.015, 0.015, 5);
    //     denoised_frames.push(denoised_frame);
    // }

    let style = ProgressStyle::default_bar();

    let denoised_frames: Vec<ms::DenseFrame> = frames
        .par_iter_mut()
        .progress_with_style(style)
        .map(|frame| {
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
            // let denoised_frame = dense.min_neighbor_denoise(0.015, 0.015, 2);
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
                    println!("End max intensity is greater than start max intensity for frame {}!", frame.index);
                    println!("Before: {}", max_intensity_start);
                    println!("After: {}", max_intensity_end);
                }

                // Allow the next one to fail if there is very low intensity to start with.
                assert!((max_intensity_end >= max_intensity_start) || (max_intensity_start < 1000));
                assert!(peak_ratio <= 1.);
            };
            denoised_frame
        })
        .collect::<Vec<_>>();

    if cfg!(feature = "viz") {
        let rec = rec.as_mut().unwrap();
        for frame in denoised_frames.iter() {
            trace!("Logging frame {}", frame.index);
            rec.set_time_sequence("frame_idx", frame.index as i64);
            log_denseframe_points(frame, rec, String::from("points/denoised")).unwrap();
        }
    }

    denoised_frames
}

pub fn read_all_ms1_denoising(path: String) -> Vec<ms::DenseFrame> {
    let reader = timsrust::FileReader::new(path).unwrap();
    info!("Reading all MS1 frames");
    let mut frames = reader.read_all_ms1_frames();

    let ims_converter = reader.get_scan_converter().unwrap();
    let mz_converter = reader.get_tof_converter().unwrap();
    let rt_converter = reader.get_frame_converter().unwrap();

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
            let dense = ms::DenseFrame::new(&frame, &ims_converter, &mz_converter, &rt_converter);
            dense
        })
        .collect();

    let min_intensity = 100u64;
    let min_n: usize = 3;

    denoise_denseframe_vec(
        denseframes,
        min_intensity,
        min_n,
    )
}

// This could probably be a macro ...
pub fn read_all_dia_denoising(path: String) -> Vec<ms::DenseFrame> {
    info!("Reading all DIA frames");
    let reader = timsrust::FileReader::new(path.clone()).unwrap();
    let mut frames = reader.read_all_ms2_frames();

    let ims_converter = reader.get_scan_converter().unwrap();
    let mz_converter = reader.get_tof_converter().unwrap();
    let rt_converter = reader.get_frame_converter().unwrap();

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
            let dense = ms::DenseFrame::new(&frame, &ims_converter, &mz_converter, &rt_converter);
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
            );
            out.extend(denoised_frames);
        }
    }

    out
}
