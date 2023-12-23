use crate::ms;
use crate::ms::TimsPeak;
use crate::quad;
use crate::quad::RadiusQuadTree;

use rayon::prelude::*;

trait DenoisableFrame {
    // Drops peaks that dont have at least one neighbor within a given mz/mobility tolerance
    fn min_neighbor_denoise(&mut self, mz_scaling: f64, ims_scaling: f64, min_n: usize) -> Self;
}

fn min_max_points(points: &[quad::Point]) -> (quad::Point, quad::Point) {
    let mut min_x = points[0].x;
    let mut max_x = points[0].x;
    let mut min_y = points[0].y;
    let mut max_y = points[0].y;

    for p in points.iter() {
        if p.x < min_x {
            min_x = p.x;
        } else if p.x > max_x {
            max_x = p.x;
        }

        if p.y < min_y {
            min_y = p.y;
        } else if p.y > max_y {
            max_y = p.y;
        }
    }

    (
        quad::Point { x: min_x, y: min_y },
        quad::Point { x: max_x, y: max_y },
    )
}

#[cfg(test)]
mod test_min_max {
    use super::*;

    #[test]
    fn test_min_max() {
        let points = vec![
            quad::Point { x: 0.0, y: 0.0 },
            quad::Point { x: 1.0, y: 1.0 },
            quad::Point { x: 2.0, y: 2.0 },
            quad::Point { x: 3.0, y: 3.0 },
            quad::Point { x: 4.0, y: 4.0 },
        ];

        let (min_point, max_point) = min_max_points(&points);

        assert_eq!(min_point, quad::Point { x: 0.0, y: 0.0 });
        assert_eq!(max_point, quad::Point { x: 4.0, y: 4.0 });
    }
}

impl DenoisableFrame for ms::DenseFrame {
    fn min_neighbor_denoise(
        &mut self,
        mz_scaling: f64,
        ims_scaling: f64,
        min_n: usize,
    ) -> ms::DenseFrame {
        // I could pre-sort here and use the window iterator,
        // to pre-filter for points with no neighbors in the mz dimension.

        // AKA could filter out the points with no mz neighbors sorting
        // and the use the tree to filter points with no mz+mobility neighbors.
        let out_frame_type = self.frame_type.clone();
        let out_rt = self.rt.clone();
        let out_index = self.index.clone();

        self.sort_by_mz();

        let mut prefiltered_peaks_bool: Vec<bool> = Vec::with_capacity(self.raw_peaks.len());

        let mut i_left = 0;
        let mut i_right = 0;
        let mut curr_i = 0;
        let mut mz_left = self.raw_peaks[0].mz;
        let mut mz_right = self.raw_peaks[0].mz;

        // 1. Slide the left index until the mz difference while sliding is more than the mz tolerance.
        // 2. Slide the right index until the mz difference while sliding is greater than the mz tolerance.
        // 3. If the number of points between the left and right index is greater than the minimum number of points, add them to the prefiltered peaks.

        while curr_i < self.raw_peaks.len() {
            let curr_mz = self.raw_peaks[curr_i].mz;
            let mut mz_diff = curr_mz - mz_left;
            if mz_diff > mz_scaling {
                // Slide the left index until the mz difference while sliding is more than the mz tolerance.
                while mz_diff > mz_scaling {
                    i_left += 1;
                    mz_left = self.raw_peaks[i_left].mz;
                    mz_diff = curr_mz - mz_left;
                }
            }

            // Slide the right index until the mz difference while sliding is greater than the mz tolerance.
            while (mz_right - curr_mz < mz_scaling) && ((i_right + 1) < self.raw_peaks.len()) {
                i_right += 1;
                mz_right = self.raw_peaks[i_right].mz;
            }

            // If the number of points between the left and right index is greater than the minimum number of points, add them to the prefiltered peaks.
            // println!("{} {}", i_left, i_right);
            prefiltered_peaks_bool.push(i_right - i_left > min_n);
            curr_i += 1;
        }

        // let skipped = prefiltered_peaks_bool
        //     .iter()
        //     .filter(|&b| !*b)
        //     .collect::<Vec<_>>()
        //     .len();

        // println!(
        //     "Skipped {} peaks out of {} total peaks",
        //     skipped,
        //     self.raw_peaks.len()
        // );

        // Filter the peaks and replace the raw peaks with the filtered peaks.
        let prefiltered_peaks = self
            .raw_peaks
            .clone()
            .into_iter()
            .zip(prefiltered_peaks_bool.into_iter())
            .filter(|(_, b)| *b)
            .map(|(peak, _)| peak)
            .collect::<Vec<_>>();

        let quad_points = prefiltered_peaks
            .iter()
            .map(|peak| quad::Point {
                x: (peak.mz / mz_scaling),
                y: (peak.mobility as f64 / ims_scaling),
            })
            .collect::<Vec<_>>();

        let (min_point, max_point) = min_max_points(&quad_points);
        let min_x = min_point.x;
        let min_y = min_point.y;
        let max_x = max_point.x;
        let max_y = max_point.y;

        let boundary = quad::Boundary::new ( 
            (max_x + min_x) / 2.0,
            (max_y + min_y) / 2.0,
            max_x - min_x,
            max_y - min_y,
         );

        let mut tree: RadiusQuadTree<'_, TimsPeak> = quad::RadiusQuadTree::new(boundary, 20, 1.);
        for (point, timspeak) in quad_points.iter().zip(prefiltered_peaks.iter()) {
            tree.insert(point.clone(), timspeak);
        }

        let mut denoised_peaks = Vec::with_capacity(prefiltered_peaks.len());
        for (point, peaks) in quad_points.iter().zip(prefiltered_peaks.iter()) {
            let mut result_counts = 0;

            // TODO: implement an 'any neighbor' method.
            tree.count_query(*point, &mut result_counts);

            if result_counts as usize >= min_n {
                denoised_peaks.push(peaks.clone());
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

pub fn read_all_ms1_denoising(path: String) -> Vec<ms::DenseFrame> {
    let reader = timsrust::FileReader::new(path).unwrap();
    let mut frames = reader.read_all_ms1_frames();

    let ims_converter = reader.get_scan_converter().unwrap();
    let mz_converter = reader.get_tof_converter().unwrap();
    let rt_converter = reader.get_frame_converter().unwrap();

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

    let denoised_frames = frames
        .par_iter_mut()
        .filter(|frame| match frame.frame_type {
            timsrust::FrameType::MS1 => true,
            _ => false,
        })
        .map(|frame| {
            let mut dense =
                ms::DenseFrame::new(frame, &ims_converter, &mz_converter, &rt_converter);

            let tot_intensity_start = dense.raw_peaks.iter().map(|peak| peak.intensity as f64).sum::<f64>();
            let denoised_frame = dense.min_neighbor_denoise(0.015, 0.015, 3);
            let tot_intensity_end = denoised_frame.raw_peaks.iter().map(|peak| peak.intensity as f64).sum::<f64>();
            let intensity_ratio = tot_intensity_end / tot_intensity_start;

            println!("Denoising frame {} with intensity ratio {}", frame.index, intensity_ratio);
            denoised_frame
        })
        .collect::<Vec<_>>();

    denoised_frames
}
