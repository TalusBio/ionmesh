use crate::ms::DenseFrame;
use crate::ms::TimsPeak;
use crate::quad;
use crate::quad::RadiusQuadTree;
use crate::{ms, tdf};

use indicatif::{ParallelProgressIterator, ProgressBar, ProgressStyle};
use log::{info, trace, warn};
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

// TODO: rename count_neigh_monotonocally_increasing
// because it can do more than just count neighbors....

#[inline(always)]
fn count_neigh_monotonocally_increasing<T, R, W>(
    elems: &[T],
    key: &dyn Fn(&T) -> R,
    max_dist: R,
    out_func: &dyn Fn(&usize, &usize) -> W,
) -> Vec<W>
where
    R: PartialOrd + Copy + std::ops::Sub<Output = R> + std::ops::Add<Output = R> + Default,
    T: Copy,
    W: Default + Copy,
{
    let mut prefiltered_peaks_bool: Vec<W> = vec![W::default(); elems.len()];

    let mut i_left = 0;
    let mut i_right = 0;
    let mut mz_left = key(&elems[0]);
    let mut mz_right = key(&elems[0]);

    // Does the cmpiler re-use the memory here?
    // let mut curr_mz = R::default();
    // let mut left_mz_diff = R::default();
    // let mut right_mz_diff = R::default();

    // 1. Slide the left index until the mz difference while sliding is more than the mz tolerance.
    // 2. Slide the right index until the mz difference while sliding is greater than the mz tolerance.
    // 3. If the number of points between the left and right index is greater than the minimum number of points, add them to the prefiltered peaks.

    let elems_len = elems.len();
    let elems_len_minus_one = elems_len - 1;
    for (curr_i, elem) in elems.iter().enumerate() {
        let curr_mz = key(elem);
        let mut left_mz_diff = curr_mz - mz_left;
        let mut right_mz_diff = mz_right - curr_mz;

        while left_mz_diff > max_dist {
            i_left += 1;
            mz_left = key(&elems[i_left]);
            left_mz_diff = curr_mz - mz_left;
        }

        // Slide the right index until the mz difference while sliding is greater than the mz tolerance.
        while (right_mz_diff < max_dist) && (i_right < elems_len) {
            i_right += 1;
            mz_right = key(&elems[i_right.min(elems_len_minus_one)]);
            right_mz_diff = mz_right - curr_mz;
        }

        // If the number of points between the left and right index is greater than the minimum number of points, add them to the prefiltered peaks.
        // println!("{} {}", i_left, i_right);
        if i_left < i_right {
            prefiltered_peaks_bool[curr_i] = out_func(&i_right, &(i_left));
        }

        if cfg!(test) {
            assert!(i_left <= i_right);
        }
    }

    prefiltered_peaks_bool
}

#[cfg(test)]
mod test_count_neigh {
    use super::*;

    #[test]
    fn test_count_neigh() {
        let elems = vec![0.0, 1.0, 2.0, 3.0, 4.0];

        let prefiltered_peaks_bool =
            count_neigh_monotonocally_increasing(&elems, &|x| *x, 1.1, &|i_right, i_left| {
                (i_right - i_left) >= 3
            });

        assert_eq!(prefiltered_peaks_bool, vec![false, true, true, true, false]);
    }
}

fn denseframe_to_quadtree_points(
    denseframe: &mut ms::DenseFrame,
    mz_scaling: f64,
    ims_scaling: f64,
    min_n: usize,
) -> (Vec<quad::Point>, Vec<TimsPeak>, quad::Boundary) {
    // Initial pre-filtering step
    denseframe.sort_by_mz();

    let num_neigh = count_neigh_monotonocally_increasing(
        &denseframe.raw_peaks,
        &|peak| peak.mz,
        mz_scaling,
        &|i_right, i_left| (i_right - i_left) >= min_n,
    );

    // Filter the peaks and replace the raw peaks with the filtered peaks.
    let prefiltered_peaks = denseframe
        .raw_peaks
        .clone()
        .into_iter()
        .zip(num_neigh.into_iter())
        .filter(|(_, b)| *b)
        .map(|(peak, _)| peak.clone()) // Clone the TimsPeak
        .collect::<Vec<_>>();

    let quad_points = prefiltered_peaks // denseframe.raw_peaks //
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

    let boundary = quad::Boundary::new(
        (max_x + min_x) / 2.0,
        (max_y + min_y) / 2.0,
        max_x - min_x,
        max_y - min_y,
    );

    (quad_points, prefiltered_peaks, boundary)
    // (quad_points, denseframe.raw_peaks.clone(), boundary)

    // NOTE: I would like to do this, but I dont know how to fix the lifetime issues...
    // let mut tree: RadiusQuadTree<'_, TimsPeak> = quad::RadiusQuadTree::new(boundary, 20, 1.);
    // for (point, timspeak) in quad_points.iter().zip(prefiltered_peaks.iter()) {
    //     tree.insert(point.clone(), timspeak);
    // }

    // (quad_points, prefiltered_peaks, tree)
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

// Pseudocode from wikipedia.
// Donate to wikipedia y'all. :3
// DBSCAN(DB, distFunc, eps, minPts) {
//     C := 0                                                  /* Cluster counter */
//     for each point P in database DB {
//         if label(P) ≠ undefined then continue               /* Previously processed in inner loop */
//         Neighbors N := RangeQuery(DB, distFunc, P, eps)     /* Find neighbors */
//         if |N| < minPts then {                              /* Density check */
//             label(P) := Noise                               /* Label as Noise */
//             continue
//         }
//         C := C + 1                                          /* next cluster label */
//         label(P) := C                                       /* Label initial point */
//         SeedSet S := N \ {P}                                /* Neighbors to expand */
//         for each point Q in S {                             /* Process every seed point Q */
//             if label(Q) = Noise then label(Q) := C          /* Change Noise to border point */
//             if label(Q) ≠ undefined then continue           /* Previously processed (e.g., border point) */
//             label(Q) := C                                   /* Label neighbor */
//             Neighbors N := RangeQuery(DB, distFunc, Q, eps) /* Find neighbors */
//             if |N| ≥ minPts then {                          /* Density check (if Q is a core point) */
//                 S := S ∪ N                                  /* Add new neighbors to seed set */
//             }
//         }
//     }
// }

// Variations ...
// 1. Use a quadtree to find neighbors
// 2. Sort the pointd by decreasing intensity (more intense points adopt first).
// 3. Use an intensity threshold intead of a minimum number of neighbors.

#[derive(Debug, PartialEq, Clone)]
enum ClusterLabel<T> {
    Unassigned,
    Noise,
    Cluster(T),
}

fn dbscan(
    denseframe: &mut ms::DenseFrame,
    mz_scaling: f64,
    ims_scaling: f64,
    min_n: usize,
    min_intensity: u64,
) -> ms::DenseFrame {
    // I could pre-sort here and use the window iterator,
    // to pre-filter for points with no neighbors in the mz dimension.

    // AKA could filter out the points with no mz neighbors sorting
    // and the use the tree to filter points with no mz+mobility neighbors.

    // NOTE: the multiple quad isolation windows in DIA are not being handled just yet.
    let out_frame_type: timsrust::FrameType = denseframe.frame_type.clone();
    let out_rt: f64 = denseframe.rt.clone();
    let out_index: usize = denseframe.index.clone();

    let (quad_points, prefiltered_peaks, boundary) =
        denseframe_to_quadtree_points(denseframe, mz_scaling, ims_scaling, min_n.saturating_sub(1));

    let mut tree = RadiusQuadTree::new(boundary, 20, 1.);

    // // Sort by decreasing intensity
    // // I am not the biggest fan of this nested tuple ...

    // let mut intensity_sorted_peaks = quad_points
    //     .iter()
    //     .zip(prefiltered_peaks.iter())
    //     .enumerate()
    //     .collect::<Vec<(usize, (&quad::Point, &ms::TimsPeak))>>();
    // intensity_sorted_peaks.sort_unstable_by(|a, b| {
    //     b.1.1.intensity.partial_cmp(&a.1.1.intensity).unwrap()
    // });

    // let isp = intensity_sorted_peaks.clone();
    // for (i, (point, peak)) in isp.iter() {
    //     tree.insert(*point.clone(), (i, peak.clone()));
    // }

    // >>>>>> this reimplement this to use an index pointer instead of
    // >>>>>> sorting!

    let quad_indices = (0..quad_points.len()).collect::<Vec<_>>();

    for (quad_point, i) in quad_points.iter().zip(quad_indices.iter()) {
        tree.insert(quad_point.clone(), i);
    }
    let mut intensity_sorted_indices = prefiltered_peaks
        .iter()
        .enumerate()
        .map(|(i, peak)| (i.clone(), peak.intensity.clone()))
        .collect::<Vec<_>>();

    intensity_sorted_indices.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

    let mut cluster_labels = vec![ClusterLabel::Unassigned; prefiltered_peaks.len()];
    let mut cluster_id = 0;

    for (point_index, _intensity) in intensity_sorted_indices.iter() {
        let point_index = *point_index;
        if cluster_labels[point_index] != ClusterLabel::Unassigned {
            continue;
        }

        let mut neighbors = Vec::new();
        let query_point = quad_points[point_index].clone();
        tree.query(query_point, &mut neighbors);

        // Do I need to care about overflows here?
        let neighbor_intensity_total = neighbors
            .iter()
            .map(|(_, i)| prefiltered_peaks[**i].intensity as u64)
            .sum::<u64>();

        if neighbors.len() < min_n || neighbor_intensity_total < min_intensity {
            cluster_labels[point_index] = ClusterLabel::Noise;
            continue;
        }

        cluster_id += 1;
        cluster_labels[point_index] = ClusterLabel::Cluster(cluster_id);
        let mut seed_set = neighbors.clone();

        const MAX_EXTENSION_DISTANCE: f64 = 5.;

        while let Some(neighbor) = seed_set.pop() {
            let neighbor_index = neighbor.1.clone();
            if cluster_labels[neighbor_index] == ClusterLabel::Noise {
                cluster_labels[neighbor_index] = ClusterLabel::Cluster(cluster_id);
            }

            if cluster_labels[neighbor_index] != ClusterLabel::Unassigned {
                continue;
            }

            cluster_labels[neighbor_index] = ClusterLabel::Cluster(cluster_id);

            let mut neighbors = Vec::new();
            tree.query(neighbor.0, &mut neighbors);

            let neighbor_intensity_total = neighbors
                .iter()
                .map(|(_, i)| prefiltered_peaks[**i].intensity as u64)
                .sum::<u64>();

            if neighbors.len() >= min_n && neighbor_intensity_total >= min_intensity {
                // Keep only the neighbors that are not already in a cluster
                neighbors = neighbors
                    .into_iter()
                    .filter(|(_, i)| match cluster_labels[**i] {
                        ClusterLabel::Cluster(_) => false,
                        _ => true,
                    })
                    .collect::<Vec<_>>();

                // Keep only the neighbors that are within the max extension distance
                // It might be worth setting a different max extension distance for the mz and mobility dimensions.
                neighbors = neighbors
                    .into_iter()
                    .filter(|(p, _)| {
                        let dist = (p.x - query_point.x).powi(2) + (p.y - query_point.y).powi(2);
                        dist <= MAX_EXTENSION_DISTANCE.powi(2)
                    })
                    .collect::<Vec<_>>();

                seed_set.extend(neighbors);
            }
        }
    }

    // Each element is a tuple representing the summed cluster intensity, mz, and mobility.
    // And will be used to calculate the weighted average of mz and mobility AND the total intensity.
    let mut cluster_vecs = vec![(0u64, 0f64, 0f64); cluster_id as usize];
    for (point_index, cluster_label) in cluster_labels.iter().enumerate() {
        match cluster_label {
            ClusterLabel::Cluster(cluster_id) => {
                let cluster_idx = *cluster_id as usize - 1;
                let timspeak = prefiltered_peaks[point_index];
                let f64_intensity = timspeak.intensity as f64;
                let cluster_vec = &mut cluster_vecs[cluster_idx];
                cluster_vec.0 += timspeak.intensity as u64;
                cluster_vec.1 += (timspeak.mz as f64) * f64_intensity;
                cluster_vec.2 += (timspeak.mobility as f64) * f64_intensity;
            }
            _ => {}
        }
    }

    let denoised_peaks = cluster_vecs
        .iter_mut()
        .map(|(cluster_intensity, cluster_mz, cluster_mobility)| {
            let cluster_intensity = cluster_intensity; // Note not averaged
            let cluster_mz = *cluster_mz / *cluster_intensity as f64;
            let cluster_mobility = *cluster_mobility / *cluster_intensity as f64;
            ms::TimsPeak {
                intensity: u32::try_from(*cluster_intensity).ok().unwrap(),
                mz: cluster_mz,
                mobility: cluster_mobility as f32,
            }
        })
        .collect::<Vec<_>>();

    // TODO add an option to keep noise points

    ms::DenseFrame {
        raw_peaks: denoised_peaks,
        index: out_index,
        rt: out_rt,
        frame_type: out_frame_type,
        sorted: None,
    }
}

fn denoise_denseframe_vec(
    mut frames: Vec<DenseFrame>,
    rt_converter: timsrust::Frame2RtConverter,
    ims_converter: timsrust::Scan2ImConverter,
    mz_converter: timsrust::Tof2MzConverter,
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
            let denoised_frame = dbscan(frame, 0.015, 0.03, min_n, min_intensity);
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
        rt_converter,
        ims_converter,
        mz_converter,
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
                rt_converter.clone(),
                ims_converter.clone(),
                mz_converter.clone(),
                min_intensity,
                min_n,
            );
            out.extend(denoised_frames);
        }
    }

    out
}
