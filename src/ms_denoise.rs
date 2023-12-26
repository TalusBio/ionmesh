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

    let boundary = quad::Boundary::new(
        (max_x + min_x) / 2.0,
        (max_y + min_y) / 2.0,
        max_x - min_x,
        max_y - min_y,
    );

    (quad_points, prefiltered_peaks, boundary)

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

        let mut tree: RadiusQuadTree<'_, TimsPeak> = RadiusQuadTree::new(boundary, 20, 1.);

        for (point, peak) in quad_points.iter().zip(prefiltered_peaks.iter()) {
            tree.insert(point.clone(), peak);
        }

        let mut denoised_peaks = Vec::with_capacity(prefiltered_peaks.len());

        let min_n = min_n as u64;
        for (point, peaks) in quad_points.iter().zip(prefiltered_peaks.iter()) {
            let mut result_counts = 0;

            // TODO: implement an 'any neighbor' method.
            tree.count_query(*point, &mut result_counts);

            if result_counts >= min_n {
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

// #[cfg(feature='viz')]
fn log_points(
    points: &[quad::Point],
    rec: &mut rerun::RecordingStream,
) -> Result<(), Box<dyn std::error::Error>> {
    use rerun::{
        demo_util::{bounce_lerp, color_spiral},
        external::glam,
    };

    rec.log(
        "random",
        &rerun::Points2D::new(points.iter().map(|point| (point.x as f32, point.y as f32))),
    )?;

    Ok(())
}

fn log_denseframe_points(
    frame: &ms::DenseFrame,
    rec: &mut rerun::RecordingStream,
    entry_path: String,
) -> Result<(), Box<dyn std::error::Error>> {
    use rerun::{
        demo_util::{bounce_lerp, color_spiral},
        external::glam,
    };

    let quad_points = frame
        .raw_peaks
        .iter()
        .map(|peak| quad::Point {
            x: (peak.mz / 10.),
            y: (100. * peak.mobility as f64),
        })
        .collect::<Vec<_>>();

    let radii = frame
        .raw_peaks
        .iter()
        .map(|peak| (peak.intensity as f32).log2() / 50.)
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
    Noise,
    Border(T),
    Core(T),
    Unassigned,
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

    let quad_indices: Vec<usize> = quad_points
        .iter()
        .enumerate()
        .map(|(i, _)| i)
        .collect::<Vec<_>>();

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
    let mut cluster_vecs: Vec<Vec<usize>> = Vec::with_capacity(prefiltered_peaks.len());
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

        // Add here intensity cutoff
        if neighbors.len() < min_n || neighbor_intensity_total < min_intensity {
            cluster_labels[point_index] = ClusterLabel::Noise;
            continue;
        }

        cluster_id += 1;
        cluster_labels[point_index] = ClusterLabel::Core(cluster_id);
        let mut cluster_vec_local = vec![point_index];
        let mut seed_set = neighbors.clone();

        while let Some(neighbor) = seed_set.pop() {
            let neighbor_index = neighbor.1.clone();
            if cluster_labels[neighbor_index] == ClusterLabel::Noise {
                cluster_labels[neighbor_index] = ClusterLabel::Border(cluster_id);
                cluster_vec_local.push(neighbor_index);
            }

            if cluster_labels[neighbor_index] != ClusterLabel::Unassigned {
                continue;
            }

            cluster_labels[neighbor_index] = ClusterLabel::Core(cluster_id);
            cluster_vec_local.push(neighbor_index);

            // Add here intensity cutoff

            let mut neighbors = Vec::new();
            tree.query(neighbor.0, &mut neighbors);

            let neighbor_intensity_total = neighbors
                .iter()
                .map(|(_, i)| prefiltered_peaks[**i].intensity as u64)
                .sum::<u64>();

            if neighbors.len() >= min_n && neighbor_intensity_total >= min_intensity {
                seed_set.extend(neighbors);
            }
        }

        cluster_vecs.push(cluster_vec_local);
    }

    let denoised_peaks = cluster_vecs
        .iter()
        .map(|cluster| {
            let mut cluster_intensity = 0;
            let mut cluster_mz = 0.0;
            let mut cluster_mobility = 0.0;
            for point_index in cluster {
                let timspeak = prefiltered_peaks[*point_index];
                let f64_intensity = timspeak.intensity as f64;
                cluster_intensity += timspeak.intensity as u64;
                cluster_mz += (timspeak.mz as f64) * f64_intensity;
                cluster_mobility += (timspeak.mobility as f64) * f64_intensity;
            }
            let cluster_intensity = cluster_intensity; // Note not averaged
            let cluster_mz = cluster_mz / cluster_intensity as f64;
            let cluster_mobility = cluster_mobility / cluster_intensity as f64;
            ms::TimsPeak {
                intensity: cluster_intensity as u32,
                mz: cluster_mz,
                mobility: cluster_mobility as f32,
            }
        })
        .collect::<Vec<_>>();
    ms::DenseFrame {
        raw_peaks: denoised_peaks,
        index: out_index,
        rt: out_rt,
        frame_type: out_frame_type,
        sorted: None,
    }
}

pub fn read_all_ms1_denoising(path: String) -> Vec<ms::DenseFrame> {
    let mut rec = Option::None;
    if cfg!(feature = "viz") {
        rec = Some(setup_recorder());
    }

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

    frames = frames
        .into_iter()
        .filter(|frame| match frame.frame_type {
            timsrust::FrameType::MS1 => true,
            _ => false,
        })
        .collect();

    // randomly viz 1/200 frames
    if cfg!(feature = "viz") {
        let rec: &mut rerun::RecordingStream = rec.as_mut().unwrap();
        let frames_keep: Vec<timsrust::Frame> = frames
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
            println!("Logging frame {}", frame.index);
            rec.set_time_sequence("frame_idx", frame.index as i64);
            let dense = ms::DenseFrame::new(frame, &ims_converter, &mz_converter, &rt_converter);
            log_denseframe_points(&dense, rec, String::from("points/Original")).unwrap();
        }
    }

    let denoised_frames: Vec<ms::DenseFrame> = frames
        .par_iter_mut()
        .map(|frame| {
            let mut dense =
                ms::DenseFrame::new(frame, &ims_converter, &mz_converter, &rt_converter);

            let tot_intensity_start = dense
                .raw_peaks
                .iter()
                .map(|peak| peak.intensity as f64)
                .sum::<f64>();
            // let denoised_frame = dense.min_neighbor_denoise(0.015, 0.015, 2);
            let denoised_frame = dbscan(&mut dense, 0.02, 0.02, 2, 500);
            let tot_intensity_end = denoised_frame
                .raw_peaks
                .iter()
                .map(|peak| peak.intensity as f64)
                .sum::<f64>();
            let intensity_ratio = tot_intensity_end / tot_intensity_start;

            println!(
                "Denoising frame {} with intensity ratio {}",
                frame.index, intensity_ratio
            );
            denoised_frame
        })
        .collect::<Vec<_>>();

    if cfg!(feature = "viz") {
        let rec = rec.as_mut().unwrap();
        for frame in denoised_frames.iter() {
            println!("Logging frame {}", frame.index);
            rec.set_time_sequence("frame_idx", frame.index as i64);
            log_denseframe_points(frame, rec, String::from("points/denoised")).unwrap();
        }
    }

    denoised_frames
}
