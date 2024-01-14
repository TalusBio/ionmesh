use crate::ms::frames::TimsPeak;
use crate::utils;
use crate::utils::within_distance_apply;

/// Density-based spatial clustering of applications with noise (DBSCAN)
///
/// This module implements a variant of dbscan with a couple of modifications
/// with respect to the vanilla implementation.
///
/// 1. Intensity usage.
///
use crate::mod_types::Float;
use crate::ms::frames;
use crate::space_generics::{HasIntensity, IndexedPoints, NDPoint};
use log::{error, trace};

use rayon::prelude::*;

use crate::kdtree::RadiusKDTree;
use crate::quad::RadiusQuadTree;

// Pseudocode from wikipedia.
// Donate to wikipedia y'all. :3
//
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

impl HasIntensity<u32> for frames::TimsPeak {
    fn intensity(&self) -> u32 {
        self.intensity
    }
}

// THIS IS A BOTTLENECK FUNCTION
fn _dbscan<'a, const N: usize>(
    tree: &'a impl IndexedPoints<'a, N, usize>,
    prefiltered_peaks: &Vec<impl HasIntensity<u32>>,
    quad_points: &Vec<NDPoint<N>>, // TODO make generic over dimensions
    min_n: usize,
    min_intensity: u64,
    intensity_sorted_indices: &Vec<(usize, u32)>,
) -> (u64, Vec<ClusterLabel<u64>>) {
    let mut cluster_labels = vec![ClusterLabel::Unassigned; prefiltered_peaks.len()];
    let mut cluster_id = 0;

    for (point_index, _intensity) in intensity_sorted_indices.iter() {
        let point_index = *point_index;
        if cluster_labels[point_index] != ClusterLabel::Unassigned {
            continue;
        }

        let query_point = quad_points[point_index].clone();
        let neighbors = tree.query_ndpoint(&query_point);

        if neighbors.len() < min_n {
            cluster_labels[point_index] = ClusterLabel::Noise;
            continue;
        }

        // Do I need to care about overflows here?
        let mut neighbor_intensity_total: u64 = 0;

        for i in neighbors.iter() {
            neighbor_intensity_total += prefiltered_peaks[**i].intensity() as u64;
        }

        if neighbor_intensity_total < min_intensity {
            cluster_labels[point_index] = ClusterLabel::Noise;
            continue;
        }

        cluster_id += 1;
        cluster_labels[point_index] = ClusterLabel::Cluster(cluster_id);
        let mut seed_set: Vec<&usize> = Vec::new();
        seed_set.extend(neighbors);

        const MAX_EXTENSION_DISTANCE: Float = 5.;

        while let Some(neighbor) = seed_set.pop() {
            let neighbor_index = neighbor.clone();
            if cluster_labels[neighbor_index] == ClusterLabel::Noise {
                cluster_labels[neighbor_index] = ClusterLabel::Cluster(cluster_id);
            }

            if cluster_labels[neighbor_index] != ClusterLabel::Unassigned {
                continue;
            }

            cluster_labels[neighbor_index] = ClusterLabel::Cluster(cluster_id);

            let neighbors = tree.query_ndpoint(&quad_points[*neighbor]);

            let neighbor_intensity: u32 = prefiltered_peaks[neighbor_index].intensity();
            let neighbor_intensity_total = neighbors
                .iter()
                .map(|i| prefiltered_peaks[**i].intensity() as u64)
                .sum::<u64>();

            if neighbors.len() >= min_n && neighbor_intensity_total >= min_intensity {
                // Keep only the neighbors that are not already in a cluster
                let local_neighbors = neighbors
                    .into_iter()
                    .filter(|i| match cluster_labels[**i] {
                        ClusterLabel::Cluster(_) => false,
                        _ => true,
                    })
                    .collect::<Vec<_>>();

                // Keep only the neighbors that are within the max extension distance
                // It might be worth setting a different max extension distance for the mz and mobility dimensions.
                let local_neighbors2 = local_neighbors
                    .into_iter()
                    .filter(|i| {
                        let going_downhill =
                            prefiltered_peaks[**i].intensity() <= neighbor_intensity;

                        let p = &quad_points[**i];
                        // Using minkowski distance with p = 1, manhattan distance.
                        let dist = (p.values[0] - query_point.values[0]).abs()
                            + (p.values[1] - query_point.values[1]).abs();
                        let within_distance = dist <= MAX_EXTENSION_DISTANCE.powi(2);
                        going_downhill && within_distance
                    })
                    .collect::<Vec<_>>();

                seed_set.extend(local_neighbors2);
            }
        }
    }

    (cluster_id, cluster_labels)
}

// pub fn dbscan(
//     denseframe: frames::DenseFrame,
//     mz_scaling: f64,
//     ims_scaling: f32,
//     min_n: usize,
//     min_intensity: u64,
// ) -> frames::DenseFrame {
//     let out_frame_type: timsrust::FrameType = denseframe.frame_type.clone();
//     let out_rt: f64 = denseframe.rt.clone();
//     let out_index: usize = denseframe.index.clone();
//
//     let (quad_points, prefiltered_peaks, boundary) =
//         denseframe_to_quadtree_points(denseframe, mz_scaling, ims_scaling, min_n.saturating_sub(1));
//
//     let mut tree = RadiusQuadTree::new(boundary, 20, 1.);
//     // let mut tree = RadiusQuadTree::new(boundary, 20, 1.);
//
//     let quad_indices = (0..quad_points.len()).collect::<Vec<_>>();
//
//     for (quad_point, i) in quad_points.iter().zip(quad_indices.iter()) {
//         tree.insert(quad_point.clone(), i);
//     }
//     let mut intensity_sorted_indices = prefiltered_peaks
//         .iter()
//         .enumerate()
//         .map(|(i, peak)| (i.clone(), peak.intensity.clone()))
//         .collect::<Vec<_>>();
//
//     intensity_sorted_indices.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
//
//     let (cluster_id, cluster_labels) = _dbscan(
//         &tree,
//         &prefiltered_peaks,
//         &quad_points,
//         min_n,
//         min_intensity,
//         &intensity_sorted_indices,
//     );
//     // Each element is a tuple representing the summed cluster intensity, mz, and mobility.
//     // And will be used to calculate the weighted average of mz and mobility AND the total intensity.
//     let mut cluster_vecs = vec![(0u64, 0f64, 0f64); cluster_id as usize];
//     for (point_index, cluster_label) in cluster_labels.iter().enumerate() {
//         match cluster_label {
//             ClusterLabel::Cluster(cluster_id) => {
//                 let cluster_idx = *cluster_id as usize - 1;
//                 let timspeak = prefiltered_peaks[point_index];
//                 let f64_intensity = timspeak.intensity as f64;
//                 let cluster_vec = &mut cluster_vecs[cluster_idx];
//                 cluster_vec.0 += timspeak.intensity as u64;
//                 cluster_vec.1 += (timspeak.mz as f64) * f64_intensity;
//                 cluster_vec.2 += (timspeak.mobility as f64) * f64_intensity;
//             }
//             _ => {}
//         }
//     }
//
//     let denoised_peaks = cluster_vecs
//         .iter_mut()
//         .map(|(cluster_intensity, cluster_mz, cluster_mobility)| {
//             let cluster_intensity = cluster_intensity; // Note not averaged
//             let cluster_mz = *cluster_mz / *cluster_intensity as f64;
//             let cluster_mobility = *cluster_mobility / *cluster_intensity as f64;
//             frames::TimsPeak {
//                 intensity: u32::try_from(*cluster_intensity).ok().unwrap(),
//                 mz: cluster_mz,
//                 mobility: cluster_mobility as f32,
//             }
//         })
//         .collect::<Vec<_>>();
//
//     // TODO add an option to keep noise points
//
//     frames::DenseFrame {
//         raw_peaks: denoised_peaks,
//         index: out_index,
//         rt: out_rt,
//         frame_type: out_frame_type,
//         sorted: None,
//     }
// }

use crate::space_generics::NDPointConverter;

/// A trait for aggregating points into a single point.
/// This is used for the final step of dbscan.
///
/// Types <T,R,S> are:
/// T: The type of the points to be aggregated.
/// R: The type of the aggregated point.
/// S: The type of the aggregator.
///
pub trait ClusterAggregator<T, R> {
    fn add(&mut self, elem: &T);
    fn aggregate(&self) -> R;
    fn combine(self, other: Self) -> Self;
}

#[derive(Default, Debug)]
struct TimsPeakAggregator {
    cluster_intensity: u64,
    cluster_mz: f64,
    cluster_mobility: f64,
    num_peaks: u64,
}

impl ClusterAggregator<TimsPeak, TimsPeak> for TimsPeakAggregator {
    fn add(&mut self, elem: &TimsPeak) {
        let f64_intensity = elem.intensity as f64;
        debug_assert!((elem.intensity as u64) < (u64::MAX - self.cluster_intensity));
        self.cluster_intensity += elem.intensity as u64;
        self.cluster_mz += (elem.mz as f64) * f64_intensity;
        self.cluster_mobility += (elem.mobility as f64) * f64_intensity;
        self.num_peaks += 1;
    }

    fn aggregate(&self) -> TimsPeak {
        let cluster_mz = self.cluster_mz / self.cluster_intensity as f64;
        let cluster_mobility = self.cluster_mobility / self.cluster_intensity as f64;
        frames::TimsPeak {
            intensity: self.cluster_intensity as u32,
            mz: cluster_mz,
            mobility: cluster_mobility as f32,
        }
    }

    fn combine(self, other: Self) -> Self {
        let out = Self {
            cluster_intensity: self.cluster_intensity + other.cluster_intensity,
            cluster_mz: self.cluster_mz + other.cluster_mz,
            cluster_mobility: self.cluster_mobility + other.cluster_mobility,
            num_peaks: self.num_peaks + other.num_peaks,
        };
        out
    }
}

fn _inner<T: Copy, G: ClusterAggregator<T, R>, R>(
    chunk: &[(usize, T)],
    max_cluster_id: usize,
    def_aggregator: &dyn Fn() -> G,
) -> Vec<Option<G>> {
    let mut cluster_vecs: Vec<Option<G>> = (0..max_cluster_id).map(|_| None).collect();

    for (cluster_idx, point) in chunk {
        if cluster_vecs[*cluster_idx].is_none() {
            cluster_vecs[*cluster_idx] = Some(def_aggregator());
        }
        cluster_vecs[*cluster_idx].as_mut().unwrap().add(&point);
    }

    cluster_vecs
}

// fn DBSCAN<C: NDPointConverter<T, D>, R, G: Default + ClusterAggregator<T,R,G>, T: HasIntensity<u32>, const D: usize>(
pub fn dbscan_generic<
    C: NDPointConverter<T, N>,
    R: Send,
    G: Sync + Send + ClusterAggregator<T, R>,
    T: HasIntensity<u32> + Send + Clone + Copy,
    F: Fn() -> G + Send + Sync,
    const N: usize,
>(
    converter: C,
    prefiltered_peaks: Vec<T>,
    min_n: usize,
    min_intensity: u64,
    def_aggregator: F,
) -> Vec<R> {
    let timer =
        utils::ContextTimer::new("dbscan_generic::conversion", true, utils::LogLevel::TRACE);
    let (ndpoints, boundary) = converter.convert_vec(&prefiltered_peaks);
    timer.stop();
    // let mut tree = RadiusKDTree::new_empty(boundary, 1000, 1.);
    // let mut tree = RadiusQuadTree::new_empty(boundary, 1000, 1.);

    let timer = utils::ContextTimer::new("dbscan_generic::tree", true, utils::LogLevel::TRACE);
    let mut tree = RadiusKDTree::new_empty(boundary, 500, 1.);
    let quad_indices = (0..ndpoints.len()).collect::<Vec<_>>();

    for (quad_point, i) in ndpoints.iter().zip(quad_indices.iter()) {
        tree.insert_ndpoint(quad_point.clone(), i);
    }
    timer.stop();

    let timer = utils::ContextTimer::new("dbscan_generic::pre-sort", true, utils::LogLevel::TRACE);
    let mut intensity_sorted_indices = prefiltered_peaks
        .iter()
        .enumerate()
        .map(|(i, peak)| (i.clone(), peak.intensity()))
        .collect::<Vec<_>>();
    // Q: Does ^^^^ need a clone? i and peak intensity ... - S

    intensity_sorted_indices.par_sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    timer.stop();

    let timer = utils::ContextTimer::new("dbscan_generic::dbscan", true, utils::LogLevel::TRACE);
    let (cluster_id, cluster_labels) = _dbscan(
        &tree,
        &prefiltered_peaks,
        &ndpoints,
        min_n,
        min_intensity,
        &intensity_sorted_indices,
    );
    timer.stop();

    let cluster_vecs: Vec<G> = if cfg!(feature = "par_dataprep") {
        let timer = utils::ContextTimer::new(
            "dbscan_generic::par_aggregation",
            true,
            utils::LogLevel::TRACE,
        );
        let out: Vec<(usize, T)> = cluster_labels
            .iter()
            .enumerate()
            .map(|(point_index, x)| match x {
                ClusterLabel::Cluster(cluster_id) => {
                    let cluster_idx = *cluster_id as usize - 1;
                    let tmp: Option<(usize, T)> =
                        Some((cluster_idx, prefiltered_peaks[point_index].clone()));
                    tmp
                }
                _ => None,
            })
            .filter(|x| x.is_some())
            .map(|x| x.unwrap())
            .collect();

        let run_closure =
            |chunk: Vec<(usize, T)>| _inner(&chunk, cluster_id.clone() as usize, &def_aggregator);
        let chunk_size = (out.len() / rayon::current_num_threads()) / 2;
        let chunk_size = chunk_size.max(1);
        let out2 = out
            .into_par_iter()
            .chunks(chunk_size)
            .map(run_closure)
            .reduce(
                || Vec::new(),
                |l, r| {
                    if l.len() == 0 {
                        r
                    } else {
                        l.into_iter()
                            .zip(r.into_iter())
                            .map(|(l, r)| match (l, r) {
                                (Some(l), Some(r)) => {
                                    let o = l.combine(r);
                                    Some(o)
                                }
                                (Some(l), None) => Some(l),
                                (None, Some(r)) => Some(r),
                                (None, None) => None,
                            })
                            .collect::<Vec<_>>()
                    }
                },
            );

        let cluster_vecs = out2
            .into_iter()
            .filter(|x| x.is_some())
            .map(|x| x.unwrap())
            .collect::<Vec<_>>();
        timer.stop();
        cluster_vecs
    } else {
        let mut cluster_vecs: Vec<G> = Vec::with_capacity(cluster_id as usize);
        for _ in (0..cluster_id).into_iter() {
            cluster_vecs.push(def_aggregator());
        }
        for (point_index, cluster_label) in cluster_labels.iter().enumerate() {
            match cluster_label {
                ClusterLabel::Cluster(cluster_id) => {
                    let cluster_idx = *cluster_id as usize - 1;
                    cluster_vecs[cluster_idx].add(&(prefiltered_peaks[point_index]));
                }
                _ => {}
            }
        }
        cluster_vecs
    };

    //     .par_iter_mut() // <<<<- This works but its slower.
    //     .map(|cluster| cluster.aggregate())
    //     .collect::<Vec<_>>()

    let timer =
        utils::ContextTimer::new("dbscan_generic::aggregation", true, utils::LogLevel::TRACE);
    let out = cluster_vecs
        .iter()
        .map(|cluster| cluster.aggregate())
        .collect::<Vec<_>>();
    timer.stop();

    out
}

struct DenseFrameConverter {
    mz_scaling: f64,
    ims_scaling: f32,
}

impl NDPointConverter<TimsPeak, 2> for DenseFrameConverter {
    fn convert(&self, elem: &TimsPeak) -> NDPoint<2> {
        NDPoint {
            values: [
                (elem.mz / self.mz_scaling) as Float,
                (elem.mobility / self.ims_scaling) as Float,
            ],
        }
    }
}

pub fn dbscan_denseframes(
    mut denseframe: frames::DenseFrame,
    mz_scaling: f64,
    ims_scaling: f32,
    min_n: usize,
    min_intensity: u64,
) -> frames::DenseFrame {
    let out_frame_type: timsrust::FrameType = denseframe.frame_type.clone();
    let out_rt: f64 = denseframe.rt.clone();
    let out_index: usize = denseframe.index.clone();

    let prefiltered_peaks = {
        denseframe.sort_by_mz();

        let keep_vector = within_distance_apply(
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
            .zip(keep_vector.into_iter())
            .filter(|(_, b)| *b)
            .map(|(peak, _)| peak) // Clone the TimsPeak
            .collect::<Vec<_>>();

        prefiltered_peaks
    };

    let converter = DenseFrameConverter {
        mz_scaling,
        ims_scaling,
    };
    let peak_vec = dbscan_generic(converter, prefiltered_peaks, min_n, min_intensity, &|| {
        TimsPeakAggregator::default()
    });

    frames::DenseFrame {
        raw_peaks: peak_vec,
        index: out_index,
        rt: out_rt,
        frame_type: out_frame_type,
        sorted: None,
    }
}
