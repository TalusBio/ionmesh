use std::collections::BTreeMap;
use std::ops::{Add, Div, Mul, Sub};

use crate::ms::frames::TimsPeak;
use crate::space::space_generics::NDPointConverter;
use crate::utils::within_distance_apply;
use crate::utils;

/// Density-based spatial clustering of applications with noise (DBSCAN)
///
/// This module implements a variant of dbscan with a couple of modifications
/// with respect to the vanilla implementation.
///
/// 1. Intensity usage.
///
use crate::mod_types::Float;
use crate::ms::frames;
use crate::space::space_generics::{HasIntensity, IndexedPoints, NDPoint};
use indicatif::ProgressIterator;
use log::{debug, info, trace};

use rayon::prelude::*;

use crate::space::kdtree::RadiusKDTree;

use num::cast::AsPrimitive;

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
pub enum ClusterLabel<T> {
    Unassigned,
    Noise,
    Cluster(T),
}

impl HasIntensity<u32> for frames::TimsPeak {
    fn intensity(&self) -> u32 {
        self.intensity
    }
}

struct FilterFunCache<'a> {
    cache: Vec<Option<BTreeMap<usize, bool>>>,
    filter_fun: Box<&'a dyn Fn(&usize, &usize) -> bool>,
    tot_queries: u64,
    cached_queries: u64,
}

impl<'a> FilterFunCache<'a> {
    fn new(filter_fun: Box<&'a dyn Fn(&usize, &usize) -> bool>, capacity: usize) -> Self {
        Self {
            cache: vec![None; capacity],
            filter_fun,
            tot_queries: 0,
            cached_queries: 0,
        }
    }

    fn get(&mut self, elem_idx: usize, reference_idx: usize) -> bool {
        // Get the value if it exists, call the functon, insert it and
        // return it if it doesn't.
        self.tot_queries += 1;

        let out: bool = match self.cache[elem_idx] {
            Some(ref map) => match map.get(&reference_idx) {
                Some(x) => {
                    self.cached_queries += 1;
                    *x
                }
                None => {
                    let out: bool = (self.filter_fun)(&elem_idx, &reference_idx);
                    self.insert(elem_idx, reference_idx, out);
                    self.insert(reference_idx, elem_idx, out);
                    out
                }
            },
            None => {
                let out = (self.filter_fun)(&elem_idx, &reference_idx);
                self.insert(elem_idx, reference_idx, out);
                self.insert(reference_idx, elem_idx, out);
                out
            }
        };
        out
    }

    fn insert(&mut self, elem_idx: usize, reference_idx: usize, value: bool) {
        match self.cache[elem_idx] {
            Some(ref mut map) => {
                _ = map.insert(reference_idx, value);
            }
            None => {
                let mut map = BTreeMap::new();
                map.insert(reference_idx, value);
                self.cache[elem_idx] = Some(map);
            }
        }
    }

    fn get_stats(&self) -> (u64, u64) {
        (self.tot_queries, self.cached_queries)
    }
}

// TODO: rename quad_points, since this no longer uses a quadtree.
// TODO: refactor to take a filter function instead of requiting
//       a min intensity and an intensity trait.
// TODO: rename the pre-filtered...
// TODO: reimplement this a two-stage pass, where the first in parallel
//       gets the neighbors and the second does the iterative aggregation.

// THIS IS A BOTTLENECK FUNCTION
fn _dbscan<
    'a,
    const N: usize,
    C: NDPointConverter<E, N>,
    I: Div<Output = I>
        + Add<Output = I>
        + Mul<Output = I>
        + Sub<Output = I>
        + Default
        + Copy
        + PartialOrd<I>
        + AsPrimitive<u64>
        + Send
        + Sync,
    E: Sync + HasIntensity<I>,
    T: IndexedPoints<'a, N, usize> + std::marker::Sync,
    FF: Fn(&E, &E) -> bool + Send + Sync + Copy,
>(
    indexed_points: &'a T,
    prefiltered_peaks: &Vec<E>,
    quad_points: &Vec<NDPoint<N>>,
    min_n: usize,
    min_intensity: u64,
    intensity_sorted_indices: &Vec<(usize, I)>,
    filter_fun: Option<FF>,
    converter: C,
    progress: bool,
    max_extension_distances: &[Float;N],
) -> (u64, Vec<ClusterLabel<u64>>) {
    let mut initial_candidates_counts = utils::RollingSDCalculator::default();
    let mut final_candidates_counts = utils::RollingSDCalculator::default();

    let mut cluster_labels = vec![ClusterLabel::Unassigned; prefiltered_peaks.len()];
    let mut cluster_id = 0;

    let mut timer = utils::ContextTimer::new("internal_dbscan", false, utils::LogLevel::DEBUG);

    let mut filter_fun_cache_timer = timer.start_sub_timer("filter_fun_cache");
    let mut outer_loop_nn_timer = timer.start_sub_timer("outer_loop_nn");
    let mut inner_loop_nn_timer = timer.start_sub_timer("inner_loop_nn");
    let mut local_neighbor_filter_timer = timer.start_sub_timer("local_neighbor_filter");
    let mut outer_intensity_calculation = timer.start_sub_timer("outer_intensity_calculation");
    let mut inner_intensity_calculation = timer.start_sub_timer("inner_intensity_calculation");

    let usize_filterfun = |a: &usize, b: &usize| {
        filter_fun.expect("filter_fun should be Some")(
            &prefiltered_peaks[*a],
            &prefiltered_peaks[*b],
        )
    };
    let mut filterfun_cache =
        FilterFunCache::new(Box::new(&usize_filterfun), prefiltered_peaks.len());
    let mut filterfun_with_cache = |elem_idx: usize, reference_idx: usize| {
        filter_fun_cache_timer.reset_start();
        let out = filterfun_cache.get(elem_idx, reference_idx);
        filter_fun_cache_timer.stop(false);
        out
    };

    let my_progbar = if progress {
        indicatif::ProgressBar::new(intensity_sorted_indices.len() as u64)
    } else {
        indicatif::ProgressBar::hidden()
    };

    for (point_index, _intensity) in intensity_sorted_indices.iter().progress_with(my_progbar) {
        let point_index = *point_index;
        if cluster_labels[point_index] != ClusterLabel::Unassigned {
            continue;
        }

        outer_loop_nn_timer.reset_start();
        let query_elems = converter.convert_to_bounds_query(&quad_points[point_index]);
        let mut neighbors = indexed_points.query_ndrange(&query_elems.0, query_elems.1);
        outer_loop_nn_timer.stop(false);

        if neighbors.len() < min_n {
            cluster_labels[point_index] = ClusterLabel::Noise;
            continue;
        }

        if filter_fun.is_some() {
            let num_initial_candidates = neighbors.len();
            neighbors.retain(|i| filterfun_with_cache(**i, point_index));
            // .filter(|i| filter_fun.unwrap()(&prefiltered_peaks[**i], &query_peak))

            let candidates_after_filter = neighbors.len();
            initial_candidates_counts.add(num_initial_candidates as f32, 1);
            final_candidates_counts.add(candidates_after_filter as f32, 1);

            if neighbors.len() < min_n {
                cluster_labels[point_index] = ClusterLabel::Noise;
                continue;
            }
        }

        // Q: Do I need to care about overflows here? - Sebastian
        outer_intensity_calculation.reset_start();
        let neighbor_intensity_total = neighbors
            .iter()
            .map(|i| prefiltered_peaks[**i].intensity().as_())
            .sum::<u64>();
        outer_intensity_calculation.stop(false);

        if neighbor_intensity_total < min_intensity {
            cluster_labels[point_index] = ClusterLabel::Noise;
            continue;
        }

        cluster_id += 1;
        cluster_labels[point_index] = ClusterLabel::Cluster(cluster_id);
        let mut seed_set: Vec<&usize> = Vec::new();
        seed_set.extend(neighbors);

        let mut internal_neighbor_additions = 0;

        while let Some(neighbor) = seed_set.pop() {
            let neighbor_index = *neighbor;
            if cluster_labels[neighbor_index] == ClusterLabel::Noise {
                cluster_labels[neighbor_index] = ClusterLabel::Cluster(cluster_id);
            }

            if cluster_labels[neighbor_index] != ClusterLabel::Unassigned {
                continue;
            }

            cluster_labels[neighbor_index] = ClusterLabel::Cluster(cluster_id);

            inner_loop_nn_timer.reset_start();
            let inner_query_elems = converter.convert_to_bounds_query(&quad_points[*neighbor]);
            let mut local_neighbors =
                indexed_points.query_ndrange(&inner_query_elems.0, inner_query_elems.1);
            inner_loop_nn_timer.stop(false);

            if filter_fun.is_some() {
                local_neighbors.retain(|i| filterfun_with_cache(**i, point_index))
                // .filter(|i| filter_fun.unwrap()(&prefiltered_peaks[**i], &query_peak))
            }

            inner_intensity_calculation.reset_start();
            let query_intensity = prefiltered_peaks[neighbor_index].intensity();
            let neighbor_intensity_total = local_neighbors
                .iter()
                .map(|i| prefiltered_peaks[**i].intensity().as_())
                .sum::<u64>();
            inner_intensity_calculation.stop(false);

            if local_neighbors.len() >= min_n && neighbor_intensity_total >= min_intensity {
                // Keep only the neighbors that are not already in a cluster
                local_neighbors.retain(|i| match cluster_labels[**i] {
                    ClusterLabel::Cluster(_) => false,
                    _ => true,
                });

                // Keep only the neighbors that are within the max extension distance
                // It might be worth setting a different max extension distance for the mz and mobility dimensions.
                local_neighbor_filter_timer.reset_start();
                local_neighbors.retain(|i| {
                    let going_downhill = prefiltered_peaks[**i].intensity() <= query_intensity;

                    let p = &quad_points[**i];
                    let query_point = query_elems.1.unwrap();
                    // Using minkowski distance with p = 1, manhattan distance.
                    let mut within_distance = true;
                    for ((p, q), max_dist) in p.values.iter().zip(query_point.values).zip(max_extension_distances.iter()) {
                        let dist = (p - q).abs();
                        within_distance = within_distance && dist <= *max_dist;
                        if !within_distance {
                            break;
                        }
                    }

                    going_downhill && within_distance
                });
                local_neighbor_filter_timer.stop(false);

                internal_neighbor_additions += local_neighbors.len();
                seed_set.extend(local_neighbors);
            }
        }
    }

    let (tot_queries, cached_queries) = filterfun_cache.get_stats();

    if tot_queries > 1000 {
        let cache_hit_rate = cached_queries as f64 / tot_queries as f64;
        info!(
            "Cache hit rate: {} / {} = {}",
            cached_queries, tot_queries, cache_hit_rate
        );

        let avg_initial_candidates = initial_candidates_counts.get_mean();
        let avg_final_candidates = final_candidates_counts.get_mean();
        debug!(
            "Avg initial candidates: {} Avg final candidates: {}",
            avg_initial_candidates, avg_final_candidates
        );
    }

    timer.stop(false);
    if timer.cumtime.as_micros() > 1000000 {
        timer.report();
        filter_fun_cache_timer.report();
        outer_loop_nn_timer.report();
        inner_loop_nn_timer.report();
        local_neighbor_filter_timer.report();
        outer_intensity_calculation.report();
        inner_intensity_calculation.report();
    }

    (cluster_id, cluster_labels)
}

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
        self.cluster_mz += elem.mz * f64_intensity;
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
            npeaks: self.num_peaks as u32,
        }
    }

    fn combine(self, other: Self) -> Self {
        Self {
            cluster_intensity: self.cluster_intensity + other.cluster_intensity,
            cluster_mz: self.cluster_mz + other.cluster_mz,
            cluster_mobility: self.cluster_mobility + other.cluster_mobility,
            num_peaks: self.num_peaks + other.num_peaks,
        }
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
        cluster_vecs[*cluster_idx].as_mut().unwrap().add(point);
    }

    cluster_vecs
}

pub fn aggregate_clusters<
    T: HasIntensity<Z> + Send + Clone + Copy,
    G: Sync + Send + ClusterAggregator<T, R>,
    R: Send,
    F: Fn() -> G + Send + Sync,
    Z: AsPrimitive<u64>
        + Send
        + Sync
        + Add<Output = Z>
        + PartialOrd
        + Div<Output = Z>
        + Mul<Output = Z>
        + Default
        + Sub<Output = Z>,
>(
    tot_clusters: u64,
    cluster_labels: Vec<ClusterLabel<u64>>,
    elements: &[T],
    def_aggregator: &F,
    log_level: utils::LogLevel,
    keep_unclustered: bool,
) -> Vec<R> {
    let cluster_vecs: Vec<G> = if cfg!(feature = "par_dataprep") {
        let mut timer =
            utils::ContextTimer::new("dbscan_generic::par_aggregation", true, log_level);
        let out: Vec<(usize, T)> = cluster_labels
            .iter()
            .enumerate()
            .filter_map(|(point_index, x)| match x {
                ClusterLabel::Cluster(cluster_id) => {
                    let cluster_idx = *cluster_id as usize - 1;
                    let tmp: Option<(usize, T)> = Some((cluster_idx, elements[point_index]));
                    tmp
                }
                _ => None,
            })
            .collect();

        let run_closure =
            |chunk: Vec<(usize, T)>| _inner(&chunk, tot_clusters as usize, &def_aggregator);
        let chunk_size = (out.len() / rayon::current_num_threads()) / 2;
        let chunk_size = chunk_size.max(1);
        let out2 = out
            .into_par_iter()
            .chunks(chunk_size)
            .map(run_closure)
            .reduce(Vec::new, |l, r| {
                if l.is_empty() {
                    r
                } else {
                    l.into_iter()
                        .zip(r)
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
            });

        let mut cluster_vecs = out2.into_iter().flatten().collect::<Vec<_>>();

        let unclustered_elems: Vec<usize> = cluster_labels
            .iter()
            .enumerate()
            .filter(|(_, x)| match x {
                ClusterLabel::Unassigned => true,
                ClusterLabel::Noise => keep_unclustered,
                _ => false,
            })
            .map(|(i, _elem)| i)
            .collect();

        // if unclustered_elems.len() > 0 {
        //     log::debug!("Total Orig elems: {}", cluster_labels.len());
        //     log::debug!("Unclustered elems: {}", unclustered_elems.len());
        //     log::debug!("Clustered elems: {}", cluster_vecs.len());
        // }

        let unclustered_elems = unclustered_elems
            .iter()
            .map(|i| {
                let mut oe = def_aggregator();
                oe.add(&elements[*i]);
                oe
            })
            .collect::<Vec<_>>();

        cluster_vecs.extend(unclustered_elems);

        timer.stop(true);
        cluster_vecs
    } else {
        let mut cluster_vecs: Vec<G> = Vec::with_capacity(tot_clusters as usize);
        let mut unclustered_points: Vec<G> = Vec::new();
        for _ in 0..tot_clusters {
            cluster_vecs.push(def_aggregator());
        }
        for (point_index, cluster_label) in cluster_labels.iter().enumerate() {
            match cluster_label {
                ClusterLabel::Cluster(cluster_id) => {
                    let cluster_idx = *cluster_id as usize - 1;
                    cluster_vecs[cluster_idx].add(&(elements[point_index]));
                }
                ClusterLabel::Noise => {
                    if keep_unclustered {
                        let mut oe = def_aggregator();
                        oe.add(&elements[point_index]);
                        unclustered_points.push(oe);
                    }
                }
                _ => {}
            }
        }
        cluster_vecs.extend(unclustered_points);
        cluster_vecs
    };

    let mut timer =
        utils::ContextTimer::new("dbscan_generic::aggregation", true, utils::LogLevel::TRACE);
    let out = cluster_vecs
        .par_iter()
        .map(|cluster| cluster.aggregate())
        .collect::<Vec<_>>();
    timer.stop(true);

    out
}

// Pretty simple function ... it uses every passed centroid, converts it to a point
// and generates a new centroid that aggregates all the points in its range.
// In contrast with the dbscan method, the elements in each cluster are not necessarily
// mutually exclusive.
fn reassign_centroid<
    'a,
    const N: usize,
    T: HasIntensity<Z> + Send + Clone + Copy,
    C: NDPointConverter<R, N>,
    I: IndexedPoints<'a, N, usize> + std::marker::Sync,
    G: Sync + Send + ClusterAggregator<T, R>,
    R: Send,
    F: Fn() -> G + Send + Sync,
    Z: AsPrimitive<u64>
        + Send
        + Sync
        + Add<Output = Z>
        + PartialOrd
        + Div<Output = Z>
        + Mul<Output = Z>
        + Default
        + Sub<Output = Z>,
>(
    centroids: Vec<R>,
    indexed_points: &'a I,
    centroid_converter: C,
    elements: &Vec<T>,
    def_aggregator: F,
    log_level: utils::LogLevel,
    expansion_factors: &[Float;N],
) -> Vec<R> {
    let mut timer = utils::ContextTimer::new("reassign_centroid", true, log_level);
    let mut out = Vec::with_capacity(centroids.len());

    for centroid in centroids {
        let query_point = centroid_converter.convert(&centroid);
        let mut query_elems = centroid_converter.convert_to_bounds_query(&query_point);
        query_elems.0.expand(expansion_factors);

        // trace!("Querying for Centroid: {:?}", query_elems.1);
        // trace!("Querying for Boundary: {:?}", query_elems.0);
        let neighbors = indexed_points.query_ndrange(&query_elems.0, query_elems.1);
        // trace!("Found {} neighbors", neighbors.len());
        let mut aggregator = def_aggregator();
        let mut num_agg = 0;
        for neighbor in neighbors {
            aggregator.add(&elements[*neighbor]);
            num_agg += 1;
        }
        trace!("Aggregated {} elements", num_agg);
        out.push(aggregator.aggregate());
    }

    timer.stop(true);
    out
}
// TODO: rename prefiltered peaks argument!
// TODO implement a version that takes a sparse distance matrix.

pub fn dbscan_generic<
    C: NDPointConverter<T, N>,
    C2: NDPointConverter<R, N>,
    R: Send,
    G: Sync + Send + ClusterAggregator<T, R>,
    T: HasIntensity<Z> + Send + Clone + Copy + Sync,
    F: Fn() -> G + Send + Sync,
    const N: usize,
    // Z is usually u32 or u64
    Z: AsPrimitive<u64>
        + Send
        + Sync
        + Add<Output = Z>
        + PartialOrd
        + Div<Output = Z>
        + Mul<Output = Z>
        + Default
        + Sub<Output = Z>,
    FF: Send + Sync + Fn(&T, &T) -> bool,
>(
    converter: C,
    prefiltered_peaks: Vec<T>,
    min_n: usize,
    min_intensity: u64,
    def_aggregator: F,
    extra_filter_fun: Option<&FF>,
    log_level: Option<utils::LogLevel>,
    keep_unclustered: bool,
    max_extension_distances: &[Float;N],
    back_converter: Option<C2>,
) -> Vec<R> {
    let show_progress = log_level.is_some();
    let log_level = match log_level {
        Some(x) => x,
        None => utils::LogLevel::TRACE,
    };

    let timer = utils::ContextTimer::new("dbscan_generic", true, log_level);
    let mut i_timer = timer.start_sub_timer("conversion");
    let (ndpoints, boundary) = converter.convert_vec(&prefiltered_peaks);
    i_timer.stop(true);

    let mut i_timer = timer.start_sub_timer("tree");
    let mut tree = RadiusKDTree::new_empty(boundary, 500, 1.);
    let quad_indices = (0..ndpoints.len()).collect::<Vec<_>>();

    for (quad_point, i) in ndpoints.iter().zip(quad_indices.iter()) {
        tree.insert_ndpoint(quad_point.clone(), i);
    }
    i_timer.stop(true);

    let mut i_timer = timer.start_sub_timer("pre-sort");
    let mut intensity_sorted_indices = prefiltered_peaks
        .iter()
        .enumerate()
        .map(|(i, peak)| (i, peak.intensity()))
        .collect::<Vec<_>>();
    // Q: Does ^^^^ need a clone? i and peak intensity ... - S

    intensity_sorted_indices.par_sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    i_timer.stop(true);

    let mut i_timer = timer.start_sub_timer("dbscan");
    let (tot_clusters, cluster_labels) = _dbscan(
        &tree,
        &prefiltered_peaks,
        &ndpoints,
        min_n,
        min_intensity,
        &intensity_sorted_indices,
        extra_filter_fun,
        converter,
        show_progress,
        max_extension_distances,
    );
    i_timer.stop(true);

    let centroids = aggregate_clusters(
        tot_clusters,
        cluster_labels,
        &prefiltered_peaks,
        &def_aggregator,
        log_level,
        keep_unclustered,
    );

    match back_converter {
        Some(bc) => {

            reassign_centroid(
                centroids,
                &tree,
                bc,
                &prefiltered_peaks,
                &def_aggregator,
                log_level,
                max_extension_distances,
            )
        }
        None => {
            centroids
        }
    }
}

// https://github.com/rust-lang/rust/issues/35121
// The never type is not stable yet....
struct BypassDenseFrameBackConverter {}

impl NDPointConverter<frames::TimsPeak, 2> for BypassDenseFrameBackConverter {
    fn convert(&self, _elem: &frames::TimsPeak) -> NDPoint<2> {
        panic!("This should never be called")
    }
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

type FFTimsPeak = fn(&TimsPeak, &TimsPeak) -> bool;
// <FF: Send + Sync + Fn(&TimsPeak, &TimsPeak) -> bool>
pub fn dbscan_denseframes(
    mut denseframe: frames::DenseFrame,
    mz_scaling: f64,
    max_mz_extension: f64,
    ims_scaling: f32,
    max_ims_extension: f32,
    min_n: usize,
    min_intensity: u64,
) -> frames::DenseFrame {
    let out_frame_type: timsrust::FrameType = denseframe.frame_type;
    let out_rt: f64 = denseframe.rt;
    let out_index: usize = denseframe.index;

    let prefiltered_peaks = {
        denseframe.sort_by_mz();

        let keep_vector = within_distance_apply(
            &denseframe.raw_peaks,
            &|peak| peak.mz,
            &mz_scaling,
            &|i_right, i_left| (i_right - i_left) >= min_n,
        );

        // Filter the peaks and replace the raw peaks with the filtered peaks.

        denseframe
            .raw_peaks
            .clone()
            .into_iter()
            .zip(keep_vector)
            .filter(|(_, b)| *b)
            .map(|(peak, _)| peak) // Clone the TimsPeak
            .collect::<Vec<_>>()
    };

    let converter = DenseFrameConverter {
        mz_scaling,
        ims_scaling,
    };
    let peak_vec: Vec<TimsPeak> = dbscan_generic(
        converter,
        prefiltered_peaks,
        min_n,
        min_intensity,
        TimsPeakAggregator::default,
        None::<&FFTimsPeak>,
        None,
        true,
        &[max_mz_extension as Float, max_ims_extension as Float],
        None::<BypassDenseFrameBackConverter>,
    );

    frames::DenseFrame {
        raw_peaks: peak_vec,
        index: out_index,
        rt: out_rt,
        frame_type: out_frame_type,
        sorted: None,
    }
}
