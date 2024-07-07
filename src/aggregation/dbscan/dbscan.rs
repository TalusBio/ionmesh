use crate::aggregation::aggregators::{aggregate_clusters, ClusterAggregator, ClusterLabel};
use crate::space::kdtree::RadiusKDTree;
use crate::space::space_generics::{
    HasIntensity, NDPoint, NDPointConverter, QueriableIndexedPoints,
};
use crate::utils;
use indicatif::ProgressIterator;
use log::{debug, info, trace};
use num::cast::AsPrimitive;
use rayon::prelude::*;
use std::ops::Add;

use crate::aggregation::dbscan::utils::FilterFunCache;

/// Density-based spatial clustering of applications with noise (DBSCAN)
///
/// This module implements a variant of dbscan with a couple of modifications
/// with respect to the vanilla implementation.
///
/// Pseudocode from wikipedia.
/// Donate to wikipedia y'all. :3
//
/// DBSCAN(DB, distFunc, eps, minPts) {
///     C := 0                                                  /* Cluster counter */
///     for each point P in database DB {
///         if label(P) ≠ undefined then continue               /* Previously processed in inner loop */
///         Neighbors N := RangeQuery(DB, distFunc, P, eps)     /* Find neighbors */
///         if |N| < minPts then {                              /* Density check */
///             label(P) := Noise                               /* Label as Noise */
///             continue
///         }
///         C := C + 1                                          /* next cluster label */
///         label(P) := C                                       /* Label initial point */
///         SeedSet S := N \ {P}                                /* Neighbors to expand */
///         for each point Q in S {                             /* Process every seed point Q */
///             if label(Q) = Noise then label(Q) := C          /* Change Noise to border point */
///             if label(Q) ≠ undefined then continue           /* Previously processed (e.g., border point) */
///             label(Q) := C                                   /* Label neighbor */
///             Neighbors N := RangeQuery(DB, distFunc, Q, eps) /* Find neighbors */
///             if |N| ≥ minPts then {                          /* Density check (if Q is a core point) */
///                 S := S ∪ N                                  /* Add new neighbors to seed set */
///             }
///         }
///     }
/// }
/// Variations ...
/// 1. Indexing is am implementation detail to find the neighbors (generic indexer)
/// 2. Sort the pointd by decreasing intensity (more intense points adopt first).
/// 3. Use an intensity threshold intead of a minimum number of neighbors.
/// 4. There are ways to define the limits to the extension of a cluster.

// TODO: rename quad_points, since this no longer uses a quadtree.
// TODO: refactor to take a filter function instead of requiting
//       a min intensity and an intensity trait.
// TODO: rename the pre-filtered...
// TODO: reimplement this a two-stage pass, where the first in parallel
//       gets the neighbors and the second does the iterative aggregation.
// THERE BE DRAGONS in this function ... I am thinking about sane ways to
// refactor it to make it more readable and maintainable.

struct DBScanTimers {
    main: utils::ContextTimer,
    filter_fun_cache_timer: utils::ContextTimer,
    outer_loop_nn_timer: utils::ContextTimer,
    inner_loop_nn_timer: utils::ContextTimer,
    local_neighbor_filter_timer: utils::ContextTimer,
    outer_intensity_calculation: utils::ContextTimer,
    inner_intensity_calculation: utils::ContextTimer,
}

impl DBScanTimers {
    fn new() -> Self {
        let mut timer = utils::ContextTimer::new("internal_dbscan", false, utils::LogLevel::DEBUG);
        let mut filter_fun_cache_timer = timer.start_sub_timer("filter_fun_cache");
        let mut outer_loop_nn_timer = timer.start_sub_timer("outer_loop_nn");
        let mut inner_loop_nn_timer = timer.start_sub_timer("inner_loop_nn");
        let mut local_neighbor_filter_timer = timer.start_sub_timer("local_neighbor_filter");
        let mut outer_intensity_calculation = timer.start_sub_timer("outer_intensity_calculation");
        let mut inner_intensity_calculation = timer.start_sub_timer("inner_intensity_calculation");
        Self {
            main: timer,
            filter_fun_cache_timer,
            outer_loop_nn_timer,
            inner_loop_nn_timer,
            local_neighbor_filter_timer,
            outer_intensity_calculation,
            inner_intensity_calculation,
        }
    }

    fn report_if_gt_us(self, min_time: f64) {
        if self.timer.cumtime.as_micros() > min_time {
            self.main.report();
            self.filter_fun_cache_timer.report();
            self.outer_loop_nn_timer.report();
            self.inner_loop_nn_timer.report();
            self.local_neighbor_filter_timer.report();
            self.outer_intensity_calculation.report();
            self.inner_intensity_calculation.report();
        }
    }
}

// THIS IS A BOTTLENECK FUNCTION
fn _dbscan<
    'a,
    const N: usize,
    C: NDPointConverter<E, N>,
    E: Sync + HasIntensity,
    T: QueriableIndexedPoints<'a, N, usize> + std::marker::Sync,
    FF: Fn(&E, &E) -> bool + Send + Sync + Copy,
>(
    indexed_points: &'a T,
    prefiltered_peaks: &Vec<E>,
    quad_points: &[NDPoint<N>],
    min_n: usize,
    min_intensity: u64,
    intensity_sorted_indices: &Vec<(usize, u64)>,
    filter_fun: Option<FF>,
    converter: C,
    progress: bool,
    max_extension_distances: &[f32; N],
) -> (u64, Vec<ClusterLabel<u64>>) {
    let mut initial_candidates_counts = utils::RollingSDCalculator::default();
    let mut final_candidates_counts = utils::RollingSDCalculator::default();

    let mut cluster_labels = vec![ClusterLabel::Unassigned; prefiltered_peaks.len()];
    let mut cluster_id = 0;
    let mut timers = DBScanTimers::new();

    let usize_filterfun = |a: &usize, b: &usize| {
        filter_fun.expect("filter_fun should be Some")(
            &prefiltered_peaks[*a],
            &prefiltered_peaks[*b],
        )
    };
    let mut filterfun_cache =
        FilterFunCache::new(Box::new(&usize_filterfun), prefiltered_peaks.len());
    let mut filterfun_with_cache = |elem_idx: usize, reference_idx: usize| {
        timers.filter_fun_cache_timer.reset_start();
        let out = filterfun_cache.get(elem_idx, reference_idx);
        timers.filter_fun_cache_timer.stop(false);
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

        timers.outer_loop_nn_timer.reset_start();
        let query_elems = converter.convert_to_bounds_query(&quad_points[point_index]);
        let mut neighbors = indexed_points.query_ndrange(&query_elems.0, query_elems.1);
        timers.outer_loop_nn_timer.stop(false);

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
        timers.outer_intensity_calculation.reset_start();
        let neighbor_intensity_total = neighbors
            .iter()
            .map(|i| prefiltered_peaks[**i].intensity().as_())
            .sum::<u64>();
        timers.outer_intensity_calculation.stop(false);

        if neighbor_intensity_total < min_intensity {
            cluster_labels[point_index] = ClusterLabel::Noise;
            continue;
        }

        cluster_id += 1;
        cluster_labels[point_index] = ClusterLabel::Cluster(cluster_id);
        let mut seed_set: Vec<&usize> = Vec::new();
        seed_set.extend(neighbors);

        while let Some(neighbor) = seed_set.pop() {
            let neighbor_index = *neighbor;
            if cluster_labels[neighbor_index] == ClusterLabel::Noise {
                cluster_labels[neighbor_index] = ClusterLabel::Cluster(cluster_id);
            }

            if cluster_labels[neighbor_index] != ClusterLabel::Unassigned {
                continue;
            }

            cluster_labels[neighbor_index] = ClusterLabel::Cluster(cluster_id);

            timers.inner_loop_nn_timer.reset_start();
            let inner_query_elems = converter.convert_to_bounds_query(&quad_points[*neighbor]);
            let mut local_neighbors =
                indexed_points.query_ndrange(&inner_query_elems.0, inner_query_elems.1);
            timers.inner_loop_nn_timer.stop(false);

            if filter_fun.is_some() {
                local_neighbors.retain(|i| filterfun_with_cache(**i, point_index))
                // .filter(|i| filter_fun.unwrap()(&prefiltered_peaks[**i], &query_peak))
            }

            timers.inner_intensity_calculation.reset_start();
            let query_intensity = prefiltered_peaks[neighbor_index].intensity();
            let neighbor_intensity_total = local_neighbors
                .iter()
                .map(|i| prefiltered_peaks[**i].intensity().as_())
                .sum::<u64>();
            timers.inner_intensity_calculation.stop(false);

            if local_neighbors.len() >= min_n && neighbor_intensity_total >= min_intensity {
                // Keep only the neighbors that are not already in a cluster
                local_neighbors
                    .retain(|i| !matches!(cluster_labels[**i], ClusterLabel::Cluster(_)));

                // Keep only the neighbors that are within the max extension distance
                // It might be worth setting a different max extension distance for the mz and mobility dimensions.
                timers.local_neighbor_filter_timer.reset_start();
                local_neighbors.retain(|i| {
                    let going_downhill = prefiltered_peaks[**i].intensity() <= query_intensity;

                    let p = &quad_points[**i];
                    let query_point = query_elems.1.unwrap();
                    // Using minkowski distance with p = 1, manhattan distance.
                    let mut within_distance = true;
                    for ((p, q), max_dist) in p
                        .values
                        .iter()
                        .zip(query_point.values)
                        .zip(max_extension_distances.iter())
                    {
                        let dist = (p - q).abs();
                        within_distance = within_distance && dist <= *max_dist;
                        if !within_distance {
                            break;
                        }
                    }

                    going_downhill && within_distance
                });
                timers.local_neighbor_filter_timer.stop(false);

                seed_set.extend(local_neighbors);
            }
        }
    }

    let (tot_queries, cached_queries) = timers.filterfun_cache.get_stats();

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

    timers.main.stop(false);
    timers.report_if_gt_us(1000000);

    (cluster_id, cluster_labels)
}

// Pretty simple function ... it uses every passed centroid, converts it to a point
// and generates a new centroid that aggregates all the points in its range.
// In contrast with the dbscan method, the elements in each cluster are not necessarily
// mutually exclusive.
fn reassign_centroid<
    'a,
    const N: usize,
    T: HasIntensity + Send + Clone + Copy,
    C: NDPointConverter<R, N>,
    I: QueriableIndexedPoints<'a, N, usize> + std::marker::Sync,
    G: Sync + Send + ClusterAggregator<T, R>,
    R: Send,
    F: Fn() -> G + Send + Sync,
>(
    centroids: Vec<R>,
    indexed_points: &'a I,
    centroid_converter: C,
    elements: &[T],
    def_aggregator: F,
    log_level: utils::LogLevel,
    expansion_factors: &[f32; N],
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
    T: HasIntensity + Send + Clone + Copy + Sync,
    F: Fn() -> G + Send + Sync,
    const N: usize,
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
    max_extension_distances: &[f32; N],
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
        Some(bc) => reassign_centroid(
            centroids,
            &tree,
            bc,
            &prefiltered_peaks,
            &def_aggregator,
            log_level,
            max_extension_distances,
        ),
        None => centroids,
    }
}
