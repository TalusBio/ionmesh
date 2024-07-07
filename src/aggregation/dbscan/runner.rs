use crate::space::space_generics::NDPointConverter;
use crate::space::space_generics::{HasIntensity, NDPoint, QueriableIndexedPoints};
use crate::utils;
use indicatif::ProgressIterator;

use rayon::prelude::*;

use crate::aggregation::aggregators::ClusterLabel;
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

#[derive(Debug, Clone)]
pub struct ClusterLabels {
    pub cluster_labels: Vec<ClusterLabel<u64>>,
    pub num_clusters: u64,
}

impl ClusterLabels {
    fn new(num_labels: usize) -> Self {
        let cluster_labels = vec![ClusterLabel::Unassigned; num_labels];
        Self {
            cluster_labels,
            num_clusters: 0,
        }
    }

    fn set_cluster(&mut self, index: usize, cluster_id: u64) {
        self.cluster_labels[index] = ClusterLabel::Cluster(cluster_id);
    }

    fn set_new_cluster(&mut self, index: usize) {
        self.num_clusters += 1;
        self.set_cluster(index, self.num_clusters);
    }

    fn set_current_cluster(&mut self, index: usize) {
        let cluster_id = self.num_clusters;
        self.set_cluster(index, cluster_id);
    }

    fn set_noise(&mut self, index: usize) {
        self.cluster_labels[index] = ClusterLabel::Noise;
    }

    fn get(&self, index: usize) -> ClusterLabel<u64> {
        self.cluster_labels[index]
    }
}

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
        let timer = utils::ContextTimer::new("internal_dbscan", false, utils::LogLevel::DEBUG);
        let filter_fun_cache_timer = timer.start_sub_timer("filter_fun_cache");
        let outer_loop_nn_timer = timer.start_sub_timer("outer_loop_nn");
        let inner_loop_nn_timer = timer.start_sub_timer("inner_loop_nn");
        let local_neighbor_filter_timer = timer.start_sub_timer("local_neighbor_filter");
        let outer_intensity_calculation = timer.start_sub_timer("outer_intensity_calculation");
        let inner_intensity_calculation = timer.start_sub_timer("inner_intensity_calculation");
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

    fn report_if_gt_us(&self, min_time: u128) {
        if self.main.cumtime.as_micros() > min_time {
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

struct CandidateCountMetrics {
    initial_candidates_counts: utils::RollingSDCalculator<f32, u64>,
    final_candidates_counts: utils::RollingSDCalculator<f32, u64>,
}

impl CandidateCountMetrics {
    fn new() -> Self {
        Self {
            initial_candidates_counts: utils::RollingSDCalculator::default(),
            final_candidates_counts: utils::RollingSDCalculator::default(),
        }
    }
}

struct DBSCANRunnerState {
    cluster_labels: ClusterLabels,
    filter_fun_cache: Option<FilterFunCache>,
    timers: DBScanTimers,
    candidate_metrics: CandidateCountMetrics,
}

impl DBSCANRunnerState {
    fn new<P>(nlabels: usize, usize_filterfun: Option<P>) -> Self
    where
        P: Fn(&usize, &usize) -> bool + Send + Sync,
    {
        let cluster_labels = ClusterLabels::new(nlabels);

        let filter_fun_cache = match usize_filterfun {
            Some(_) => Some(FilterFunCache::new(nlabels)),
            None => None,
        };
        //FilterFunCache::new(Box::new(&usize_filterfun), nlabels);
        let timers = DBScanTimers::new();
        let candidate_metrics = CandidateCountMetrics::new();

        Self {
            cluster_labels,
            filter_fun_cache,
            timers,
            candidate_metrics,
        }
    }

    fn create_progress_bar(&self, len: usize, visible: bool) -> indicatif::ProgressBar {
        if visible {
            indicatif::ProgressBar::new(len as u64)
        } else {
            indicatif::ProgressBar::hidden()
        }
    }
}

//trait FilterFunction: for<'a, 'b> Fn<(&'a E, &'b E)> + Sized{}

struct DBSCANRunner<'a, const N: usize, C, E> {
    min_n: usize,
    min_intensity: u64,
    filter_fun: Option<&'a (dyn Fn(&E, &E) -> bool + Send + Sync)>,
    converter: C,
    progress: bool,
    max_extension_distances: &'a [f32; N],
}

struct DBSCANPoints<'a, const N: usize, E> {
    prefiltered_peaks: &'a Vec<E>,
    intensity_sorted_indices: &'a Vec<(usize, u64)>,
    indexed_points: &'a (dyn QueriableIndexedPoints<'a, N, usize> + std::marker::Sync),
    quad_points: &'a [NDPoint<N>],
}

// C: NDPointConverter<T, N>,
// C2: NDPointConverter<R, N>,
// R: Send,
// G: Sync + Send + ClusterAggregator<T, R>,
// T: HasIntensity + Send + Clone + Copy + Sync,
// F: Fn() -> G + Send + Sync,
// const N: usize,
// FF: Send + Sync + Fn(&T, &T) -> bool,

impl<'a, 'b: 'a, const N: usize, C, E> DBSCANRunner<'a, N, C, E>
where
    C: NDPointConverter<E, N>,
    E: Sync + HasIntensity,
{
    fn run(
        &self,
        prefiltered_peaks: &'b Vec<E>,
        intensity_sorted_indices: &'b Vec<(usize, u64)>,
        indexed_points: &'b (dyn QueriableIndexedPoints<'a, N, usize> + std::marker::Sync),
        quad_points: &'b [NDPoint<N>],
    ) -> ClusterLabels {
        let usize_filterfun = match self.filter_fun {
            Some(filterfun) => {
                let cl = |a: &usize, b: &usize| {
                    filterfun(&prefiltered_peaks[*a], &prefiltered_peaks[*b])
                };
                let bind = Some(cl);
                bind
            }
            None => None,
        };
        // |a: &usize, b: &usize| {
        //    (self.filter_fun)(&prefiltered_peaks[*a], &prefiltered_peaks[*b])
        // };
        let mut state = DBSCANRunnerState::new(intensity_sorted_indices.len(), usize_filterfun);

        let points: DBSCANPoints<N, E> = DBSCANPoints {
            prefiltered_peaks,
            intensity_sorted_indices,
            indexed_points,
            quad_points,
        };
        // Q: if filter fun is required ... why is it an option?
        state = self.process_points(state, &points);
        state = self.report_timers(state);

        self.take_cluster_labels(state)
    }

    fn report_timers(&self, mut state: DBSCANRunnerState) -> DBSCANRunnerState {
        state.timers.main.stop(false);
        state.timers.report_if_gt_us(1000000);
        state
    }

    fn take_cluster_labels(&self, state: DBSCANRunnerState) -> ClusterLabels {
        state.cluster_labels
    }

    fn process_points(
        &self,
        mut state: DBSCANRunnerState,
        points: &DBSCANPoints<'a, N, E>,
    ) -> DBSCANRunnerState {
        let my_progbar =
            state.create_progress_bar(points.intensity_sorted_indices.len(), self.progress);

        for (point_index, _intensity) in points
            .intensity_sorted_indices
            .iter()
            .progress_with(my_progbar)
        {
            self.process_single_point(
                *point_index,
                &points,
                &mut state.cluster_labels,
                &mut state.filter_fun_cache,
                &mut state.timers,
                &mut state.candidate_metrics,
            );
        }
        state
    }

    fn process_single_point(
        &self,
        point_index: usize,
        points: &DBSCANPoints<'a, N, E>,
        cluster_labels: &mut ClusterLabels,
        filter_fun_cache: &mut Option<FilterFunCache>,
        timers: &mut DBScanTimers,
        cc_metrics: &mut CandidateCountMetrics,
    ) {
        if cluster_labels.get(point_index) != ClusterLabel::Unassigned {
            return;
        }

        let (neighbors, ref_point) =
            self.find_neighbors(point_index, points, filter_fun_cache, timers, cc_metrics);
        if !self.is_core_point(&neighbors, points.prefiltered_peaks, timers) {
            cluster_labels.set_noise(point_index);
            return;
        }

        self.expand_cluster(
            point_index,
            ref_point.unwrap(),
            neighbors,
            points,
            cluster_labels,
            filter_fun_cache,
            timers,
        );
    }

    fn find_neighbors(
        &self,
        point_index: usize,
        points: &DBSCANPoints<'a, N, E>,
        filter_fun_cache: &mut Option<FilterFunCache>,
        timers: &mut DBScanTimers,
        cc_metrics: &mut CandidateCountMetrics,
    ) -> (Vec<usize>, Option<&NDPoint<N>>) {
        timers.outer_loop_nn_timer.reset_start();
        let query_elems = self
            .converter
            .convert_to_bounds_query(&points.quad_points[point_index]);
        let mut candidate_neighbors = points
            .indexed_points
            .query_ndrange(&query_elems.0, query_elems.1)
            .iter()
            .map(|x| **x)
            .collect::<Vec<_>>();
        timers.outer_loop_nn_timer.stop(false);

        if filter_fun_cache.is_none() {
            return (candidate_neighbors, query_elems.1);
        }

        let num_initial_candidates = candidate_neighbors.len();
        candidate_neighbors.retain(|i| {
            let tmp = filter_fun_cache.as_mut().unwrap();
            let res_in_cache = tmp.get(*i, point_index);
            match res_in_cache {
                Some(res) => res,
                None => {
                    let res = (self.filter_fun.unwrap())(
                        &points.prefiltered_peaks[*i],
                        &points.prefiltered_peaks[point_index],
                    );
                    tmp.set(*i, point_index, res);
                    res
                }
            }
        });

        let neighbors = candidate_neighbors;
        let candidates_after_filter = neighbors.len();
        cc_metrics
            .initial_candidates_counts
            .add(num_initial_candidates as f32, 1);
        cc_metrics
            .final_candidates_counts
            .add(candidates_after_filter as f32, 1);

        (neighbors, query_elems.1)
    }

    fn is_core_point(
        &self,
        neighbors: &[usize],
        prefiltered_peaks: &'a Vec<E>,
        timers: &mut DBScanTimers,
    ) -> bool {
        timers.outer_intensity_calculation.reset_start();
        let neighbor_intensity_total = neighbors
            .iter()
            .map(|i| prefiltered_peaks[*i].intensity())
            .sum::<u64>();
        timers.outer_intensity_calculation.stop(false);
        return neighbor_intensity_total >= self.min_intensity;
    }

    fn expand_cluster(
        &self,
        point_index: usize,
        query_point: &NDPoint<N>,
        neighbors: Vec<usize>,
        points: &DBSCANPoints<'a, N, E>,
        cluster_labels: &mut ClusterLabels,
        filter_fun_cache: &mut Option<FilterFunCache>,
        timers: &mut DBScanTimers,
    ) {
        cluster_labels.set_new_cluster(point_index);

        let mut seed_set: Vec<usize> = Vec::new();
        seed_set.extend(neighbors);

        while let Some(neighbor) = seed_set.pop() {
            let neighbor_index = neighbor;
            if cluster_labels.get(neighbor_index) == ClusterLabel::Noise {
                cluster_labels.set_current_cluster(neighbor_index);
            }

            if cluster_labels.get(neighbor_index) != ClusterLabel::Unassigned {
                continue;
            }

            cluster_labels.set_current_cluster(neighbor_index);

            timers.inner_loop_nn_timer.reset_start();
            let inner_query_elems = self
                .converter
                .convert_to_bounds_query(&points.quad_points[neighbor]);
            let mut local_neighbors = points
                .indexed_points
                .query_ndrange(&inner_query_elems.0, inner_query_elems.1);
            timers.inner_loop_nn_timer.stop(false);

            if filter_fun_cache.is_some() {
                local_neighbors.retain(|i| {
                    let cache = filter_fun_cache.as_mut().unwrap();
                    let res = cache.get(**i, point_index);
                    match res {
                        Some(res) => res,
                        None => {
                            let res = (self.filter_fun.unwrap())(
                                &points.prefiltered_peaks[**i],
                                &points.prefiltered_peaks[point_index],
                            );
                            cache.set(**i, point_index, res);
                            res
                        }
                    }
                });
            }

            timers.inner_intensity_calculation.reset_start();
            let query_intensity = points.prefiltered_peaks[neighbor_index].intensity();
            let neighbor_intensity_total = local_neighbors
                .iter()
                .map(|i| points.prefiltered_peaks[**i].intensity())
                .sum::<u64>();
            timers.inner_intensity_calculation.stop(false);

            if local_neighbors.len() >= self.min_n && neighbor_intensity_total >= self.min_intensity
            {
                local_neighbors
                    .retain(|i| !matches!(cluster_labels.get(**i), ClusterLabel::Cluster(_)));

                timers.local_neighbor_filter_timer.reset_start();
                local_neighbors.retain(|i| {
                    let going_downhill =
                        points.prefiltered_peaks[**i].intensity() <= query_intensity;

                    let p: &NDPoint<N> = &points.quad_points[**i];
                    let mut within_distance = true;
                    for ((p, q), max_dist) in p
                        .values
                        .iter()
                        .zip(query_point.values)
                        .zip(self.max_extension_distances.iter())
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
}

pub fn _dbscan<
    'a,
    const N: usize,
    C: NDPointConverter<E, N>,
    E: Sync + HasIntensity,
    T: QueriableIndexedPoints<'a, N, usize> + std::marker::Sync,
>(
    indexed_points: &'a T,
    prefiltered_peaks: &'a Vec<E>,
    quad_points: &'a [NDPoint<N>],
    min_n: usize,
    min_intensity: u64,
    intensity_sorted_indices: &'a Vec<(usize, u64)>,
    filter_fun: Option<&'a (dyn Fn(&E, &E) -> bool + Send + Sync)>,
    converter: C,
    progress: bool,
    max_extension_distances: &'a [f32; N],
) -> ClusterLabels {
    let runner = DBSCANRunner {
        min_n,
        min_intensity,
        converter,
        progress,
        filter_fun: filter_fun,
        max_extension_distances,
    };

    let cluster_labels = runner.run(
        prefiltered_peaks,
        intensity_sorted_indices,
        indexed_points,
        quad_points,
    );

    cluster_labels
}
