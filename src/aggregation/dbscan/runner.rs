use crate::space::space_generics::{
    convert_to_bounds_query, AsNDPointsAtIndex, DistantAtIndex, HasIntensity, IntenseAtIndex,
    NDPoint, QueriableIndexedPoints,
};
use crate::space::space_generics::{AsAggregableAtIndex, NDPointConverter};
use std::marker::PhantomData;
use std::ops::Index;

use crate::utils;
use indicatif::ProgressIterator;

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

    fn set_cluster(
        &mut self,
        index: usize,
        cluster_id: u64,
    ) {
        self.cluster_labels[index] = ClusterLabel::Cluster(cluster_id);
    }

    fn set_new_cluster(
        &mut self,
        index: usize,
    ) {
        self.num_clusters += 1;
        self.set_cluster(index, self.num_clusters);
    }

    fn set_current_cluster(
        &mut self,
        index: usize,
    ) {
        let cluster_id = self.num_clusters;
        self.set_cluster(index, cluster_id);
    }

    fn set_noise(
        &mut self,
        index: usize,
    ) {
        self.cluster_labels[index] = ClusterLabel::Noise;
    }

    fn get(
        &self,
        index: usize,
    ) -> ClusterLabel<u64> {
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

    fn report_if_gt_us(
        &self,
        min_time: u128,
    ) {
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
    fn new<P>(
        nlabels: usize,
        usize_filterfun: Option<P>,
    ) -> Self
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

    fn create_progress_bar(
        &self,
        len: usize,
        visible: bool,
    ) -> indicatif::ProgressBar {
        if visible {
            indicatif::ProgressBar::new(len as u64)
        } else {
            indicatif::ProgressBar::hidden()
        }
    }
}

struct DBSCANRunner<'a, const N: usize, D> {
    min_n: usize,
    min_intensity: u64,
    filter_fun: Option<&'a (dyn Fn(&D) -> bool + Send + Sync)>,
    progress: bool,
    max_extension_distances: &'a [f32; N],
}

struct DBSCANPoints<'a, const N: usize, PP, PE, DAI, E>
where
    PP: IntenseAtIndex + std::marker::Send + ?Sized,
    PE: AsNDPointsAtIndex<N> + ?Sized,
    DAI: DistantAtIndex<E> + ?Sized,
{
    raw_elements: &'a PP, // &'a Vec<E>,
    intensity_sorted_indices: Vec<(usize, u64)>,
    indexed_points: &'a (dyn QueriableIndexedPoints<'a, N> + std::marker::Sync),
    projected_elements: &'a PE, // [NDPoint<N>],
    raw_dist: &'a DAI,
    _phantom_metric: PhantomData<E>,
}

impl<'a, const N: usize, PP, QQ, D, E> DBSCANPoints<'a, N, PP, QQ, D, E>
where
    PP: IntenseAtIndex + std::marker::Send + ?Sized,
    QQ: AsNDPointsAtIndex<N> + ?Sized,
    D: DistantAtIndex<E> + ?Sized,
{
    fn get_intensity_at_index(
        &self,
        index: usize,
    ) -> u64 {
        self.raw_elements.intensity_at_index(index)
    }

    fn get_ndpoint_at_index(
        &self,
        index: usize,
    ) -> NDPoint<N> {
        self.projected_elements.get_ndpoint(index)
    }

    fn get_distance_at_indices(
        &self,
        a: usize,
        b: usize,
    ) -> E {
        self.raw_dist.distance_at_indices(a, b)
    }
}

impl<'a, 'b: 'a, const N: usize, D> DBSCANRunner<'a, N, D>
where
    D: Sync,
{
    fn run<PP, PE, DAI>(
        &self,
        raw_elements: &'b PP, // Vec<E>, // trait impl Index<usize, Output=E>
        intensity_sorted_indices: Vec<(usize, u64)>,
        indexed_points: &'b (dyn QueriableIndexedPoints<'a, N> + std::marker::Sync),
        projected_elements: &'b PE, //[NDPoint<N>], // trait impl AsNDPointAtIndex<usize, Output=NDPoint<N>>
        raw_distance_calculator: &'b DAI,
    ) -> ClusterLabels
    where
        PP: IntenseAtIndex + Send + Sync + ?Sized,
        PE: AsNDPointsAtIndex<N> + ?Sized,
        DAI: DistantAtIndex<D> + Send + Sync + ?Sized,
    {
        let usize_filterfun = match self.filter_fun {
            Some(filterfun) => {
                let cl = |a: &usize, b: &usize| {
                    filterfun(&raw_distance_calculator.distance_at_indices(*a, *b))
                };
                let bind = Some(cl);
                bind
            },
            None => None,
        };

        let mut state = DBSCANRunnerState::new(intensity_sorted_indices.len(), usize_filterfun);

        let points: DBSCANPoints<N, PP, PE, DAI, D> = DBSCANPoints {
            raw_elements,
            intensity_sorted_indices: intensity_sorted_indices,
            indexed_points,
            projected_elements,
            raw_dist: raw_distance_calculator,
            _phantom_metric: PhantomData,
        };
        // Q: if filter fun is required ... why is it an option?
        state = self.process_points(state, &points);
        state = self.report_timers(state);

        self.take_cluster_labels(state)
    }

    fn report_timers(
        &self,
        mut state: DBSCANRunnerState,
    ) -> DBSCANRunnerState {
        state.timers.main.stop(false);
        state.timers.report_if_gt_us(1000000);
        state
    }

    fn take_cluster_labels(
        &self,
        state: DBSCANRunnerState,
    ) -> ClusterLabels {
        state.cluster_labels
    }

    fn process_points<PP, PE, DAI>(
        &self,
        mut state: DBSCANRunnerState,
        points: &DBSCANPoints<'a, N, PP, PE, DAI, D>,
    ) -> DBSCANRunnerState
    where
        PP: IntenseAtIndex + Send + ?Sized,
        PE: AsNDPointsAtIndex<N> + ?Sized,
        DAI: DistantAtIndex<D> + Send + Sync + ?Sized,
    {
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

    /// This method gets applied to every point in decreasing intensity order.
    fn process_single_point<PP, PE, DAI>(
        &self,
        point_index: usize,
        points: &DBSCANPoints<'a, N, PP, PE, DAI, D>,
        cluster_labels: &mut ClusterLabels,
        filter_fun_cache: &mut Option<FilterFunCache>,
        timers: &mut DBScanTimers,
        cc_metrics: &mut CandidateCountMetrics,
    ) where
        PP: IntenseAtIndex + Send + ?Sized,
        PE: AsNDPointsAtIndex<N> + ?Sized,
        DAI: DistantAtIndex<D> + Send + Sync + ?Sized,
    {
        if cluster_labels.get(point_index) != ClusterLabel::Unassigned {
            return;
        }

        let neighbors = self.find_main_loop_neighbors(
            point_index,
            points,
            filter_fun_cache,
            timers,
            cc_metrics,
        );
        if !self.is_core_point(&neighbors, points.raw_elements, timers) {
            cluster_labels.set_noise(point_index);
            return;
        }

        self.main_loop_expand_cluster(
            point_index,
            neighbors,
            points,
            cluster_labels,
            filter_fun_cache,
            timers,
        );
    }

    fn find_main_loop_neighbors<PP, PE, DAI>(
        &self,
        point_index: usize,
        points: &DBSCANPoints<'a, N, PP, PE, DAI, D>,
        filter_fun_cache: &mut Option<FilterFunCache>,
        timers: &mut DBScanTimers,
        cc_metrics: &mut CandidateCountMetrics,
    ) -> Vec<usize>
    where
        PP: IntenseAtIndex + Send + ?Sized,
        PE: AsNDPointsAtIndex<N> + ?Sized,
        DAI: DistantAtIndex<D> + Send + Sync + ?Sized,
    {
        timers.outer_loop_nn_timer.reset_start();
        let binding = points.projected_elements.get_ndpoint(point_index);
        let query_elems = convert_to_bounds_query(&binding);
        let mut candidate_neighbors = points
            .indexed_points
            .query_ndrange(&query_elems.0, query_elems.1)
            .iter()
            .map(|x| *x)
            .collect::<Vec<_>>();
        timers.outer_loop_nn_timer.stop(false);

        if filter_fun_cache.is_none() {
            return candidate_neighbors;
        }

        let num_initial_candidates = candidate_neighbors.len();
        candidate_neighbors.retain(|i| {
            let tmp = filter_fun_cache.as_mut().unwrap();
            let res_in_cache = tmp.get(*i, point_index);
            match res_in_cache {
                Some(res) => res,
                None => {
                    let res = (self.filter_fun.unwrap())(
                        &points.get_distance_at_indices(*i, point_index),
                    );
                    tmp.set(*i, point_index, res);
                    res
                },
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

        neighbors
    }

    fn is_core_point<PP>(
        &self,
        neighbors: &[usize],
        raw_elements: &'a PP,
        timers: &mut DBScanTimers,
    ) -> bool
    where
        PP: IntenseAtIndex + Send + ?Sized,
    {
        timers.outer_intensity_calculation.reset_start();
        let neighbor_intensity_total = neighbors
            .iter()
            .map(|i| raw_elements.intensity_at_index(*i))
            .sum::<u64>();
        timers.outer_intensity_calculation.stop(false);
        return neighbor_intensity_total >= self.min_intensity;
    }

    fn main_loop_expand_cluster<PP, PE, DAI>(
        &self,
        apex_point_index: usize,
        neighbors: Vec<usize>,
        points: &DBSCANPoints<'a, N, PP, PE, DAI, D>,
        cluster_labels: &mut ClusterLabels,
        filter_fun_cache: &mut Option<FilterFunCache>,
        timers: &mut DBScanTimers,
    ) where
        PP: IntenseAtIndex + Send + ?Sized,
        PE: AsNDPointsAtIndex<N> + ?Sized,
        DAI: DistantAtIndex<D> + Send + Sync + ?Sized,
    {
        cluster_labels.set_new_cluster(apex_point_index);
        let mut seed_set: Vec<usize> = neighbors;

        while let Some(neighbor_index) = seed_set.pop() {
            if !self.process_neighbor(neighbor_index, cluster_labels) {
                continue;
            }

            let local_neighbors = self.find_local_neighbors(neighbor_index, points, timers);
            let filtered_neighbors = self.filter_neighbors_inner_loop(
                local_neighbors,
                apex_point_index,
                neighbor_index,
                points,
                cluster_labels,
                filter_fun_cache,
                timers,
            );

            seed_set.extend(filtered_neighbors);
        }
    }

    fn process_neighbor(
        &self,
        neighbor_index: usize,
        cluster_labels: &mut ClusterLabels,
    ) -> bool {
        match cluster_labels.get(neighbor_index) {
            ClusterLabel::Noise => {
                cluster_labels.set_current_cluster(neighbor_index);
                true
            },
            ClusterLabel::Unassigned => {
                cluster_labels.set_current_cluster(neighbor_index);
                true
            },
            ClusterLabel::Cluster(_) => false,
        }
    }

    fn find_local_neighbors<PP, PE, DAI>(
        &self,
        neighbor_index: usize,
        points: &DBSCANPoints<'a, N, PP, PE, DAI, D>,
        timers: &mut DBScanTimers,
    ) -> Vec<usize>
    where
        PP: IntenseAtIndex + Send + ?Sized,
        PE: AsNDPointsAtIndex<N> + ?Sized,
        DAI: DistantAtIndex<D> + Send + Sync + ?Sized,
    {
        timers.inner_loop_nn_timer.reset_start();
        let binding = points.projected_elements.get_ndpoint(neighbor_index);
        let inner_query_elems = convert_to_bounds_query(&binding);
        let local_neighbors: Vec<usize> = points
            .indexed_points
            .query_ndrange(&inner_query_elems.0, inner_query_elems.1)
            .iter_mut()
            .map(|x| *x)
            .collect::<Vec<_>>();
        timers.inner_loop_nn_timer.stop(false);
        local_neighbors
    }

    fn filter_neighbors_inner_loop<PP, PE, DAI>(
        &self,
        local_neighbors: Vec<usize>,
        cluster_apex_point_index: usize,
        current_center_point_index: usize,
        points: &DBSCANPoints<'a, N, PP, PE, DAI, D>,
        cluster_labels: &ClusterLabels,
        filter_fun_cache: &mut Option<FilterFunCache>,
        timers: &mut DBScanTimers,
    ) -> Vec<usize>
    where
        PP: IntenseAtIndex + Send + ?Sized,
        PE: AsNDPointsAtIndex<N> + ?Sized,
        DAI: DistantAtIndex<D> + Send + Sync + ?Sized,
    {
        let filtered = self.apply_filter_fun(
            local_neighbors,
            cluster_apex_point_index,
            points,
            filter_fun_cache,
        );

        if !self.is_extension_core_point(&filtered, current_center_point_index, points, timers) {
            return Vec::new();
        }

        let unassigned = self.filter_unassigned(filtered, cluster_labels);
        let unassigned_in_global_distance =
            self.filter_by_apex_distance(unassigned, cluster_apex_point_index, points, timers);
        self.filter_by_local_intensity_and_distance(
            unassigned_in_global_distance,
            current_center_point_index,
            points,
            timers,
        )
    }

    fn filter_by_apex_distance<PP, PE, DAI>(
        &self,
        mut neighbors: Vec<usize>,
        apex_point_index: usize,
        points: &DBSCANPoints<'a, N, PP, PE, DAI, D>,
        timers: &mut DBScanTimers,
    ) -> Vec<usize>
    where
        PP: IntenseAtIndex + Send + ?Sized,
        PE: AsNDPointsAtIndex<N> + ?Sized,
        DAI: DistantAtIndex<D> + Send + Sync + ?Sized,
    {
        timers.local_neighbor_filter_timer.reset_start();
        let query_point = &points.projected_elements.get_ndpoint(apex_point_index);
        neighbors.retain(|&i| {
            self.is_within_max_distance(&points.projected_elements.get_ndpoint(i), query_point)
        });
        timers.local_neighbor_filter_timer.stop(false);
        neighbors
    }

    fn is_extension_core_point<PP, PE, DAI>(
        &self,
        neighbors: &[usize],
        current_center_point_index: usize,
        points: &DBSCANPoints<'a, N, PP, PE, DAI, D>,
        timers: &mut DBScanTimers,
    ) -> bool
    where
        PP: IntenseAtIndex + Send + ?Sized,
        PE: AsNDPointsAtIndex<N> + ?Sized,
        DAI: DistantAtIndex<D> + Send + Sync + ?Sized,
    {
        timers.inner_intensity_calculation.reset_start();
        let mut neighbor_intensity_total: u64 = neighbors
            .iter()
            .map(|&i| points.raw_elements.intensity_at_index(i))
            .sum();

        neighbor_intensity_total += points
            .raw_elements
            .intensity_at_index(current_center_point_index);
        timers.inner_intensity_calculation.stop(false);

        neighbors.len() >= self.min_n && neighbor_intensity_total >= self.min_intensity
    }

    /// This is mean to apply additional filter logic that considers
    /// elements that are not only represented by the 'space' of the points
    /// or the intensity.
    ///
    /// Some examples might be if every point represents say ... a chromatogram
    /// one could pass a function that checks if the chromatograms a high correlation.
    /// Because two might share the same point in space, intensity is not really
    /// relevant but co-elution might be critical.
    fn apply_filter_fun<PP, PE, DAI>(
        &self,
        local_neighbors: Vec<usize>,
        point_index: usize,
        points: &DBSCANPoints<'a, N, PP, PE, DAI, D>,
        filter_fun_cache: &mut Option<FilterFunCache>,
    ) -> Vec<usize>
    where
        PP: IntenseAtIndex + Send + ?Sized,
        PE: AsNDPointsAtIndex<N> + ?Sized,
        DAI: DistantAtIndex<D> + Send + Sync + ?Sized,
    {
        if let Some(cache) = filter_fun_cache {
            local_neighbors
                .into_iter()
                .filter(|&i| {
                    cache.get(i, point_index).unwrap_or_else(|| {
                        let res = (self.filter_fun.unwrap())(
                            &points.get_distance_at_indices(i, point_index),
                        );
                        cache.set(i, point_index, res);
                        res
                    })
                })
                .collect()
        } else {
            local_neighbors
        }
    }

    fn filter_unassigned(
        &self,
        mut neighbors: Vec<usize>,
        cluster_labels: &ClusterLabels,
    ) -> Vec<usize> {
        neighbors.retain(|&i| matches!(cluster_labels.get(i), ClusterLabel::Unassigned));
        neighbors
    }

    fn filter_by_local_intensity_and_distance<PP, PE, DAI>(
        &self,
        mut neighbors: Vec<usize>,
        neighbor_index: usize,
        points: &DBSCANPoints<'a, N, PP, PE, DAI, D>,
        timers: &mut DBScanTimers,
    ) -> Vec<usize>
    where
        PP: IntenseAtIndex + Send + ?Sized,
        PE: AsNDPointsAtIndex<N> + ?Sized,
        DAI: DistantAtIndex<D> + Send + Sync + ?Sized,
    {
        timers.local_neighbor_filter_timer.reset_start();
        let query_intensity = points.raw_elements.intensity_at_index(neighbor_index);
        let query_point = &points.projected_elements.get_ndpoint(neighbor_index);

        neighbors.retain(|&i| {
            let going_downhill = points.raw_elements.intensity_at_index(i) <= query_intensity;
            let within_distance =
                self.is_within_max_distance(&points.projected_elements.get_ndpoint(i), query_point);
            going_downhill && within_distance
        });

        timers.local_neighbor_filter_timer.stop(false);
        neighbors
    }

    fn is_within_max_distance(
        &self,
        p: &NDPoint<N>,
        query_point: &NDPoint<N>,
    ) -> bool {
        p.values
            .iter()
            .zip(query_point.values)
            .zip(self.max_extension_distances.iter())
            .all(|((p, q), max_dist)| (p - q).abs() <= *max_dist)
    }
}

pub fn dbscan_label_clusters<
    'a,
    const N: usize,
    RE: IntenseAtIndex + DistantAtIndex<D> + Send + Sync + AsAggregableAtIndex<E> + ?Sized,
    T: QueriableIndexedPoints<'a, N> + Send + std::marker::Sync,
    PE: AsNDPointsAtIndex<N> + Send + Sync + ?Sized,
    D: Send + Sync,
    E: Send + Sync + Copy,
>(
    indexed_points: &'a T,
    raw_elements: &'a RE,
    projected_elements: &'a PE, // [NDPoint<N>],
    min_n: usize,
    min_intensity: u64,
    intensity_sorted_indices: Vec<(usize, u64)>,
    filter_fun: Option<&'a (dyn Fn(&D) -> bool + Send + Sync)>,
    progress: bool,
    max_extension_distances: &'a [f32; N],
) -> ClusterLabels {
    let runner = DBSCANRunner {
        min_n,
        min_intensity,
        progress,
        filter_fun: filter_fun,
        max_extension_distances,
    };

    let cluster_labels = runner.run(
        raw_elements,
        intensity_sorted_indices,
        indexed_points,
        projected_elements,
        raw_elements,
    );

    cluster_labels
}
