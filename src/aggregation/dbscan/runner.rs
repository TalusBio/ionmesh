use crate::space::space_generics::AsAggregableAtIndex;
use crate::space::space_generics::{
    AsNDPointsAtIndex, DistantAtIndex, IntenseAtIndex, NDBoundary, NDPoint, QueriableIndexedPoints,
};
use crate::utils;
use core::fmt::Debug;
use indicatif::ProgressIterator;
use log::debug;
use std::marker::PhantomData;
use std::sync::Arc;

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
    // TODO aux timers can probably be a hashmap
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

        let filter_fun_cache = usize_filterfun.map(|_| FilterFunCache::new(nlabels));
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

struct DBSCANRunner<'a, const N: usize, D, FF>
where
    FF: Fn(&D) -> bool + Send + Sync + ?Sized,
    D: Send + Sync,
{
    min_n: usize,
    min_intensity: u64,
    filter_fun: Option<&'a FF>,
    progress: bool,
    max_extension_distances: &'a [f32; N],
    _phantom: PhantomData<D>,
}

impl<'a, const N: usize, D, FF> Debug for DBSCANRunner<'a, N, D, FF>
where
    FF: Fn(&D) -> bool + Send + Sync + ?Sized,
    D: Send + Sync,
{
    fn fmt(
        &self,
        f: &mut std::fmt::Formatter<'_>,
    ) -> std::fmt::Result {
        f.debug_struct("DBSCANRunner")
            .field("min_n", &self.min_n)
            .field("min_intensity", &self.min_intensity)
            .field("filter_fun", &"Some<Fn(&f32) -> bool>???")
            .field("progress", &self.progress)
            .field("max_extension_distances", &self.max_extension_distances)
            .finish()
    }
}

#[derive(Clone)]
struct DBSCANPoints<'a, const N: usize, PP, PE, DAI, E, QIP>
where
    PP: IntenseAtIndex + Send + Sync + ?Sized,
    PE: AsNDPointsAtIndex<N> + Send + Sync + ?Sized,
    DAI: DistantAtIndex<E> + ?Sized,
    QIP: QueriableIndexedPoints<N> + Sync,
{
    raw_elements: &'a PP, // &'a Vec<E>,
    intensity_sorted_indices: Vec<(usize, u64)>,
    indexed_points: &'a QIP,
    projected_elements: &'a PE, // [NDPoint<N>],
    raw_dist: &'a DAI,
    _phantom_metric: PhantomData<E>,
}

impl<'a, const N: usize, PP, QQ, DAI, E, QIP> QueriableIndexedPoints<N>
    for DBSCANPoints<'a, N, PP, QQ, DAI, E, QIP>
where
    PP: IntenseAtIndex + Send + Sync + ?Sized,
    QQ: AsNDPointsAtIndex<N> + Send + Sync + ?Sized,
    DAI: DistantAtIndex<E> + ?Sized,
    QIP: QueriableIndexedPoints<N> + Sync,
{
    fn query_ndpoint(
        &self,
        point: &NDPoint<N>,
    ) -> Vec<usize> {
        self.indexed_points.query_ndpoint(point)
    }

    fn query_ndrange(
        &self,
        boundary: &NDBoundary<N>,
        reference_point: Option<&NDPoint<N>>,
    ) -> Vec<usize> {
        self.indexed_points.query_ndrange(boundary, reference_point)
    }
}

impl<'a, const N: usize, PP, QQ, DAI, E, QIP> DistantAtIndex<E>
    for DBSCANPoints<'a, N, PP, QQ, DAI, E, QIP>
where
    PP: IntenseAtIndex + Sync + Send + ?Sized,
    QQ: AsNDPointsAtIndex<N> + Send + Sync + ?Sized,
    DAI: DistantAtIndex<E> + ?Sized,
    QIP: QueriableIndexedPoints<N> + std::marker::Sync,
{
    fn distance_at_indices(
        &self,
        a: usize,
        b: usize,
    ) -> E {
        self.raw_dist.distance_at_indices(a, b)
    }
}

impl<'a, const N: usize, PP, QQ, D, E, QIP> IntenseAtIndex
    for DBSCANPoints<'a, N, PP, QQ, D, E, QIP>
where
    PP: IntenseAtIndex + std::marker::Send + Sync + ?Sized,
    QQ: AsNDPointsAtIndex<N> + Send + Sync + ?Sized,
    D: DistantAtIndex<E> + ?Sized,
    QIP: QueriableIndexedPoints<N> + std::marker::Sync,
{
    fn intensity_at_index(
        &self,
        index: usize,
    ) -> u64 {
        self.raw_elements.intensity_at_index(index)
    }

    fn weight_at_index(
        &self,
        index: usize,
    ) -> u64 {
        self.raw_elements.weight_at_index(index)
    }

    fn intensity_index_length(&self) -> usize {
        self.raw_elements.intensity_index_length()
    }
}

impl<'a, const N: usize, PP, QQ, D, E, QIP> AsNDPointsAtIndex<N>
    for DBSCANPoints<'a, N, PP, QQ, D, E, QIP>
where
    PP: IntenseAtIndex + std::marker::Send + Sync + ?Sized,
    QQ: AsNDPointsAtIndex<N> + Send + Sync + ?Sized,
    D: DistantAtIndex<E> + ?Sized,
    QIP: QueriableIndexedPoints<N> + std::marker::Sync,
{
    fn get_ndpoint(
        &self,
        index: usize,
    ) -> NDPoint<N> {
        self.projected_elements.get_ndpoint(index)
    }

    fn num_ndpoints(&self) -> usize {
        self.projected_elements.num_ndpoints()
    }
}

impl<'c, 'b: 'c, 'a: 'b, const N: usize, D, FF> DBSCANRunner<'b, N, D, FF>
where
    D: Sync + Send + 'a,
    FF: Fn(&D) -> bool + Send + Sync + 'a + ?Sized,
{
    fn run<PP, PE, DAI, QIP>(
        &self,
        raw_elements: &'a PP, // Vec<E>, // trait impl Index<usize, Output=E>
        intensity_sorted_indices: Vec<(usize, u64)>,
        indexed_points: &'a QIP,
        projected_elements: &'a PE, //[NDPoint<N>], // trait impl AsNDPointAtIndex<usize, Output=NDPoint<N>>
        raw_distance_calculator: &'a DAI,
    ) -> ClusterLabels
    where
        PP: IntenseAtIndex + Send + Sync + ?Sized,
        PE: AsNDPointsAtIndex<N> + Send + Sync + ?Sized,
        DAI: DistantAtIndex<D> + Send + Sync + ?Sized,
        QIP: QueriableIndexedPoints<N> + std::marker::Sync + std::fmt::Debug,
    {
        if self.progress {
            debug!("Starting DBSCAN");
            debug!("Params: {:?}", self);
        }
        let usize_filterfun = match self.filter_fun {
            Some(filterfun) => {
                let cl = |a: &usize, b: &usize| {
                    filterfun(&raw_distance_calculator.distance_at_indices(*a, *b))
                };

                Some(cl)
            },
            None => None,
        };

        let mut state = DBSCANRunnerState::new(intensity_sorted_indices.len(), usize_filterfun);

        debug_assert!(intensity_sorted_indices.len() == raw_elements.intensity_index_length());
        debug_assert!(intensity_sorted_indices.len() == projected_elements.num_ndpoints());
        // trace!("Index: {:?}", indexed_points);

        let points: DBSCANPoints<N, PP, PE, DAI, D, QIP> = DBSCANPoints {
            raw_elements,
            intensity_sorted_indices,
            indexed_points,
            projected_elements,
            raw_dist: raw_distance_calculator,
            _phantom_metric: PhantomData,
        };
        // Q: if filter fun is required ... why is it an option?
        state = self.process_points(state, Arc::new(points));
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
        if self.progress {
            debug!("Finished DBSCAN");
            debug!(
                "Exporting Num clusters: {}",
                state.cluster_labels.num_clusters
            );
        }
        state.cluster_labels
    }

    fn process_points<PP, PE, DAI, QIP>(
        &self,
        mut state: DBSCANRunnerState,
        points: Arc<DBSCANPoints<'a, N, PP, PE, DAI, D, QIP>>,
    ) -> DBSCANRunnerState
    where
        PP: IntenseAtIndex + Send + Sync + ?Sized,
        PE: AsNDPointsAtIndex<N> + Sync + Send + ?Sized,
        DAI: DistantAtIndex<D> + Send + Sync + ?Sized,
        QIP: QueriableIndexedPoints<N> + std::marker::Sync,
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
                Arc::clone(&points),
                &mut state.cluster_labels,
                &mut state.filter_fun_cache,
                &mut state.timers,
                &mut state.candidate_metrics,
            );
        }
        state
    }

    /// This method gets applied to every point in decreasing intensity order.
    fn process_single_point<PP, PE, DAI, QIP>(
        &'b self,
        point_index: usize,
        points: Arc<DBSCANPoints<'a, N, PP, PE, DAI, D, QIP>>,
        cluster_labels: &mut ClusterLabels,
        filter_fun_cache: &mut Option<FilterFunCache>,
        timers: &mut DBScanTimers,
        cc_metrics: &mut CandidateCountMetrics,
    ) where
        PP: IntenseAtIndex + Send + Sync + ?Sized,
        PE: AsNDPointsAtIndex<N> + Send + Sync + ?Sized,
        DAI: DistantAtIndex<D> + Send + Sync + ?Sized,
        QIP: QueriableIndexedPoints<N> + Sync,
    {
        if cluster_labels.get(point_index) != ClusterLabel::Unassigned {
            return;
        }

        let neighbors = self.find_main_loop_neighbors(
            point_index,
            Arc::clone(&points),
            filter_fun_cache,
            timers,
            cc_metrics,
        );

        // trace!("Neighbors: {:?}", neighbors);

        if !self.is_core_point(&neighbors, Arc::clone(&points), timers) {
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

    fn find_main_loop_neighbors<PTS>(
        &self,
        point_index: usize,
        points: Arc<PTS>,
        filter_fun_cache: &mut Option<FilterFunCache>,
        timers: &mut DBScanTimers,
        cc_metrics: &mut CandidateCountMetrics,
    ) -> Vec<usize>
    where
        PTS: AsNDPointsAtIndex<N>
            + DistantAtIndex<D>
            + QueriableIndexedPoints<N>
            + IntenseAtIndex
            + Send
            + Sync
            + ?Sized,
    {
        timers.outer_loop_nn_timer.reset_start();
        let binding = points.get_ndpoint(point_index);
        let mut candidate_neighbors = points.query_ndpoint(&binding);
        // Every point should have at least itself as a neighbor.
        debug_assert!(
            !candidate_neighbors.is_empty(),
            "No neighbors found, {}, {:?}, at least itself should be a neighbor",
            point_index,
            binding
        );

        // trace!("Query elems: {:?}", query_elems);
        // trace!("Candidate neighbors: {:?}", candidate_neighbors);
        if cfg!(debug_assertions) {
            let max_i = candidate_neighbors.iter().max().unwrap();
            // Make sure all generated neighbors are within the bounds.
            assert!(
                *max_i < points.num_ndpoints(),
                "Index: {} out of proj elems bounds",
                max_i,
            );
            assert!(
                *max_i < points.intensity_index_length(),
                "Index: {} out of intensity bounds",
                max_i
            );
        }
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
                    let res =
                        (self.filter_fun.unwrap())(&points.distance_at_indices(*i, point_index));
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
        points: Arc<PP>,
        timers: &mut DBScanTimers,
    ) -> bool
    where
        PP: IntenseAtIndex + Send + ?Sized,
    {
        timers.outer_intensity_calculation.reset_start();
        let neighbor_intensity_total = neighbors
            .iter()
            .map(|i| points.intensity_at_index(*i))
            .sum::<u64>();
        timers.outer_intensity_calculation.stop(false);
        neighbor_intensity_total >= self.min_intensity
    }

    fn main_loop_expand_cluster<PP, PE, DAI, QIP>(
        &self,
        apex_point_index: usize,
        neighbors: Vec<usize>,
        points: Arc<DBSCANPoints<'a, N, PP, PE, DAI, D, QIP>>,
        cluster_labels: &mut ClusterLabels,
        filter_fun_cache: &mut Option<FilterFunCache>,
        timers: &mut DBScanTimers,
    ) where
        PP: IntenseAtIndex + Sync + Send + ?Sized,
        PE: AsNDPointsAtIndex<N> + Send + Sync + ?Sized,
        DAI: DistantAtIndex<D> + Send + Sync + ?Sized,
        QIP: QueriableIndexedPoints<N> + std::marker::Sync,
    {
        cluster_labels.set_new_cluster(apex_point_index);
        let mut seed_set: Vec<usize> = neighbors;

        while let Some(neighbor_index) = seed_set.pop() {
            if !self.process_neighbor(neighbor_index, cluster_labels) {
                continue;
            }

            let local_neighbors =
                self.find_local_neighbors(neighbor_index, Arc::clone(&points), timers);
            let filtered_neighbors = self.filter_neighbors_inner_loop(
                local_neighbors,
                apex_point_index,
                neighbor_index,
                Arc::clone(&points),
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

    fn find_local_neighbors<PTS>(
        &self,
        neighbor_index: usize,
        points: Arc<PTS>,
        timers: &mut DBScanTimers,
    ) -> Vec<usize>
    where
        PTS: AsNDPointsAtIndex<N> + ?Sized + QueriableIndexedPoints<N> + std::marker::Sync + 'a,
    {
        timers.inner_loop_nn_timer.reset_start();
        let binding = Arc::clone(&points).get_ndpoint(neighbor_index);
        let local_neighbors: Vec<usize> = points.query_ndpoint(&binding).to_vec();
        // Should I warn if nothing is gotten here?
        // every point should have at least itself as a neighbor ...
        debug_assert!(!local_neighbors.is_empty());
        timers.inner_loop_nn_timer.stop(false);
        local_neighbors
    }

    fn filter_neighbors_inner_loop<PTS>(
        &self,
        local_neighbors: Vec<usize>,
        cluster_apex_point_index: usize,
        current_center_point_index: usize,
        points: Arc<PTS>,
        cluster_labels: &ClusterLabels,
        filter_fun_cache: &mut Option<FilterFunCache>,
        timers: &mut DBScanTimers,
    ) -> Vec<usize>
    where
        PTS:
            IntenseAtIndex + Send + AsNDPointsAtIndex<N> + DistantAtIndex<D> + Send + Sync + ?Sized,
    {
        let filtered = self.apply_filter_fun(
            local_neighbors,
            cluster_apex_point_index,
            Arc::clone(&points),
            filter_fun_cache,
        );

        if !self.is_extension_core_point(
            &filtered,
            current_center_point_index,
            Arc::clone(&points),
            timers,
        ) {
            return Vec::new();
        }

        let unassigned = self.filter_unassigned(filtered, cluster_labels);
        let unassigned_in_global_distance = self.filter_by_apex_distance(
            unassigned,
            cluster_apex_point_index,
            Arc::clone(&points),
            timers,
        );
        self.filter_by_local_intensity_and_distance(
            unassigned_in_global_distance,
            current_center_point_index,
            Arc::clone(&points),
            timers,
        )
    }

    fn filter_by_apex_distance<PTS>(
        &self,
        mut neighbors: Vec<usize>,
        apex_point_index: usize,
        points: Arc<PTS>,
        timers: &mut DBScanTimers,
    ) -> Vec<usize>
    where
        PTS: AsNDPointsAtIndex<N> + ?Sized,
    {
        timers.local_neighbor_filter_timer.reset_start();
        let query_point = &points.get_ndpoint(apex_point_index);
        neighbors.retain(|&i| self.is_within_max_distance(&points.get_ndpoint(i), query_point));
        timers.local_neighbor_filter_timer.stop(false);
        neighbors
    }

    fn is_extension_core_point<PTS>(
        &self,
        neighbors: &[usize],
        current_center_point_index: usize,
        points: Arc<PTS>,
        timers: &mut DBScanTimers,
    ) -> bool
    where
        PTS: IntenseAtIndex + Sync + Send + ?Sized,
    {
        timers.inner_intensity_calculation.reset_start();
        let mut neighbor_intensity_total: u64 = neighbors
            .iter()
            .map(|&i| points.intensity_at_index(i))
            .sum();

        neighbor_intensity_total += points.intensity_at_index(current_center_point_index);
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
    fn apply_filter_fun<PTS>(
        &self,
        local_neighbors: Vec<usize>,
        point_index: usize,
        points: Arc<PTS>,
        filter_fun_cache: &mut Option<FilterFunCache>,
    ) -> Vec<usize>
    where
        PTS: DistantAtIndex<D> + IntenseAtIndex + Sync + Send + ?Sized,
    {
        if let Some(cache) = filter_fun_cache {
            local_neighbors
                .into_iter()
                .filter(|&i| {
                    cache.get(i, point_index).unwrap_or_else(|| {
                        let res =
                            (self.filter_fun.unwrap())(&points.distance_at_indices(i, point_index));
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

    fn filter_by_local_intensity_and_distance<PTS>(
        &self,
        mut neighbors: Vec<usize>,
        neighbor_index: usize,
        points: Arc<PTS>,
        timers: &mut DBScanTimers,
    ) -> Vec<usize>
    where
        PTS: IntenseAtIndex + AsNDPointsAtIndex<N> + Sync + Send + ?Sized,
    {
        timers.local_neighbor_filter_timer.reset_start();
        let query_intensity = points.intensity_at_index(neighbor_index);
        let query_point = &points.get_ndpoint(neighbor_index);

        neighbors.retain(|&i| {
            let going_downhill = points.intensity_at_index(i) <= query_intensity;
            let within_distance = self.is_within_max_distance(&points.get_ndpoint(i), query_point);
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
    T: QueriableIndexedPoints<N> + Send + std::marker::Sync + std::fmt::Debug,
    PE: AsNDPointsAtIndex<N> + Send + Sync + ?Sized,
    D: Send + Sync,
    E: Send + Sync + Copy,
    FF: Fn(&D) -> bool + Send + Sync + ?Sized,
>(
    indexed_points: &'a T,
    raw_elements: &'a RE,
    projected_elements: &'a PE, // [NDPoint<N>],
    min_n: usize,
    min_intensity: u64,
    intensity_sorted_indices: Vec<(usize, u64)>,
    filter_fun: Option<&'a FF>,
    progress: bool,
    max_extension_distances: &'a [f32; N],
) -> ClusterLabels {
    let runner = DBSCANRunner {
        min_n,
        min_intensity,
        progress,
        filter_fun,
        max_extension_distances,
        _phantom: PhantomData::<D>,
    };

    runner.run(
        raw_elements,
        intensity_sorted_indices,
        indexed_points,
        projected_elements,
        raw_elements,
    )
}
