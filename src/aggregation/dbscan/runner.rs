use std::process::Output;

use crate::space::space_generics::NDPointConverter;
use crate::space::space_generics::{HasIntensity, NDPoint, QueriableIndexedPoints};
use crate::utils;
use crate::utils::within_distance_apply;
use indicatif::ProgressIterator;
use log::{debug, info, trace};

use rayon::prelude::*;

use crate::aggregation::aggregators::{
    aggregate_clusters, ClusterAggregator, ClusterLabel, TimsPeakAggregator,
};
use crate::space::kdtree::RadiusKDTree;

use crate::aggregation::dbscan::utils::FilterFunCache;

struct ClusterLabels {
    cluster_labels: Vec<ClusterLabel<u64>>,
    num_clusters: u64,
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

    fn report_if_gt_us(self, min_time: u128) {
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

struct DBSCANRunnerState<'a> {
    cluster_labels: ClusterLabels,
    filter_fun_cache: FilterFunCache<'a>,
    timers: DBScanTimers,
    candidate_metrics: CandidateCountMetrics,
}

impl DBSCANRunnerState<'_> {
    fn new<'a>(
        nlabels: usize,
        min_n: usize,
        usize_filterfun: &dyn Fn(&usize, &usize) -> bool,
    ) -> Self {
        let mut cluster_labels = ClusterLabels::new(nlabels);
        let filter_fun_cache = FilterFunCache::new(Box::new(&usize_filterfun), nlabels);
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
    filter_fun: &'a (dyn Fn(&E, &E) -> bool + Send + Sync),
    converter: C,
    progress: bool,
    max_extension_distances: &'a [f32; N],
    state: Option<DBSCANRunnerState<'a>>,
}

// C: NDPointConverter<T, N>,
// C2: NDPointConverter<R, N>,
// R: Send,
// G: Sync + Send + ClusterAggregator<T, R>,
// T: HasIntensity + Send + Clone + Copy + Sync,
// F: Fn() -> G + Send + Sync,
// const N: usize,
// FF: Send + Sync + Fn(&T, &T) -> bool,

impl<'a, const N: usize, C, E> DBSCANRunner<'a, N, C, E>
where
    C: NDPointConverter<E, N>,
    E: Sync + HasIntensity,
    //T: QueriableIndexedPoints<'a, N, usize> + std::marker::Sync,
{
    fn run(
        &self,
        prefiltered_peaks: &'a Vec<E>,
        intensity_sorted_indices: &'a Vec<(usize, f64)>,
    ) -> ClusterLabels {
        let usize_filterfun = |a: &usize, b: &usize| {
            (self.filter_fun)(&prefiltered_peaks[*a], &prefiltered_peaks[*b])
        };
        self.state = Some(DBSCANRunnerState::new(
            intensity_sorted_indices.len(),
            self.min_n,
            &usize_filterfun,
        ));

        let mut state = self.state.expect("State is created in this function.");
        // Q: if filter fun is required ... why is it an option?
        self.process_points(state, prefiltered_peaks, intensity_sorted_indices);

        state.timers.main.stop(false);
        state.timers.report_if_gt_us(1000000);
        state.cluster_labels
    }

    fn process_points(
        &self,
        mut state: DBSCANRunnerState<'a>,
        prefiltered_peaks: &'a Vec<E>,
        intensity_sorted_indices: &'a Vec<(usize, f64)>,
    ) {
        let my_progbar = state.create_progress_bar(intensity_sorted_indices.len(), self.progress);

        for (point_index, _intensity) in intensity_sorted_indices.iter().progress_with(my_progbar) {
            self.process_single_point(
                *point_index,
                prefiltered_peaks,
                &mut state.cluster_labels,
                &mut state.filter_fun_cache,
                &mut state.timers,
                &mut state.candidate_metrics,
            );
        }
    }

    fn process_single_point(
        &self,
        point_index: usize,
        prefiltered_peaks: &'a Vec<E>,
        cluster_labels: &mut ClusterLabels,
        filter_fun_cache: &mut FilterFunCache<'a>,
        timers: &mut DBScanTimers,
        cc_metrics: &mut CandidateCountMetrics,
    ) {
        if cluster_labels.get(point_index) != ClusterLabel::Unassigned {
            return;
        }

        let neighbors = self.find_neighbors(
            point_index,
            prefiltered_peaks,
            filter_fun_cache,
            timers,
            cc_metrics,
        );
        if !self.is_core_point(&neighbors, prefiltered_peaks, timers) {
            cluster_labels.set_noise(point_index);
            return;
        }

        self.expand_cluster(
            point_index,
            neighbors,
            prefiltered_peaks,
            cluster_labels,
            filter_fun_cache,
            timers,
        );
    }

    fn find_neighbors(
        &self,
        point_index: usize,
        prefiltered_peaks: &'a Vec<E>,
        filter_fun_cache: &mut FilterFunCache<'a>,
        timers: &mut DBScanTimers,
        cc_metrics: &mut CandidateCountMetrics,
    ) -> Vec<usize> {
        timers.outer_loop_nn_timer.reset_start();
        let query_elems = self
            .converter
            .convert_to_bounds_query(&quad_points[point_index]);
        let mut candidate_neighbors = self
            .indexed_points
            .query_ndrange(&query_elems.0, query_elems.1);
        timers.outer_loop_nn_timer.stop(false);

        let num_initial_candidates = candidate_neighbors.len();
        candidate_neighbors.retain(|i| filter_fun_cache(**i, point_index));

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

    fn is_core_point(
        &self,
        neighbors: &[usize],
        prefiltered_peaks: &'a Vec<E>,
        timers: &mut DBScanTimers,
    ) -> bool {
        timers.outer_intensity_calculation.reset_start();
        let neighbor_intensity_total = neighbors
            .iter()
            .map(|i| prefiltered_peaks[**i].intensity().as_())
            .sum::<u64>();
        timers.outer_intensity_calculation.stop(false);
        return neighbor_intensity_total >= self.min_intensity;
    }

    fn expand_cluster(
        &self,
        point_index: usize,
        mut neighbors: Vec<usize>,
        prefiltered_peaks: &'a Vec<E>,
        cluster_labels: &mut ClusterLabels,
        filter_fun_cache: &mut FilterFunCache<'a>,
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
            let inner_query_elems = converter.convert_to_bounds_query(&quad_points[*neighbor]);
            let mut local_neighbors =
                indexed_points.query_ndrange(&inner_query_elems.0, inner_query_elems.1);
            timers.inner_loop_nn_timer.stop(false);

            local_neighbors.retain(|i| filterfun_with_cache(**i, point_index));

            timers.inner_intensity_calculation.reset_start();
            let query_intensity = prefiltered_peaks[neighbor_index].intensity();
            let neighbor_intensity_total = local_neighbors
                .iter()
                .map(|i| prefiltered_peaks[**i].intensity().as_())
                .sum::<u64>();
            timers.inner_intensity_calculation.stop(false);

            if local_neighbors.len() >= min_n && neighbor_intensity_total >= min_intensity {
                local_neighbors
                    .retain(|i| !matches!(cluster_labels[**i], ClusterLabel::Cluster(_)));

                timers.local_neighbor_filter_timer.reset_start();
                local_neighbors.retain(|i| {
                    let going_downhill = prefiltered_peaks[**i].intensity() <= query_intensity;

                    let p = &quad_points[**i];
                    let query_point = query_elems.1.unwrap();
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

fn _dbscan<'a, const N: usize, C, I, E, T, FF>(
    indexed_points: &'a T,
    prefiltered_peaks: &'a Vec<E>,
    quad_points: &'a [NDPoint<N>],
    min_n: usize,
    min_intensity: u64,
    intensity_sorted_indices: &'a Vec<(usize, I)>,
    filter_fun: Option<FF>,
    converter: C,
    progress: bool,
    max_extension_distances: &'a [f32; N],
) -> (u64, Vec<ClusterLabel<u64>>) {
    let runner = DBSCANRunner::new(
        indexed_points,
        quad_points,
        min_n,
        min_intensity,
        filter_fun,
        converter,
        progress,
        max_extension_distances,
    );

    let mut cluster_labels = vec![ClusterLabel::Unassigned; prefiltered_peaks.len()];

    let cluster_id = runner.run(
        prefiltered_peaks,
        intensity_sorted_indices,
        &mut cluster_labels,
    );

    (cluster_id, cluster_labels)
}
