use crate::aggregation::aggregators::{aggregate_clusters, ClusterAggregator, ClusterLabel};
use crate::space::kdtree::RadiusKDTree;
use crate::space::space_generics::{
    convert_to_bounds_query, AsAggregableAtIndex, AsNDPointsAtIndex, DistantAtIndex, HasIntensity,
    IntenseAtIndex, NDPoint, NDPointConverter, QueriableIndexedPoints,
};
use crate::utils::{self, ContextTimer};
use log::{debug, info, trace};
use rayon::prelude::*;
use std::ops::{Add, Index};

use crate::aggregation::dbscan::runner::dbscan_label_clusters;

// Pretty simple function ... it uses every passed centroid, converts it to a point
// and generates a new centroid that aggregates all the points in its range.
// In contrast with the dbscan method, the elements in each cluster are not necessarily
// mutually exclusive.
fn reassign_centroid<
    'a,
    const N: usize,
    T: Send + Clone + Copy,
    C: NDPointConverter<R, N>,
    I: QueriableIndexedPoints<'a, N> + std::marker::Sync,
    G: Sync + Send + ClusterAggregator<T, R>,
    R: Send,
    RE: Send + Sync + AsAggregableAtIndex<T> + ?Sized,
    F: Fn() -> G + Send + Sync,
>(
    centroids: Vec<R>,
    indexed_points: &'a I,
    centroid_converter: C,
    elements: &RE,
    def_aggregator: F,
    log_level: utils::LogLevel,
    expansion_factors: &[f32; N],
) -> Vec<R> {
    let mut timer = utils::ContextTimer::new("reassign_centroid", true, log_level);
    let mut out = Vec::with_capacity(centroids.len());

    for centroid in centroids {
        let query_point = centroid_converter.convert(&centroid);
        let mut query_elems = convert_to_bounds_query(&query_point);
        query_elems.0.expand(expansion_factors);

        // trace!("Querying for Centroid: {:?}", query_elems.1);
        // trace!("Querying for Boundary: {:?}", query_elems.0);
        let neighbors = indexed_points.query_ndrange(&query_elems.0, query_elems.1);
        // trace!("Found {} neighbors", neighbors.len());
        let mut aggregator = def_aggregator();
        let mut num_agg = 0;
        for neighbor in neighbors {
            aggregator.add(&elements.get_aggregable_at_index(neighbor));
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

impl<const N: usize> AsNDPointsAtIndex<N> for Vec<NDPoint<N>> {
    fn get_ndpoint(
        &self,
        index: usize,
    ) -> NDPoint<N> {
        self[index]
    }

    fn num_ndpoints(&self) -> usize {
        self.len()
    }
}

pub fn dbscan_generic<
    C: NDPointConverter<T, N>,
    C2: NDPointConverter<R, N>,
    R: Send,
    G: Sync + Send + ClusterAggregator<T, R>,
    T: HasIntensity + Send + Clone + Copy + Sync,
    RE: IntenseAtIndex
        + DistantAtIndex<D>
        + IntoIterator<Item = T>
        + Send
        + Sync
        + AsAggregableAtIndex<T>
        + std::fmt::Debug
        + ?Sized,
    F: Fn() -> G + Send + Sync,
    D: Send + Sync,
    const N: usize,
>(
    converter: C,
    prefiltered_peaks: &RE,
    min_n: usize,
    min_intensity: u64,
    def_aggregator: F,
    extra_filter_fun: Option<&(dyn Fn(&D) -> bool + Send + Sync)>,
    log_level: Option<utils::LogLevel>,
    keep_unclustered: bool,
    max_extension_distances: &[f32; N],
    back_converter: Option<C2>,
) -> Vec<R>
where
    <RE as IntoIterator>::IntoIter: ExactSizeIterator,
{
    let show_progress = log_level.is_some();
    let log_level = match log_level {
        Some(x) => x,
        None => utils::LogLevel::TRACE,
    };

    let timer = utils::ContextTimer::new("dbscan_generic", true, log_level);
    let mut i_timer = timer.start_sub_timer("conversion");
    let (ndpoints, boundary) = converter.convert_aggregables(prefiltered_peaks);
    i_timer.stop(true);

    let mut i_timer = timer.start_sub_timer("tree");
    let mut tree = RadiusKDTree::new_empty(boundary, 500, 1.);
    let quad_indices = (0..ndpoints.len()).collect::<Vec<_>>();

    for (quad_point, i) in ndpoints.iter().zip(quad_indices.iter()) {
        tree.insert_ndpoint(quad_point.clone(), i);
    }
    i_timer.stop(true);

    let centroids = dbscan_aggregate(
        prefiltered_peaks,
        &ndpoints,
        &tree,
        timer,
        min_n,
        min_intensity,
        &def_aggregator,
        extra_filter_fun,
        log_level,
        keep_unclustered,
        max_extension_distances,
        show_progress,
    );

    match back_converter {
        Some(bc) => reassign_centroid(
            centroids,
            &tree,
            bc,
            prefiltered_peaks,
            &def_aggregator,
            log_level,
            max_extension_distances,
        ),
        None => centroids,
    }
}

pub fn dbscan_aggregate<
    'a,
    const N: usize,
    RE: IntenseAtIndex
        + DistantAtIndex<D>
        + AsAggregableAtIndex<T>
        + Send
        + Sync
        + std::fmt::Debug
        + ?Sized,
    IND: QueriableIndexedPoints<'a, N> + std::marker::Sync + Send + std::fmt::Debug,
    NAI: AsNDPointsAtIndex<N> + std::marker::Sync + Send,
    T: HasIntensity + Send + Clone + Copy + Sync,
    D: Send + Sync,
    G: Sync + Send + ClusterAggregator<T, R>,
    R: Send,
    F: Fn() -> G + Send + Sync,
>(
    prefiltered_peaks: &'a RE,
    ndpoints: &'a NAI,
    index: &'a IND,
    timer: ContextTimer,
    min_n: usize,
    min_intensity: u64,
    def_aggregator: F,
    extra_filter_fun: Option<&'a (dyn Fn(&D) -> bool + Send + Sync)>,
    log_level: utils::LogLevel,
    keep_unclustered: bool,
    max_extension_distances: &'a [f32; N],
    show_progress: bool,
) -> Vec<R> {
    let mut i_timer = timer.start_sub_timer("pre-sort");
    let intensity_sorted_indices = prefiltered_peaks.intensity_sorted_indices();

    i_timer.stop(true);

    let mut i_timer = timer.start_sub_timer("dbscan");
    let cluster_labels = dbscan_label_clusters(
        index,
        prefiltered_peaks,
        ndpoints,
        min_n,
        min_intensity,
        intensity_sorted_indices,
        extra_filter_fun,
        show_progress,
        max_extension_distances,
    );
    i_timer.stop(true);

    let centroids = aggregate_clusters(
        cluster_labels.num_clusters,
        cluster_labels.cluster_labels,
        prefiltered_peaks,
        &def_aggregator,
        log_level,
        keep_unclustered,
    );
    centroids
}
