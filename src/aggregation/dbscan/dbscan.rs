use crate::aggregation::aggregators::{aggregate_clusters, ClusterAggregator, ClusterLabel};
use crate::space::kdtree::RadiusKDTree;
use crate::space::space_generics::{
    convert_to_bounds_query, DistantAtIndex, HasIntensity, IntenseAtIndex, NDPointConverter,
    QueriableIndexedPoints,
};
use crate::utils;
use log::{debug, info, trace};
use rayon::prelude::*;
use std::ops::Add;

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
        let mut query_elems = convert_to_bounds_query(&query_point);
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
    D: Send + Sync,
    const N: usize,
>(
    converter: C,
    prefiltered_peaks: Vec<T>,
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
    Vec<T>: IntenseAtIndex + DistantAtIndex<D>,
{
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
    let cluster_labels = dbscan_label_clusters(
        &tree,
        &prefiltered_peaks,
        ndpoints.as_slice(),
        min_n,
        min_intensity,
        &intensity_sorted_indices,
        extra_filter_fun,
        show_progress,
        max_extension_distances,
    );
    i_timer.stop(true);

    let centroids = aggregate_clusters(
        cluster_labels.num_clusters,
        cluster_labels.cluster_labels,
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
