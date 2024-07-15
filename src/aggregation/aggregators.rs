use crate::ms::frames::TimsPeak;
use crate::space::space_generics::HasIntensity;
use crate::utils;

use rayon::prelude::*;

// I Dont really like having this here but I am not sure where else to
// define it ... since its needed by the aggregation functions
#[derive(Debug, PartialEq, Clone, Copy)]
pub enum ClusterLabel<T> {
    Unassigned,
    Noise,
    Cluster(T),
}

/// A trait for aggregating points into a single point.
/// This is used for the final step of dbscan.
///
/// Types <T,R,S> are:
/// T: The type of the points to be aggregated.
/// R: The type of the aggregated point.
/// S: The type of the aggregator.
///
pub trait ClusterAggregator<T, R>: Send + Sync {
    fn add(
        &mut self,
        elem: &T,
    );
    fn aggregate(&self) -> R;
    fn combine(
        self,
        other: Self,
    ) -> Self;
}

#[derive(Default, Debug)]
pub struct TimsPeakAggregator {
    pub cluster_intensity: u64,
    pub cluster_mz: f64,
    pub cluster_mobility: f64,
    pub num_peaks: u64,
}

impl ClusterAggregator<TimsPeak, TimsPeak> for TimsPeakAggregator {
    fn add(
        &mut self,
        elem: &TimsPeak,
    ) {
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
        TimsPeak {
            intensity: self.cluster_intensity as u32,
            mz: cluster_mz,
            mobility: cluster_mobility as f32,
            npeaks: self.num_peaks as u32,
        }
    }

    fn combine(
        self,
        other: Self,
    ) -> Self {
        Self {
            cluster_intensity: self.cluster_intensity + other.cluster_intensity,
            cluster_mz: self.cluster_mz + other.cluster_mz,
            cluster_mobility: self.cluster_mobility + other.cluster_mobility,
            num_peaks: self.num_peaks + other.num_peaks,
        }
    }
}

pub fn aggregate_clusters<
    T: Send + Clone + Copy,
    G: Sync + Send + ClusterAggregator<T, R>,
    R: Send,
    F: Fn() -> G + Send + Sync,
>(
    tot_clusters: u64,
    cluster_labels: Vec<ClusterLabel<u64>>,
    elements: &[T],
    def_aggregator: &F,
    log_level: utils::LogLevel,
    keep_unclustered: bool,
) -> Vec<R> {
    let cluster_vecs: Vec<G> = if cfg!(feature = "par_dataprep") {
        parallel_aggregate_clusters(
            tot_clusters,
            cluster_labels,
            elements,
            def_aggregator,
            log_level,
            keep_unclustered,
        )
    } else {
        serial_aggregate_clusters(
            tot_clusters,
            cluster_labels,
            elements,
            def_aggregator,
            keep_unclustered,
        )
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

fn parallel_aggregate_clusters<
    T: Send + Clone + Copy,
    G: Sync + Send + ClusterAggregator<T, R>,
    R: Send,
    F: Fn() -> G + Send + Sync,
>(
    tot_clusters: u64,
    cluster_labels: Vec<ClusterLabel<u64>>,
    elements: &[T],
    def_aggregator: &F,
    log_level: utils::LogLevel,
    keep_unclustered: bool,
) -> Vec<G> {
    let mut timer = utils::ContextTimer::new("dbscan_generic::par_aggregation", true, log_level);
    let out: Vec<(usize, T)> = cluster_labels
        .iter()
        .enumerate()
        .filter_map(|(point_index, x)| match x {
            ClusterLabel::Cluster(cluster_id) => {
                let cluster_idx = *cluster_id as usize - 1;
                let tmp: Option<(usize, T)> = Some((cluster_idx, elements[point_index]));
                tmp
            },
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
                        },
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
}

fn serial_aggregate_clusters<
    T: Send + Clone + Copy,
    G: Sync + Send + ClusterAggregator<T, R>,
    R: Send,
    F: Fn() -> G + Send + Sync,
>(
    tot_clusters: u64,
    cluster_labels: Vec<ClusterLabel<u64>>,
    elements: &[T],
    def_aggregator: &F,
    keep_unclustered: bool,
) -> Vec<G> {
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
            },
            ClusterLabel::Noise => {
                if keep_unclustered {
                    let mut oe = def_aggregator();
                    oe.add(&elements[point_index]);
                    unclustered_points.push(oe);
                }
            },
            _ => {},
        }
    }
    cluster_vecs.extend(unclustered_points);
    cluster_vecs
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
