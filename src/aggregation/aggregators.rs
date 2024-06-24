
use crate::ms::frames::TimsPeak;

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
pub struct TimsPeakAggregator {
    pub cluster_intensity: u64,
    pub cluster_mz: f64,
    pub cluster_mobility: f64,
    pub num_peaks: u64,
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
        TimsPeak {
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
