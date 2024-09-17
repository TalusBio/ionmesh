use std::error::Error;
use std::io::Write;
use std::path::Path;

use log::{
    debug,
    info,
    warn,
};
use rayon::iter::IntoParallelIterator;
use rayon::prelude::*;
use serde::ser::{
    SerializeStruct,
    Serializer,
};
use serde::{
    Deserialize,
    Serialize,
};

use crate::aggregation::aggregators::{
    aggregate_clusters,
    ClusterAggregator,
};
use crate::aggregation::chromatograms::{
    BTreeChromatogram,
    ChromatogramArray,
    NUM_LOCAL_CHROMATOGRAM_BINS,
};
use crate::aggregation::dbscan::runner::dbscan_label_clusters;
use crate::aggregation::queriable_collections::queriable_indexed_points::{
    QueriableTimeTimsPeaks,
    TimeTimsPeakScaling,
};
use crate::ms::frames::DenseFrameWindow;
use crate::space::space_generics::{
    HasIntensity,
    TraceLike,
};
use crate::utils;
use crate::utils::RollingSDCalculator;

type QuadLowHigh = (f64, f64);

#[derive(Debug, Serialize, Deserialize, Clone, Copy)]
pub struct TracingConfig {
    pub mz_scaling: f32,
    pub rt_scaling: f32,
    pub ims_scaling: f32,
    pub max_mz_expansion_ratio: f32,
    pub max_rt_expansion_ratio: f32,
    pub max_ims_expansion_ratio: f32,
    pub min_n: u8,
    pub min_neighbor_intensity: u32,
}

impl Default for TracingConfig {
    fn default() -> Self {
        TracingConfig {
            mz_scaling: 0.015,
            rt_scaling: 2.4,
            ims_scaling: 0.02,
            max_mz_expansion_ratio: 1.,
            max_rt_expansion_ratio: 1.5,
            max_ims_expansion_ratio: 4.,
            min_n: 3,
            min_neighbor_intensity: 450,
        }
    }
}

// Serialize
#[derive(Debug, Clone, Copy)]
pub struct BaseTrace {
    pub mz: f64,
    pub mz_std: f64,
    pub intensity: u64,
    pub rt: f32,
    pub rt_std: f32,
    pub rt_kurtosis: f32,
    pub rt_skew: f32,
    pub rt_start: f32,
    pub rt_end: f32,
    pub mobility: f32,
    pub num_agg: usize,
    pub quad_low: f32,
    pub quad_high: f32,
    pub quad_center: f32,
    pub chromatogram: ChromatogramArray<f32, NUM_LOCAL_CHROMATOGRAM_BINS>,
    pub num_rt_points: usize,
    pub num_tot_points: usize,
}

impl Serialize for BaseTrace {
    fn serialize<S>(
        &self,
        serializer: S,
    ) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let chromatogram = self.chromatogram.chromatogram.to_vec();
        let mut state = serializer.serialize_struct("BaseTrace", 15)?;
        state.serialize_field("mz", &self.mz)?;
        state.serialize_field("mz_std", &self.mz_std)?;
        state.serialize_field("intensity", &self.intensity)?;
        state.serialize_field("rt", &self.rt)?;
        state.serialize_field("rt_std", &self.rt_std)?;
        state.serialize_field("rt_kurtosis", &self.rt_kurtosis)?;
        state.serialize_field("rt_skew", &self.rt_skew)?;
        state.serialize_field("rt_start", &self.rt_start)?;
        state.serialize_field("rt_end", &self.rt_end)?;
        state.serialize_field("mobility", &self.mobility)?;
        state.serialize_field("num_agg", &self.num_agg)?;
        state.serialize_field("num_tot_points", &self.num_tot_points)?;
        state.serialize_field("quad_low", &self.quad_low)?;
        state.serialize_field("quad_high", &self.quad_high)?;
        state.serialize_field("quad_center", &self.quad_center)?;
        state.serialize_field("chromatogram", &format!("{:?}", chromatogram))?;
        state.end()
    }
}

pub fn write_trace_csv(
    traces: &Vec<BaseTrace>,
    path: impl AsRef<Path>,
) -> Result<(), Box<dyn Error>> {
    let mut wtr = csv::Writer::from_path(path).unwrap();
    for trace in traces {
        wtr.serialize(trace)?;
    }
    let _ = wtr.flush();
    Ok(())
}

#[derive(Debug, Clone, Copy)]
pub struct TimeTimsPeak {
    pub mz: f64,
    pub intensity: u64,
    pub rt: f32,
    pub ims: f32,
    pub quad_low_high: QuadLowHigh,
    pub n_peaks: u32,
}

impl HasIntensity for TimeTimsPeak {
    fn intensity(&self) -> u64 {
        self.intensity
    }
}

pub fn iou(
    a: &(f32, f32),
    b: &(f32, f32),
) -> f32 {
    let min_ends = a.1.min(b.1);
    let max_starts = a.0.max(b.0);

    let max_ends = a.1.max(b.1);
    let min_starts = a.0.min(b.0);

    let intersection = min_ends - max_starts;
    let union = max_ends - min_starts;

    intersection / union
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_giou() {
        let a = (0., 10.);
        let b = (5., 15.);
        let c = (10., 15.);
        let d = (15., 20.);

        let giou_ab = iou(&a, &b);
        assert!(giou_ab > 0.33);
        assert!(giou_ab < 0.34);

        let giou_ac = iou(&a, &c);
        assert!(giou_ac == 0.);

        let giou_ad = iou(&a, &d);
        assert_eq!(giou_ad, -0.25);
    }
}

impl BaseTrace {
    pub fn rt_iou(
        &self,
        other: &BaseTrace,
    ) -> f32 {
        // TODO change this to be the measured peak width ...
        let width_a = self.rt_std.max(0.7);
        let width_b: f32 = other.rt_std.max(0.7);
        let a = (self.rt - width_a, self.rt + width_a);
        let b = (other.rt - width_b, other.rt + width_b);
        iou(&a, &b)
    }
}

impl HasIntensity for BaseTrace {
    fn intensity(&self) -> u64 {
        self.intensity
    }
}

// TODO consider if this trait is actually requried ...
impl TraceLike<f32> for BaseTrace {
    fn get_mz(&self) -> f64 {
        self.mz
    }
    fn get_intensity(&self) -> u64 {
        self.intensity
    }
    fn get_rt(&self) -> f32 {
        self.rt
    }
    fn get_ims(&self) -> f32 {
        self.mobility
    }
    fn get_quad_low_high(&self) -> QuadLowHigh {
        (self.mz, self.mz)
    }
}

pub fn calculate_cycle_time(frames: &[DenseFrameWindow]) -> f64 {
    let rts = frames.iter().map(|x| x.frame.rt).collect::<Vec<f64>>();
    let rt_diffs = rts.windows(2).map(|x| x[1] - x[0]).collect::<Vec<f64>>();
    let cycle_time = rt_diffs.iter().sum::<f64>() / rt_diffs.len() as f64;
    cycle_time
}

pub fn combine_traces(
    grouped_denseframe_windows: Vec<Vec<DenseFrameWindow>>,
    config: TracingConfig,
) -> Vec<Vec<BaseTrace>> {
    // mz_scaling: f64,
    // rt_scaling: f64,
    // ims_scaling: f64,
    // min_n: usize,
    // min_intensity: u32,
    // Grouping by quad windows + group id

    let mut timer = utils::ContextTimer::new("Tracing peaks in time", true, utils::LogLevel::INFO);

    // rt_binsize: f32,

    let grouped_windows: Vec<(f64, Vec<TimeTimsPeak>)> = grouped_denseframe_windows
        .into_iter()
        .map(|x| {
            let cycle_time = calculate_cycle_time(&x);
            let o = _flatten_denseframe_vec(x);
            (cycle_time, o)
        })
        .collect();

    let combine_lambda = |cycle_time: f64, x: Vec<TimeTimsPeak>| {
        combine_single_window_traces2(
            x,
            config.mz_scaling.into(),
            config.max_mz_expansion_ratio,
            config.rt_scaling.into(),
            config.max_rt_expansion_ratio,
            config.ims_scaling.into(),
            config.max_ims_expansion_ratio,
            config.min_n.into(),
            config.min_neighbor_intensity,
            cycle_time as f32,
        )
    };

    // Combine the traces
    let out: Vec<Vec<BaseTrace>> = if cfg!(feature = "less_parallel") {
        warn!("Running in single-threaded mode");
        grouped_windows
            .into_iter()
            .map(|x| combine_lambda(x.0, x.1))
            .collect()
    } else {
        grouped_windows
            .into_par_iter()
            .map(|x| combine_lambda(x.0, x.1))
            .collect()
    };

    info!("Total Combined traces: {}", out.len());
    timer.stop(true);

    out
}

#[derive(Debug, Clone)]
struct TraceAggregator {
    mz: RollingSDCalculator<f64, u64>,
    intensity: u64,
    rt: RollingSDCalculator<f64, u64>,
    ims: RollingSDCalculator<f64, u64>,
    num_rt_peaks: usize,
    num_peaks: usize,
    quad_low_high: QuadLowHigh,
    btree_chromatogram: BTreeChromatogram,
}

impl ClusterAggregator<TimeTimsPeak, BaseTrace> for TraceAggregator {
    fn add(
        &mut self,
        peak: &TimeTimsPeak,
    ) {
        let _f64_intensity = peak.intensity as f64;
        self.mz.add(peak.mz, peak.intensity);
        debug_assert!(peak.intensity < u64::MAX - self.intensity);
        self.intensity += peak.intensity;
        self.rt.add(peak.rt as f64, peak.intensity);
        self.ims.add(peak.ims as f64, peak.intensity);
        self.btree_chromatogram.add(peak.rt, peak.intensity);
        self.num_rt_peaks += 1;
        self.num_peaks += peak.n_peaks as usize;
    }

    fn aggregate(&self) -> BaseTrace {
        let mz = self.mz.get_mean();
        let rt = self.rt.get_mean() as f32;
        let ims = self.ims.get_mean() as f32;
        let min_rt = match self.rt.get_min() {
            Some(x) => x as f32,
            None => rt,
        };
        let max_rt = match self.rt.get_max() {
            Some(x) => x as f32,
            None => rt,
        };

        //  The chromatogram is an array centered on the retention time
        let num_rt_points = self.btree_chromatogram.btree.len();
        let chromatogram: ChromatogramArray<f32, NUM_LOCAL_CHROMATOGRAM_BINS> =
            self.btree_chromatogram.as_chromatogram_array(Some(rt));

        // let apex = chromatogram.chromatogram.iter().enumerate().max_by_key(|x| (x.1 * 100.) as i32).unwrap().0;
        // let apex_offset = (apex as f32 - (NUM_LOCAL_CHROMATOGRAM_BINS as f32 / 2.)) * self.btree_chromatogram.rt_binsize;

        BaseTrace {
            mz,
            mz_std: self.mz.get_sd(),
            intensity: self.intensity,
            rt,
            rt_std: self.rt.get_sd() as f32,
            rt_kurtosis: self.rt.get_kurtosis() as f32,
            rt_skew: self.rt.get_skew() as f32,
            rt_start: min_rt,
            rt_end: max_rt,
            mobility: ims,
            num_agg: self.num_peaks,
            quad_low: self.quad_low_high.0 as f32,
            quad_high: self.quad_low_high.1 as f32,
            quad_center: (self.quad_low_high.0 + self.quad_low_high.1) as f32 / 2.,
            chromatogram,
            num_rt_points,
            num_tot_points: self.num_peaks,
        }
    }

    fn combine(
        self,
        other: Self,
    ) -> Self {
        let mut mz = self.mz;
        let mut rt = self.rt;
        let mut ims = self.ims;
        let mut btree_chromatogram = self.btree_chromatogram;

        mz.merge(&other.mz);
        rt.merge(&other.rt);
        ims.merge(&other.ims);
        btree_chromatogram.adopt(&other.btree_chromatogram);

        TraceAggregator {
            mz,
            intensity: self.intensity + other.intensity,
            rt,
            ims,
            num_peaks: self.num_peaks + other.num_peaks,
            num_rt_peaks: self.num_rt_peaks + other.num_rt_peaks,
            quad_low_high: self.quad_low_high,
            btree_chromatogram,
        }
    }
}

fn _flatten_denseframe_vec(denseframe_windows: Vec<DenseFrameWindow>) -> Vec<TimeTimsPeak> {
    denseframe_windows
        .into_iter()
        .flat_map(|dfw| {
            let mut out = Vec::new();
            for peak in dfw.frame.raw_peaks {
                let mz_start = dfw.quadrupole_setting.isolation_mz
                    - (dfw.quadrupole_setting.isolation_width / 2.);
                let mz_end = dfw.quadrupole_setting.isolation_mz
                    + (dfw.quadrupole_setting.isolation_width / 2.);
                out.push(TimeTimsPeak {
                    mz: peak.mz,
                    intensity: peak.intensity as u64,
                    rt: dfw.frame.rt as f32,
                    ims: peak.mobility,
                    quad_low_high: (mz_start, mz_end),
                    n_peaks: 1,
                });
            }
            out
        })
        .collect::<Vec<_>>()
}

fn combine_single_window_traces2(
    prefiltered_peaks: Vec<TimeTimsPeak>,
    mz_scaling: f64,
    max_mz_expansion_ratio: f32,
    rt_scaling: f64,
    max_rt_expansion_ratio: f32,
    ims_scaling: f64,
    max_ims_expansion_ratio: f32,
    min_n: usize,
    min_intensity: u32,
    rt_binsize: f32,
) -> Vec<BaseTrace> {
    let timer = utils::ContextTimer::new("dbscan_wt2", true, utils::LogLevel::DEBUG);
    info!("Peaks in window: {}", prefiltered_peaks.len());
    let scalings = TimeTimsPeakScaling {
        mz_scaling: mz_scaling as f32,
        rt_scaling: rt_scaling as f32,
        ims_scaling: ims_scaling as f32,
        quad_scaling: 1.,
    };
    let window_quad_low_high = (
        prefiltered_peaks[0].quad_low_high.0,
        prefiltered_peaks[0].quad_low_high.1,
    );
    let index = QueriableTimeTimsPeaks::new(prefiltered_peaks, scalings);
    let intensity_sorted_indices = index.get_intensity_sorted_indices();
    let max_extension_distances: [f32; 3] = [
        max_mz_expansion_ratio * mz_scaling as f32,
        max_rt_expansion_ratio * rt_scaling as f32,
        max_ims_expansion_ratio * ims_scaling as f32,
    ];

    let mut i_timer = timer.start_sub_timer("dbscan");
    let cluster_labels = dbscan_label_clusters(
        &index,
        &index,
        &index,
        min_n,
        min_intensity.into(),
        intensity_sorted_indices,
        None::<&(dyn Fn(&f32) -> bool + Send + Sync)>,
        true,
        &max_extension_distances,
    );

    i_timer.stop(true);

    let def_aggregator = || TraceAggregator {
        mz: RollingSDCalculator::default(),
        intensity: 0,
        rt: RollingSDCalculator::default(),
        ims: RollingSDCalculator::default(),
        num_peaks: 0,
        num_rt_peaks: 0,
        quad_low_high: window_quad_low_high,
        btree_chromatogram: BTreeChromatogram::new_lazy(rt_binsize),
    };

    let centroids = aggregate_clusters(
        cluster_labels.num_clusters,
        cluster_labels.cluster_labels,
        &index,
        &def_aggregator,
        utils::LogLevel::TRACE,
        false,
    );

    debug!("Combined traces: {}", centroids.len());
    centroids
}
