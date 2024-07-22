use crate::aggregation::aggregators::{aggregate_clusters, ClusterAggregator};
use crate::aggregation::chromatograms::{
    BTreeChromatogram, ChromatogramArray, NUM_LOCAL_CHROMATOGRAM_BINS,
};
use crate::aggregation::dbscan::dbscan::dbscan_generic;
use crate::aggregation::dbscan::runner::dbscan_label_clusters;
use crate::ms::frames::DenseFrameWindow;
use crate::space::space_generics::{
    AsAggregableAtIndex, AsNDPointsAtIndex, DistantAtIndex, HasIntensity, NDPoint,
    NDPointConverter, QueriableIndexedPoints, TraceLike,
};
use crate::space::space_generics::{IntenseAtIndex, NDBoundary};
use crate::utils;
use crate::utils::{binary_search_slice, RollingSDCalculator};

use core::panic;
use log::{debug, error, info, warn};
use rayon::iter::IntoParallelIterator;
use rayon::prelude::*;
use serde::ser::{SerializeStruct, Serializer};
use serde::{Deserialize, Serialize};
use std::error::Error;
use std::io::Write;
use std::path::Path;

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

pub fn combine_traces(
    grouped_denseframe_windows: Vec<Vec<DenseFrameWindow>>,
    config: TracingConfig,
    rt_binsize: f32,
) -> Vec<BaseTrace> {
    // mz_scaling: f64,
    // rt_scaling: f64,
    // ims_scaling: f64,
    // min_n: usize,
    // min_intensity: u32,
    // Grouping by quad windows + group id

    let mut timer = utils::ContextTimer::new("Tracing peaks in time", true, utils::LogLevel::INFO);

    let grouped_windows: Vec<Vec<TimeTimsPeak>> = grouped_denseframe_windows
        .into_iter()
        .map(_flatten_denseframe_vec)
        .collect();

    let combine_lambda = |x: Vec<TimeTimsPeak>| {
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
            rt_binsize,
        )
    };

    // Combine the traces
    let out: Vec<BaseTrace> = if cfg!(feature = "less_parallel") {
        warn!("Running in single-threaded mode");
        grouped_windows
            .into_iter()
            .map(combine_lambda)
            .flatten()
            .collect()
    } else {
        grouped_windows
            .into_par_iter()
            .map(combine_lambda)
            .flatten()
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

#[derive(Debug, Default)]
struct TimeTimsPeakConverter {
    // Takes  DenseFrameWindow
    mz_scaling: f64,
    rt_scaling: f64,
    ims_scaling: f64,
}

impl NDPointConverter<TimeTimsPeak, 3> for TimeTimsPeakConverter {
    fn convert(
        &self,
        elem: &TimeTimsPeak,
    ) -> NDPoint<3> {
        NDPoint {
            values: [
                (elem.mz / self.mz_scaling) as f32,
                (elem.rt as f64 / self.rt_scaling) as f32,
                (elem.ims as f64 / self.ims_scaling) as f32,
            ],
        }
    }
}

struct BypassBaseTraceBackConverter {}

impl NDPointConverter<BaseTrace, 3> for BypassBaseTraceBackConverter {
    fn convert(
        &self,
        _elem: &BaseTrace,
    ) -> NDPoint<3> {
        panic!("This should never be called");
    }
}

fn _flatten_denseframe_vec(denseframe_windows: Vec<DenseFrameWindow>) -> Vec<TimeTimsPeak> {
    denseframe_windows
        .into_iter()
        .flat_map(|dfw| {
            let mut out = Vec::new();
            for peak in dfw.frame.raw_peaks {
                out.push(TimeTimsPeak {
                    mz: peak.mz,
                    intensity: peak.intensity as u64,
                    rt: dfw.frame.rt as f32,
                    ims: peak.mobility,
                    quad_low_high: (dfw.mz_start, dfw.mz_end),
                    n_peaks: 1,
                });
            }
            out
        })
        .collect::<Vec<_>>()
}

impl IntenseAtIndex for Vec<TimeTimsPeak> {
    fn intensity_at_index(
        &self,
        index: usize,
    ) -> u64 {
        self[index].intensity
    }

    fn intensity_index_length(&self) -> usize {
        self.len()
    }
}

impl AsAggregableAtIndex<TimeTimsPeak> for Vec<TimeTimsPeak> {
    fn get_aggregable_at_index(
        &self,
        index: usize,
    ) -> TimeTimsPeak {
        self[index]
    }

    fn num_aggregable(&self) -> usize {
        self.len()
    }
}

impl DistantAtIndex<f32> for Vec<TimeTimsPeak> {
    fn distance_at_indices(
        &self,
        index: usize,
        other: usize,
    ) -> f32 {
        panic!("I dont think this is called ever ...");
    }
}

// Needed to specify the generic in dbscan_generic
type FFTimeTimsPeak = fn(&TimeTimsPeak, &TimeTimsPeak) -> bool;

#[derive(Debug)]
struct TimeTimsPeakScaling {
    mz_scaling: f32,
    rt_scaling: f32,
    ims_scaling: f32,
    quad_scaling: f32,
}

#[derive(Debug)]
struct QueriableTimeTimsPeaks {
    peaks: Vec<TimeTimsPeak>,
    min_bucket_mz_vals: Vec<f32>,
    bucket_size: usize,
    scalings: TimeTimsPeakScaling,
}

impl QueriableTimeTimsPeaks {
    fn new(
        mut peaks: Vec<TimeTimsPeak>,
        scalings: TimeTimsPeakScaling,
    ) -> Self {
        const BUCKET_SIZE: usize = 16384;
        // // Sort all of our theoretical fragments by m/z, from low to high
        peaks.par_sort_unstable_by(|a, b| a.mz.partial_cmp(&b.mz).unwrap());

        let mut min_bucket_mz_vals = peaks
            .par_chunks_mut(BUCKET_SIZE)
            .map(|bucket| {
                let min = bucket[0].mz;
                bucket.par_sort_unstable_by(|a, b| a.rt.partial_cmp(&b.rt).unwrap());
                min as f32
            })
            .collect::<Vec<_>>();

        // Get the max value of the last bucket
        let max_bucket_mz = peaks[peaks.len().saturating_sub(BUCKET_SIZE)..peaks.len()]
            .iter()
            .max_by(|a, b| a.mz.partial_cmp(&b.mz).unwrap())
            .unwrap()
            .mz as f32;
        min_bucket_mz_vals.push(max_bucket_mz);

        QueriableTimeTimsPeaks {
            peaks,
            min_bucket_mz_vals,
            bucket_size: BUCKET_SIZE,
            scalings,
        }
    }

    fn get_bucket_at(
        &self,
        index: usize,
    ) -> Result<&[TimeTimsPeak], ()> {
        let page_start = index * self.bucket_size;
        if page_start >= self.peaks.len() {
            return Err(());
        }
        let page_end = (page_start + self.bucket_size).min(self.peaks.len());
        let tmp = &self.peaks[page_start..page_end];

        if cfg!(debug_assertions) {
            // Make sure all rts are sorted within the bucket
            for i in 1..tmp.len() {
                if tmp[i - 1].rt > tmp[i].rt {
                    panic!("RTs are not sorted within the bucket");
                }
            }
        }
        Ok(tmp)
    }

    fn get_intensity_sorted_indices(&self) -> Vec<(usize, u64)> {
        let mut indices: Vec<(usize, u64)> = (0..self.peaks.len())
            .map(|i| (i, self.peaks[i].intensity))
            .collect();
        indices.par_sort_unstable_by_key(|&x| x.1);

        debug_assert!(indices.len() == self.peaks.len());
        if cfg!(debug_assertions) {
            if indices.len() > 1 {
                for i in 1..indices.len() {
                    if indices[i - 1].1 > indices[i].1 {
                        panic!("Indices are not sorted");
                    }
                }
            }
        }
        indices
    }
}

impl AsNDPointsAtIndex<3> for QueriableTimeTimsPeaks {
    fn get_ndpoint(
        &self,
        index: usize,
    ) -> NDPoint<3> {
        NDPoint {
            values: [
                self.peaks[index].mz as f32,
                self.peaks[index].rt,
                self.peaks[index].ims,
            ],
        }
    }

    fn num_ndpoints(&self) -> usize {
        self.peaks.len()
    }
}

impl IntenseAtIndex for QueriableTimeTimsPeaks {
    fn intensity_at_index(
        &self,
        index: usize,
    ) -> u64 {
        self.peaks[index].intensity
    }

    fn intensity_index_length(&self) -> usize {
        self.peaks.len()
    }
}

impl AsAggregableAtIndex<TimeTimsPeak> for QueriableTimeTimsPeaks {
    fn get_aggregable_at_index(
        &self,
        index: usize,
    ) -> TimeTimsPeak {
        self.peaks[index]
    }

    fn num_aggregable(&self) -> usize {
        self.peaks.len()
    }
}

impl DistantAtIndex<f32> for QueriableTimeTimsPeaks {
    fn distance_at_indices(
        &self,
        index: usize,
        other: usize,
    ) -> f32 {
        let a = self.peaks[index];
        let b = self.peaks[other];
        let mz = (a.mz - b.mz) as f32 / self.scalings.mz_scaling;
        let rt = (a.rt - b.rt) as f32 / self.scalings.rt_scaling;
        let ims = (a.ims - b.ims) as f32 / self.scalings.ims_scaling;
        (mz * mz + rt * rt + ims * ims).sqrt()
    }
}

impl QueriableIndexedPoints<3> for QueriableTimeTimsPeaks {
    fn query_ndpoint(
        &self,
        point: &NDPoint<3>,
    ) -> Vec<usize> {
        let boundary = NDBoundary::new(
            [
                (point.values[0] - self.scalings.mz_scaling) - f32::EPSILON,
                (point.values[1] - self.scalings.rt_scaling),
                (point.values[2] - self.scalings.ims_scaling) - f32::EPSILON,
            ],
            [
                (point.values[0] + self.scalings.mz_scaling) + f32::EPSILON,
                (point.values[1] + self.scalings.rt_scaling),
                (point.values[2] + self.scalings.ims_scaling) + f32::EPSILON,
            ],
        );
        let out = self.query_ndrange(&boundary, None);
        out
    }

    fn query_ndrange(
        &self,
        boundary: &NDBoundary<3>,
        reference_point: Option<&NDPoint<3>>,
    ) -> Vec<usize> {
        let mut out = Vec::new();
        let mz_range = (boundary.starts[0], boundary.ends[0]);
        let mz_range_f64 = (boundary.starts[0] as f64, boundary.ends[0] as f64);
        let rt_range = (boundary.starts[1], boundary.ends[1]);
        let ims_range = (boundary.starts[2], boundary.ends[2]);

        let (bstart, bend) = binary_search_slice(
            &self.min_bucket_mz_vals,
            |a, b| a.total_cmp(b),
            mz_range.0,
            mz_range.1,
        );

        let bstart = bstart.saturating_sub(1);
        let bend_new = bend.saturating_add(1).min(self.min_bucket_mz_vals.len());

        for bnum in bstart..bend_new {
            let c_bucket = self.get_bucket_at(bnum);
            if c_bucket.is_err() {
                continue;
            }
            let c_bucket = c_bucket.unwrap();
            let page_start = bnum * self.bucket_size;

            let (istart, iend) =
                binary_search_slice(c_bucket, |a, b| a.rt.total_cmp(&b), rt_range.0, rt_range.1);

            for (j, peak) in self.peaks[(istart + page_start)..(iend + page_start)]
                .iter()
                .enumerate()
            {
                debug_assert!(
                    peak.rt >= rt_range.0 && peak.rt <= rt_range.1,
                    "RT out of range -> {} {} {}; istart {}, page_starrt {}, j {}; window rts: {:?}",
                    peak.rt,
                    rt_range.0,
                    rt_range.1,
                    istart,
                    page_start,
                    j,
                    &self.peaks[(j + istart + page_start).saturating_sub(5)
                        ..(j + istart + page_start + 5).min(self.peaks.len())]
                        .iter()
                        .map(|x| x.rt)
                        .collect::<Vec<f32>>()
                );
                if peak.ims >= ims_range.0 && peak.ims <= ims_range.1 {
                    if peak.mz as f32 >= mz_range.0 && peak.mz as f32 <= mz_range.1 {
                        out.push(j + istart + page_start);
                    }
                }
            }
        }

        out
    }
}

// QueriableIndexedPoints<N>

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

    let centroids = aggregate_clusters(
        cluster_labels.num_clusters,
        cluster_labels.cluster_labels,
        &index,
        &|| TraceAggregator {
            mz: RollingSDCalculator::default(),
            intensity: 0,
            rt: RollingSDCalculator::default(),
            ims: RollingSDCalculator::default(),
            num_peaks: 0,
            num_rt_peaks: 0,
            quad_low_high: window_quad_low_high,
            btree_chromatogram: BTreeChromatogram::new_lazy(rt_binsize),
        },
        utils::LogLevel::TRACE,
        false,
    );

    debug!("Combined traces: {}", centroids.len());
    centroids
}

// TODO maybe this can be a builder-> executor pattern
fn combine_single_window_traces(
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
    info!("Peaks in window: {}", prefiltered_peaks.len());
    let converter: TimeTimsPeakConverter = TimeTimsPeakConverter {
        mz_scaling,
        rt_scaling,
        ims_scaling,
    };
    let window_quad_low_high = (
        prefiltered_peaks[0].quad_low_high.0,
        prefiltered_peaks[0].quad_low_high.1,
    );
    let max_extension_distances: [f32; 3] = [
        max_mz_expansion_ratio,
        max_rt_expansion_ratio,
        max_ims_expansion_ratio,
    ];
    warn!("Assuming all quad windows are the same!!! (fine for diaPASEF)");

    // TODO make dbscan_generic a runner-class
    let out_traces: Vec<BaseTrace> = dbscan_generic(
        converter,
        &prefiltered_peaks,
        min_n,
        min_intensity.into(),
        || TraceAggregator {
            mz: RollingSDCalculator::default(),
            intensity: 0,
            rt: RollingSDCalculator::default(),
            ims: RollingSDCalculator::default(),
            num_peaks: 0,
            num_rt_peaks: 0,
            quad_low_high: window_quad_low_high,
            btree_chromatogram: BTreeChromatogram::new_lazy(rt_binsize),
        },
        None::<&(dyn Fn(&f32) -> bool + Send + Sync)>,
        None,
        false,
        &max_extension_distances,
        None::<BypassBaseTraceBackConverter>,
    );

    debug!("Combined traces: {}", out_traces.len());
    out_traces
}

// NOW ... combine traces into pseudospectra

/// Peaks are mz-intensity pairs
type Peak = (f64, u64);

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PseudoSpectrum {
    pub peaks: Vec<Peak>,
    pub rt: f32,
    pub rt_min: f32,
    pub rt_max: f32,
    pub rt_std: f32,
    pub rt_skew: f32,
    pub ims: f32,
    pub ims_std: f32,
    pub ims_skew: f32,
    pub quad_low: f32,
    pub quad_high: f32,
}

#[derive(Debug)]
pub struct PseudoSpectrumAggregator {
    peaks: Vec<Peak>,
    intensity: u64,
    rt: RollingSDCalculator<f64, u64>,
    ims: RollingSDCalculator<f64, u64>,
    quad_low: RollingSDCalculator<f32, u64>,
    quad_high: RollingSDCalculator<f32, u64>,
}

impl Default for PseudoSpectrumAggregator {
    fn default() -> Self {
        let nv = Vec::new();
        PseudoSpectrumAggregator {
            peaks: nv,
            intensity: 0,
            rt: RollingSDCalculator::default(),
            ims: RollingSDCalculator::default(),
            // I am adding here because in the future I want to support
            // the weird pasef modes.
            quad_low: RollingSDCalculator::default(),
            quad_high: RollingSDCalculator::default(),
        }
    }
}

impl<'a> ClusterAggregator<BaseTrace, PseudoSpectrum> for PseudoSpectrumAggregator {
    fn add(
        &mut self,
        peak: &BaseTrace,
    ) {
        debug_assert!(peak.intensity < u64::MAX - self.intensity);

        self.rt.add(peak.rt as f64, peak.intensity);
        self.ims.add(peak.mobility as f64, peak.intensity);
        self.quad_low.add(peak.quad_low, peak.intensity);
        self.quad_high.add(peak.quad_high, peak.intensity);
        self.peaks.push((peak.mz, peak.intensity));
    }

    fn aggregate(&self) -> PseudoSpectrum {
        // TECHNICALLY this can error out if there are no elements...
        let rt = self.rt.get_mean() as f32;
        let ims = self.ims.get_mean() as f32;
        let rt_skew = self.rt.get_skew() as f32;
        let ims_skew = self.ims.get_skew() as f32;
        let rt_std = self.rt.get_sd() as f32;
        let ims_std = self.ims.get_sd() as f32;
        let quad_low_high = (self.quad_low.get_mean(), self.quad_high.get_mean());

        PseudoSpectrum {
            peaks: self.peaks.clone(),
            rt,
            ims,
            rt_min: self.rt.get_min().unwrap() as f32,
            rt_max: self.rt.get_max().unwrap() as f32,
            rt_std,
            ims_std,
            rt_skew,
            ims_skew,
            quad_low: quad_low_high.0,
            quad_high: quad_low_high.1,
        }
    }

    fn combine(
        self,
        other: Self,
    ) -> Self {
        let mut peaks = self.peaks.clone();
        peaks.extend(other.peaks.clone());
        let mut rt = self.rt;
        let mut ims = self.ims;
        let mut quad_low = self.quad_low;
        let mut quad_high = self.quad_high;

        rt.merge(&other.rt);
        ims.merge(&other.ims);
        quad_low.merge(&other.quad_low);
        quad_high.merge(&other.quad_high);

        PseudoSpectrumAggregator {
            peaks,
            intensity: self.intensity + other.intensity,
            rt,
            ims,
            quad_low,
            quad_high,
        }
    }
}

struct BaseTraceConverter {
    rt_scaling: f64,
    ims_scaling: f64,
    quad_scaling: f64,
}

impl NDPointConverter<BaseTrace, 3> for BaseTraceConverter {
    fn convert(
        &self,
        elem: &BaseTrace,
    ) -> NDPoint<3> {
        // let rt_start_use = (elem.rt - elem.rt_std).min(elem.rt - self.peak_width_prior as f32);
        // let rt_end_use = (elem.rt + elem.rt_std).max(elem.rt + self.peak_width_prior as f32);
        // let rt_start_end_scaling = self.rt_scaling * self.rt_start_end_ratio;
        let quad_center = (elem.quad_low + elem.quad_high) / 2.;
        NDPoint {
            values: [
                (elem.rt as f64 / self.rt_scaling) as f32,
                (elem.mobility as f64 / self.ims_scaling) as f32,
                (quad_center as f64 / self.quad_scaling) as f32,
            ],
        }
    }
}

struct PseudoScanBackConverter {
    rt_scaling: f64,
    ims_scaling: f64,
    quad_scaling: f64,
}

impl NDPointConverter<PseudoSpectrum, 3> for PseudoScanBackConverter {
    fn convert(
        &self,
        elem: &PseudoSpectrum,
    ) -> NDPoint<3> {
        let quad_mid = (elem.quad_low + elem.quad_high) / 2.;
        NDPoint {
            values: [
                (elem.rt as f64 / self.rt_scaling) as f32,
                (elem.ims as f64 / self.ims_scaling) as f32,
                (quad_mid as f64 / self.quad_scaling) as f32,
            ],
        }
    }
}

#[derive(Debug, Serialize, Deserialize, Clone, Copy)]
pub struct PseudoscanGenerationConfig {
    pub rt_scaling: f32,
    pub quad_scaling: f32,
    pub ims_scaling: f32,
    pub max_rt_expansion_ratio: f32,
    pub max_quad_expansion_ratio: f32,
    pub max_ims_expansion_ratio: f32,
    pub min_n: u8,
    pub min_neighbor_intensity: u32,
}

impl Default for PseudoscanGenerationConfig {
    fn default() -> Self {
        PseudoscanGenerationConfig {
            rt_scaling: 2.4,
            quad_scaling: 5.,
            ims_scaling: 0.015,
            max_rt_expansion_ratio: 5.,
            max_quad_expansion_ratio: 1.,
            max_ims_expansion_ratio: 2.,
            min_n: 6,
            min_neighbor_intensity: 6000,
        }
    }
}

impl IntenseAtIndex for Vec<BaseTrace> {
    fn intensity_at_index(
        &self,
        index: usize,
    ) -> u64 {
        self[index].intensity
    }

    fn intensity_index_length(&self) -> usize {
        self.len()
    }
}

impl AsAggregableAtIndex<BaseTrace> for Vec<BaseTrace> {
    fn get_aggregable_at_index(
        &self,
        index: usize,
    ) -> BaseTrace {
        self[index]
    }

    fn num_aggregable(&self) -> usize {
        self.len()
    }
}

struct BaseTraceDistance {
    quad_diff: f32,
    iou: f32,
    cosine: f32,
}

impl DistantAtIndex<BaseTraceDistance> for Vec<BaseTrace> {
    fn distance_at_indices(
        &self,
        index: usize,
        other: usize,
    ) -> BaseTraceDistance {
        let quad_diff = (self[index].quad_center - self[other].quad_center).abs();
        let iou = self[index].rt_iou(&self[other]);
        // Q: What can cause an error here??
        let cosine = self[index]
            .chromatogram
            .cosine_similarity(&self[other].chromatogram)
            .unwrap();
        BaseTraceDistance {
            quad_diff,
            iou,
            cosine,
        }
    }
}

pub fn combine_pseudospectra(
    traces: Vec<BaseTrace>,
    config: PseudoscanGenerationConfig,
) -> Vec<PseudoSpectrum> {
    let mut timer =
        utils::ContextTimer::new("Combining pseudospectra", true, utils::LogLevel::INFO);

    let converter = BaseTraceConverter {
        rt_scaling: config.rt_scaling.into(),
        ims_scaling: config.ims_scaling.into(),
        quad_scaling: config.quad_scaling.into(),
        // rt_start_end_ratio: 2.,
        // peak_width_prior: 0.75,
    };

    const IOU_THRESH: f32 = 0.1;
    const COSINE_THRESH: f32 = 0.8;
    let extra_filter_fun = |x: &BaseTraceDistance| {
        let close_in_quad = (x.quad_diff).abs() < 5.0;
        let within_iou_tolerance = x.iou > IOU_THRESH;
        let within_cosine_tolerance = x.cosine > COSINE_THRESH;

        return close_in_quad && within_iou_tolerance && within_cosine_tolerance;
    };

    let back_converter = PseudoScanBackConverter {
        rt_scaling: config.rt_scaling.into(),
        ims_scaling: config.ims_scaling.into(),
        quad_scaling: config.quad_scaling.into(),
    };
    let max_extension_distances: [f32; 3] = [
        config.max_rt_expansion_ratio,
        config.max_ims_expansion_ratio,
        config.max_quad_expansion_ratio,
    ];

    let foo: Vec<PseudoSpectrum> = dbscan_generic(
        converter,
        &traces,
        config.min_n.into(),
        config.min_neighbor_intensity.into(),
        PseudoSpectrumAggregator::default,
        Some(&extra_filter_fun),
        Some(utils::LogLevel::INFO),
        false,
        &max_extension_distances,
        Some(back_converter),
    );

    info!("Combined pseudospectra: {}", foo.len());
    timer.stop(true);
    foo
}

pub fn write_pseudoscans_json(
    pseudocscans: &[PseudoSpectrum],
    out_path: impl AsRef<Path>,
) -> Result<(), Box<dyn Error>> {
    info!(
        "Writting pseudoscans to json: {}",
        out_path.as_ref().display()
    );
    let mut file = std::fs::File::create(out_path)?;
    file.write("[".as_bytes())?;
    let mut is_first = true;
    for x in pseudocscans {
        let json = serde_json::to_string(&x)?;
        if is_first {
            is_first = false;
        } else {
            file.write(",\n".as_bytes())?;
        }
        file.write(json.as_bytes())?;
    }
    file.write("]".as_bytes())?;

    Ok(())
}

// pub fn read_pseudoscans_json(
//     in_path: impl AsRef<Path>,
// ) -> Result<Vec<PseudoSpectrum>, Box<dyn Error>> {
//     info!("Reading pseudoscans from json {}", in_path.as_ref().display());
//     let file = std::fs::File::open(in_path)?;
//     let reader = std::io::BufReader::new(file);
//     let out: Vec<PseudoSpectrum> = serde_json::from_reader(reader)?;
//     Ok(out)
// }
