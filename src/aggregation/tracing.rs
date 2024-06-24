use crate::aggregation::dbscan::dbscan_generic;
use crate::aggregation::aggregators::ClusterAggregator;
use crate::ms::frames::DenseFrameWindow;
use crate::space::space_generics::{HasIntensity, NDPoint, NDPointConverter, TraceLike};
use crate::utils;
use crate::utils::RollingSDCalculator;
use crate::space::space_generics::NDBoundary;
use crate::aggregation::chromatograms::{BTreeChromatogram, ChromatogramArray, NUM_LOCAL_CHROMATOGRAM_BINS};

use log::{debug, error, info, warn};
use rayon::iter::IntoParallelIterator;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use serde::ser::{Serializer, SerializeStruct};
use core::panic;
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
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
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

pub fn write_trace_csv(traces: &Vec<BaseTrace>, path: impl AsRef<Path>) -> Result<(), Box<dyn Error>> {
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

impl HasIntensity<u32> for TimeTimsPeak {
    fn intensity(&self) -> u32 {
        let o = self.intensity.try_into();
        match o {
            Ok(x) => x,
            Err(_) => {
                error!("Intensity overflowed u32");
                u32::MAX
            }
        }
    }
}

pub fn iou(a: &(f32, f32), b: &(f32, f32)) -> f32 {
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
    pub fn rt_iou(&self, other: &BaseTrace) -> f32 {
        // TODO change this to be the measured peak width ...
        let width_a = self.rt_std.max(0.7);
        let width_b: f32 = other.rt_std.max(0.7);
        let a = (self.rt - width_a, self.rt + width_a);
        let b = (other.rt - width_b, other.rt + width_b);
        iou(&a, &b)
    }
}

impl HasIntensity<u64> for BaseTrace {
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
    denseframe_windows: Vec<DenseFrameWindow>,
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

    let mut grouped_windows: Vec<Vec<Option<Vec<DenseFrameWindow>>>> = Vec::new();
    for dfw in denseframe_windows {
        let dia_group = dfw.group_id;
        let quad_group = dfw.quad_group_id;

        while grouped_windows.len() <= dia_group {
            grouped_windows.push(Vec::new());
        }

        while grouped_windows[dia_group].len() <= quad_group {
            grouped_windows[dia_group].push(None);
        }

        if grouped_windows[dia_group][quad_group].is_none() {
            grouped_windows[dia_group][quad_group] = Some(Vec::new());
        } else {
            grouped_windows[dia_group][quad_group]
                .as_mut()
                .unwrap()
                .push(dfw);
        }
    }

    // Flatten one level
    let grouped_windows: Vec<Vec<DenseFrameWindow>> =
        grouped_windows.into_iter().flatten().flatten().collect();

    let grouped_windows: Vec<Vec<TimeTimsPeak>> = grouped_windows
        .into_iter()
        .map(_flatten_denseframe_vec)
        .collect();

    // Combine the traces
    let out: Vec<BaseTrace> = grouped_windows
        .into_par_iter()
        .map(|x| {
            _combine_single_window_traces(
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
        })
        .flatten()
        .collect();

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
    fn add(&mut self, peak: &TimeTimsPeak) {
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
        let chromatogram: ChromatogramArray<f32, NUM_LOCAL_CHROMATOGRAM_BINS> = self.btree_chromatogram.as_chromatogram_array(Some(rt));

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

    fn combine(self, other: Self) -> Self {
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
    fn convert(&self, elem: &TimeTimsPeak) -> NDPoint<3> {
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
    fn convert(&self, _elem: &BaseTrace) -> NDPoint<3> {
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

// Needed to specify the generic in dbscan_generic
type FFTimeTimsPeak = fn(&TimeTimsPeak, &TimeTimsPeak) -> bool;


// TODO maybe this can be a builder-> executor pattern
fn _combine_single_window_traces(
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
    debug!("Prefiltered peaks: {}", prefiltered_peaks.len());
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
        prefiltered_peaks,
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
        None::<&FFTimeTimsPeak>,
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
    fn add(&mut self, peak: &BaseTrace) {
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

    fn combine(self, other: Self) -> Self {
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
    fn convert(&self, elem: &BaseTrace) -> NDPoint<3> {
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

    fn convert_to_bounds_query<'a>(
        &self,
        point: &'a NDPoint<3>,
    ) -> (
        NDBoundary<3>,
        Option<&'a NDPoint<3>>,
    ) {
        const NUM_DIMENTIONS: usize = 3;
        // let range_center = (point.values[1] + point.values[2]) / 2.;
        let mut starts = point.values;
        let mut ends = point.values;
        for i in 0..NUM_DIMENTIONS {
            starts[i] -= 1.;
            ends[i] += 1.;
        }

        // // KEY =                 [-------]
        // // Allowed ends =            [------]
        // // Allowed starts =  [------]

        // ends[1] = range_center;
        // starts[2] = range_center;

        let bounds = NDBoundary::new(starts, ends);
        (bounds, Some(point))
    }
}

struct PseudoScanBackConverter {
    rt_scaling: f64,
    ims_scaling: f64,
    quad_scaling: f64,
}

impl NDPointConverter<PseudoSpectrum, 3> for PseudoScanBackConverter {
    fn convert(&self, elem: &PseudoSpectrum) -> NDPoint<3> {
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
    let extra_filter_fun = |x: &BaseTrace, y: &BaseTrace| {
        let close_in_quad = (x.quad_center - y.quad_center).abs() < 5.0;
        if !close_in_quad {
            return false;
        }

        let iou = x.rt_iou(y);
        let within_iou_tolerance = iou > IOU_THRESH;

        let cosine = x.chromatogram.cosine_similarity(&y.chromatogram).unwrap();
        let within_cosine_tolerance = cosine > COSINE_THRESH;

        within_iou_tolerance && within_cosine_tolerance
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
        traces,
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
