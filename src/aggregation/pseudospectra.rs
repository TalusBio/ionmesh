use std::error::Error;
use std::io::Write;
use std::path::Path;

use log::info;
use rayon::prelude::*;
use serde::{
    Deserialize,
    Serialize,
};

use super::tracing::BaseTrace;
use crate::aggregation::aggregators::ClusterAggregator;
use crate::aggregation::dbscan::dbscan::{
    dbscan_aggregate,
    reassign_centroid,
};
use crate::aggregation::queriable_collections::queriable_traces::{
    BaseTraceDistance,
    TraceScalings,
};
use crate::aggregation::queriable_collections::QueriableTraces;
use crate::space::space_generics::{
    HasIntensity,
    NDPoint,
    NDPointConverter,
};
use crate::utils;
use crate::utils::RollingSDCalculator;

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

impl NDPointConverter<PseudoSpectrum, 2> for PseudoscanGenerationConfig {
    fn convert(
        &self,
        elem: &PseudoSpectrum,
    ) -> NDPoint<2> {
        // let quad_mid = (elem.quad_low + elem.quad_high) / 2.;
        NDPoint {
            values: [elem.rt, elem.ims],
        }
    }
}

pub fn combine_pseudospectra(
    traces: Vec<Vec<BaseTrace>>,
    config: PseudoscanGenerationConfig,
) -> Vec<PseudoSpectrum> {
    traces
        .into_iter()
        .flat_map(|x| combine_single_pseudospectra_window(x, config))
        .collect()
}

pub fn combine_single_pseudospectra_window(
    traces: Vec<BaseTrace>,
    config: PseudoscanGenerationConfig,
) -> Vec<PseudoSpectrum> {
    let mut timer =
        utils::ContextTimer::new("Combining pseudospectra??", true, utils::LogLevel::INFO);

    // let converter = BaseTraceConverter {
    //     rt_scaling: config.rt_scaling.into(),
    //     ims_scaling: config.ims_scaling.into(),
    //     quad_scaling: config.quad_scaling.into(),
    //     // rt_start_end_ratio: 2.,
    //     // peak_width_prior: 0.75,
    // };

    const IOU_THRESH: f32 = 0.3;
    const COSINE_THRESH: f32 = 0.8;
    let extra_filter_fun = |x: &BaseTraceDistance| {
        let close_in_quad = (x.quad_diff).abs() < 5.0;
        let within_iou_tolerance = x.iou > IOU_THRESH;
        let within_cosine_tolerance = x.cosine > COSINE_THRESH;

        close_in_quad && within_iou_tolerance && within_cosine_tolerance
    };

    let max_extension_distances: [f32; 2] = [
        config.max_rt_expansion_ratio * config.rt_scaling,
        config.max_ims_expansion_ratio * config.ims_scaling,
        // config.max_quad_expansion_ratio,
    ];

    let scalings = TraceScalings {
        rt_scaling: config.rt_scaling.into(),
        ims_scaling: config.ims_scaling.into(),
        quad_scaling: config.quad_scaling.into(),
    };

    let qtt = QueriableTraces::new(traces, scalings);
    let agg_timer = timer.start_sub_timer("aggregation");
    let mut agg1 = dbscan_aggregate(
        &qtt,
        &qtt,
        &qtt,
        agg_timer,
        config.min_n.into(),
        config.min_neighbor_intensity.into(),
        PseudoSpectrumAggregator::default,
        Some(&extra_filter_fun),
        utils::LogLevel::INFO,
        false,
        &max_extension_distances,
        true,
    );
    agg1.retain(|x| x.peaks.len() > 3);

    let reassign_max_distances = [config.rt_scaling, config.ims_scaling];

    let ranking_lambda = |p: &PseudoSpectrum, a: &BaseTrace, b: &BaseTrace| {
        let rt_diff = (p.rt - a.rt).abs() / config.rt_scaling;
        let ims_diff = (p.ims - a.mobility).abs() / config.ims_scaling;
        let quad_diff = (p.quad_low - a.quad_low).abs();
        let rt_diff_b = (p.rt - b.rt).abs() / config.rt_scaling;
        let ims_diff_b = (p.ims - b.mobility).abs() / config.ims_scaling;
        let quad_diff_b = (p.quad_low - b.quad_low).abs();
        let diff = rt_diff + ims_diff + quad_diff;
        let diff_b = rt_diff_b + ims_diff_b + quad_diff_b;

        diff.total_cmp(&diff_b)
    };
    let agg2 = reassign_centroid(
        agg1,
        &qtt,
        config,
        &qtt,
        PseudoSpectrumAggregator::default,
        utils::LogLevel::INFO,
        &reassign_max_distances,
        Some(300),
        Some(&ranking_lambda),
    );

    info!("Combined pseudospectra: {}", agg2.len());
    timer.stop(true);
    agg2
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
    file.write_all("[".as_bytes())?;
    let mut is_first = true;
    for x in pseudocscans {
        let json = serde_json::to_string(&x)?;
        if is_first {
            is_first = false;
        } else {
            file.write_all(",\n".as_bytes())?;
        }
        file.write_all(json.as_bytes())?;
    }
    file.write_all("]".as_bytes())?;

    Ok(())
}

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
