use crate::dbscan::{dbscan_generic, ClusterAggregator};
use crate::mod_types::Float;
use crate::ms::frames::{DenseFrame, DenseFrameWindow, TimsPeak};
use crate::space_generics::{HasIntensity, NDPoint, NDPointConverter, TraceLike};
use crate::utils;
use crate::utils::RollingSDCalculator;
use crate::visualization::RerunPlottable;

use log::{debug, error, info, warn};
use rayon::iter::IntoParallelIterator;
use rayon::prelude::*;
use rerun::Time;
use serde::Serialize;
use std::error::Error;
use std::io::Write;

type QuadLowHigh = (f64, f64);

#[derive(Debug, Clone, Copy, Serialize)]
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
}

pub fn write_trace_csv(traces: &Vec<BaseTrace>, path: &String) -> Result<(), Box<dyn Error>> {
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
    let (a_start, a_end) = a;
    let (b_start, b_end) = b;

    let intersection = if a_start > b_end || b_start > a_end {
        0.
    } else {
        let start = a_start.max(*b_start);
        let end = a_end.min(*b_end);
        end - start
    };

    let union = (a_end - a_start) + (b_end - b_start) - intersection;

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

        let giou_ab = iou(&a, &b);
        assert!(giou_ab > 0.33);
        assert!(giou_ab < 0.34);

        let giou_ac = iou(&a, &c);
        assert!(giou_ac == 0.);
    }
}

impl BaseTrace {
    pub fn rt_iou(&self, other: &BaseTrace) -> f32 {
        let a = (self.rt_start, self.rt_end);
        let b = (other.rt_start, other.rt_end);
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
    mz_scaling: f64,
    rt_scaling: f64,
    ims_scaling: f64,
    min_n: usize,
    min_intensity: u32,
    record_stream: &mut Option<rerun::RecordingStream>,
) -> Vec<BaseTrace> {
    // Grouping by quad windows + group id

    let timer = utils::ContextTimer::new("Tracing peaks in time", true, utils::LogLevel::INFO);

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
    let grouped_windows: Vec<Vec<DenseFrameWindow>> = grouped_windows
        .into_iter()
        .flatten()
        .filter_map(|x| x)
        .collect();

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
                mz_scaling,
                rt_scaling,
                ims_scaling,
                min_n,
                min_intensity,
            )
        })
        .flatten()
        .collect();

    info!("Total Combined traces: {}", out.len());
    timer.stop();

    if let Some(stream) = record_stream.as_mut() {
        let _ = out.plot(stream, String::from("points/combined"), None, None);
    }

    out
}

impl RerunPlottable<Option<usize>> for Vec<BaseTrace> {
    fn plot(
        &self,
        rec: &mut rerun::RecordingStream,
        entry_path: String,
        log_time_in_seconds: Option<f32>,
        required_extras: Option<usize>,
    ) -> Result<(), Box<dyn std::error::Error>> {
        // Sort by retention time and make groups of 1s
        let mut outs = Vec::new();
        let mut sorted_traces = (*self).clone();
        sorted_traces.sort_by(|a, b| a.rt.partial_cmp(&b.rt).unwrap());

        let mut groups: Vec<Vec<BaseTrace>> = Vec::new();

        let mut group: Vec<BaseTrace> = Vec::new();
        let mut last_second = sorted_traces[0].rt as u32;
        for trace in sorted_traces {
            let curr_second = trace.rt as u32;
            if curr_second != last_second {
                groups.push(group.clone());
                group = Vec::new();
            }
            last_second = curr_second;
            group.push(trace);
        }

        // For each group
        // Plot the group
        for group in groups {
            let mut peaks = Vec::new();
            for trace in group {
                peaks.push(TimsPeak {
                    mz: trace.mz,
                    intensity: trace.intensity.try_into().unwrap_or(u32::MAX),
                    mobility: trace.mobility,
                })
            }

            // Pack them into a denseframe
            let df = DenseFrame {
                raw_peaks: peaks,
                rt: last_second as f64,
                index: 555,
                frame_type: timsrust::FrameType::Unknown,
                sorted: None,
            };

            // Plot the denseframe
            let out = df.plot(
                rec,
                entry_path.clone(),
                log_time_in_seconds,
                required_extras,
            );
            if out.is_err() {
                error!("Error plotting pseudo-denseframe: {:?}", out);
            } else {
                info!("Plotted pseudo-denseframe");
            }
            outs.push(out);
        }

        Ok(())
    }
}

#[derive(Debug, Clone)]
struct TraceAggregator {
    mz: RollingSDCalculator<f64, u64>,
    intensity: u64,
    rt: RollingSDCalculator<f64, u64>,
    ims: RollingSDCalculator<f64, u64>,
    num_peaks: usize,
    quad_low_high: QuadLowHigh,
}

impl ClusterAggregator<TimeTimsPeak, BaseTrace> for TraceAggregator {
    fn add(&mut self, peak: &TimeTimsPeak) {
        let f64_intensity = peak.intensity as f64;
        self.mz.add(peak.mz, peak.intensity);
        debug_assert!(peak.intensity < u64::MAX - self.intensity);
        self.intensity += peak.intensity;
        self.rt.add(peak.rt as f64, peak.intensity);
        self.ims.add(peak.ims as f64, peak.intensity);
        self.num_peaks += 1;
    }

    fn aggregate(&self) -> BaseTrace {
        let mz = self.mz.get_mean() as f64;
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

        BaseTrace {
            mz: mz,
            mz_std: self.mz.get_sd() as f64,
            intensity: self.intensity.clone(),
            rt: rt,
            rt_std: self.rt.get_sd() as f32,
            rt_kurtosis: self.rt.get_kurtosis() as f32,
            rt_skew: self.rt.get_skew() as f32,
            rt_start: min_rt,
            rt_end: max_rt,
            mobility: ims,
            num_agg: self.num_peaks,
            quad_low: self.quad_low_high.0 as f32,
            quad_high: self.quad_low_high.1 as f32,
        }
    }

    fn combine(self, other: Self) -> Self {
        let mut mz = self.mz.clone();
        let mut rt = self.rt.clone();
        let mut ims = self.ims.clone();

        mz.merge(&other.mz);
        rt.merge(&other.rt);
        ims.merge(&other.ims);

        let out = TraceAggregator {
            mz: mz,
            intensity: self.intensity + other.intensity,
            rt: rt,
            ims: ims,
            num_peaks: self.num_peaks + other.num_peaks,
            quad_low_high: self.quad_low_high,
        };
        out
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
                (elem.mz / self.mz_scaling) as Float,
                (elem.rt as f64 / self.rt_scaling) as Float,
                (elem.ims as f64 / self.ims_scaling) as Float,
            ],
        }
    }
}

fn _flatten_denseframe_vec(denseframe_windows: Vec<DenseFrameWindow>) -> Vec<TimeTimsPeak> {
    denseframe_windows
        .into_iter()
        .map(|dfw| {
            let mut out = Vec::new();
            for peak in dfw.frame.raw_peaks {
                out.push(TimeTimsPeak {
                    mz: peak.mz,
                    intensity: peak.intensity as u64,
                    rt: dfw.frame.rt as f32,
                    ims: peak.mobility as f32,
                    quad_low_high: (dfw.mz_start, dfw.mz_end),
                });
            }
            out
        })
        .flatten()
        .collect::<Vec<_>>()
}

fn _combine_single_window_traces(
    prefiltered_peaks: Vec<TimeTimsPeak>,
    mz_scaling: f64,
    rt_scaling: f64,
    ims_scaling: f64,
    min_n: usize,
    min_intensity: u32,
) -> Vec<BaseTrace> {
    debug!("Prefiltered peaks: {}", prefiltered_peaks.len());
    let converter = TimeTimsPeakConverter {
        mz_scaling,
        rt_scaling,
        ims_scaling,
    };
    let window_quad_low_high = (
        prefiltered_peaks[0].quad_low_high.0,
        prefiltered_peaks[0].quad_low_high.1,
    );
    warn!("Assuming all quad windows are the same!!! (fine for diaPASEF)");

    let foo: Vec<BaseTrace> = dbscan_generic(
        converter,
        prefiltered_peaks,
        min_n,
        min_intensity.into(),
        &|| TraceAggregator {
            mz: RollingSDCalculator::default(),
            intensity: 0,
            rt: RollingSDCalculator::default(),
            ims: RollingSDCalculator::default(),
            num_peaks: 0,
            quad_low_high: window_quad_low_high.clone(),
        },
        None::<&Box<dyn Fn(&TimeTimsPeak, &TimeTimsPeak) -> bool>>,
    );

    debug!("Combined traces: {}", foo.len());
    foo
}

// NOW ... combine traces into pseudospectra

/// Peaks are mz-intensity pairs
type Peak = (f64, u64);

#[derive(Debug, Clone, Serialize)]
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
            rt: rt,
            ims: ims,
            rt_min: self.rt.get_min().unwrap() as f32,
            rt_max: self.rt.get_max().unwrap() as f32,
            rt_std: rt_std,
            ims_std: ims_std,
            rt_skew: rt_skew,
            ims_skew: ims_skew,
            quad_low: quad_low_high.0,
            quad_high: quad_low_high.1,
        }
    }

    fn combine(self, other: Self) -> Self {
        let mut peaks = self.peaks.clone();
        peaks.extend(other.peaks.clone());
        let mut rt = self.rt.clone();
        let mut ims = self.ims.clone();
        let mut quad_low = self.quad_low.clone();
        let mut quad_high = self.quad_high.clone();

        rt.merge(&other.rt);
        ims.merge(&other.ims);
        quad_low.merge(&other.quad_low);
        quad_high.merge(&other.quad_high);

        let out = PseudoSpectrumAggregator {
            peaks: peaks,
            intensity: self.intensity + other.intensity,
            rt: rt,
            ims: ims,
            quad_low: quad_low,
            quad_high: quad_high,
        };
        out
    }
}

struct BaseTraceConverter {
    rt_scaling: f64,
    ims_scaling: f64,
    quad_scaling: f64,
}

impl NDPointConverter<BaseTrace, 4> for BaseTraceConverter {
    fn convert(&self, elem: &BaseTrace) -> NDPoint<4> {
        NDPoint {
            values: [
                (elem.rt as f64 / self.rt_scaling) as Float,
                (elem.mobility as f64 / self.ims_scaling) as Float,
                (elem.quad_low as f64 / self.quad_scaling) as Float,
                (elem.quad_high as f64 / self.quad_scaling) as Float,
            ],
        }
    }
}

pub fn combine_pseudospectra(
    traces: Vec<BaseTrace>,
    rt_scaling: f64,
    ims_scaling: f64,
    quad_scaling: f64,
    min_intensity: u32,
    min_n: usize,
    record_stream: &mut Option<rerun::RecordingStream>,
) -> Vec<PseudoSpectrum> {
    let timer = utils::ContextTimer::new("Combining pseudospectra", true, utils::LogLevel::INFO);

    let converter = BaseTraceConverter {
        rt_scaling,
        ims_scaling,
        quad_scaling,
    };

    const IOU_THRESH: f32 = 0.5;
    let extra_filter_fun = |x: &BaseTrace, y: &BaseTrace| {
        let iou = x.rt_iou(y);
        iou > IOU_THRESH
    };
    let foo: Vec<PseudoSpectrum> = dbscan_generic(
        converter,
        traces,
        min_n,
        min_intensity.into(),
        &|| PseudoSpectrumAggregator::default(),
        Some(&extra_filter_fun),
    );

    info!("Combined pseudospectra: {}", foo.len());
    timer.stop();

    if let Some(stream) = record_stream.as_mut() {
        warn!("Plotting pseudospectra is not implemented yet");
        // let _ = foo.plot(stream, String::from("points/pseudospectra"), None, None);
    }

    foo
}

pub fn write_pseudoscans_json(
    pseudocscans: &[PseudoSpectrum],
    out_path: String,
) -> Result<(), Box<dyn Error>> {
    info!("Writting pseudoscans to json");
    let json = serde_json::to_string(&pseudocscans)?;

    let mut file = std::fs::File::create(out_path)?;
    file.write(json.as_bytes())?;
    Ok(())
}
