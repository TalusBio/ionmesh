use crate::dbscan::{dbscan_generic, ClusterAggregator};
use crate::mod_types::Float;
use crate::ms::frames::{DenseFrame, DenseFrameWindow, TimsPeak};
use crate::space_generics::{HasIntensity, NDPoint, NDPointConverter, TraceLike};
use crate::visualization::RerunPlottable;

use log::{error, info};
use rayon::iter::IntoParallelIterator;

#[derive(Debug, Clone)]
pub struct BaseTrace {
    pub mz: f64,
    pub intensity: u64,
    pub rt: f32,
    pub mobility: f32,
}

#[derive(Debug, Clone)]
pub struct TimeTimsPeak {
    pub mz: f64,
    pub intensity: u64,
    pub rt: f32,
    pub ims: f32,
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
}

pub fn combine_traces(
    denseframe_windows: Vec<DenseFrameWindow>,
    mz_scaling: f64,
    rt_scaling: f64,
    ims_scaling: f64,
    record_stream: &mut Option<rerun::RecordingStream>,
) -> Vec<BaseTrace> {
    // Grouping by quad windows + group id
    info!("Combining traces");
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
        .into_iter()
        .map(|x| _combine_single_window_traces(x, mz_scaling, rt_scaling, ims_scaling))
        .flatten()
        .collect();

    if let Some(stream) = record_stream.as_mut() {
        let _ = out.plot(stream, String::from("points/combined"), None, None);
    }
    info!("Total Combined traces: {}", out.len());
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

#[derive(Debug, Clone, Default)]
struct TraceAggregator {
    mz: f64,
    intensity: u64,
    rt: f64,
    ims: f64,
    num_peaks: usize,
}

impl ClusterAggregator<TimeTimsPeak, BaseTrace> for TraceAggregator {
    fn add(&mut self, peak: &TimeTimsPeak) {
        let f64_intensity = peak.intensity as f64;
        self.mz += peak.mz * f64_intensity;
        debug_assert!(peak.intensity < u64::MAX - self.intensity);
        self.intensity += peak.intensity;
        self.rt += (peak.rt as f64) * f64_intensity;
        self.ims += (peak.ims as f64) * f64_intensity;
        self.num_peaks += 1;
    }

    fn aggregate(&self) -> BaseTrace {
        let mz = self.mz / self.intensity as f64;
        let rt = self.rt / self.intensity as f64;
        let ims = self.ims / self.intensity as f64;

        BaseTrace {
            mz: mz,
            intensity: self.intensity.clone(),
            rt: rt as f32,
            mobility: ims as f32,
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
) -> Vec<BaseTrace> {
    info!("Prefiltered peaks: {}", prefiltered_peaks.len());
    let min_n = 3;
    let min_intensity = 200;
    let converter = TimeTimsPeakConverter {
        mz_scaling,
        rt_scaling,
        ims_scaling,
    };
    let foo: Vec<BaseTrace> =
        dbscan_generic(converter, prefiltered_peaks, min_n, min_intensity, &|| {
            TraceAggregator::default()
        });

    info!("Combined traces: {}", foo.len());
    foo
}
