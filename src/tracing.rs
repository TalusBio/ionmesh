use crate::ms::frames::DenseFrameWindow;
use crate::space_generics::TraceLike;

pub struct BaseTrace {
    pub mz: f64,
    pub intensity: f64,
    pub rt: f32,
    pub mobility: f32,
}

pub struct TimeTimsPeak {
    pub mz: f64,
    pub intensity: f64,
    pub rt: f32,
    pub ims: f32,
}

impl TraceLike<TimeTimsPeak, f32> for BaseTrace {
    fn get_mz(&self) -> f64 {
        self.mz
    }
    fn get_intensity(&self) -> f64 {
        self.intensity
    }
    fn get_rt(&self) -> f32 {
        self.rt
    }
    fn get_ims(&self) -> f32 {
        self.mobility
    }

    fn from_peaks(peaks: Vec<TimeTimsPeak>) -> Self {
        let mut mz = 0.;
        let mut intensity = 0.;
        let mut rt = 0.;
        let mut mobility = 0.;
        let num_peaks_f64 = peaks.len() as f64;

        for peak in peaks {
            mz += peak.mz;
            intensity += peak.intensity as f64;
            rt += peak.rt;
            mobility += peak.ims;
        }

        mz /= num_peaks_f64.clone();
        intensity /= num_peaks_f64.clone();
        rt /= num_peaks_f64 as f32;
        mobility /= num_peaks_f64 as f32;

        BaseTrace {
            mz,
            intensity,
            rt,
            mobility,
        }
    }
}

pub fn combine_traces<T: TraceLike<TimeTimsPeak, f32>>(
    denseframe_windows: Vec<DenseFrameWindow>,
) -> Vec<T> {
    // Grouping by quad windows + group id
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

    // Combine the traces

    Vec::new()
}

fn _combine_single_window_traces<T: TraceLike<TimeTimsPeak, f32>>(
    denseframe_windows: Vec<DenseFrameWindow>,
) -> Vec<T> {
    Vec::new()
}
