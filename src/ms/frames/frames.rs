use std::sync::Arc;

use serde::Serialize;
pub use timsrust::{
    AcquisitionType,
    MSLevel,
    QuadrupoleSettings,
};

use crate::space::space_generics::HasIntensity;

#[derive(Debug, Clone, Copy, Serialize)]
pub struct TimsPeak {
    pub intensity: u32,
    pub mz: f64,
    pub mobility: f32,
    pub npeaks: u32,
}

impl HasIntensity for TimsPeak {
    fn intensity(&self) -> u64 {
        self.intensity as u64
    }
}

#[derive(Debug, Clone, Copy)]
pub struct RawTimsPeak {
    pub intensity: u32,
    pub tof_index: u32,
    pub scan_index: usize,
}

#[derive(Debug, Clone, Copy)]
pub struct RawTimsPeakReference<'a> {
    pub intensity: &'a u32,
    pub tof_index: &'a u32,
    pub scan_index: &'a usize,
}

impl HasIntensity for RawTimsPeak {
    fn intensity(&self) -> u64 {
        self.intensity as u64
    }
}

impl<'a> HasIntensity for RawTimsPeakReference<'a> {
    fn intensity(&self) -> u64 {
        *self.intensity as u64
    }
}

#[derive(Debug, Clone, Copy)]
pub enum SortingOrder {
    None,
    Mz,
    Mobility,
    Intensity,
}

#[derive(Debug, Clone, Serialize)]
pub struct DenseFrame {
    pub raw_peaks: Vec<TimsPeak>,
    pub index: usize,
    pub rt: f64,
    pub window_group_id: u8,
    pub intensity_correction_factor: f64,

    #[serde(skip_serializing)]
    pub acquisition_type: AcquisitionType,

    #[serde(skip_serializing)]
    pub ms_level: MSLevel,

    #[serde(skip_serializing)]
    pub sorted: Option<SortingOrder>,
}

#[derive(Debug, Clone, Serialize, PartialEq)]
pub struct SingleQuadrupoleSettings {
    pub parent_index: usize,
    pub scan_start: usize,
    pub scan_end: usize,
    pub isolation_mz: f64,
    pub isolation_max: f64,
    pub isolation_min: f64,
    pub isolation_width: f64,
    pub collision_energy: f64,
}

impl SingleQuadrupoleSettings {
    pub fn from_quad_settings(
        quad_settings: Arc<QuadrupoleSettings>,
        index: usize,
    ) -> Self {
        let isolation_mz = quad_settings.isolation_mz[index];
        let isolation_width = quad_settings.isolation_width[index];
        let isolation_max = isolation_mz + (isolation_width / 2.);
        let isolation_min = isolation_mz - (isolation_width / 2.);
        let collision_energy = quad_settings.collision_energy[index];

        Self {
            parent_index: quad_settings.index,
            scan_start: quad_settings.scan_starts[index],
            scan_end: quad_settings.scan_ends[index],
            isolation_mz,
            isolation_max,
            isolation_min,
            isolation_width,
            collision_energy,
        }
    }
}

impl PartialOrd for SingleQuadrupoleSettings {
    fn partial_cmp(
        &self,
        other: &Self,
    ) -> Option<std::cmp::Ordering> {
        match self.parent_index.partial_cmp(&other.parent_index) {
            Some(std::cmp::Ordering::Equal) => self.scan_start.partial_cmp(&other.scan_start),
            x => x,
        }
    }
}
