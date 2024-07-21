use serde::Serialize;
pub use timsrust::Frame;
pub use timsrust::FrameType;
pub use timsrust::{
    ConvertableIndex, FileReader, Frame2RtConverter, Scan2ImConverter, Tof2MzConverter,
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

    #[serde(skip_serializing)]
    pub frame_type: FrameType,

    #[serde(skip_serializing)]
    pub sorted: Option<SortingOrder>,
}

/// Information on the context of a window in a frame.
///
/// This adds to a frame slice the context of the what isolation was used
/// to generate the frame slice.
#[derive(Debug, Clone, Serialize)]
pub struct FrameMsMsWindowInfo {
    pub mz_start: f32,
    pub mz_end: f32,
    pub window_group_id: usize,
    pub within_window_quad_group_id: usize,
    pub global_quad_row_id: usize,
}
