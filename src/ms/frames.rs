pub use timsrust::Frame;
pub use timsrust::FrameType;
pub use timsrust::{
    ConvertableIndex, FileReader, Frame2RtConverter, Scan2ImConverter, Tof2MzConverter,
};

use crate::mod_types::Float;
use crate::space_generics::NDPoint;
use crate::visualization::RerunPlottable;

#[derive(Debug, Clone, Copy)]
pub struct TimsPeak {
    pub intensity: u32,
    pub mz: f64,
    pub mobility: f32,
}

#[derive(Debug, Clone, Copy)]
pub struct RawTimsPeak {
    pub intensity: u32,
    pub tof_index: u32,
    pub scan_index: usize,
}

#[derive(Debug, Clone, Copy)]
pub enum SortingOrder {
    Mz,
    Mobility,
    Intensity,
}

/// Unprocessed data from a 'Frame'.
///
/// 1. every tof-index + intensity represents a peak.
/// 2. Scan offsets are monotonically increasing.
/// 3. Peaks are arranged in increasing m/z order WITHIN a scan.
/// 4. Getting the peaks for scan #x in the frame is done by subsetting
///    the tof indices and intensities.
///     - scan_1_intensities = intensities[scan_offsets[1]:scan_offsets[2]]
///     - scan_x_intensities = intensities[scan_offsets[x]:scan_offsets[x+1]]
/// 5. The m/z values are a function of the tof indices (the measured m/z
///    of tof index `x` will be the same within the same run/instrument
///    calibration)
///
/// Frame                  Example values
///    - Scan offsets.    [0,0,0,0,0,3,5,6 ...] n=number of scans
///    - tof indices.     [100, 101, 102, 10, 20, 30 ...] len = len(intensities)
///    - intensities.     [123, 111, 12 ,  3,  4,  1 ...] len = len(tof indices)
///    - index            34
///    - rt               65.34
#[derive(Debug, Clone)]
pub struct FrameWindow {
    /// A vector of length (s) where contiguous elements represent
    ///
    pub scan_offsets: Vec<u64>,
    pub tof_indices: Vec<u32>,
    pub intensities: Vec<u32>,
    pub index: usize,
    pub rt: f64,
    pub frame_type: FrameType,
    // From this point on they are local implementations
    // Before they are used from the timsrust crate.
    pub scan_start: usize,
    pub group_id: usize,
    pub quad_group_id: usize,
}

#[derive(Debug, Clone)]
pub struct DenseFrame {
    pub raw_peaks: Vec<TimsPeak>,
    pub index: usize,
    pub rt: f64,
    pub frame_type: FrameType,
    pub sorted: Option<SortingOrder>,
}

pub struct DenseFrameWindow {
    pub frame: DenseFrame,
    pub ims_start: f32,
    pub ims_end: f32,
    pub mz_start: f64,
    pub mz_end: f64,
    pub group_id: usize,
    pub quad_group_id: usize,
}

impl DenseFrame {
    pub fn new(
        frame: &Frame,
        ims_converter: &Scan2ImConverter,
        mz_converter: &Tof2MzConverter,
    ) -> DenseFrame {
        let mut expanded_scan_indices = Vec::with_capacity(frame.tof_indices.len());
        let mut last_scan_offset = frame.scan_offsets[0].clone();
        for (scan_index, index_offset) in frame.scan_offsets[1..].iter().enumerate() {
            let num_tofs = index_offset - last_scan_offset;

            let ims = ims_converter.convert(scan_index as u32) as f32;
            expanded_scan_indices.extend(vec![ims; num_tofs as usize]);
            last_scan_offset = index_offset.clone();
        }

        let peaks = expanded_scan_indices
            .iter()
            .zip(frame.tof_indices.iter())
            .zip(frame.intensities.iter())
            .map(|((scan_index, tof_index), intensity)| TimsPeak {
                intensity: *intensity,
                mz: mz_converter.convert(*tof_index),
                mobility: *scan_index,
            })
            .collect::<Vec<_>>();

        let index = frame.index;
        let rt = frame.rt;
        let frame_type = frame.frame_type;

        DenseFrame {
            raw_peaks: peaks,
            index,
            rt,
            frame_type,
            sorted: None,
        }
    }

    fn from_frame_window(
        frame_window: FrameWindow,
        ims_converter: &Scan2ImConverter,
        mz_converter: &Tof2MzConverter,
    ) -> DenseFrame {
        let mut expanded_scan_indices = Vec::with_capacity(frame_window.tof_indices.len());
        let mut last_scan_offset = frame_window.scan_offsets[0].clone();
        for (scan_index, index_offset) in frame_window.scan_offsets[1..].iter().enumerate() {
            let num_tofs = index_offset - last_scan_offset;
            let scan_index_use = (scan_index + frame_window.scan_start) as u32;

            let ims = ims_converter.convert(scan_index_use) as f32;
            expanded_scan_indices.extend(vec![ims; num_tofs as usize]);
            last_scan_offset = index_offset.clone();
        }
        debug_assert!(last_scan_offset == frame_window.tof_indices.len() as u64);

        let peaks = expanded_scan_indices
            .iter()
            .zip(frame_window.tof_indices.iter())
            .zip(frame_window.intensities.iter())
            .map(|((scan_index, tof_index), intensity)| TimsPeak {
                intensity: *intensity,
                mz: mz_converter.convert(*tof_index),
                mobility: *scan_index,
            })
            .collect::<Vec<_>>();

        let index = frame_window.index;
        let rt = frame_window.rt;
        let frame_type = frame_window.frame_type;

        DenseFrame {
            raw_peaks: peaks,
            index,
            rt,
            frame_type,
            sorted: None,
        }
    }

    fn concatenate(mut self, other: DenseFrame) -> DenseFrame {
        self.raw_peaks.extend(other.raw_peaks);
        self.sorted = None;
        self
    }

    pub fn sort_by_mz(&mut self) {
        match self.sorted {
            Some(SortingOrder::Mz) => return,
            _ => {
                self.raw_peaks
                    .sort_unstable_by(|a, b| a.mz.partial_cmp(&b.mz).unwrap());
                self.sorted = Some(SortingOrder::Mz);
                return;
            }
        }
    }

    pub fn sort_by_mobility(&mut self) {
        match self.sorted {
            Some(SortingOrder::Mobility) => return,
            _ => {
                self.raw_peaks
                    .sort_unstable_by(|a, b| a.mobility.partial_cmp(&b.mobility).unwrap());
                self.sorted = Some(SortingOrder::Mobility);
                return;
            }
        }
    }
}

impl RerunPlottable for DenseFrame {
    fn plot(
        &self,
        rec: &mut rerun::RecordingStream,
        entry_path: String,
        log_time_in_seconds: f32,
    ) -> Result<(), Box<dyn std::error::Error>> {
        rec.set_time_seconds("rt_seconds", log_time_in_seconds);
        let quad_points = self
            .raw_peaks
            .iter()
            .map(|peak| NDPoint {
                values: [(peak.mz / 10.) as Float, (100. * peak.mobility as Float)],
            })
            .collect::<Vec<_>>();

        let max_intensity = self
            .raw_peaks
            .iter()
            .map(|peak| peak.intensity)
            .max()
            .unwrap_or(0) as f32;

        let radii = self
            .raw_peaks
            .iter()
            .map(|peak| (peak.intensity as f32) / max_intensity)
            .collect::<Vec<_>>();

        rec.log(
            entry_path,
            &rerun::Points2D::new(
                quad_points
                    .iter()
                    .map(|point| (point.values[0] as f32, point.values[1] as f32)),
            )
            .with_radii(radii),
        )?;

        Ok(())
    }
}
