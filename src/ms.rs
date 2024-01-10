pub use timsrust::Frame;
pub use timsrust::FrameType;
pub use timsrust::{
    ConvertableIndex, FileReader, Frame2RtConverter, Scan2ImConverter, Tof2MzConverter,
};

use crate::quad::{Boundary, Point};

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

#[derive(Debug, Clone)]
pub struct DenseFrame {
    pub raw_peaks: Vec<TimsPeak>,
    pub index: usize,
    pub rt: f64,
    pub frame_type: FrameType,
    pub sorted: Option<SortingOrder>,
}

// binary search ...
fn binary_search<T: PartialOrd>(vec: &[T], target: &T) -> Option<usize> {
    let mut low = 0;
    let mut high = vec.len() - 1;

    while low <= high {
        let mid = (low + high) / 2;
        if vec[mid] > *target {
            high = mid - 1;
        } else if vec[mid] < *target {
            low = mid + 1;
        } else {
            return Some(mid);
        }
    }
    None
}

pub struct DenseFrameWindow {
    pub frame: DenseFrame,
    pub ims_start: f32,
    pub ims_end: f32,
    pub mz_start: f64,
    pub mz_end: f64,
    pub group_id: usize,
    pub quad_group_id: usize,
    bounds: Boundary,
}

impl DenseFrameWindow {
    pub fn new(
        frame: DenseFrame,
        ims_start: f32,
        ims_end: f32,
        mz_start: f64,
        mz_end: f64,
        group_id: usize,
        quad_group_id: usize,
    ) -> DenseFrameWindow {
        let bounds = Boundary::from_xxyy(ims_start.into(), ims_end.into(), mz_start.into(), mz_end.into());
        DenseFrameWindow {
            frame,
            ims_start,
            ims_end,
            mz_start,
            mz_end,
            group_id,
            quad_group_id,
            bounds,
        }
    }

    pub fn contains(&self, ims: f32, mz: f64) -> bool {
        let point_use = Point{ x: ims.into(), y: mz.into() };
        self.bounds.contains(&point_use)
    }
}

// From: https://github.com/mbhall88/psdm/blob/0c8c4be5e4a6d566193b688824197fac2d233108/src/lib.rs#L13-L41
// MIT licensed
trait SortExt<T> {
    fn argsort(&self) -> Vec<usize>;
    fn sort_by_indices(&mut self, indices: &mut Vec<usize>);
}

impl<T: Ord + Clone> SortExt<T> for Vec<T> {
    fn argsort(&self) -> Vec<usize> {
        let mut indices = (0..self.len()).collect::<Vec<_>>();
        indices.sort_by_key(|&i| &self[i]);
        indices
    }

    fn sort_by_indices(&mut self, indices: &mut Vec<usize>) {
        for idx in 0..self.len() {
            if indices[idx] != usize::MAX {
                let mut current_idx = idx;
                loop {
                    let target_idx = indices[current_idx];
                    indices[current_idx] = usize::MAX;
                    if indices[target_idx] == usize::MAX {
                        break;
                    }
                    self.swap(current_idx, target_idx);
                    current_idx = target_idx;
                }
            }
        }
    }
}

impl DenseFrame {
    pub fn new(
        frame: &Frame,
        ims_converter: &Scan2ImConverter,
        mz_converter: &Tof2MzConverter,
        rt_converter: &Frame2RtConverter,
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

#[cfg(test)]
mod test_argsort {
    use super::*;

    #[test]
    fn test_reorder_vec() {
        let mut vec1 = vec![4, 1, 3, 2, 5];
        let mut vec2 = vec!["p", "q", "r", "s", "t"];
        let mut inds = vec1.argsort();
        vec1.sort_by_indices(&mut inds.clone());
        vec2.sort_by_indices(&mut inds);
        assert_eq!(vec1, vec![1, 2, 3, 4, 5]);
        assert_eq!(vec2, vec!["q", "s", "r", "p", "t"]);
    }
}
