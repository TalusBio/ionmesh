use log::trace;
use serde::Serialize;
use timsrust::converters::{
    ConvertableDomain,
    Scan2ImConverter,
    Tof2MzConverter,
};

use super::{
    ExpandedFrameSlice,
    TimsPeak,
};
use crate::aggregation::aggregators::ClusterAggregator;
use crate::space::space_generics::{
    AsAggregableAtIndex,
    AsNDPointsAtIndex,
    DistantAtIndex,
    IntenseAtIndex,
    NDPoint,
    QueriableIndexedPoints,
};

#[derive(Debug, Serialize)]
pub struct FrameSliceWindow<'a> {
    pub window: &'a [ExpandedFrameSlice],
    pub reference_index: usize,
    pub cum_lengths: Vec<usize>,
}

#[derive(Debug, Clone, Copy)]
pub struct MaybeIntenseRawPeak {
    pub intensity: u32,
    pub tof_index: u32,
    pub scan_index: usize,
    pub weight_only: bool,
}

impl FrameSliceWindow<'_> {
    pub fn new(window: &[ExpandedFrameSlice]) -> FrameSliceWindow<'_> {
        let cum_lengths = window
            .iter()
            .map(|x| x.num_ndpoints())
            .scan(0, |acc, x| {
                *acc += x;
                Some(*acc)
            })
            .collect();
        trace!("Cumulative lengths: {:?}", cum_lengths);
        FrameSliceWindow {
            window,
            reference_index: window.len() / 2,
            cum_lengths,
        }
    }
    fn get_window_index(
        &self,
        index: usize,
    ) -> (usize, usize) {
        let mut pos = 0;
        let mut last_cum_length = 0;
        for (i, cum_length) in self.cum_lengths.iter().enumerate() {
            if index < *cum_length {
                pos = i;
                break;
            }
            last_cum_length = *cum_length;
        }

        debug_assert!(
            index < *self.cum_lengths.last().unwrap(),
            "Index out of bounds, generated index: {}, pos: {}, cum_lengths: {:?}",
            index,
            pos,
            self.cum_lengths
        );
        let within_window_index = index - last_cum_length;

        if cfg!(debug_assertions) {
            assert!(
                self.window[pos].intensities.len() > within_window_index,
                "Index out of bounds, generated index: {}, within_window_index: {}, pos: {}, \
                 cum_lengths: {:?}",
                index,
                within_window_index,
                pos,
                self.cum_lengths,
            );
        }

        (pos, within_window_index)
    }
}

impl AsAggregableAtIndex<MaybeIntenseRawPeak> for FrameSliceWindow<'_> {
    fn get_aggregable_at_index(
        &self,
        index: usize,
    ) -> MaybeIntenseRawPeak {
        let (pos, within_window_index) = self.get_window_index(index);
        let tmp = &self.window[pos];
        let tof = tmp.tof_indices[within_window_index];
        let int = tmp.intensities[within_window_index];
        let scan = tmp.scan_numbers[within_window_index];

        MaybeIntenseRawPeak {
            intensity: int,
            tof_index: tof,
            scan_index: scan,
            weight_only: pos != self.reference_index,
        }
    }

    fn num_aggregable(&self) -> usize {
        *self.cum_lengths.last().unwrap()
    }
}

impl IntenseAtIndex for FrameSliceWindow<'_> {
    fn intensity_at_index(
        &self,
        index: usize,
    ) -> u64 {
        let (pos, within_window_index) = self.get_window_index(index);
        if pos == self.reference_index {
            self.window[self.reference_index].intensity_at_index(within_window_index)
        } else {
            0
        }
    }

    fn weight_at_index(
        &self,
        index: usize,
    ) -> u64 {
        let (pos, within_window_index) = self.get_window_index(index);
        self.window[pos].weight_at_index(within_window_index)
    }

    fn intensity_index_length(&self) -> usize {
        *self.cum_lengths.last().unwrap()
    }
}

impl<'a> QueriableIndexedPoints<2> for FrameSliceWindow<'a> {
    fn query_ndpoint(
        &self,
        point: &NDPoint<2>,
    ) -> Vec<usize> {
        let mut out = Vec::new();
        let mut last_cum_length = 0;
        for (_i, (frame, cum_length)) in self.window.iter().zip(self.cum_lengths.iter()).enumerate()
        {
            let local_outs = frame.query_ndpoint(point);
            for ii in local_outs {
                out.push(ii + last_cum_length);
            }
            last_cum_length = *cum_length;
        }
        out
    }

    fn query_ndrange(
        &self,
        boundary: &crate::space::space_generics::NDBoundary<2>,
        reference_point: Option<&NDPoint<2>>,
    ) -> Vec<usize> {
        let mut out = Vec::new();
        let last = *self.cum_lengths.last().unwrap();
        let mut last_cum_length = 0;
        for (frame, cum_length) in self.window.iter().zip(self.cum_lengths.iter()) {
            let local_outs = frame.query_ndrange(boundary, reference_point);
            for ii in local_outs {
                let pi = ii + last_cum_length;
                debug_assert!(pi < last, "Index out of bounds: {}, last: {}", pi, last);
                out.push(pi);
            }
            last_cum_length = *cum_length;
        }

        out
    }
}

impl DistantAtIndex<f32> for FrameSliceWindow<'_> {
    fn distance_at_indices(
        &self,
        index: usize,
        other: usize,
    ) -> f32 {
        let (_pos, _within_window_index) = self.get_window_index(index);
        let (_pos_other, _within_window_index_other) = self.get_window_index(other);
        panic!("unimplemented");
        0.
    }
}

impl AsNDPointsAtIndex<2> for FrameSliceWindow<'_> {
    fn get_ndpoint(
        &self,
        index: usize,
    ) -> NDPoint<2> {
        let (pos, within_window_index) = self.get_window_index(index);
        self.window[pos].get_ndpoint(within_window_index)
    }

    fn num_ndpoints(&self) -> usize {
        *self.cum_lengths.last().unwrap()
    }
}

#[derive(Default, Debug, Clone, Copy)]
pub struct RawWeightedTimsPeakAggregator {
    pub cumulative_weighted_cluster_tof: u64,
    pub cumulative_weighted_cluster_scan: u64,
    pub cumulative_cluster_weight: u64,
    pub cumulative_cluster_intensity: u64,
    pub num_peaks: u64,
    pub num_intense_peaks: u64,
}

#[derive(Debug, Clone, Copy)]
pub struct RawScaleTimsPeak {
    pub intensity: f64,
    pub tof_index: f64,
    pub scan_index: f64,
    pub npeaks: u64,
}

impl RawScaleTimsPeak {
    pub fn to_timspeak(
        &self,
        mz_converter: &Tof2MzConverter,
        ims_converter: &Scan2ImConverter,
    ) -> TimsPeak {
        TimsPeak {
            intensity: self.intensity as u32,
            mz: mz_converter.convert(self.tof_index),
            mobility: ims_converter.convert(self.scan_index) as f32,
            npeaks: self.npeaks as u32,
        }
    }
}

impl ClusterAggregator<MaybeIntenseRawPeak, RawScaleTimsPeak> for RawWeightedTimsPeakAggregator {
    // Calculate the weight-weighted average of the cluster
    // for mz and ims. The intensity is kept as is.
    fn add(
        &mut self,
        elem: &MaybeIntenseRawPeak,
    ) {
        self.cumulative_cluster_intensity +=
            if elem.weight_only { 0 } else { elem.intensity } as u64;
        self.cumulative_cluster_weight += elem.intensity as u64;
        self.cumulative_weighted_cluster_tof += elem.tof_index as u64 * elem.intensity as u64;
        self.cumulative_weighted_cluster_scan += elem.scan_index as u64 * elem.intensity as u64;
        self.num_peaks += 1;
        if !elem.weight_only {
            self.num_intense_peaks += 1;
        };
    }

    fn aggregate(&self) -> RawScaleTimsPeak {
        // Use raw
        RawScaleTimsPeak {
            intensity: self.cumulative_cluster_intensity as f64,
            tof_index: self.cumulative_weighted_cluster_tof as f64
                / self.cumulative_cluster_weight as f64,
            scan_index: self.cumulative_weighted_cluster_scan as f64
                / self.cumulative_cluster_weight as f64,
            npeaks: self.num_intense_peaks,
        }
    }

    fn combine(
        self,
        other: Self,
    ) -> Self {
        Self {
            cumulative_weighted_cluster_tof: self.cumulative_weighted_cluster_tof
                + other.cumulative_weighted_cluster_tof,
            cumulative_weighted_cluster_scan: self.cumulative_weighted_cluster_scan
                + other.cumulative_weighted_cluster_scan,
            cumulative_cluster_weight: self.cumulative_cluster_weight
                + other.cumulative_cluster_weight,
            cumulative_cluster_intensity: self.cumulative_cluster_intensity
                + other.cumulative_cluster_intensity,
            num_peaks: self.num_peaks + other.num_peaks,
            num_intense_peaks: self.num_intense_peaks + other.num_intense_peaks,
        }
    }
}
