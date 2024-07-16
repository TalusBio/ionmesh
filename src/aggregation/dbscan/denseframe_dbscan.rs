use crate::aggregation::aggregators::TimsPeakAggregator;
use crate::aggregation::converters::{BypassDenseFrameBackConverter, DenseFrameConverter};
use crate::aggregation::dbscan::dbscan::dbscan_generic;
use crate::ms::frames::{DenseFrame, TimsPeak};
use crate::space::space_generics::{DistantAtIndex, IntenseAtIndex};
use crate::utils::within_distance_apply;

// <FF: Send + Sync + Fn(&TimsPeak, &TimsPeak) -> bool>
pub fn dbscan_denseframe(
    mut denseframe: DenseFrame,
    mz_scaling: f64,
    max_mz_extension: f64,
    ims_scaling: f32,
    max_ims_extension: f32,
    min_n: usize,
    min_intensity: u64,
) -> DenseFrame {
    let out_frame_type: timsrust::FrameType = denseframe.frame_type;
    let out_rt: f64 = denseframe.rt;
    let out_index: usize = denseframe.index;

    let prefiltered_peaks = {
        denseframe.sort_by_mz();

        let keep_vector = within_distance_apply(
            &denseframe.raw_peaks,
            &|peak| peak.mz,
            &mz_scaling,
            &|i_right, i_left| (i_right - i_left) >= min_n,
        );

        // Filter the peaks and replace the raw peaks with the filtered peaks.

        denseframe
            .raw_peaks
            .clone()
            .into_iter()
            .zip(keep_vector)
            .filter(|(_, b)| *b)
            .map(|(peak, _)| peak) // Clone the TimsPeak
            .collect::<Vec<_>>()
    };

    let converter = DenseFrameConverter {
        mz_scaling,
        ims_scaling,
    };
    let peak_vec: Vec<TimsPeak> = dbscan_generic(
        converter,
        prefiltered_peaks,
        min_n,
        min_intensity,
        TimsPeakAggregator::default,
        None::<&(dyn Fn(&f32) -> bool + Send + Sync)>,
        None,
        true,
        &[max_mz_extension as f32, max_ims_extension],
        None::<BypassDenseFrameBackConverter>,
    );

    DenseFrame {
        raw_peaks: peak_vec,
        index: out_index,
        rt: out_rt,
        frame_type: out_frame_type,
        sorted: None,
    }
}

impl IntenseAtIndex for Vec<TimsPeak> {
    fn intensity_at_index(
        &self,
        index: usize,
    ) -> u64 {
        self[index].intensity as u64
    }
}

impl DistantAtIndex<f32> for Vec<TimsPeak> {
    fn distance_at_indices(
        &self,
        index: usize,
        other: usize,
    ) -> f32 {
        panic!("I dont think this is called ever ...");
        let mut sum = 0.0;
        let diff_mz = (self[index].mz - self[other].mz) as f32;
        sum += diff_mz * diff_mz;
        let diff_ims = self[index].mobility - self[other].mobility;
        sum += diff_ims * diff_ims;
        sum.sqrt()
    }
}
