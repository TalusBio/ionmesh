pub use crate::{
    aggregation::tracing::TimeTimsPeak,
    space::space_generics::{
        AsAggregableAtIndex, AsNDPointsAtIndex, DistantAtIndex, IntenseAtIndex, NDBoundary,
        NDPoint, QueriableIndexedPoints,
    },
    utils::binary_search_slice,
};
use rayon::prelude::*;

#[derive(Debug)]
pub struct TimeTimsPeakScaling {
    pub mz_scaling: f32,
    pub rt_scaling: f32,
    pub ims_scaling: f32,
    pub quad_scaling: f32,
}

#[derive(Debug)]
pub struct QueriableTimeTimsPeaks {
    peaks: Vec<TimeTimsPeak>,
    min_bucket_mz_vals: Vec<f32>,
    bucket_size: usize,
    scalings: TimeTimsPeakScaling,
}

impl QueriableTimeTimsPeaks {
    pub fn new(
        mut peaks: Vec<TimeTimsPeak>,
        scalings: TimeTimsPeakScaling,
    ) -> Self {
        const BUCKET_SIZE: usize = 16384;
        // // Sort all of our peaks by m/z, from low to high
        peaks.par_sort_unstable_by(|a, b| a.mz.partial_cmp(&b.mz).unwrap());

        let mut min_bucket_mz_vals = peaks
            .par_chunks_mut(BUCKET_SIZE)
            .map(|bucket| {
                let min = bucket[0].mz;
                bucket.par_sort_unstable_by(|a, b| a.rt.partial_cmp(&b.rt).unwrap());
                min as f32
            })
            .collect::<Vec<_>>();

        // Get the max value of the last bucket
        let max_bucket_mz = peaks[peaks.len().saturating_sub(BUCKET_SIZE)..peaks.len()]
            .iter()
            .max_by(|a, b| a.mz.partial_cmp(&b.mz).unwrap())
            .unwrap()
            .mz as f32;
        min_bucket_mz_vals.push(max_bucket_mz);

        QueriableTimeTimsPeaks {
            peaks,
            min_bucket_mz_vals,
            bucket_size: BUCKET_SIZE,
            scalings,
        }
    }

    fn get_bucket_at(
        &self,
        index: usize,
    ) -> Result<&[TimeTimsPeak], ()> {
        let page_start = index * self.bucket_size;
        if page_start >= self.peaks.len() {
            return Err(());
        }
        let page_end = (page_start + self.bucket_size).min(self.peaks.len());
        let tmp = &self.peaks[page_start..page_end];

        if cfg!(debug_assertions) {
            // Check every 100 random queries ...
            if rand::random::<usize>() % 100 == 0 {
                let mut last_rt = 0.;
                for i in 0..tmp.len() {
                    if tmp[i].rt < last_rt {
                        panic!("RTs are not sorted within the bucket");
                    }
                    last_rt = tmp[i].rt;
                }
            }
        }
        Ok(tmp)
    }

    pub fn get_intensity_sorted_indices(&self) -> Vec<(usize, u64)> {
        let mut indices: Vec<(usize, u64)> = (0..self.peaks.len())
            .map(|i| (i, self.peaks[i].intensity))
            .collect();
        indices.par_sort_unstable_by_key(|&x| x.1);

        debug_assert!(indices.len() == self.peaks.len());
        if cfg!(debug_assertions) {
            if indices.len() > 1 {
                for i in 1..indices.len() {
                    if indices[i - 1].1 > indices[i].1 {
                        panic!("Indices are not sorted");
                    }
                }
            }
        }
        indices
    }
}

impl AsNDPointsAtIndex<3> for QueriableTimeTimsPeaks {
    fn get_ndpoint(
        &self,
        index: usize,
    ) -> NDPoint<3> {
        NDPoint {
            values: [
                self.peaks[index].mz as f32,
                self.peaks[index].rt,
                self.peaks[index].ims,
            ],
        }
    }

    fn num_ndpoints(&self) -> usize {
        self.peaks.len()
    }
}

impl IntenseAtIndex for QueriableTimeTimsPeaks {
    fn intensity_at_index(
        &self,
        index: usize,
    ) -> u64 {
        self.peaks[index].intensity
    }

    fn intensity_index_length(&self) -> usize {
        self.peaks.len()
    }
}

impl AsAggregableAtIndex<TimeTimsPeak> for QueriableTimeTimsPeaks {
    fn get_aggregable_at_index(
        &self,
        index: usize,
    ) -> TimeTimsPeak {
        self.peaks[index]
    }

    fn num_aggregable(&self) -> usize {
        self.peaks.len()
    }
}

impl DistantAtIndex<f32> for QueriableTimeTimsPeaks {
    fn distance_at_indices(
        &self,
        index: usize,
        other: usize,
    ) -> f32 {
        let a = self.peaks[index];
        let b = self.peaks[other];
        let mz = (a.mz - b.mz) as f32 / self.scalings.mz_scaling;
        let rt = (a.rt - b.rt) as f32 / self.scalings.rt_scaling;
        let ims = (a.ims - b.ims) as f32 / self.scalings.ims_scaling;
        (mz * mz + rt * rt + ims * ims).sqrt()
    }
}

impl QueriableIndexedPoints<3> for QueriableTimeTimsPeaks {
    fn query_ndpoint(
        &self,
        point: &NDPoint<3>,
    ) -> Vec<usize> {
        let boundary = NDBoundary::new(
            [
                (point.values[0] - self.scalings.mz_scaling) - f32::EPSILON,
                (point.values[1] - self.scalings.rt_scaling),
                (point.values[2] - self.scalings.ims_scaling) - f32::EPSILON,
            ],
            [
                (point.values[0] + self.scalings.mz_scaling) + f32::EPSILON,
                (point.values[1] + self.scalings.rt_scaling),
                (point.values[2] + self.scalings.ims_scaling) + f32::EPSILON,
            ],
        );
        let out = self.query_ndrange(&boundary, None);
        out
    }

    fn query_ndrange(
        &self,
        boundary: &NDBoundary<3>,
        reference_point: Option<&NDPoint<3>>,
    ) -> Vec<usize> {
        let mut out = Vec::new();
        let mz_range = (boundary.starts[0], boundary.ends[0]);
        let mz_range_f64 = (boundary.starts[0] as f64, boundary.ends[0] as f64);
        let rt_range = (boundary.starts[1], boundary.ends[1]);
        let ims_range = (boundary.starts[2], boundary.ends[2]);

        let (bstart, bend) = binary_search_slice(
            &self.min_bucket_mz_vals,
            |a, b| a.total_cmp(b),
            mz_range.0,
            mz_range.1,
        );

        let bstart = bstart.saturating_sub(1);
        let bend_new = bend.saturating_add(1).min(self.min_bucket_mz_vals.len());

        for bnum in bstart..bend_new {
            let c_bucket = self.get_bucket_at(bnum);
            if c_bucket.is_err() {
                continue;
            }
            let c_bucket = c_bucket.unwrap();
            let page_start = bnum * self.bucket_size;

            let (istart, iend) =
                binary_search_slice(c_bucket, |a, b| a.rt.total_cmp(&b), rt_range.0, rt_range.1);

            for (j, peak) in self.peaks[(istart + page_start)..(iend + page_start)]
                .iter()
                .enumerate()
            {
                debug_assert!(
                    peak.rt >= rt_range.0 && peak.rt <= rt_range.1,
                    "RT out of range -> {} {} {}; istart {}, page_starrt {}, j {}; window rts: {:?}",
                    peak.rt,
                    rt_range.0,
                    rt_range.1,
                    istart,
                    page_start,
                    j,
                    &self.peaks[(j + istart + page_start).saturating_sub(5)
                        ..(j + istart + page_start + 5).min(self.peaks.len())]
                        .iter()
                        .map(|x| x.rt)
                        .collect::<Vec<f32>>()
                );
                if peak.ims >= ims_range.0 && peak.ims <= ims_range.1 {
                    if peak.mz as f32 >= mz_range.0 && peak.mz as f32 <= mz_range.1 {
                        out.push(j + istart + page_start);
                    }
                }
            }
        }

        out
    }
}
