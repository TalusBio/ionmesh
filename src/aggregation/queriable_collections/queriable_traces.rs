use log::info;
use rayon::prelude::*;

use crate::aggregation::tracing::BaseTrace;
pub use crate::space::space_generics::{
    AsAggregableAtIndex,
    AsNDPointsAtIndex,
    DistantAtIndex,
    IntenseAtIndex,
    NDBoundary,
    NDPoint,
    QueriableIndexedPoints,
};
pub use crate::utils::binary_search_slice;

#[derive(Debug)]
pub struct TraceScalings {
    pub rt_scaling: f64,
    pub ims_scaling: f64,
    pub quad_scaling: f64,
}

#[derive(Debug)]
pub struct QueriableTraces {
    traces: Vec<BaseTrace>,
    min_bucket_rt_vals: Vec<f32>,
    bucket_size: usize,
    scalings: TraceScalings,
}

impl QueriableTraces {
    pub fn new(
        mut traces: Vec<BaseTrace>,
        scalings: TraceScalings,
    ) -> Self {
        const BUCKET_SIZE: usize = 16384 / 2;
        // Sort all of our peaks by rt, from low to high
        traces.par_sort_unstable_by(|a, b| a.rt.partial_cmp(&b.rt).unwrap());

        let mut min_bucket_rt_vals = traces
            .par_chunks_mut(BUCKET_SIZE)
            .map(|bucket| {
                let min = bucket[0].rt;
                bucket.par_sort_unstable_by(|a, b| a.mobility.partial_cmp(&b.mobility).unwrap());
                min
            })
            .collect::<Vec<_>>();

        // Get the max value of the last bucket
        let max_bucket_rt = traces[traces.len().saturating_sub(BUCKET_SIZE)..traces.len()]
            .iter()
            .max_by(|a, b| a.rt.partial_cmp(&b.rt).unwrap())
            .unwrap()
            .rt;
        min_bucket_rt_vals.push(max_bucket_rt);

        QueriableTraces {
            traces,
            min_bucket_rt_vals,
            bucket_size: BUCKET_SIZE,
            scalings,
        }
    }

    fn get_bucket_at(
        &self,
        index: usize,
    ) -> Result<&[BaseTrace], ()> {
        let page_start = index * self.bucket_size;
        if page_start >= self.traces.len() {
            return Err(());
        }
        let page_end = (page_start + self.bucket_size).min(self.traces.len());
        let tmp = &self.traces[page_start..page_end];

        if cfg!(debug_assertions) && rand::random::<usize>() % 100 == 0 {
            // Make sure all rts are sorted within the bucket
            for i in 1..tmp.len() {
                if tmp[i - 1].mobility > tmp[i].mobility {
                    panic!("RTs are not sorted within the bucket");
                }
            }
        }
        Ok(tmp)
    }

    pub fn get_intensity_sorted_indices(&self) -> Vec<(usize, u64)> {
        let mut indices: Vec<(usize, u64)> = (0..self.traces.len())
            .map(|i| (i, self.traces[i].intensity))
            .collect();

        indices.par_sort_unstable_by_key(|&x| x.1);
        indices
    }
}

impl AsNDPointsAtIndex<2> for QueriableTraces {
    fn get_ndpoint(
        &self,
        index: usize,
    ) -> NDPoint<2> {
        NDPoint {
            values: [self.traces[index].rt, self.traces[index].mobility],
        }
    }
    fn num_ndpoints(&self) -> usize {
        self.traces.len()
    }
}

impl IntenseAtIndex for QueriableTraces {
    fn intensity_at_index(
        &self,
        index: usize,
    ) -> u64 {
        self.traces[index].intensity
    }

    fn intensity_index_length(&self) -> usize {
        self.traces.len()
    }
}

impl AsAggregableAtIndex<BaseTrace> for QueriableTraces {
    fn get_aggregable_at_index(
        &self,
        index: usize,
    ) -> BaseTrace {
        self.traces[index]
    }

    fn num_aggregable(&self) -> usize {
        self.traces.len()
    }
}

pub struct BaseTraceDistance {
    pub quad_diff: f32,
    pub iou: f32,
    pub cosine: f32,
}

impl DistantAtIndex<BaseTraceDistance> for QueriableTraces {
    fn distance_at_indices(
        &self,
        index: usize,
        other: usize,
    ) -> BaseTraceDistance {
        let quad_diff = (self.traces[index].quad_center - self.traces[other].quad_center).abs();
        let iou = self.traces[index].rt_iou(&self.traces[other]);
        // Q: What can cause an error here??
        let cosine = self.traces[index]
            .chromatogram
            .cosine_similarity(&self.traces[other].chromatogram)
            .unwrap();
        BaseTraceDistance {
            quad_diff,
            iou,
            cosine,
        }
    }
}

impl QueriableIndexedPoints<2> for QueriableTraces {
    fn query_ndpoint(
        &self,
        point: &NDPoint<2>,
    ) -> Vec<usize> {
        let (bounds, point) = self.convert_to_bounds_query(point);
        self.query_ndrange(&bounds, point)
    }

    fn convert_to_bounds_query<'a>(
        &'a self,
        point: &'a NDPoint<2>,
    ) -> (NDBoundary<2>, Option<&NDPoint<2>>) {
        let rt = point.values[0];
        let mobility = point.values[1];
        let bounds = NDBoundary::new(
            [
                rt - self.scalings.rt_scaling as f32,
                mobility - self.scalings.ims_scaling as f32,
            ],
            [
                rt + self.scalings.rt_scaling as f32,
                mobility + self.scalings.ims_scaling as f32,
            ],
        );
        (bounds, Some(point))
    }

    fn query_ndrange(
        &self,
        boundary: &NDBoundary<2>,
        reference_point: Option<&NDPoint<2>>,
    ) -> Vec<usize> {
        let start_rt = boundary.starts[0];
        let end_rt = boundary.ends[0];

        let start_ims = boundary.starts[1];
        let end_ims = boundary.ends[1];

        let mut out = Vec::new();
        let (start_bucket, end_bucket) = binary_search_slice(
            &self.min_bucket_rt_vals,
            |a, b| a.total_cmp(b),
            start_rt,
            end_rt,
        );

        let bstart = start_bucket.saturating_sub(1);
        let bend_new = end_bucket
            .saturating_add(1)
            .min(self.min_bucket_rt_vals.len());

        for bucket_index in bstart..bend_new {
            let bucket = match self.get_bucket_at(bucket_index) {
                Ok(x) => x,
                Err(()) => continue,
            };

            let (ibstart, ibend) = binary_search_slice(
                bucket,
                |a, b| a.mobility.partial_cmp(b).unwrap(),
                start_ims,
                end_ims,
            );

            for (ti, trace) in bucket[ibstart..ibend].iter().enumerate() {
                if trace.rt < start_rt || trace.rt > end_rt {
                    continue;
                }
                if trace.mobility < start_ims || trace.mobility > end_ims {
                    continue;
                }
                if let Some(reference_point) = reference_point {
                    let dist = (reference_point.values[0] - trace.rt).abs()
                        + (reference_point.values[1] - trace.mobility).abs();
                    if dist > self.scalings.rt_scaling as f32 + self.scalings.ims_scaling as f32 {
                        continue;
                    }
                }
                let pi = ti + ibstart + bucket_index * self.bucket_size;
                debug_assert!(pi < self.traces.len());
                out.push(pi);
            }
        }

        if out.is_empty() {
            info!(
                "No traces found for query: \n{:?} -> {:?}\n",
                boundary, reference_point
            );
        }

        out
    }
}
