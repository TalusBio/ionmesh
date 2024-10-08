use rayon::prelude::ParallelSliceMut;

#[derive(Debug, Clone, Copy)]
pub struct NDBoundary<const DIMENSIONALITY: usize> {
    pub starts: [f32; DIMENSIONALITY],
    pub ends: [f32; DIMENSIONALITY],
    pub widths: [f32; DIMENSIONALITY],
    pub centers: [f32; DIMENSIONALITY],
}

impl<const D: usize> NDBoundary<D> {
    pub fn new(
        starts: [f32; D],
        ends: [f32; D],
    ) -> NDBoundary<D> {
        let mut widths = [0.0; D];
        let mut centers = [0.0; D];
        for i in 0..D {
            widths[i] = ends[i] - starts[i];
            centers[i] = (ends[i] + starts[i]) / 2.0;
        }

        if cfg!(debug_assertions) {
            for i in 0..D {
                if starts[i] > ends[i] {
                    panic!("Starts must be less than ends");
                }
            }
        }

        NDBoundary {
            starts,
            ends,
            widths,
            centers,
        }
    }

    pub fn contains(
        &self,
        point: &NDPoint<D>,
    ) -> bool {
        for i in 0..D {
            // if point.values[i] < self.starts[i] || point.values[i] >= self.ends[i] {
            if point.values[i] < self.starts[i] || point.values[i] > self.ends[i] {
                return false;
            }
        }
        true
    }

    pub fn intersects(
        &self,
        other: &NDBoundary<D>,
    ) -> bool {
        for i in 0..D {
            if self.starts[i] >= other.ends[i] || self.ends[i] <= other.starts[i] {
                return false;
            }
        }
        true
    }

    pub fn from_ndpoints(points: &[NDPoint<D>]) -> NDBoundary<D> {
        let mut starts = [f32::MAX; D];
        let mut ends = [f32::MIN; D];

        for point in points.iter() {
            for i in 0..D {
                if point.values[i] < starts[i] {
                    starts[i] = point.values[i] - f32::EPSILON;
                }
                if point.values[i] > ends[i] {
                    ends[i] = point.values[i] + f32::EPSILON;
                }
            }
        }

        NDBoundary::new(starts, ends)
    }

    pub fn expand_relative(
        &mut self,
        factors: &[f32; D],
    ) {
        for (i, ef) in factors.iter().enumerate() {
            let mut half_width = self.widths[i] / 2.0;
            let center = self.centers[i];

            half_width *= ef;

            self.starts[i] = center - half_width;
            self.ends[i] = center + half_width;
            self.widths[i] = self.ends[i];
            self.centers[i] = (self.ends[i] + self.starts[i]) / 2.0;
        }
    }
    pub fn expand_absolute(
        &mut self,
        factors: &[f32; D],
    ) {
        for (i, ef) in factors.iter().enumerate() {
            let new_start = self.starts[i] - ef;
            let new_end = self.ends[i] + ef;
            let new_center = (new_start + new_end) / 2.0;
            let new_width = new_end - new_start;

            self.starts[i] = new_start;
            self.ends[i] = new_end;
            self.widths[i] = new_width;
            self.centers[i] = new_center;
        }
    }
}

// #[derive(Debug, Clone, Copy)]
// Oddly enough ... adding copy makes it slower ...
#[derive(Debug, Clone, Copy)]
pub struct NDPoint<const DIMENSIONALITY: usize> {
    pub values: [f32; DIMENSIONALITY],
}

pub trait QueriableIndexedPoints<const N: usize> {
    fn query_ndpoint(
        &self,
        point: &NDPoint<N>,
    ) -> Vec<usize> {
        let (bounds, reference_point) = self.convert_to_bounds_query(point);
        self.query_ndrange(&bounds, reference_point)
    }
    fn query_ndrange(
        &self,
        boundary: &NDBoundary<N>,
        reference_point: Option<&NDPoint<N>>,
    ) -> Vec<usize>;
    fn convert_to_bounds_query<'a>(
        &'a self,
        point: &'a NDPoint<N>,
    ) -> (NDBoundary<N>, Option<&NDPoint<N>>) {
        let bounds = NDBoundary::new(
            point
                .values
                .iter()
                .map(|x| *x - 1.)
                .collect::<Vec<_>>()
                .try_into()
                .unwrap(),
            point
                .values
                .iter()
                .map(|x| *x + 1.)
                .collect::<Vec<_>>()
                .try_into()
                .unwrap(),
        );

        (bounds, Some(point))
    }
}

pub trait AsNDPointsAtIndex<const D: usize> {
    fn get_ndpoint(
        &self,
        index: usize,
    ) -> NDPoint<D>;
    fn num_ndpoints(&self) -> usize;
}

impl<const D: usize> AsNDPointsAtIndex<D> for [NDPoint<D>] {
    fn get_ndpoint(
        &self,
        index: usize,
    ) -> NDPoint<D> {
        self[index]
    }

    fn num_ndpoints(&self) -> usize {
        self.len()
    }
}

pub trait HasIntensity: Sync + Send {
    fn intensity(&self) -> u64;
    fn weight(&self) -> u64 {
        self.intensity()
    }
}

pub trait IntenseAtIndex {
    fn intensity_at_index(
        &self,
        index: usize,
    ) -> u64;
    fn weight_at_index(
        &self,
        index: usize,
    ) -> u64 {
        self.intensity_at_index(index)
    }
    fn intensity_index_length(&self) -> usize;
    fn intensity_sorted_indices(&self) -> Vec<(usize, u64)> {
        let mut indices: Vec<(usize, u64)> = (0..self.intensity_index_length())
            .map(|i| (i, self.intensity_at_index(i)))
            .collect();
        indices.par_sort_unstable_by_key(|&x| x.1);

        debug_assert!(indices.len() == self.intensity_index_length());
        if cfg!(debug_assertions) && indices.len() > 1 {
            for i in 1..indices.len() {
                if indices[i - 1].1 > indices[i].1 {
                    panic!("Indices are not sorted");
                }
            }
        }
        indices
    }
}

pub trait AsAggregableAtIndex<T>
where
    // T: HasIntensity + Copy,
    // I am not sure how I want to express this in the type system.
    T: Copy,
{
    fn get_aggregable_at_index(
        &self,
        index: usize,
    ) -> T;

    fn num_aggregable(&self) -> usize;
}

impl<T> IntenseAtIndex for [T]
where
    T: HasIntensity + Copy,
{
    fn intensity_at_index(
        &self,
        index: usize,
    ) -> u64 {
        self[index].intensity()
    }
    fn intensity_index_length(&self) -> usize {
        self.len()
    }
}

pub trait DistantAtIndex<T> {
    fn distance_at_indices(
        &self,
        index: usize,
        other: usize,
    ) -> T;
}

impl<const N: usize> DistantAtIndex<f32> for [NDPoint<N>] {
    fn distance_at_indices(
        &self,
        index: usize,
        other: usize,
    ) -> f32 {
        let mut sum = 0.0;
        for i in 0..N {
            let diff = self[index].values[i] - self[other].values[i];
            sum += diff * diff;
        }
        sum.sqrt()
    }
}

pub trait TraceLike<R: std::convert::Into<f64>> {
    fn get_mz(&self) -> f64;
    fn get_intensity(&self) -> u64;
    fn get_rt(&self) -> R;
    fn get_ims(&self) -> R;
    fn get_quad_low_high(&self) -> (f64, f64);
}

pub trait NDPointConverter<T, const D: usize> {
    fn convert(
        &self,
        elem: &T,
    ) -> NDPoint<D>;
    fn convert_aggregables<IT>(
        &self,
        elems: &IT,
    ) -> (Vec<NDPoint<D>>, NDBoundary<D>)
    where
        IT: AsAggregableAtIndex<T> + ?Sized,
        T: Copy,
    {
        let points = (0..elems.num_aggregable())
            .map(|i| self.convert(&elems.get_aggregable_at_index(i)))
            .collect::<Vec<_>>();
        let boundary = NDBoundary::from_ndpoints(&points);
        (points, boundary)
    }
}
