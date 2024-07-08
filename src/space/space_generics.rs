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

    pub fn expand(
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
}

// #[derive(Debug, Clone, Copy)]
// Oddly enough ... adding copy makes it slower ...
#[derive(Debug, Clone)]
pub struct NDPoint<const DIMENSIONALITY: usize> {
    pub values: [f32; DIMENSIONALITY],
}

// Q: is there any instance where T is not usize?
pub trait QueriableIndexedPoints<'a, const N: usize, T> {
    fn query_ndpoint(
        &'a self,
        point: &NDPoint<N>,
    ) -> Vec<&'a T>;
    fn query_ndrange(
        &'a self,
        boundary: &NDBoundary<N>,
        reference_point: Option<&NDPoint<N>>,
    ) -> Vec<&'a T>;
}

pub trait AsNDPoints<const D: usize> {
    fn get_ndpoint(
        &self,
        index: usize,
    ) -> NDPoint<D>;
    fn num_ndpoints(&self) -> usize;
    fn intensity_at(
        &self,
        index: usize,
    ) -> u64;
    fn weight_at(
        &self,
        index: usize,
    ) -> u64 {
        self.intensity_at(index)
    }
}

pub trait HasIntensity: Sync {
    fn intensity(&self) -> u64;
    fn weight(&self) -> u64 {
        self.intensity()
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
    fn convert_vec(
        &self,
        elems: &[T],
    ) -> (Vec<NDPoint<D>>, NDBoundary<D>) {
        let points = elems
            .iter()
            .map(|elem| self.convert(elem))
            .collect::<Vec<_>>();
        let boundary = NDBoundary::from_ndpoints(&points);
        (points, boundary)
    }
    fn convert_to_bounds_query<'a>(
        &self,
        point: &'a NDPoint<D>,
    ) -> (NDBoundary<D>, Option<&'a NDPoint<D>>) {
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
