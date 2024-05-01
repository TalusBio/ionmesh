use crate::mod_types::Float;
// f32 or f64 depending on compilation

const EPSILON: Float = Float::EPSILON;

#[derive(Debug, Clone, Copy)]
pub struct NDBoundary<const DIMENSIONALITY: usize> {
    pub starts: [Float; DIMENSIONALITY],
    pub ends: [Float; DIMENSIONALITY],
    pub widths: [Float; DIMENSIONALITY],
    pub centers: [Float; DIMENSIONALITY],
}

impl<const D: usize> NDBoundary<D> {
    pub fn new(starts: [Float; D], ends: [Float; D]) -> NDBoundary<D> {
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

    pub fn contains(&self, point: &NDPoint<D>) -> bool {
        for i in 0..D {
            // if point.values[i] < self.starts[i] || point.values[i] >= self.ends[i] {
            if point.values[i] < self.starts[i] || point.values[i] > self.ends[i] {
                return false;
            }
        }
        true
    }

    pub fn intersects(&self, other: &NDBoundary<D>) -> bool {
        for i in 0..D {
            if self.starts[i] >= other.ends[i] || self.ends[i] <= other.starts[i] {
                return false;
            }
        }
        true
    }

    pub fn from_ndpoints(points: &[NDPoint<D>]) -> NDBoundary<D> {
        let mut starts = [Float::MAX; D];
        let mut ends = [Float::MIN; D];

        for point in points.iter() {
            for i in 0..D {
                if point.values[i] < starts[i] {
                    starts[i] = point.values[i] - EPSILON;
                }
                if point.values[i] > ends[i] {
                    ends[i] = point.values[i] + EPSILON;
                }
            }
        }

        NDBoundary::new(starts, ends)
    }
}

// #[derive(Debug, Clone, Copy)]
// Oddly enough ... adding copy makes it slower ...
#[derive(Debug, Clone)]
pub struct NDPoint<const DIMENSIONALITY: usize> {
    pub values: [Float; DIMENSIONALITY],
}

pub trait IndexedPoints<'a, const N: usize, T> {
    fn query_ndpoint(&'a self, point: &NDPoint<N>) -> Vec<&'a T>;
    fn query_ndrange(
        &'a self,
        boundary: &NDBoundary<N>,
        reference_point: Option<&NDPoint<N>>,
    ) -> Vec<&'a T>;
}

pub trait HasIntensity<T>
where
    T: Copy
        + PartialOrd
        + std::ops::Add<Output = T>
        + std::ops::Sub<Output = T>
        + std::ops::Mul<Output = T>
        + std::ops::Div<Output = T>
        + Default,
{
    fn intensity(&self) -> T;
}

pub trait TraceLike<R: std::convert::Into<f64>> {
    fn get_mz(&self) -> f64;
    fn get_intensity(&self) -> u64;
    fn get_rt(&self) -> R;
    fn get_ims(&self) -> R;
    fn get_quad_low_high(&self) -> (f64, f64);
}

pub trait NDPointConverter<T, const D: usize> {
    fn convert(&self, elem: &T) -> NDPoint<D>;
    fn convert_vec(&self, elems: &[T]) -> (Vec<NDPoint<D>>, NDBoundary<D>) {
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
