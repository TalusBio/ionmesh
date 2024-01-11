use crate::mod_types::Float;
// f32 or f64 depending on compilation

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
            if point.values[i] < self.starts[i] || point.values[i] >= self.ends[i] {
                return false;
            }
        }
        return true;
    }

    pub fn intersects(&self, other: &NDBoundary<D>) -> bool {
        for i in 0..D {
            if self.starts[i] >= other.ends[i] || self.ends[i] <= other.starts[i] {
                return false;
            }
        }
        return true;
    }

    pub fn from_ndpoints(points: &[NDPoint<D>]) -> NDBoundary<D> {
        let mut starts = [Float::MAX; D];
        let mut ends = [Float::MIN; D];

        for point in points.iter() {
            for i in 0..D {
                if point.values[i] < starts[i] {
                    starts[i] = point.values[i];
                }
                if point.values[i] > ends[i] {
                    ends[i] = point.values[i];
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

trait IntoNDPoint<const DIMENSIONALITY: usize> {
    fn into_nd_point(&self) -> NDPoint<DIMENSIONALITY>;
}
