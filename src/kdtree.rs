
use crate::mod_types::Float;
// f32 or f64 depending on compilation

#[derive(Debug, Clone, Copy)]
struct NDBoundary<const DIMENSIONALITY: usize> {
    pub starts : [Float; DIMENSIONALITY],
    pub ends : [Float; DIMENSIONALITY]
}

impl <const D:usize>NDBoundary<D> {
    fn contains(&self, point: &NDPoint<D>) -> bool {
        for i in 0..D {
            if point.values[i] < self.starts[i] || point.values[i] >= self.ends[i] {
                return false;
            }
        }
        return true;
    }

    fn intersects(&self, other: &NDBoundary<D>) -> bool {
        for i in 0..D {
            if self.starts[i] >= other.ends[i] || self.ends[i] <= other.starts[i] {
                return false;
            }
        }
        return true;
    }
}

#[derive(Debug, Clone, Copy)]
struct NDPoint<const DIMENSIONALITY: usize> {
    values : [Float; DIMENSIONALITY]
}

trait IntoNDPoint<const DIMENSIONALITY: usize> {
    fn into_nd_point(&self) -> NDPoint<DIMENSIONALITY>;
}

// Implements a kdtree with several minor differences.
#[derive(Debug, Clone)]
pub struct RadiusKDTree<'a, T, const DIMENSIONALITY: usize> {
    boundary: NDBoundary<DIMENSIONALITY>,
    capacity: usize,
    radius: Float,
    points: Vec<(NDPoint<DIMENSIONALITY>, &'a T)>,
    high_split: Option<Box<RadiusKDTree<'a, T, DIMENSIONALITY>>>,
    low_split: Option<Box<RadiusKDTree<'a, T, DIMENSIONALITY>>>,
    division_axis: Option<usize>,

    // Since ranges are [closed, open) by convention,
    // I could think of the splits to be [low_bounds, division_value)
    // and [division_value, high_bounds).
    division_value: Option<Float>,
    count: usize,
}

impl<'a, const D: usize,T> RadiusKDTree<'a, T, D> {
    pub fn insert_ndpoint(&mut self, point: NDPoint<D>, value: &'a T) {
        if cfg!(debug_assertions) && !self.boundary.contains(&point) {
            panic!("Point {:?} is not contained in the boundary of this tree", point);
        }

        self.count += 1;

        // I am not the biggest fan of this nesting of if else statements...
        if self.division_value.is_none() {
            if self.points.len() < self.capacity {
                self.points.push((point, value));
            } else {
                let split_check = self.split();
                
                if split_check.is_err() {
                    self.points.push((point, value));
                }
                // Ok case handled in the next chunk.
            }
        };

        if self.division_value.is_some() {
            let division_value = self.division_value.unwrap();
            let division_axis = self.division_axis.unwrap();
            if point.values[division_axis] < division_value {
                self.low_split.as_mut().unwrap().insert_ndpoint(point, value);
            } else {
                self.high_split.as_mut().unwrap().insert_ndpoint(point, value);
            }
        }
    }

    fn split(&mut self) -> Result<(), &'static str>{
        let mut low_bounds = self.boundary.starts;
        let mut high_bounds = self.boundary.ends;
        let mut division_axis = 0;
        let mut division_value = self.boundary.starts[0];
        let mut longest_axis = 0;
        let mut longest_axis_length = self.boundary.ends[0] - self.boundary.starts[0];
        let mut any_change = false;

        for i in 1..D {
            let axis_length = self.boundary.ends[i] - self.boundary.starts[i];
            if axis_length < self.radius {
                continue;
            }

            if axis_length > longest_axis_length {
                // Check that the actual values in the points have a range
                // > 0, otherwise skip dimension.
                const EPS: Float = 1e-4;
                let mut axis_val_first = self.points.first().unwrap().0.values[i].clone();
                let mut keep = false;
                for point in self.points.iter() {
                    let diff = (point.0.values[i] - axis_val_first).abs();
                    if diff > EPS {
                        keep = true;
                        break;
                    }
                }
                if !keep {
                    continue;
                }

                longest_axis_length = axis_length;
                longest_axis = i;
                any_change = true;
            }
        }

        if !any_change {
            return Err("All dimensions have a range of 0");
        }

        division_axis = longest_axis;
        division_value = (self.boundary.ends[division_axis] + self.boundary.starts[division_axis]) / 2.0;
        low_bounds[division_axis] = self.boundary.starts[division_axis];
        high_bounds[division_axis] = self.boundary.ends[division_axis];

        let low_boundary = NDBoundary {
            starts: low_bounds,
            ends: high_bounds,
        };

        let high_boundary = NDBoundary {
            starts: low_bounds,
            ends: high_bounds,
        };

        let mut low_split = RadiusKDTree {
            boundary: low_boundary,
            capacity: self.capacity,
            radius: self.radius,
            points: Vec::new(),
            high_split: None,
            low_split: None,
            division_axis: None,
            division_value: None,
            count: 0,
        };

        let mut high_split = RadiusKDTree {
            boundary: high_boundary,
            capacity: self.capacity,
            radius: self.radius,
            points: Vec::new(),
            high_split: None,
            low_split: None,
            division_axis: None,
            division_value: None,
            count: 0,
        };

        while let Some(elem) = self.points.pop() {
            let (point, value) = elem;
            if point.values[division_axis] < division_value {
                low_split.insert_ndpoint(point, value);
            } else {
                high_split.insert_ndpoint(point, value);
            }
        }

        self.low_split = Some(Box::new(low_split));
        self.high_split = Some(Box::new(high_split));
        self.division_axis = Some(division_axis);
        self.division_value = Some(division_value);

        debug_assert_eq!(self.points.len(), 0);
        Ok(())
    }

}