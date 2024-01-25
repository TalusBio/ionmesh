use crate::mod_types::Float;
use crate::space::space_generics::{IndexedPoints, NDBoundary, NDPoint};
use log::warn;

const EPSILON: Float = Float::EPSILON;
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
    level: usize,
}

// WARN right now this is over-flowing ...

impl<'a, const D: usize, T> RadiusKDTree<'a, T, D> {
    pub fn new_empty(
        boundary: NDBoundary<D>,
        capacity: usize,
        radius: Float,
    ) -> RadiusKDTree<'a, T, D> {
        RadiusKDTree {
            boundary,
            capacity,
            radius,
            points: Vec::new(),
            high_split: None,
            low_split: None,
            division_axis: None,
            division_value: None,
            count: 0,
            level: 0,
        }
    }

    pub fn insert_ndpoint(&mut self, point: NDPoint<D>, value: &'a T) {
        if cfg!(debug_assertions) && !self.boundary.contains(&point) {
            panic!(
                "Point {:?} is not contained in the boundary of this tree ({:?})",
                point, self.boundary
            );
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
                } else {
                    self.insert_ndpoint(point, value);
                }
            }
        } else {
            let division_value = self.division_value.unwrap();
            let division_axis = self.division_axis.unwrap();
            if point.values[division_axis] < division_value {
                self.low_split
                    .as_mut()
                    .unwrap()
                    .insert_ndpoint(point, value);
            } else {
                self.high_split
                    .as_mut()
                    .unwrap()
                    .insert_ndpoint(point, value);
            }
        }
    }

    fn split(&mut self) -> Result<(), &'static str> {
        let _low_bounds = self.boundary.starts;
        let _high_bounds = self.boundary.ends;
        let mut longest_axis: Option<usize> = None;
        let mut longest_axis_length: Option<Float> = None;

        for i in 0..D {
            let axis_length = self.boundary.widths[i];
            if axis_length < self.radius {
                continue;
            }

            if longest_axis_length.is_none() || axis_length > longest_axis_length.unwrap() {
                // Check that the actual values in the points have a range
                // > 0, otherwise skip dimension.
                let axis_val_first = self.points.first().unwrap().0.values[i];
                let mut keep = false;
                for point in self.points.iter() {
                    let diff = (point.0.values[i] - axis_val_first).abs();
                    if diff > EPSILON {
                        keep = true;
                        break;
                    }
                }
                if !keep {
                    continue;
                }

                longest_axis_length = Some(axis_length);
                longest_axis = Some(i);
            }
        }

        if longest_axis.is_none() {
            return Err("All dimensions have a range of 0");
        }

        let division_axis = longest_axis.unwrap();
        let division_value = self.boundary.centers[division_axis];

        let low_boundary_starts = self.boundary.starts;
        let mut low_boundary_ends = self.boundary.ends;
        low_boundary_ends[division_axis] = division_value;

        let mut high_boundary_starts = self.boundary.starts;
        let high_boundary_ends = self.boundary.ends;
        high_boundary_starts[division_axis] = division_value;

        let low_boundary = NDBoundary::new(low_boundary_starts, low_boundary_ends);
        let high_boundary = NDBoundary::new(high_boundary_starts, high_boundary_ends);

        let mut low_split = RadiusKDTree::new_empty(low_boundary, self.capacity, self.radius);
        let mut high_split = RadiusKDTree::new_empty(high_boundary, self.capacity, self.radius);

        let new_level = self.level + 1;
        low_split.level = new_level;
        high_split.level = new_level;

        self.low_split = Some(Box::new(low_split));
        self.high_split = Some(Box::new(high_split));
        self.division_axis = Some(division_axis);
        self.division_value = Some(division_value);

        while let Some(elem) = self.points.pop() {
            if elem.0.values[division_axis] < division_value {
                self.low_split
                    .as_mut()
                    .unwrap()
                    .insert_ndpoint(elem.0, elem.1);
            } else {
                self.high_split
                    .as_mut()
                    .unwrap()
                    .insert_ndpoint(elem.0, elem.1);
            }
        }

        if new_level > 100 {
            // Should this be a warn?
            warn!("Splitting at level {} on axis {}", new_level, division_axis);
            warn!("Division value is {}", division_value);
            warn!("Curr bounds are {:?}", self.boundary);
            warn!(
                "Number of points on each child is {} and {}",
                self.low_split.as_ref().unwrap().count,
                self.high_split.as_ref().unwrap().count
            );
        }

        debug_assert_eq!(self.points.len(), 0);
        Ok(())
    }

    pub fn query(&'a self, point: &NDPoint<D>) -> Vec<&'a T> {
        let candidates: Vec<(&NDPoint<D>, &T)> = self.query_range(&NDBoundary::new(
            point
                .values
                .iter()
                .map(|x| x - self.radius)
                .collect::<Vec<Float>>()
                .try_into()
                .unwrap(),
            point
                .values
                .iter()
                .map(|x| x + self.radius)
                .collect::<Vec<Float>>()
                .try_into()
                .unwrap(),
        ));
        let out = self.refine_query(point, candidates);

        out
    }

    pub fn query_range(&'a self, boundary: &NDBoundary<D>) -> Vec<(&NDPoint<D>, &'a T)> {
        let mut result = Vec::new();
        if !self.boundary.intersects(boundary) {
            return result;
        }

        if self.division_value.is_none() {
            for elem in self.points.iter() {
                if boundary.contains(&elem.0) {
                    result.push((&elem.0, elem.1));
                }
            }
        }

        if self.division_value.is_some() {
            let division_value = self.division_value.unwrap();
            let division_axis = self.division_axis.unwrap();
            if boundary.starts[division_axis] < division_value {
                result.extend(self.low_split.as_ref().unwrap().query_range(boundary));
            }
            if boundary.ends[division_axis] >= division_value {
                result.extend(self.high_split.as_ref().unwrap().query_range(boundary));
            }
        }

        result
    }

    /// Calculates the manhattan distance between the query point and the
    /// candidate points. If the distance is less than the radius, the candidate
    /// point is kept.
    fn refine_query(&self, point: &NDPoint<D>, candidates: Vec<(&'a NDPoint<D>, &'a T)>) -> Vec<&'a T> {
        let out: Vec<&'a T> = candidates
            .into_iter()
            .filter(|x| {
                let dist =
                    x.0.values
                        .iter()
                        .zip(point.values.iter())
                        .map(|(x, y)| (x - y).abs())
                        .sum::<Float>();
                dist < self.radius
            })
            .map(|x| x.1)
            .collect();
        out
    }
}

impl<'a, T, const D: usize> IndexedPoints<'a, D, T> for RadiusKDTree<'a, T, D> {
    fn query_ndpoint(&'a self, point: &NDPoint<D>) -> Vec<&'a T> {
        self.query(point)
    }

    fn query_ndrange(&'a self, boundary: &NDBoundary<D>, reference_point: Option<&NDPoint<D>>) -> Vec<&'a T> {
        let candidates = self.query_range(boundary);
        if let Some(point) = reference_point {
            self.refine_query(point, candidates)
        } else {
            candidates.iter().map(|x| x.1).collect()
        }
    }
}
