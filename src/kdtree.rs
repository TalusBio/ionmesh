use crate::mod_types::Float;
// f32 or f64 depending on compilation

#[derive(Debug, Clone, Copy)]
pub struct NDBoundary<const DIMENSIONALITY: usize> {
    pub starts: [Float; DIMENSIONALITY],
    pub ends: [Float; DIMENSIONALITY],
}

impl<const D: usize> NDBoundary<D> {
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

    fn from_ndpoints(points: &[NDPoint<D>]) -> NDBoundary<D> {
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

        NDBoundary { starts, ends }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct NDPoint<const DIMENSIONALITY: usize> {
    values: [Float; DIMENSIONALITY],
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
        }
    }

    pub fn insert_ndpoint(&mut self, point: NDPoint<D>, value: &'a T) {
        if cfg!(debug_assertions) && !self.boundary.contains(&point) {
            panic!(
                "Point {:?} is not contained in the boundary of this tree",
                point
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
                }
                // Ok case handled in the next chunk.
            }
        };

        if self.division_value.is_some() {
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
        let mut low_bounds = self.boundary.starts;
        let mut high_bounds = self.boundary.ends;
        let mut longest_axis: Option<usize> = None;
        let mut longest_axis_length: Option<Float> = None;

        for i in 1..D {
            let axis_length = self.boundary.ends[i] - self.boundary.starts[i];
            if axis_length < self.radius {
                continue;
            }

            if longest_axis_length.is_none() || axis_length > longest_axis_length.unwrap() {
                // Check that the actual values in the points have a range
                // > 0, otherwise skip dimension.
                const EPS: Float = 1e-4;
                let axis_val_first = self.points.first().unwrap().0.values[i];
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

                longest_axis_length = Some(axis_length);
                longest_axis = Some(i);
            }
        }

        if longest_axis.is_none() {
            return Err("All dimensions have a range of 0");
        }

        let division_axis = longest_axis.unwrap();
        let division_value =
            (self.boundary.ends[division_axis] + self.boundary.starts[division_axis]) / 2.0;

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

        let mut low_split = RadiusKDTree::new_empty(low_boundary, self.capacity, self.radius);
        let mut high_split = RadiusKDTree::new_empty(high_boundary, self.capacity, self.radius);

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

    pub fn query(&'a self, point: NDPoint<D>, result: &mut Vec<&'a T>) {
        let mut candidates = Vec::new();
        self.query_range(
            NDBoundary {
                starts: point
                    .values
                    .iter()
                    .map(|x| x - self.radius)
                    .collect::<Vec<Float>>()
                    .try_into()
                    .unwrap(),
                ends: point
                    .values
                    .iter()
                    .map(|x| x + self.radius)
                    .collect::<Vec<Float>>()
                    .try_into()
                    .unwrap(),
            },
            &mut candidates,
        );

        let out: Vec<&T> = candidates
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

        result.extend(out);
    }

    pub fn query_range(
        &'a self,
        boundary: NDBoundary<D>,
        result: &mut Vec<(&'a NDPoint<D>, &'a T)>,
    ) {
        if !self.boundary.intersects(&boundary) {
            return;
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
                self.low_split
                    .as_ref()
                    .unwrap()
                    .query_range(boundary, result);
            }
            if boundary.ends[division_axis] >= division_value {
                self.high_split
                    .as_ref()
                    .unwrap()
                    .query_range(boundary, result);
            }
        }
    }
}
