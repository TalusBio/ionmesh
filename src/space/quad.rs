use crate::space::space_generics::{IndexedPoints, NDBoundary, NDPoint};
use core::panic;
use log::trace;

#[derive(Debug, Clone)]
pub struct RadiusQuadTree<'a, T> {
    boundary: NDBoundary<2>,
    capacity: usize,
    radius: f32,
    points: Vec<(NDPoint<2>, &'a T)>,
    northeast: Option<Box<RadiusQuadTree<'a, T>>>,
    northwest: Option<Box<RadiusQuadTree<'a, T>>>,
    southeast: Option<Box<RadiusQuadTree<'a, T>>>,
    southwest: Option<Box<RadiusQuadTree<'a, T>>>,
    division_point: Option<NDPoint<2>>,
    count: usize,
    depth: usize,
}

impl<'a, T> RadiusQuadTree<'a, T> {
    pub fn new_empty(
        boundary: NDBoundary<2>,
        capacity: usize,
        radius: f32,
    ) -> RadiusQuadTree<'a, T> {
        RadiusQuadTree {
            boundary,
            capacity,
            radius,
            points: Vec::new(),
            northeast: None,
            northwest: None,
            southeast: None,
            southwest: None,
            division_point: Option::None,
            count: 0,
            depth: 0,
        }
    }

    pub fn insert_ndpoint(&mut self, point: NDPoint<2>, data: &'a T) {
        self.insert(point, data);
    }

    pub fn insert(&mut self, point: NDPoint<2>, data: &'a T) {
        if cfg!(debug_assertions) && !self.boundary.contains(&point) {
            println!(
                "(Error??) Point outside of boundary {:?} {:?}",
                point, self.boundary
            );
            // print xs and ys
            println!("x: {:?} y: {:?}", point.values[0], point.values[1]);
            println!(
                "xmin: {:?} xmax: {:?}",
                self.boundary.starts[0], self.boundary.ends[0]
            );
            println!(
                "ymin: {:?} ymax: {:?}",
                self.boundary.starts[1], self.boundary.ends[1]
            );
            panic!("Point outside of boundary");
        }

        self.count += 1;

        if self.division_point.is_none() {
            let distance_squared = (point.values[0] - self.boundary.centers[0]).powi(2)
                + (point.values[1] - self.boundary.centers[1]).powi(2);

            // This can be pre-computed
            // let radius_squared = self.radius.powi(2);
            let radius_squared = (self.radius * 2.).powi(2);

            // This means any sub-division will be smaller than the radius
            let query_contained = radius_squared > distance_squared;
            if (self.points.len() < self.capacity) || query_contained {
                self.points.push((point, data));
            } else {
                self.subdivide();
                self.insert(point, data);
            }
        } else {
            let div_x = self.division_point.as_ref().unwrap().values[0];
            let div_y = self.division_point.as_ref().unwrap().values[1];

            if point.values[0] > div_x {
                if point.values[1] > div_y {
                    self.northeast.as_mut().unwrap().insert(point, data);
                } else {
                    self.southeast.as_mut().unwrap().insert(point, data);
                }
            } else if point.values[1] > div_y {
                self.northwest.as_mut().unwrap().insert(point, data);
            } else {
                self.southwest.as_mut().unwrap().insert(point, data);
            }
        }
    }

    pub fn subdivide(&mut self) {
        let division_point = NDPoint {
            values: self.boundary.centers,
        };

        //     |-----------------[c0, e1]-------------  ends
        //     |                    |                    |
        //     |                    |                    |
        //     |      NW            |        NE          |
        //     |                    |                    |
        //     |                    |                    |
        //   [s0, c1] ----------- center -------------- [e0, c1]
        //     |                    |                    |
        //     |                    |                    |
        //     |     SW             |       SE           |
        //     |                    |                    |
        //     |                    |                    |
        //    start  ----------- [c0, s1] ---------------|

        // Define boundaries for each quadrant
        let ne_boundary = NDBoundary::new(self.boundary.centers, self.boundary.ends);
        let nw_boundary = NDBoundary::new(
            [self.boundary.starts[0], self.boundary.centers[1]],
            [self.boundary.centers[0], self.boundary.ends[1]],
        );
        let se_boundary = NDBoundary::new(
            [self.boundary.centers[0], self.boundary.starts[1]],
            [self.boundary.ends[0], self.boundary.centers[1]],
        );
        let sw_boundary = NDBoundary::new(self.boundary.starts, self.boundary.centers);

        // println!("boundary {:?}", self.boundary);
        // println!("ne_boundary {:?}", ne_boundary);
        // println!("nw_boundary {:?}", nw_boundary);
        // println!("se_boundary {:?}", se_boundary);
        // println!("sw_boundary {:?}", sw_boundary);

        let new_level = self.depth + 1;

        // Create sub-trees for each quadrant
        self.northeast = Some(Box::new(RadiusQuadTree::new_empty(
            ne_boundary,
            self.capacity,
            self.radius,
        )));
        self.northeast.as_mut().unwrap().depth = new_level;
        self.northwest = Some(Box::new(RadiusQuadTree::new_empty(
            nw_boundary,
            self.capacity,
            self.radius,
        )));
        self.northwest.as_mut().unwrap().depth = new_level;
        self.southeast = Some(Box::new(RadiusQuadTree::new_empty(
            se_boundary,
            self.capacity,
            self.radius,
        )));
        self.southeast.as_mut().unwrap().depth = new_level;
        self.southwest = Some(Box::new(RadiusQuadTree::new_empty(
            sw_boundary,
            self.capacity,
            self.radius,
        )));
        self.southwest.as_mut().unwrap().depth = new_level;

        self.division_point = Some(division_point);
        trace!(
            "Subdividing at level {} at point {:?}",
            new_level,
            &self.division_point
        );

        // Move points into sub-trees
        while let Some(point) = self.points.pop() {
            self.insert(point.0, point.1);
        }
        self.points.clear();
    }

    pub fn query(&'a self, point: &NDPoint<2>) -> Vec<(&'a NDPoint<2>, &'a T)> {
        let mut result = Vec::new();
        let range = NDBoundary::new(
            [point.values[0] - self.radius, point.values[1] - self.radius],
            [point.values[0] + self.radius, point.values[1] + self.radius],
        );
        self.query_range(&range, &mut result);
        let out = self.refine_query(point, result);

        out
    }

    fn refine_query(
        &'a self,
        point: &NDPoint<2>,
        candidates: Vec<(&'a NDPoint<2>, &'a T)>,
    ) -> Vec<(&NDPoint<2>, &T)> {
        let mut result = Vec::new();
        let radius_squared = self.radius.powi(2);

        for (candidate_point, candidate_data) in candidates.into_iter() {
            let distance_squared = (candidate_point.values[0] - point.values[0]).powi(2)
                + (candidate_point.values[1] - point.values[1]).powi(2);
            if distance_squared <= radius_squared {
                result.push((candidate_point, candidate_data));
            }
        }

        result
    }

    // This function is used a lot so any optimization here will have a big impact.
    pub fn query_range(&'a self, range: &NDBoundary<2>, result: &mut Vec<(&'a NDPoint<2>, &'a T)>) {
        if !self.boundary.intersects(range) || self.count == 0 {
            return;
        }

        if self.division_point.is_none() {
            // There might be some optimization possible if we divide further the trees,
            // we could check if the range is fully contained in the boundary and if so
            // we could skip the containment checks.
            //
            for (point, data) in self.points.iter() {
                if range.contains(point) {
                    let dist = (point.values[0] - range.centers[0]).abs()
                        + (point.values[1] - range.centers[1]).abs();
                    if dist <= self.radius {
                        result.push((point, data));
                    }
                }
            }
        } else {
            assert_eq!(self.points.len(), 0);
            self.northeast.as_ref().unwrap().query_range(range, result);
            self.northwest.as_ref().unwrap().query_range(range, result);
            self.southeast.as_ref().unwrap().query_range(range, result);
            self.southwest.as_ref().unwrap().query_range(range, result);
        }
    }
}

// TODO: rename count_neigh_monotonocally_increasing
// because it can do more than just count neighbors....

impl<'a, T> IndexedPoints<'a, 2, T> for RadiusQuadTree<'a, T> {
    fn query_ndpoint(&'a self, point: &NDPoint<2>) -> Vec<&'a T> {
        self.query(point)
            .into_iter()
            .map(|x| x.1)
            .collect::<Vec<_>>()
    }

    fn query_ndrange(
        &'a self,
        boundary: &NDBoundary<2>,
        reference_point: Option<&NDPoint<2>>,
    ) -> Vec<&'a T> {
        let mut result = Vec::new();
        self.query_range(boundary, &mut result);

        match reference_point {
            Some(point) => self.refine_query(point, result),
            None => result,
        }
        .into_iter()
        .map(|x| x.1)
        .collect::<Vec<_>>()
    }
}
