use crate::mod_types::Float;
use crate::ms::{DenseFrame, TimsPeak};
use crate::space_generics::{NDBoundary, NDPoint};
use core::panic;

#[derive(Debug, Clone, Copy, serde::Serialize, PartialEq)]
pub struct Point {
    pub x: Float,
    pub y: Float,
}

#[derive(Debug, Clone, Copy, serde::Serialize)]
pub struct Boundary {
    pub x_center: Float,
    pub y_center: Float,
    pub width: Float,
    pub height: Float,
    xmin: Float,
    xmax: Float,
    ymin: Float,
    ymax: Float,
}

const EPS: Float = 1e-6;
impl Boundary {
    pub fn new(x_center: Float, y_center: Float, width: Float, height: Float) -> Boundary {
        Boundary {
            x_center,
            y_center,
            width,
            height,
            xmin: x_center - (width / 2.0) - EPS,
            xmax: x_center + (width / 2.0) + EPS,
            ymin: y_center - (height / 2.0) - EPS,
            ymax: y_center + (height / 2.0) + EPS,
        }
    }

    pub fn from_xxyy(xmin: Float, xmax: Float, ymin: Float, ymax: Float) -> Boundary {
        let x_center = (xmin + xmax) / 2.0;
        let y_center = (ymin + ymax) / 2.0;
        let width = xmax - xmin;
        let height = ymax - ymin;
        Boundary {
            x_center,
            y_center,
            width,
            height,
            xmin: xmin - EPS,
            xmax: xmax + EPS,
            ymin: ymin - EPS,
            ymax: ymax + EPS,
        }
    }
}

#[derive(Debug, Clone)]
pub struct RadiusQuadTree<'a, T> {
    boundary: NDBoundary<2>,
    capacity: usize,
    radius: Float,
    points: Vec<(NDPoint<2>, &'a T)>,
    northeast: Option<Box<RadiusQuadTree<'a, T>>>,
    northwest: Option<Box<RadiusQuadTree<'a, T>>>,
    southeast: Option<Box<RadiusQuadTree<'a, T>>>,
    southwest: Option<Box<RadiusQuadTree<'a, T>>>,
    division_point: Option<NDPoint<2>>,
    count: usize,
}

impl<'a, T> RadiusQuadTree<'a, T> {
    pub fn new(boundary: NDBoundary<2>, capacity: usize, radius: Float) -> RadiusQuadTree<'a, T> {
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
        }
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
            if self.points.len() < self.capacity {
                self.points.push((point, data));
            } else {
                if query_contained {
                    self.points.push((point, data));
                } else {
                    self.subdivide();
                }
            }
        };

        if self.division_point.is_some() {
            let div_point = self.division_point.unwrap();
            let div_x = div_point.values[0];
            let div_y = div_point.values[1];

            if point.values[0] >= div_x {
                if point.values[1] >= div_y {
                    self.northeast.as_mut().unwrap().insert(point, data);
                } else {
                    self.southeast.as_mut().unwrap().insert(point, data);
                }
            } else {
                if point.values[1] >= div_y {
                    self.northwest.as_mut().unwrap().insert(point, data);
                } else {
                    self.southwest.as_mut().unwrap().insert(point, data);
                }
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

        // Create sub-trees for each quadrant
        self.northeast = Some(Box::new(RadiusQuadTree::new(
            ne_boundary,
            self.capacity,
            self.radius,
        )));
        self.northwest = Some(Box::new(RadiusQuadTree::new(
            nw_boundary,
            self.capacity,
            self.radius,
        )));
        self.southeast = Some(Box::new(RadiusQuadTree::new(
            se_boundary,
            self.capacity,
            self.radius,
        )));
        self.southwest = Some(Box::new(RadiusQuadTree::new(
            sw_boundary,
            self.capacity,
            self.radius,
        )));

        self.division_point = Some(division_point);

        // Move points into sub-trees
        let mut i = 0;
        while i < self.points.len() {
            let (point, data) = self.points[i];
            self.insert(point, data);
            i += 1;
        }
        self.points.clear();
    }

    pub fn query(&self, point: NDPoint<2>, result: &mut Vec<(NDPoint<2>, &'a T)>) {
        let range = NDBoundary::new(
            [point.values[0] - self.radius, point.values[1] - self.radius],
            [point.values[0] + self.radius, point.values[1] + self.radius],
        );
        self.query_range(&range, result);
    }

    pub fn count_query(&self, point: NDPoint<2>, count_keeper: &mut u64) {
        let range = NDBoundary::new(
            [point.values[0] - self.radius, point.values[1] - self.radius],
            [point.values[0] + self.radius, point.values[1] + self.radius],
        );
        self.count_query_range(&range, count_keeper);
    }

    pub fn count_query_range(&self, range: &NDBoundary<2>, count_keeper: &mut u64) {
        if !self.boundary.intersects(range) || self.count == 0 {
            return;
        }

        let mut local_count = 0;
        for &(point, _) in &self.points {
            if range.contains(&point) {
                local_count += 1;
            }
        }

        *count_keeper += local_count;

        if self.division_point.is_some() {
            self.northeast
                .as_ref()
                .unwrap()
                .count_query_range(range, count_keeper);
            self.northwest
                .as_ref()
                .unwrap()
                .count_query_range(range, count_keeper);
            self.southeast
                .as_ref()
                .unwrap()
                .count_query_range(range, count_keeper);
            self.southwest
                .as_ref()
                .unwrap()
                .count_query_range(range, count_keeper);
        }
    }

    // This function is used a lot so any optimization here will have a big impact.
    pub fn query_range(&self, range: &NDBoundary<2>, result: &mut Vec<(NDPoint<2>, &'a T)>) {
        if !self.boundary.intersects(range) || self.count == 0 {
            return;
        }

        // There might be some optimization possible if we divide further the trees,
        // we could check if the range is fully contained in the boundary and if so
        // we could skip the containment checks.
        //
        for &(point, data) in &self.points {
            if range.contains(&point) {
                result.push((point, data));
            }
        }

        if self.division_point.is_some() {
            self.northeast.as_ref().unwrap().query_range(range, result);
            self.northwest.as_ref().unwrap().query_range(range, result);
            self.southeast.as_ref().unwrap().query_range(range, result);
            self.southwest.as_ref().unwrap().query_range(range, result);
        }
    }
}

impl Boundary {
    pub fn contains(&self, point: &Point) -> bool {
        point.x >= self.xmin && point.x <= self.xmax && point.y >= self.ymin && point.y <= self.ymax
    }

    pub fn intersection(&self, other: &Boundary) -> Float {
        // Returns the fraction of the area of self that is overlapped by other.

        // Top left corner
        let max_xmin = self.xmin.max(other.xmin);
        let max_ymin = self.ymin.max(other.ymin);

        // Bottom right corner
        let min_xmin = self.xmax.min(other.xmax);
        let min_ymin = self.ymax.min(other.ymax);

        let overlap = (min_xmin - max_xmin).max(0.0) * (min_ymin - max_ymin).max(0.0);

        let intersection = overlap / (self.width * self.height);
        intersection
    }

    pub fn intersects(&self, other: &Boundary) -> bool {
        // Returns true if the two boundaries intersect.
        self.xmin <= other.xmax
            && self.xmax >= other.xmin
            && self.ymin <= other.ymax
            && self.ymax >= other.ymin
    }
}

#[cfg(test)]
mod test_boundary {
    use super::*;

    #[test]
    fn test_contains() {
        let boundary = Boundary::new(0.0, 0.0, 50.0, 50.0);
        let point = Point { x: 25.0, y: 25.0 };
        assert!(boundary.contains(&point));
    }

    #[test]
    fn test_not_contains() {
        let boundary = Boundary::new(0.0, 0.0, 50.0, 50.0);
        let point = Point { x: 75.0, y: 75.0 };
        assert!(!boundary.contains(&point));
    }

    #[test]
    fn test_intersects() {
        let boundary = Boundary::new(0.0, 0.0, 50.0, 50.0);
        let other = Boundary::new(25.0, 25.0, 50.0, 50.0);
        assert!(boundary.intersects(&other));
    }

    #[test]
    fn test_not_intersects() {
        let boundary = Boundary::new(0.0, 0.0, 50.0, 50.0);
        let other = Boundary::new(75.0, 75.0, 50.0, 50.0);
        assert!(!boundary.intersects(&other));
    }

    #[test]
    fn test_intersection() {
        let boundary = Boundary::new(0.0, 0.0, 50.0, 50.0);
        let other = Boundary::new(25.0, 25.0, 50.0, 50.0);
        let expect = 0.25;
        let max_diff = 1e-6;
        assert!(boundary.intersection(&other) - expect < max_diff);
    }

    #[test]
    fn test_no_intersection() {
        let boundary = Boundary::new(0.0, 0.0, 50.0, 50.0);
        let other = Boundary::new(75.0, 75.0, 50.0, 50.0);
        assert_eq!(boundary.intersection(&other), 0.0);
    }

    #[test]
    fn test_intersection_inside() {
        let boundary = Boundary::new(0.0, 0.0, 20.0, 10.0);
        let other = Boundary::new(0.0, 0.0, 10.0, 10.0);

        let expect = 0.5;
        let max_diff = 1e-6;

        assert!(boundary.intersection(&other) - expect < max_diff);

        let expect = 1.00;
        assert!(other.intersection(&boundary) - expect < max_diff);
    }
}

pub fn denseframe_to_quadtree_points(
    denseframe: &mut DenseFrame,
    mz_scaling: f64,
    ims_scaling: f32,
    min_n: usize,
) -> (Vec<NDPoint<2>>, Vec<TimsPeak>, NDBoundary<2>) {
    // Initial pre-filtering step
    denseframe.sort_by_mz();

    let num_neigh = count_neigh_monotonocally_increasing(
        &denseframe.raw_peaks,
        &|peak| peak.mz,
        mz_scaling,
        &|i_right, i_left| (i_right - i_left) >= min_n,
    );

    // Filter the peaks and replace the raw peaks with the filtered peaks.
    let prefiltered_peaks = denseframe
        .raw_peaks
        .clone()
        .into_iter()
        .zip(num_neigh.into_iter())
        .filter(|(_, b)| *b)
        .map(|(peak, _)| peak.clone()) // Clone the TimsPeak
        .collect::<Vec<_>>();

    let quad_points = prefiltered_peaks // denseframe.raw_peaks //
        .iter()
        .map(|peak| NDPoint {
            values: [
                (peak.mz / mz_scaling) as Float,
                (peak.mobility / ims_scaling) as Float,
            ],
        })
        .collect::<Vec<_>>();

    let boundary = NDBoundary::from_ndpoints(&quad_points);
    (quad_points, prefiltered_peaks, boundary)
    // (quad_points, denseframe.raw_peaks.clone(), boundary)

    // NOTE: I would like to do this, but I dont know how to fix the lifetime issues...
    // let mut tree: RadiusQuadTree<'_, TimsPeak> = quad::RadiusQuadTree::new(boundary, 20, 1.);
    // for (point, timspeak) in quad_points.iter().zip(prefiltered_peaks.iter()) {
    //     tree.insert(point.clone(), timspeak);
    // }

    // (quad_points, prefiltered_peaks, tree)
}

// TODO: rename count_neigh_monotonocally_increasing
// because it can do more than just count neighbors....

#[inline(always)]
fn count_neigh_monotonocally_increasing<T, R, W>(
    elems: &[T],
    key: &dyn Fn(&T) -> R,
    max_dist: R,
    out_func: &dyn Fn(&usize, &usize) -> W,
) -> Vec<W>
where
    R: PartialOrd + Copy + std::ops::Sub<Output = R> + std::ops::Add<Output = R> + Default,
    T: Copy,
    W: Default + Copy,
{
    let mut prefiltered_peaks_bool: Vec<W> = vec![W::default(); elems.len()];

    let mut i_left = 0;
    let mut i_right = 0;
    let mut mz_left = key(&elems[0]);
    let mut mz_right = key(&elems[0]);

    // Does the cmpiler re-use the memory here?
    // let mut curr_mz = R::default();
    // let mut left_mz_diff = R::default();
    // let mut right_mz_diff = R::default();

    // 1. Slide the left index until the mz difference while sliding is more than the mz tolerance.
    // 2. Slide the right index until the mz difference while sliding is greater than the mz tolerance.
    // 3. If the number of points between the left and right index is greater than the minimum number of points, add them to the prefiltered peaks.

    let elems_len = elems.len();
    let elems_len_minus_one = elems_len - 1;
    for (curr_i, elem) in elems.iter().enumerate() {
        let curr_mz = key(elem);
        let mut left_mz_diff = curr_mz - mz_left;
        let mut right_mz_diff = mz_right - curr_mz;

        while left_mz_diff > max_dist {
            i_left += 1;
            mz_left = key(&elems[i_left]);
            left_mz_diff = curr_mz - mz_left;
        }

        // Slide the right index until the mz difference while sliding is greater than the mz tolerance.
        while (right_mz_diff < max_dist) && (i_right < elems_len) {
            i_right += 1;
            mz_right = key(&elems[i_right.min(elems_len_minus_one)]);
            right_mz_diff = mz_right - curr_mz;
        }

        // If the number of points between the left and right index is greater than the minimum number of points, add them to the prefiltered peaks.
        // println!("{} {}", i_left, i_right);
        if i_left < i_right {
            prefiltered_peaks_bool[curr_i] = out_func(&i_right, &(i_left));
        }

        if cfg!(test) {
            assert!(i_left <= i_right);
        }
    }

    prefiltered_peaks_bool
}

#[cfg(test)]
mod test_count_neigh {
    use super::*;

    #[test]
    fn test_count_neigh() {
        let elems = vec![0.0, 1.0, 2.0, 3.0, 4.0];

        let prefiltered_peaks_bool =
            count_neigh_monotonocally_increasing(&elems, &|x| *x, 1.1, &|i_right, i_left| {
                (i_right - i_left) >= 3
            });

        assert_eq!(prefiltered_peaks_bool, vec![false, true, true, true, false]);
    }
}

fn min_max_points(points: &[Point]) -> (Point, Point) {
    let mut min_x = points[0].x;
    let mut max_x = points[0].x;
    let mut min_y = points[0].y;
    let mut max_y = points[0].y;

    for p in points.iter() {
        if p.x < min_x {
            min_x = p.x;
        } else if p.x > max_x {
            max_x = p.x;
        }

        if p.y < min_y {
            min_y = p.y;
        } else if p.y > max_y {
            max_y = p.y;
        }
    }

    (Point { x: min_x, y: min_y }, Point { x: max_x, y: max_y })
}

#[cfg(test)]
mod test_min_max {
    use super::*;

    #[test]
    fn test_min_max() {
        let points = vec![
            Point { x: 0.0, y: 0.0 },
            Point { x: 1.0, y: 1.0 },
            Point { x: 2.0, y: 2.0 },
            Point { x: 3.0, y: 3.0 },
            Point { x: 4.0, y: 4.0 },
        ];

        let (min_point, max_point) = min_max_points(&points);

        assert_eq!(min_point, Point { x: 0.0, y: 0.0 });
        assert_eq!(max_point, Point { x: 4.0, y: 4.0 });
    }
}
