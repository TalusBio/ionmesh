use core::panic;

#[derive(Debug, Clone, Copy, serde::Serialize, PartialEq)]
pub struct Point {
    pub x: f64,
    pub y: f64,
}

#[derive(Debug, Clone, Copy, serde::Serialize)]
pub struct Boundary {
    pub x_center: f64,
    pub y_center: f64,
    pub width: f64,
    pub height: f64,
    xmin: f64,
    xmax: f64,
    ymin: f64,
    ymax: f64,
}

impl Boundary {
    pub fn new(x_center: f64, y_center: f64, width: f64, height: f64) -> Boundary {
        const EPS: f64 = 1e-6;
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
}

#[derive(Debug)]
pub struct RadiusQuadTree<'a, T> {
    boundary: Boundary,
    capacity: usize,
    radius: f64,
    points: Vec<(Point, &'a T)>,
    northeast: Option<Box<RadiusQuadTree<'a, T>>>,
    northwest: Option<Box<RadiusQuadTree<'a, T>>>,
    southeast: Option<Box<RadiusQuadTree<'a, T>>>,
    southwest: Option<Box<RadiusQuadTree<'a, T>>>,
    division_point: Option<Point>,
    count: usize,
}

impl<'a, T> RadiusQuadTree<'a, T> {
    pub fn new(boundary: Boundary, capacity: usize, radius: f64) -> RadiusQuadTree<'a, T> {
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

    pub fn insert(&mut self, point: Point, data: &'a T) {
        if !self.boundary.contains(&point) {
            // Should this be an error?
            println!(
                "(Error??) Point outside of boundary {:?} {:?}",
                point, self.boundary
            );
            // print xs and ys
            //
            println!("x: {:?} y: {:?}", point.x, point.y);
            println!(
                "xmin: {:?} xmax: {:?}",
                self.boundary.xmin, self.boundary.xmax
            );
            println!(
                "ymin: {:?} ymax: {:?}",
                self.boundary.ymin, self.boundary.ymax
            );
            panic!("Point outside of boundary");
            return;
        }

        self.count += 1;

        if self.division_point.is_none() {
            let distance_squared = (point.x - self.boundary.x_center).powi(2)
                + (point.y - self.boundary.y_center).powi(2);

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
            let div_x = self.division_point.unwrap().x;
            let div_y = self.division_point.unwrap().y;

            if point.x >= div_x {
                if point.y >= div_y {
                    self.northeast.as_mut().unwrap().insert(point, data);
                } else {
                    self.southeast.as_mut().unwrap().insert(point, data);
                }
            } else {
                if point.y >= div_y {
                    self.northwest.as_mut().unwrap().insert(point, data);
                } else {
                    self.southwest.as_mut().unwrap().insert(point, data);
                }
            }
        }
    }

    pub fn subdivide(&mut self) {
        // println!("Subdividing");
        let x = self.boundary.x_center;
        let y = self.boundary.y_center;
        let w = self.boundary.width / 2.0;
        let h = self.boundary.height / 2.0;
        let w_offset = self.boundary.width / 4.0;
        let h_offset = self.boundary.height / 4.0;

        let division_point = Point { x: x, y: y };

        // Define boundaries for each quadrant
        let ne_boundary = Boundary::new(x + w_offset, y + h_offset, w, h);
        let nw_boundary = Boundary::new(x - w_offset, y + h_offset, w, h);
        let se_boundary = Boundary::new(x + w_offset, y - h_offset, w, h);
        let sw_boundary = Boundary::new(x - w_offset, y - h_offset, w, h);

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

    pub fn query(&self, point: Point, result: &mut Vec<(Point, &'a T)>) {
        let range = Boundary::new(point.x, point.y, self.radius, self.radius);
        self.query_range(&range, result);
    }

    pub fn count_query(&self, point: Point, count_keeper: &mut u64) {
        let range = Boundary::new(point.x, point.y, self.radius, self.radius);
        self.count_query_range(&range, count_keeper);
    }

    pub fn count_query_range(&self, range: &Boundary, count_keeper: &mut u64) {
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
    pub fn query_range(&self, range: &Boundary, result: &mut Vec<(Point, &'a T)>) {
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

    pub fn to_json(&self) -> String {
        let mut json = String::new();
        json.push_str("{\n");
        json.push_str(&format!(
            "\"boundary\": {},",
            serde_json::to_string(&self.boundary).unwrap()
        ));
        json.push_str(&format!("\"capacity\": {},", self.capacity));
        json.push_str(&format!("\"radius\": {},", self.radius));

        if self.points.len() > 0 {
            json.push_str(&format!("\"points\": ["));
            for (point, _) in &self.points {
                json.push_str(&format!("{},", serde_json::to_string(&point).unwrap()));
            }
            // remove trailing comma
            json.pop();
            json.push_str("],");
        }
        json.push_str(&format!(
            "\"northeast\": {},",
            match &self.northeast {
                Some(tree) => tree.to_json(),
                None => "null".to_string(),
            }
        ));
        json.push_str(&format!(
            "\"northwest\": {},",
            match &self.northwest {
                Some(tree) => tree.to_json(),
                None => "null".to_string(),
            }
        ));
        json.push_str(&format!(
            "\"southeast\": {},",
            match &self.southeast {
                Some(tree) => tree.to_json(),
                None => "null".to_string(),
            }
        ));
        json.push_str(&format!(
            "\"southwest\": {}",
            match &self.southwest {
                Some(tree) => tree.to_json(),
                None => "null".to_string(),
            }
        ));
        json.push_str("}\n");
        json
    }
}

impl Boundary {
    pub fn contains(&self, point: &Point) -> bool {
        point.x >= self.xmin && point.x <= self.xmax && point.y >= self.ymin && point.y <= self.ymax
    }

    pub fn intersection(&self, other: &Boundary) -> f64 {
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
