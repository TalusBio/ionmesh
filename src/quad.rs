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
            println!("Point outside of boundary {:?} {:?}", point, self.boundary);
            return;
        }

        self.count += 1;

        if self.division_point.is_none() {
            let distance_squared = (point.x - self.boundary.x_center).powi(2)
                + (point.y - self.boundary.y_center).powi(2);
            let radius_squared = self.radius.powi(2);
            if self.points.len() < self.capacity || distance_squared <= radius_squared {
                self.points.push((point, data));
            } else {
                self.subdivide();
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
        println!("Subdividing");
        let x = self.boundary.x_center;
        let y = self.boundary.y_center;
        let w = self.boundary.width / 2.0;
        let h = self.boundary.height / 2.0;
        let w_offset = self.boundary.width / 4.0;
        let h_offset = self.boundary.height / 4.0;

        let division_point = Point { x: x, y: y };

        // Define boundaries for each quadrant
        let ne_boundary = Boundary {
            x_center: x + w_offset,
            y_center: y + h_offset,
            width: w,
            height: h,
        };
        let nw_boundary = Boundary {
            x_center: x - w_offset,
            y_center: y + h_offset,
            width: w,
            height: h,
        };
        let se_boundary = Boundary {
            x_center: x + w_offset,
            y_center: y - h_offset,
            width: w,
            height: h,
        };
        let sw_boundary = Boundary {
            x_center: x - w_offset,
            y_center: y - h_offset,
            width: w,
            height: h,
        };

        println!("boundary {:?}", self.boundary);
        println!("ne_boundary {:?}", ne_boundary);
        println!("nw_boundary {:?}", nw_boundary);
        println!("se_boundary {:?}", se_boundary);
        println!("sw_boundary {:?}", sw_boundary);

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
        let range = Boundary {
            x_center: point.x,
            y_center: point.y,
            width: self.radius,
            height: self.radius,
        };
        self.query_range(&range, result);
    }

    pub fn query_range(&self, range: &Boundary, result: &mut Vec<(Point, &'a T)>) {
        if !self.boundary.intersects(range) || self.count == 0 {
            return;
        }

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
        let xmin = self.x_center - self.width;
        let xmax = self.x_center + self.width;
        let ymin = self.y_center - self.height;
        let ymax = self.y_center + self.height;

        point.x >= xmin && point.x <= xmax && point.y >= ymin && point.y <= ymax
    }

    pub fn intersects(&self, other: &Boundary) -> bool {
        let xmin = self.x_center - self.width;
        let xmax = self.x_center + self.width;
        let ymin = self.y_center - self.height;
        let ymax = self.y_center + self.height;

        let xmin2 = other.x_center - other.width;
        let xmax2 = other.x_center + other.width;
        let ymin2 = other.y_center - other.height;
        let ymax2 = other.y_center + other.height;

        !(xmin > xmax2 || xmax < xmin2 || ymin > ymax2 || ymax < ymin2)
    }
}
