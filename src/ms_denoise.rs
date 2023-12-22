use crate::ms;
use crate::ms::TimsPeak;
use crate::quad;
use crate::quad::RadiusQuadTree;

trait DenoisableFrame {
    // Drops peaks that dont have at least one neighbor within a given mz/mobility tolerance
    fn min_neighbor_denoise(&self, mz_scaling: f64, ims_scaling: f64, min_n: usize) -> Self;
}

fn min_max_points(points: &[quad::Point]) -> (quad::Point, quad::Point) {
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

    (quad::Point { x: min_x, y: min_y }, quad::Point { x: max_x, y: max_y })
}

#[cfg(test)]
mod test_min_max {
    use super::*;

    #[test]
    fn test_min_max() {
        let points = vec![
            quad::Point { x: 0.0, y: 0.0 },
            quad::Point { x: 1.0, y: 1.0 },
            quad::Point { x: 2.0, y: 2.0 },
            quad::Point { x: 3.0, y: 3.0 },
            quad::Point { x: 4.0, y: 4.0 },
        ];

        let (min_point, max_point) = min_max_points(&points);

        assert_eq!(min_point, quad::Point { x: 0.0, y: 0.0 });
        assert_eq!(max_point, quad::Point { x: 4.0, y: 4.0 });
    }
}

impl DenoisableFrame for ms::DenseFrame {
    fn min_neighbor_denoise(
        &self,
        mz_scaling: f64,
        ims_scaling: f64,
        min_n: usize,
    ) -> ms::DenseFrame {
        // I could pre-sort here and use the window iterator,
        // to pre-filter for points with no neighbors in the mz dimension.

        // AKA could filter out the points with no mz neighbors sorting
        // and the use the tree to filter points with no mz+mobility neighbors.

        let quad_points = self
            .raw_peaks
            .iter()
            .map(|peak| quad::Point {
                x: (peak.mz / mz_scaling),
                y: (peak.mobility as f64 / ims_scaling),
            })
            .collect::<Vec<_>>();

        let (min_point, max_point) = min_max_points(&quad_points);
        let min_x = min_point.x;
        let min_y = min_point.y;
        let max_x = max_point.x;
        let max_y = max_point.y;

        let boundary = quad::Boundary {
            x_center: (max_x + min_x) / 2.0,
            y_center: (max_y + min_y) / 2.0,
            width: max_x - min_x,
            height: max_y - min_y,
        };

        let mut tree: RadiusQuadTree<'_, TimsPeak> = quad::RadiusQuadTree::new(boundary, 15, 1.);
        for (point, timspeak) in quad_points.iter().zip(self.raw_peaks.iter()) {
            tree.insert(point.clone(), timspeak);
        }

        let mut denoised_peaks = Vec::new();
        for (point, peaks) in quad_points.iter().zip(self.raw_peaks.iter()) {
            let mut results = Vec::new();

            // TODO: implement an 'any neighbor' method.
            tree.query(*point, &mut results);

            if results.len() >= min_n {
                denoised_peaks.push(peaks.clone());
            }
        }

        ms::DenseFrame {
            raw_peaks: denoised_peaks,
            index: self.index,
            rt: self.rt,
            frame_type: self.frame_type,
            sorted: None,
        }
    }
}

fn read_all_ms1_denoising(path: String) -> Vec<ms::DenseFrame> {
    let reader = timsrust::FileReader::new(path).unwrap();
    let mut frames = reader.read_all_ms1_frames();
    let ims_converter = reader.get_scan_converter().unwrap();
    let mz_converter = reader.get_tof_converter().unwrap();
    let rt_converter = reader.get_frame_converter().unwrap();

    let mut denoised_frames = Vec::new();

    // TODO: parallelize this
    for frame in frames.iter_mut() {
        let dense = ms::DenseFrame::new(frame, &ims_converter, &mz_converter, &rt_converter);
        let denoised_frame = dense.min_neighbor_denoise(0.015, 0.015, 5);
        denoised_frames.push(denoised_frame);
    }

    denoised_frames
}
