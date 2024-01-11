use crate::ms;
use crate::quad;
use crate::quad::{denseframe_to_quadtree_points, RadiusQuadTree};

// Pseudocode from wikipedia.
// Donate to wikipedia y'all. :3
//
// DBSCAN(DB, distFunc, eps, minPts) {
//     C := 0                                                  /* Cluster counter */
//     for each point P in database DB {
//         if label(P) ≠ undefined then continue               /* Previously processed in inner loop */
//         Neighbors N := RangeQuery(DB, distFunc, P, eps)     /* Find neighbors */
//         if |N| < minPts then {                              /* Density check */
//             label(P) := Noise                               /* Label as Noise */
//             continue
//         }
//         C := C + 1                                          /* next cluster label */
//         label(P) := C                                       /* Label initial point */
//         SeedSet S := N \ {P}                                /* Neighbors to expand */
//         for each point Q in S {                             /* Process every seed point Q */
//             if label(Q) = Noise then label(Q) := C          /* Change Noise to border point */
//             if label(Q) ≠ undefined then continue           /* Previously processed (e.g., border point) */
//             label(Q) := C                                   /* Label neighbor */
//             Neighbors N := RangeQuery(DB, distFunc, Q, eps) /* Find neighbors */
//             if |N| ≥ minPts then {                          /* Density check (if Q is a core point) */
//                 S := S ∪ N                                  /* Add new neighbors to seed set */
//             }
//         }
//     }
// }

// Variations ...
// 1. Use a quadtree to find neighbors
// 2. Sort the pointd by decreasing intensity (more intense points adopt first).
// 3. Use an intensity threshold intead of a minimum number of neighbors.

#[derive(Debug, PartialEq, Clone)]
enum ClusterLabel<T> {
    Unassigned,
    Noise,
    Cluster(T),
}

trait HasIntensity<T>
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

impl HasIntensity<u32> for ms::TimsPeak {
    fn intensity(&self) -> u32 {
        self.intensity
    }
}

fn _dbscan<'a>(
    tree: &RadiusQuadTree<'a, usize>,
    prefiltered_peaks: &Vec<impl HasIntensity<u32>>,
    quad_points: &Vec<quad::Point>,
    min_n: usize,
    min_intensity: u64,
    intensity_sorted_indices: &Vec<(usize, u32)>,
) -> (u64, Vec<ClusterLabel<u64>>) {
    let mut cluster_labels = vec![ClusterLabel::Unassigned; prefiltered_peaks.len()];
    let mut cluster_id = 0;

    for (point_index, _intensity) in intensity_sorted_indices.iter() {
        let point_index = *point_index;
        if cluster_labels[point_index] != ClusterLabel::Unassigned {
            continue;
        }

        let mut neighbors = Vec::new();
        let query_point = quad_points[point_index].clone();
        tree.query(query_point, &mut neighbors);

        // Do I need to care about overflows here?
        let neighbor_intensity_total: u64 = neighbors
            .iter()
            .map(|(_, i)| prefiltered_peaks[**i].intensity() as u64)
            .sum::<u64>();

        if neighbors.len() < min_n || neighbor_intensity_total < min_intensity {
            cluster_labels[point_index] = ClusterLabel::Noise;
            continue;
        }

        cluster_id += 1;
        cluster_labels[point_index] = ClusterLabel::Cluster(cluster_id);
        let mut seed_set = neighbors.clone();

        const MAX_EXTENSION_DISTANCE: f64 = 5.;

        while let Some(neighbor) = seed_set.pop() {
            let neighbor_index = neighbor.1.clone();
            if cluster_labels[neighbor_index] == ClusterLabel::Noise {
                cluster_labels[neighbor_index] = ClusterLabel::Cluster(cluster_id);
            }

            if cluster_labels[neighbor_index] != ClusterLabel::Unassigned {
                continue;
            }

            cluster_labels[neighbor_index] = ClusterLabel::Cluster(cluster_id);

            let mut neighbors = Vec::new();
            tree.query(neighbor.0, &mut neighbors);

            let neighbor_intensity_total = neighbors
                .iter()
                .map(|(_, i)| prefiltered_peaks[**i].intensity() as u64)
                .sum::<u64>();

            if neighbors.len() >= min_n && neighbor_intensity_total >= min_intensity {
                // Keep only the neighbors that are not already in a cluster
                neighbors = neighbors
                    .into_iter()
                    .filter(|(_, i)| match cluster_labels[**i] {
                        ClusterLabel::Cluster(_) => false,
                        _ => true,
                    })
                    .collect::<Vec<_>>();

                // Keep only the neighbors that are within the max extension distance
                // It might be worth setting a different max extension distance for the mz and mobility dimensions.
                neighbors = neighbors
                    .into_iter()
                    .filter(|(p, _)| {
                        let dist = (p.x - query_point.x).powi(2) + (p.y - query_point.y).powi(2);
                        dist <= MAX_EXTENSION_DISTANCE.powi(2)
                    })
                    .collect::<Vec<_>>();

                seed_set.extend(neighbors);
            }
        }
    }

    (cluster_id, cluster_labels)
}

pub fn dbscan(
    denseframe: &mut ms::DenseFrame,
    mz_scaling: f64,
    ims_scaling: f64,
    min_n: usize,
    min_intensity: u64,
) -> ms::DenseFrame {
    // I could pre-sort here and use the window iterator,
    // to pre-filter for points with no neighbors in the mz dimension.

    // AKA could filter out the points with no mz neighbors sorting
    // and the use the tree to filter points with no mz+mobility neighbors.

    // NOTE: the multiple quad isolation windows in DIA are not being handled just yet.
    let out_frame_type: timsrust::FrameType = denseframe.frame_type.clone();
    let out_rt: f64 = denseframe.rt.clone();
    let out_index: usize = denseframe.index.clone();

    let (quad_points, prefiltered_peaks, boundary) =
        denseframe_to_quadtree_points(denseframe, mz_scaling, ims_scaling, min_n.saturating_sub(1));

    let mut tree = RadiusQuadTree::new(boundary, 20, 1.);

    let quad_indices = (0..quad_points.len()).collect::<Vec<_>>();

    for (quad_point, i) in quad_points.iter().zip(quad_indices.iter()) {
        tree.insert(quad_point.clone(), i);
    }
    let mut intensity_sorted_indices = prefiltered_peaks
        .iter()
        .enumerate()
        .map(|(i, peak)| (i.clone(), peak.intensity.clone()))
        .collect::<Vec<_>>();

    intensity_sorted_indices.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

    let (cluster_id, cluster_labels) = _dbscan(
        &tree,
        &prefiltered_peaks,
        &quad_points,
        min_n,
        min_intensity,
        &intensity_sorted_indices,
    );
    // Each element is a tuple representing the summed cluster intensity, mz, and mobility.
    // And will be used to calculate the weighted average of mz and mobility AND the total intensity.
    let mut cluster_vecs = vec![(0u64, 0f64, 0f64); cluster_id as usize];
    for (point_index, cluster_label) in cluster_labels.iter().enumerate() {
        match cluster_label {
            ClusterLabel::Cluster(cluster_id) => {
                let cluster_idx = *cluster_id as usize - 1;
                let timspeak = prefiltered_peaks[point_index];
                let f64_intensity = timspeak.intensity as f64;
                let cluster_vec = &mut cluster_vecs[cluster_idx];
                cluster_vec.0 += timspeak.intensity as u64;
                cluster_vec.1 += (timspeak.mz as f64) * f64_intensity;
                cluster_vec.2 += (timspeak.mobility as f64) * f64_intensity;
            }
            _ => {}
        }
    }

    let denoised_peaks = cluster_vecs
        .iter_mut()
        .map(|(cluster_intensity, cluster_mz, cluster_mobility)| {
            let cluster_intensity = cluster_intensity; // Note not averaged
            let cluster_mz = *cluster_mz / *cluster_intensity as f64;
            let cluster_mobility = *cluster_mobility / *cluster_intensity as f64;
            ms::TimsPeak {
                intensity: u32::try_from(*cluster_intensity).ok().unwrap(),
                mz: cluster_mz,
                mobility: cluster_mobility as f32,
            }
        })
        .collect::<Vec<_>>();

    // TODO add an option to keep noise points

    ms::DenseFrame {
        raw_peaks: denoised_peaks,
        index: out_index,
        rt: out_rt,
        frame_type: out_frame_type,
        sorted: None,
    }
}
