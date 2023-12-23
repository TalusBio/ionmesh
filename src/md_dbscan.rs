
use crate::quad;


// Pseudocode from wikipedia
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

fn dbscan() {}