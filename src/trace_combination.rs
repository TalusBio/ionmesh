use crate::mod_types::Float;
use crate::tracing::BaseTrace;
use crate::utils::within_distance_apply;

/// This is an attempt to use dbscan ... again to cluster the traces into  pseudo-spectra (or peptides)
///
/// The idea is to first calculate the distances using a sliding window of the traces.
///     - The first distance will just be a generalized iou of the traces.
/// Once calculated it will implement an indexed space interface (that allows query)
///

/// This is an index that represent a sparse matrix of similarities between traces.
///
/// The main idea is that for a number of points N, the `similarities` Vec has length N.
/// Each element of `similarities` is a Vec of tuples of (index, similarity) where index
/// is the index of the other trace and the similarity is the similarity between the two traces.
///
/// Therefore, the entry for similarities[i], where there is an entry (w, s) shold also have
/// an entry in similarities[w] of (i, s)
struct TraceSimilarityIndex {
    similarities: Vec<Option<Vec<(usize, Float)>>>,
}
