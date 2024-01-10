// This whole module is greatly inspired by how sage does quant.

// Percent of the retention time window to use for extraction.
// 1.0 is 100% just to be clear.
// This defines the whole width, chich means that half will be
// used before and half after the expected apex.
const EXTRACTION_WINDOW_PCT: f64 = 0.02;

// Number of bins in which the grid will be divided.
const NUM_BINS: usize = 20;
