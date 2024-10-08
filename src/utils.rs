use std::cmp::Ordering;
use std::fmt::Debug;
use std::time::{
    Duration,
    Instant,
};

use log::{
    debug,
    info,
    trace,
    warn,
};
use num::cast::AsPrimitive;

pub struct ContextTimer {
    start: Instant,
    name: String,
    level: LogLevel,
    report_start: bool,
    pub cumtime: Duration,
}

#[derive(Debug, Clone, Copy)]
pub enum LogLevel {
    INFO,
    DEBUG,
    TRACE,
}

impl ContextTimer {
    pub fn new(
        name: &str,
        report_start: bool,
        level: LogLevel,
    ) -> ContextTimer {
        let out = ContextTimer {
            start: Instant::now(),
            name: name.to_string(),
            level,
            report_start,
            cumtime: Duration::new(0, 0),
        };
        if report_start {
            out.start_msg();
        }
        out
    }

    pub fn reset_start(&mut self) {
        self.start = Instant::now();
    }

    fn start_msg(&self) {
        match self.level {
            LogLevel::INFO => info!("Started: '{}'", self.name),
            LogLevel::DEBUG => debug!("Started: '{}'", self.name),
            LogLevel::TRACE => trace!("Started: '{}'", self.name),
        }
    }

    pub fn stop(
        &mut self,
        report: bool,
    ) -> Duration {
        let duration = self.start.elapsed();
        self.cumtime += duration;
        if report {
            self.report();
        }
        duration
    }

    pub fn report(&self) {
        let duration_us = self.cumtime.as_micros() as f64;
        match self.level {
            // Time to get comfortable writting macros??
            LogLevel::INFO => info!(
                "Time elapsed in '{}' is: {:.02}s",
                self.name,
                duration_us / 1000000.
            ),
            LogLevel::DEBUG => debug!(
                "Time elapsed in '{}' is: {:.02}s",
                self.name,
                duration_us / 1000000.
            ),
            LogLevel::TRACE => trace!(
                "Time elapsed in '{}' is: {:.02}s",
                self.name,
                duration_us / 1000000.
            ),
        }
    }

    pub fn start_sub_timer(
        &self,
        name: &str,
    ) -> ContextTimer {
        ContextTimer::new(
            &format!("{}::{}", self.name, name),
            self.report_start,
            self.level,
        )
    }
}

/// Applies a function to all elements within a certain distance of each element.
///
/// Provided a slice of elements (assumed to be sorted by the key function),
/// a key function. For every element in the slice, a function will be applied
/// with the indices of the first and last element within the distance of the
/// current element.
pub fn within_distance_apply<T, R: Clone, W>(
    elems: &[T],
    key: &dyn Fn(&T) -> R,
    max_dist: &R,
    out_func: &dyn Fn(&usize, &usize) -> W,
) -> Vec<W>
where
    R: PartialOrd + Copy + std::ops::Sub<Output = R> + std::ops::Add<Output = R> + Default,
    T: Copy,
    W: Default + Copy,
{
    // TODO: rename all internal variables ... they made sense before this
    // was a generic function.
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

    let max_dist = *max_dist;
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
            within_distance_apply(&elems, &|x| *x, &1.1, &|i_right, i_left| {
                (i_right - i_left) >= 3
            });

        assert_eq!(prefiltered_peaks_bool, vec![false, true, true, true, false]);
    }
}

// Initial implementaition was based on the formulas here:
// https://en.wikipedia.org/wiki/Bessel%27s_correction
// Donate to wikipedia!!! <3

// Final implementation is based on the formulas here:
// https://doi.org/10.1007/s00180-015-0637-z
// and the c implementation here:
// https://www.johndcook.com/blog/skewness_kurtosis/

/// Calculate the variance of a set of data points without storing them.
///
/// Generic Types:
///
/// 1. *T* - The type of the data points. *NOTE* this does not have to
/// be the same type as the desired output type, but rather the representation
/// of the data points while being aggregated.
///
/// 2. *W* - The type of the weights.
#[derive(Debug, Default, Clone, Copy)]
pub struct RollingSDCalculator<T, W> {
    n: u64,
    w_sum: W,
    sqare_w_sum: W,
    m1: T,
    m2: T,
    m3: T,
    m4: T,
    min: Option<T>,
    max: Option<T>,
}

// TODO evaluate the numeric accuracy of this implementation.

impl<T, W> RollingSDCalculator<T, W>
where
    // NOTE: W > T needs to be 'losless' (u32 -> f64)
    // but T > W does not need to be.
    T: std::ops::Sub<Output = T>
        + std::ops::Mul<Output = T>
        + std::ops::Div<Output = T>
        + std::ops::Add<Output = T>
        + std::ops::AddAssign
        + std::cmp::PartialOrd
        + Into<f64>
        + Copy
        + Default
        + 'static,
    W: std::ops::Sub<Output = W>
        + std::ops::Mul<Output = W>
        + std::ops::Div<Output = W>
        + std::ops::Add<Output = W>
        + std::ops::AddAssign
        + std::cmp::PartialOrd
        + Copy
        + Default
        + AsPrimitive<T>
        + 'static,
    u64: AsPrimitive<T>,
    f64: AsPrimitive<T>,
{
    pub fn add(
        &mut self,
        x: T,
        w: W,
    ) {
        // Check for overflows
        self.merge(&Self {
            n: 1,
            w_sum: w,
            sqare_w_sum: w * w,
            m1: x,
            m2: 0.as_(),
            m3: 0.as_(),
            m4: 0.as_(),
            min: Some(x),
            max: Some(x),
        });
    }

    pub fn get_variance(&self) -> T {
        self.m2 / self.w_sum.as_()
    }

    pub fn get_sd(&self) -> T {
        self.get_variance().into().sqrt().as_()
    }

    pub fn get_variance_bessel(&self) -> T {
        // Mod here to use the avg weight
        // instead of 1 for the correction
        // self.s_ / (self.w_sum - T::one())
        let avg_weight = self.w_sum.as_() / self.n.as_();
        self.m2 / (self.w_sum.as_() - avg_weight)
    }

    pub fn get_variance_reliability(&self) -> T {
        self.m2 / (self.w_sum.as_() - (self.sqare_w_sum.as_() / self.w_sum.as_()))
    }

    pub fn get_mean(&self) -> T {
        self.m1
    }

    pub fn get_skew(&self) -> T {
        if self.m2 == 0.as_() {
            return 0.as_();
        }
        self.w_sum.as_().into().sqrt().as_() * self.m3 / (self.m2.into().powf(1.5).as_())
    }

    pub fn get_kurtosis(&self) -> T {
        self.w_sum.as_() * self.m4 / (self.m2 * self.m2)
    }

    pub fn get_min(&self) -> Option<T> {
        self.min
    }

    pub fn get_max(&self) -> Option<T> {
        self.max
    }

    pub fn merge(
        &mut self,
        other: &Self,
    ) {
        // There is for sure some optimization to be done here.
        // But right now the math is the hard part ...  would definitely pay off
        let a = *self;
        let b = other;

        let combined_n = a.n + b.n;
        let combined_weight = a.w_sum + b.w_sum;

        let combined_weight_as = combined_weight.as_();
        let a_weight = a.w_sum.as_();
        let b_weight = b.w_sum.as_();

        let delta = b.m1 - a.m1;
        let delta2 = delta * delta;
        let delta3 = delta * delta2;
        let delta4 = delta2 * delta2;

        let combined_m1 = (a_weight * a.m1 + b_weight * b.m1) / combined_weight_as;
        let combined_m2 = a.m2 + b.m2 + delta2 * a_weight * b_weight / combined_weight_as;

        let mut combined_m3 = a.m3
            + b.m3
            + delta3 * a_weight * b_weight * (a_weight - b_weight)
                / (combined_weight_as * combined_weight_as);
        combined_m3 += 3.0.as_() * delta * (a_weight * b.m2 - b_weight * a.m2) / combined_weight_as;

        let mut combined_m4 = a.m4
            + b.m4
            + delta4
                * a_weight
                * b_weight
                * (a_weight * a_weight - a_weight * b_weight + b_weight * b_weight)
                / (combined_weight_as * combined_weight_as * combined_weight_as);
        combined_m4 +=
            6.0.as_() * delta2 * (a_weight * a_weight * b.m2 + b_weight * b_weight * a.m2)
                / (combined_weight_as * combined_weight_as)
                + 4.0.as_() * delta * (a_weight * b.m3 - b_weight * a.m3) / combined_weight_as;

        let mut min_use = a.min;
        if b.min < min_use || min_use.is_none() {
            min_use = b.min;
        }

        let mut max_use = a.max;
        if b.max > max_use || max_use.is_none() {
            max_use = b.max;
        }

        self.min = min_use;
        self.max = max_use;
        self.n = combined_n;
        self.w_sum = combined_weight;
        self.sqare_w_sum = a.sqare_w_sum + b.sqare_w_sum;
        self.m1 = combined_m1;
        self.m2 = combined_m2;
        self.m3 = combined_m3;
        self.m4 = combined_m4;
    }
}

#[derive(Debug, Clone, Copy)]
pub struct Stats {
    pub mean: f64,
    pub sd: f64,
    pub skew: f64,
    pub kurtosis: f64,
    pub n: u64,
    pub min: Option<f64>,
    pub max: Option<f64>,
}

pub fn get_stats(data: &[f64]) -> Stats {
    let mut sd_calc = RollingSDCalculator::<f64, u64>::default();
    for x in data.iter() {
        sd_calc.add(*x, 1);
    }
    Stats {
        mean: sd_calc.get_mean(),
        sd: sd_calc.get_sd(),
        skew: sd_calc.get_skew(),
        kurtosis: sd_calc.get_kurtosis(),
        n: sd_calc.n,
        min: sd_calc.get_min(),
        max: sd_calc.get_max(),
    }
}

/// This has been shamelessly copied and very minorly modified from sage.
/// https://github.com/lazear/sage/blob/93a9a8a7c9f717238fc6c582c0dd501a56159be7/crates/sage/src/database.rs#L498
/// Althought it really feels like this should be in the standard lib.
///
/// Usage:
/// ```rust
/// use ionmesh::utils::binary_search_slice;
/// let data: [f64; 11]= [1.0, 1.5, 1.5, 1.5, 1.5, 2.0, 2.5, 3.0, 3.0, 3.5, 4.0];
/// let (left, right) = binary_search_slice(&data, |a: &f64, b| a.total_cmp(b), 1.5, 3.25);
/// assert!(data[left] == 1.5);
/// assert!(data[right] > 3.25);
/// assert_eq!(
///     &data[left..right],
///     &[1.5, 1.5, 1.5, 1.5, 2.0, 2.5, 3.0, 3.0]
/// );
/// let empty: [f64; 0] = [];
/// let (left, right) = binary_search_slice(&empty, |a: &f64, b| a.total_cmp(b), 1.5, 3.25);
/// assert_eq!(left, 0);
/// assert_eq!(right, 0);
/// let (left, right) = binary_search_slice(&data, |a: &f64, b| a.total_cmp(b), -100., -99.);
/// assert_eq!(left, 0);
/// assert_eq!(right, 0);
/// assert_eq!(&data[left..right], &empty);
/// let (left, right) = binary_search_slice(&data, |a: &f64, b| a.total_cmp(b), 100., 101.);
/// assert_eq!(left, data.len());
/// assert_eq!(right, data.len());
/// assert_eq!(&data[left..right], &empty);
/// let data: [f64; 7]= [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0];
/// let (left, right) = binary_search_slice(&data, |a: &f64, b| a.total_cmp(b), 1.5, 3.25);
/// assert!(data[left] == 1.5);
/// assert!(data[right] > 3.25);
/// assert!(data[right-1] < 3.25);
/// assert_eq!(
///     &data[left..right],
///     &[1.5, 2.0, 2.5, 3.0]
/// );
/// ```
///
#[inline]
pub fn binary_search_slice<T, F, S>(
    slice: &[T],
    key: F,
    low: S,
    high: S,
) -> (usize, usize)
where
    F: Fn(&T, &S) -> Ordering,
    T: Debug,
{
    let left_idx = match slice.binary_search_by(|a| key(a, &low)) {
        Ok(mut idx) | Err(mut idx) => {
            if idx == slice.len() {
                // This is very non-elegant ... pretty sure I need to split
                // the ok-err cases to make a more elegant solution.
                return (idx, idx);
            }
            let mut any_nonless = false;
            while idx != 0 && key(&slice[idx], &low) != Ordering::Less {
                any_nonless = true;
                idx -= 1;
            }
            if any_nonless {
                idx = idx.saturating_add(1);
            }
            idx
        },
    };

    let right_idx = match slice[left_idx..].binary_search_by(|a| key(a, &high)) {
        Ok(idx) | Err(idx) => {
            let mut idx = idx + left_idx;
            while idx < slice.len() && key(&slice[idx], &high) != Ordering::Greater {
                idx = idx.saturating_add(1);
            }
            idx.min(slice.len())
        },
    };
    if cfg!(debug_assertions) {
        // This makes sure the slice is indexable by the indices.
        let _foo = &slice[left_idx..right_idx];
    };
    (left_idx, right_idx)
}

/// Serializes to json the object if debug assertions are
/// enabled and an env variable with the frequency is set.
/// the env variable should be named `IONMESH_DEBUG_JSON_FREQUENCY`
/// Also derive the bath to ave to from the env variable `IONMESH_DEBUG_JSON_PATH`
/// which is created if it does not exist.
/// The object is serialized to a file named `{name}.json`
pub fn maybe_save_json_if_debugging<T>(
    obj: &T,
    name: &str,
    force: bool,
) -> bool
where
    T: serde::Serialize,
{
    if cfg!(debug_assertions) {
        let freq = std::env::var("IONMESH_DEBUG_JSON_FREQUENCY");
        if let Ok(freq) = freq {
            let freq = freq.parse::<usize>().unwrap();
            if (force || (freq > 0)) && (force || (rand::random::<usize>() % freq == 0)) {
                let json = serde_json::to_string_pretty(obj).unwrap();
                let path = std::env::var("IONMESH_DEBUG_JSON_PATH");
                let path = if let Ok(path) = path {
                    if !std::path::Path::new(&path).exists() {
                        std::fs::create_dir_all(&path).unwrap();
                    }
                    std::path::Path::new(&path).join(format!("{}.json", name))
                } else {
                    warn!("IONMESH_DEBUG_JSON_PATH not set, saving to current directory");
                    std::path::Path::new(".").join(format!("{}.json", name))
                };
                info!("Saving json to {:?}", path);

                std::fs::write(path, json).unwrap();
                return true;
            }
        }
    }
    false
}

#[cfg(test)]
mod test_rolling_sd {
    use super::*;

    // All of these should have:
    // - population variance of 10.0
    // - Mean of 9.0
    //
    //  skews kurtosis
    //   0.000 1.780
    const ASCOMBES_QX: [f64; 11] = [10., 8., 13., 9., 11., 14., 6., 4., 12., 7., 5.];
    //   2.846 9.100
    const ASCOMBES_QY: [f64; 11] = [8., 8., 8., 8., 8., 8., 8., 19., 8., 8., 8.];

    // All of these should have:
    // - population variance of 3.75 +- 0.01
    // - Mean of 7.5
    //
    //  skews kurtosis
    //  -0.055 2.179
    const ASCOMBES_Q1: [f64; 11] = [
        8.04, 6.95, 7.58, 8.81, 8.33, 9.96, 7.24, 4.26, 10.84, 4.82, 5.68,
    ];
    //  -1.129 3.007
    const ASCOMBES_Q2: [f64; 11] = [
        9.14, 8.14, 8.74, 8.77, 9.26, 8.10, 6.13, 3.10, 9.13, 7.26, 4.74,
    ];
    //   1.592 5.130
    const ASCOMBES_Q3: [f64; 11] = [
        7.46, 6.77, 12.74, 7.11, 7.81, 8.84, 6.08, 5.39, 8.15, 6.42, 5.73,
    ];
    //   1.293 4.390
    const ASCOMBES_Q5: [f64; 11] = [
        6.58, 5.76, 7.71, 8.84, 8.47, 7.04, 5.25, 12.50, 5.56, 7.91, 6.89,
    ];

    fn assert_close(
        a: f64,
        b: f64,
    ) {
        assert!((a - b).abs() < 1e-3, "{} != {}", a, b);
    }

    #[test]
    fn test_rolling_sd() {
        let mut sd_calc = RollingSDCalculator::<f64, u64>::default();

        sd_calc.add(1.0, 1);
        sd_calc.add(1.0, 1);

        assert_eq!(sd_calc.get_mean(), 1.0);
        assert_eq!(sd_calc.get_variance(), 0.);
        assert_eq!(sd_calc.get_variance_bessel(), 0.);
        assert_eq!(sd_calc.get_variance_reliability(), 0.);

        let mut sd_calc = RollingSDCalculator::<f64, u64>::default();
        sd_calc.add(1.0, 1);
        sd_calc.add(0.0, 1);
        assert_eq!(sd_calc.get_mean(), 0.5);

        let mut sd_calc = RollingSDCalculator::<f64, u64>::default();
        sd_calc.add(1.0, 200);
        sd_calc.add(0.0, 200);
        assert_eq!(sd_calc.get_mean(), 0.5);

        let mut sd_calc = RollingSDCalculator::<f64, u64>::default();
        sd_calc.add(1.0, 1);
        sd_calc.add(0.0, 2);
        assert_close(sd_calc.get_mean(), 1. / 3.);

        let mut sd_calc = RollingSDCalculator::<f64, u64>::default();
        sd_calc.add(1.0, 1);
        sd_calc.add(0.0, 2);
        assert_close(sd_calc.get_mean(), 1. / 3.);

        let mut sd_calc = RollingSDCalculator::<f64, u64>::default();
        for x in ASCOMBES_Q1.iter() {
            sd_calc.add(*x, 1);
        }
        assert_close(sd_calc.get_mean(), 7.50);
        assert_close(sd_calc.get_variance(), 3.752);
        assert_close(sd_calc.get_skew(), -0.055);
        assert_close(sd_calc.get_kurtosis(), 2.179);

        // Now using ascombes Q4 to check for weighted values
        let mut sd_calc = RollingSDCalculator::<f64, u64>::default();
        sd_calc.add(8.0, 1000);
        sd_calc.add(19.0, 100);
        assert_close(sd_calc.get_mean(), 9.0);
        assert_close(sd_calc.get_variance(), 10.0);

        assert_close(sd_calc.max.unwrap(), 19.0);
        assert_close(sd_calc.min.unwrap(), 8.0);

        // Now using ascombes QY to check for weighted values
        // With respect to last, the weight type should not influence the result
        let mut sd_calc = RollingSDCalculator::<f64, f64>::default();
        sd_calc.add(8.0, 1000.);
        sd_calc.add(19.0, 100.);
        assert_close(sd_calc.get_mean(), 9.0);
        assert_close(sd_calc.get_variance(), 10.0);
    }
}
