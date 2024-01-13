use num::cast::AsPrimitive;

pub fn within_distance_apply<T, R, W>(
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
            within_distance_apply(&elems, &|x| *x, 1.1, &|i_right, i_left| {
                (i_right - i_left) >= 3
            });

        assert_eq!(prefiltered_peaks_bool, vec![false, true, true, true, false]);
    }
}

// def online_variance(data):
//     n = 0
//     mean = 0.0
//     M2 = 0.0
//
//     for x in data:
//         n += 1
//         delta = x - mean
//         mean += delta/n
//         M2 += delta*(x - mean)
//
//     if n < 2:
//         return float('nan')
//     else:
//         return M2 / (n - 1)

// https://en.wikipedia.org/wiki/Bessel%27s_correction
// Donate to wikipedia!!! <3
//
// def weighted_incremental_variance(data_weight_pairs):
//     w_sum = w_sum2 = mean = S = 0
//
//     for x, w in data_weight_pairs:
//         w_sum = w_sum + w
//         w_sum2 = w_sum2 + w**2
//         mean_old = mean
//         mean = mean_old + (w / w_sum) * (x - mean_old)
//         S = S + w * (x - mean_old) * (x - mean)
//
//     population_variance = S / w_sum
//     # Bessel's correction for weighted samples
//     # Frequency weights
//     sample_frequency_variance = S / (w_sum - 1)
//     # Reliability weights
//     sample_reliability_variance = S / (w_sum - w_sum2 / w_sum)


/// Calculate the variance of a set of data points without storing them.
/// 
/// Generic Types:
/// 
/// 1. *T* - The type of the data points. *NOTE* this does not have to
/// be the same type as the desired output type, but rather the representation
/// of the data points while being aggregated.
/// 
/// 2. *W* - The type of the weights.
#[derive(Debug, Default, Clone)]
pub struct RollingSDCalculator<T, W> {
    w_sum: W,
    w_sum2: W,
    mean: T,
    s_: T,
    count: u64,
}

// TODO evaluate the numeric accuracy of this implementation.

impl<T, W> RollingSDCalculator<T, W>
where
// NOTE: W > T needs to be 'losless' (u32 -> f63)
// but T > W does not need to be.
    T: std::ops::Sub<Output = T>
        + std::ops::Mul<Output = T>
        + std::ops::Div<Output = T>
        + std::ops::Add<Output = T>
        + std::ops::AddAssign
        + std::cmp::PartialOrd
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
{
    pub fn add(&mut self, x: T, w: W) {
        // Check for overflows
        debug_assert!(self.w_sum < self.w_sum + w);
        debug_assert!(self.w_sum2 < self.w_sum2 + w * w);
        self.w_sum += w;
        self.w_sum2 += w * w;
        let mean_old = self.mean;
        self.mean = mean_old + ((w.as_() / self.w_sum.as_()) * (x - mean_old));
        self.s_ += w.as_() * (x - mean_old) * (x - self.mean);
        self.count += 1;
    }

    pub fn get_variance(&self) -> T {
        self.s_ / self.w_sum.as_()
    }

    pub fn get_variance_bessel(&self) -> T {
        // Mod here to use the avg weight
        // instead of 1 for the correction
        // self.s_ / (self.w_sum - T::one())
        let avg_weight  = self.w_sum.as_() / self.count.as_();
        self.s_ / (self.w_sum.as_() - avg_weight)
    }

    pub fn get_variance_reliability(&self) -> T {
        self.s_ / (self.w_sum.as_() - (self.w_sum2.as_() / self.w_sum.as_()))
    }

    pub fn get_mean(&self) -> T {
        self.mean
    }
}

#[cfg(test)]
mod test_rolling_sd {
    use super::*;

    #[test]
    fn test_rolling_sd() {
        let mut sd_calc = RollingSDCalculator::<f64, u64>::default();

        sd_calc.add(1.0, 1);
        sd_calc.add(2.0, 1);
        sd_calc.add(3.0, 1);
        sd_calc.add(4.0, 1);
        sd_calc.add(5.0, 1);

        assert_eq!(sd_calc.get_variance(), 2.5);
        assert_eq!(sd_calc.get_variance_bessel(), 2.0);
        assert_eq!(sd_calc.get_variance_reliability(), 2.0);
    }

    #[test]
    fn test_rolling_constants_addition(){
        let mut sd_calc = RollingSDCalculator::<f64, u64>::default();

        sd_calc.add(2.0, 1);
        sd_calc.add(2.0, 1);
        sd_calc.add(2.0, 1);
        sd_calc.add(2.0, 1);
        sd_calc.add(2.0, 1);

        assert!(sd_calc.mean == 2.0);
        assert_eq!(sd_calc.get_variance(), 0.0);
        assert_eq!(sd_calc.get_variance_bessel(), 0.0);
        assert_eq!(sd_calc.get_variance_reliability(), 0.0);
    }
}