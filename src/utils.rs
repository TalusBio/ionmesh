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
