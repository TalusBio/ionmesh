use log::warn;
use num_traits::AsPrimitive;

use std::collections::BTreeMap;
use std::ops::{Add, AddAssign, Mul};

// Needs to be odd
pub const NUM_LOCAL_CHROMATOGRAM_BINS: usize = 21;

#[derive(Debug, Clone)]
pub struct BTreeChromatogram {
    pub btree: BTreeMap<i32, u64>,
    pub rt_binsize: f32,
    pub rt_bin_offset: Option<f32>,
}

#[derive(Debug, Clone, Copy)]
pub struct ChromatogramArray<
    T: Mul<Output = T> + AddAssign + Default + AsPrimitive<f32>,
    const NBINS: usize,
> {
    pub chromatogram: [T; NBINS],
    pub rt_binsize: f32,
    pub rt_bin_offset: Option<f32>,
}

impl BTreeChromatogram {
    /// Create a new BTreeChromatogram with the given RT binsize and bin offset.
    ///
    /// The values in bin = 0 will be in the range [bin_offset, bin_offset + binsize)
    ///
    pub fn new(
        rt_binsize: f32,
        rt_bin_offset: f32,
    ) -> Self {
        BTreeChromatogram {
            btree: BTreeMap::new(),
            rt_binsize,
            rt_bin_offset: Some(rt_bin_offset),
        }
    }

    pub fn new_lazy(rt_binsize: f32) -> Self {
        BTreeChromatogram {
            btree: BTreeMap::new(),
            rt_binsize,
            rt_bin_offset: None,
        }
    }

    fn rt_to_bin(
        &self,
        rt: f32,
    ) -> i32 {
        ((rt - self.rt_bin_offset.unwrap()) / self.rt_binsize).floor() as i32
    }

    fn bin_to_rt(
        &self,
        bin: i32,
    ) -> f32 {
        (bin as f32 * self.rt_binsize) + self.rt_bin_offset.unwrap()
    }

    pub fn add(
        &mut self,
        rt: f32,
        intensity: u64,
    ) {
        let add_rt = rt + f32::EPSILON;
        if self.rt_bin_offset.is_none() {
            self.rt_bin_offset = Some(rt - (self.rt_binsize / 2.));
        }
        let bin = self.rt_to_bin(add_rt);
        let entry = self.btree.entry(bin).or_insert(0);
        *entry += intensity;
    }

    pub fn get_bin(
        &self,
        bin: &i32,
    ) -> Option<&u64> {
        self.btree.get(bin)
    }

    pub fn get_at_rt(
        &self,
        rt: f32,
    ) -> Option<&u64> {
        let bin = self.rt_to_bin(rt);
        self.btree.get(&bin)
    }

    fn rt_range(&self) -> Option<(f32, f32)> {
        let (min, max) = self.int_range()?;
        let _bo = self.rt_bin_offset.expect("Bin offset not set");
        Some((self.bin_to_rt(min), self.bin_to_rt(max)))
    }

    fn int_range(&self) -> Option<(i32, i32)> {
        let min = self.btree.keys().next();
        match min {
            Some(min) => {
                let max = *self.btree.keys().last().unwrap();
                Some((*min, max))
            },
            None => None,
        }
    }

    pub fn adopt(
        &mut self,
        other: &Self,
    ) {
        if self.rt_bin_offset.is_none() {
            self.rt_bin_offset = other.rt_bin_offset;
        }
        // Iterate over the other elements, convert back to RT
        // and add to self
        for (bin, intensity) in &other.btree {
            let other_rt = other.bin_to_rt(*bin);
            self.add(other_rt, *intensity);
        }
    }

    fn cosine_similarity(
        &self,
        other: &Self,
    ) -> Option<f32> {
        // Check that the bin size is almost the same
        let binsize_diff = (self.rt_binsize - other.rt_binsize).abs();
        if binsize_diff > 0.01 {
            return None;
        }

        // This would be the offset needed to align the two chromatograms
        // in terms of bins. In other words bin number 0 in self would
        // be bin number `other_vs_self_offset` in other.
        // This line will also return None if either of the chromatograms
        // has no bin offset set.
        let other_vs_self_offset =
            ((other.rt_bin_offset? - self.rt_bin_offset?) / self.rt_binsize) as i32;

        let (min, max) = self.int_range()?;
        let (min_o, max_o) = other.int_range()?;
        let min = min.max(min_o - other_vs_self_offset);
        let max = max.min(max_o - other_vs_self_offset);

        debug_assert!(min <= max);

        let mut dot = 0;
        let mut mag_a = 0;
        let mut mag_b = 0;
        for i in min..max {
            let a = *self.btree.get(&i).unwrap_or(&0);
            let b = *other.btree.get(&(i + other_vs_self_offset)).unwrap_or(&0);
            dot += a * b;
            mag_a += a * a;
            mag_b += b * b;
        }

        let mag_a = (mag_a as f32).sqrt();
        let mag_b = (mag_b as f32).sqrt();
        let cosine = dot as f32 / (mag_a * mag_b);
        Some(cosine)
    }

    fn total_intensity(&self) -> u64 {
        self.btree.values().sum()
    }

    pub fn as_chromatogram_array(
        &self,
        center_rt: Option<f32>,
    ) -> ChromatogramArray<f32, NUM_LOCAL_CHROMATOGRAM_BINS> {
        let mut chromatogram_arr = [0.; NUM_LOCAL_CHROMATOGRAM_BINS];

        let max_chr_arr_width = NUM_LOCAL_CHROMATOGRAM_BINS as f32 * self.rt_binsize;
        let curr_width = self.rt_range().unwrap().1 - self.rt_range().unwrap().0;

        // The chromatogram uses the bin size of the chromatogram btree
        // but re-centers it to the mean RT of the trace
        if !self.btree.is_empty() {
            let int_center =
                ((center_rt.unwrap_or(0.) - self.rt_bin_offset.unwrap()) / self.rt_binsize) as i32;
            let left_start = int_center - (NUM_LOCAL_CHROMATOGRAM_BINS / 2) as i32;

            for (i, item) in chromatogram_arr
                .iter_mut()
                .enumerate()
                .take(NUM_LOCAL_CHROMATOGRAM_BINS)
            {
                let bin = left_start + i as i32;
                *item = *self.btree.get(&bin).unwrap_or(&0) as f32;
            }
        }

        let out = ChromatogramArray {
            chromatogram: chromatogram_arr,
            rt_binsize: self.rt_binsize,
            rt_bin_offset: self.rt_bin_offset,
        };

        // Warn if the range is larger than the 2x width of the chromatogram
        // array
        if curr_width > max_chr_arr_width * 2. {
            warn!(
                "Warning: Chromatogram range is larger than 2x the width of the chromatogram array {} vs {} at RT: {}",
                curr_width,
                max_chr_arr_width,
                out.rt_bin_offset.unwrap());
            let arr_intensities = out.total_intensity();
            let btree_intensities = self.total_intensity() as f32;
            let ratio = arr_intensities / btree_intensities;
            warn!(
                "Array intensities: {}, Btree intensities: {}, Ratio: {}",
                arr_intensities, btree_intensities, ratio
            );
        }

        out
    }
}

impl<
        T: Mul<Output = T>
            + Add<Output = T>
            + AddAssign
            + Default
            + AsPrimitive<f32>
            + for<'a> std::iter::Sum<&'a T>,
        const NBINS: usize,
    > ChromatogramArray<T, NBINS>
{
    pub fn cosine_similarity(
        &self,
        other: &Self,
    ) -> Option<f32> {
        // Check that the bin size is almost the same
        let binsize_diff = (self.rt_binsize - other.rt_binsize).abs();
        if binsize_diff > 0.01 {
            return None;
        }

        // This would be the offset needed to align the two chromatograms
        // in terms of bins. In other words bin number 0 in self would
        // be bin number `other_vs_self_offset` in other.
        // This line will also return None if either of the chromatograms
        // has no bin offset set.
        let other_vs_self_offset =
            ((other.rt_bin_offset? - self.rt_bin_offset?) / self.rt_binsize) as i32;

        let mut dot = T::default();
        let mut mag_a = T::default();
        let mut mag_b = T::default();
        for i in 0..NBINS {
            let other_index = i + other_vs_self_offset as usize;
            if other_index >= other.chromatogram.len() {
                continue;
            }

            let a = self.chromatogram[i];
            let b = other.chromatogram[other_index];
            dot += a * b;
            mag_a += a * a;
            mag_b += b * b;
        }

        let mag_a: f32 = (mag_a.as_()).sqrt();
        let mag_b: f32 = (mag_b.as_()).sqrt();
        let cosine = dot.as_() / (mag_a * mag_b);
        Some(cosine)
    }

    pub fn total_intensity(&self) -> T {
        self.chromatogram.iter().sum()
    }
}

#[cfg(test)]
mod chromatogram_tests {
    use super::*;

    #[test]
    fn test_chromatogram() {
        let mut c = BTreeChromatogram::new(1., 0.);
        c.add(0., 1);
        c.add(1., 1);
        c.add(2., 1);
        c.add(2., 1);
        c.add(3., 1);

        assert_eq!(c.get_bin(&0), Some(&1));
        assert!(c.get_bin(&-1).is_none());
        let neg_one_rt = c.bin_to_rt(-1);
        assert!((-1.01..=-0.99).contains(&neg_one_rt));

        let mut c2 = BTreeChromatogram::new(1., 0.);
        c2.add(0., 1);
        c2.add(1., 1);
        c2.add(2., 1);
        c2.add(2., 1);
        c2.add(3., 1);

        let cosine = c.cosine_similarity(&c2).unwrap();
        let cosine = (cosine * 1000.).round() / 1000.;
        assert_eq!(cosine, 1.);

        c.add(4., 1);
        let cosine = c.cosine_similarity(&c2).unwrap();
        let cosine = (cosine * 1000.).round() / 1000.;
        assert_eq!(cosine, 1.);

        c.add(2., 20);
        let cosine = c.cosine_similarity(&c2).unwrap();
        assert!(cosine <= 0.9, "Cosine: {}", cosine);
    }

    #[test]
    fn test_chromatogram_array_cosine() {
        let mut c = ChromatogramArray::<i32, 5> {
            chromatogram: [0; 5],
            rt_binsize: 1.,
            rt_bin_offset: Some(0.),
        };
        c.chromatogram[0] = 1;
        c.chromatogram[1] = 1;
        c.chromatogram[2] = 1;
        c.chromatogram[3] = 1;
        c.chromatogram[4] = 1;

        let mut c2 = ChromatogramArray::<i32, 5> {
            chromatogram: [0; 5],
            rt_binsize: 1.,
            rt_bin_offset: Some(0.),
        };
        c2.chromatogram[0] = 1;
        c2.chromatogram[1] = 1;
        c2.chromatogram[2] = 1;
        c2.chromatogram[3] = 1;
        c2.chromatogram[4] = 1;

        let cosine = c.cosine_similarity(&c2).unwrap();
        let cosine = (cosine * 1000.).round() / 1000.;
        assert_eq!(cosine, 1.);

        c.chromatogram[4] = 20;
        let cosine = c.cosine_similarity(&c2).unwrap();
        assert!(cosine <= 0.9, "Cosine: {}", cosine);
    }

    #[test]
    fn test_diff_offsets() {
        let mut c2 = BTreeChromatogram::new(1., 1.1);

        // With bin offset of 1.1 and binsize 1.0, bin 0 is [1.1, 2.1)
        c2.add(0., 1);
        c2.add(1., 2);
        c2.add(2., 3); // 2 rt is bin 0
        c2.add(2., 4);
        c2.add(3., 5);

        assert_eq!(c2.btree.get(&0), Some(&7));
        assert_eq!(c2.btree.get(&-1), Some(&2));
    }

    #[test]
    fn test_diff_offset_merge() {
        let mut c = BTreeChromatogram::new(1., -0.45);
        // With bin offset of -0.45 and binsize 1.0, bin 0 is [-0.45, 0.55)

        c.add(0., 1);
        c.add(1., 2);
        c.add(2., 3);
        c.add(5., 5);

        let mut c2 = BTreeChromatogram::new(1., 1.55);
        // With bin offset of 1.55 and binsize 1.0, bin 0 is [1.55, 2.55)

        c2.add(0., 11);
        c2.add(1., 12);
        c2.add(2., 13);
        c2.add(3., 15);

        c.adopt(&c2);

        let rt_0_out = c.get_at_rt(0.);
        assert_eq!(rt_0_out, Some(&12));
        assert_eq!(c.get_bin(&0), Some(&12));

        assert_eq!(c.get_bin(&1), Some(&14));
        assert_eq!(c.get_bin(&2), Some(&16));
        assert_eq!(c.get_bin(&4), None);
        assert_eq!(c.get_bin(&5), Some(&5));
    }
}
