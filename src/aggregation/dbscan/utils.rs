use std::collections::BTreeMap;

pub struct FilterFunCache {
    cache: Vec<Option<BTreeMap<usize, bool>>>,
    tot_queries: u64,
    cached_queries: u64,
}

impl FilterFunCache {
    pub fn new(capacity: usize) -> Self {
        Self {
            cache: vec![None; capacity],
            tot_queries: 0,
            cached_queries: 0,
        }
    }

    pub fn get(
        &mut self,
        elem_idx: usize,
        reference_idx: usize,
    ) -> Option<bool> {
        self.tot_queries += 1;

        let out: Option<bool> = match self.cache[elem_idx] {
            Some(ref map) => match map.get(&reference_idx) {
                Some(x) => {
                    self.cached_queries += 1;
                    Some(*x)
                },
                None => None,
            },
            None => None,
        };
        out
    }

    pub fn set(
        &mut self,
        elem_idx: usize,
        reference_idx: usize,
        value: bool,
    ) {
        self.insert_both_ways(elem_idx, reference_idx, value);
    }

    fn insert_both_ways(
        &mut self,
        elem_idx: usize,
        reference_idx: usize,
        value: bool,
    ) {
        self.insert(elem_idx, reference_idx, value);
        self.insert(reference_idx, elem_idx, value);
    }

    fn insert(
        &mut self,
        elem_idx: usize,
        reference_idx: usize,
        value: bool,
    ) {
        match self.cache[elem_idx] {
            Some(ref mut map) => {
                _ = map.insert(reference_idx, value);
            },
            None => {
                let mut map = BTreeMap::new();
                map.insert(reference_idx, value);
                self.cache[elem_idx] = Some(map);
            },
        }
    }

    fn get_stats(&self) -> (u64, u64) {
        (self.tot_queries, self.cached_queries)
    }
}
