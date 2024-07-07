use std::collections::BTreeMap;

pub struct FilterFunCache<'a> {
    cache: Vec<Option<BTreeMap<usize, bool>>>,
    filter_fun: Box<&'a dyn Fn(&usize, &usize) -> bool>,
    tot_queries: u64,
    cached_queries: u64,
}

impl<'a> FilterFunCache<'a> {
    pub fn new(filter_fun: Box<&'a dyn Fn(&usize, &usize) -> bool>, capacity: usize) -> Self {
        Self {
            cache: vec![None; capacity],
            filter_fun,
            tot_queries: 0,
            cached_queries: 0,
        }
    }

    pub fn get(&mut self, elem_idx: usize, reference_idx: usize) -> bool {
        // Get the value if it exists, call the functon, insert it and
        // return it if it doesn't.
        self.tot_queries += 1;

        let out: bool = match self.cache[elem_idx] {
            Some(ref map) => match map.get(&reference_idx) {
                Some(x) => {
                    self.cached_queries += 1;
                    *x
                }
                None => {
                    let out: bool = (self.filter_fun)(&elem_idx, &reference_idx);
                    self.insert(elem_idx, reference_idx, out);
                    self.insert(reference_idx, elem_idx, out);
                    out
                }
            },
            None => {
                let out = (self.filter_fun)(&elem_idx, &reference_idx);
                self.insert(elem_idx, reference_idx, out);
                self.insert(reference_idx, elem_idx, out);
                out
            }
        };
        out
    }

    fn insert(&mut self, elem_idx: usize, reference_idx: usize, value: bool) {
        match self.cache[elem_idx] {
            Some(ref mut map) => {
                _ = map.insert(reference_idx, value);
            }
            None => {
                let mut map = BTreeMap::new();
                map.insert(reference_idx, value);
                self.cache[elem_idx] = Some(map);
            }
        }
    }

    fn get_stats(&self) -> (u64, u64) {
        (self.tot_queries, self.cached_queries)
    }
}
