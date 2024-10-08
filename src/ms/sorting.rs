// From: https://github.com/mbhall88/psdm/blob/0c8c4be5e4a6d566193b688824197fac2d233108/src/lib.rs#L13-L41
// MIT licensed
trait SortExt<T> {
    fn argsort(&self) -> Vec<usize>;
    fn sort_by_indices(
        &mut self,
        indices: &mut Vec<usize>,
    );
}

impl<T: Ord + Clone> SortExt<T> for Vec<T> {
    fn argsort(&self) -> Vec<usize> {
        let mut indices = (0..self.len()).collect::<Vec<_>>();
        indices.sort_by_key(|&i| &self[i]);
        indices
    }

    fn sort_by_indices(
        &mut self,
        indices: &mut Vec<usize>,
    ) {
        for idx in 0..self.len() {
            if indices[idx] != usize::MAX {
                let mut current_idx = idx;
                loop {
                    let target_idx = indices[current_idx];
                    indices[current_idx] = usize::MAX;
                    if indices[target_idx] == usize::MAX {
                        break;
                    }
                    self.swap(current_idx, target_idx);
                    current_idx = target_idx;
                }
            }
        }
    }
}
#[cfg(test)]
mod test_argsort {
    use super::*;

    #[test]
    fn test_reorder_vec() {
        let mut vec1 = vec![4, 1, 3, 2, 5];
        let mut vec2 = vec!["p", "q", "r", "s", "t"];
        let mut inds = vec1.argsort();
        vec1.sort_by_indices(&mut inds.clone());
        vec2.sort_by_indices(&mut inds);
        assert_eq!(vec1, vec![1, 2, 3, 4, 5]);
        assert_eq!(vec2, vec!["q", "s", "r", "p", "t"]);
    }
}
