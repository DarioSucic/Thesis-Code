
/// Returns a tuple containing the index and value of the minimum element
pub fn argmin<T: PartialOrd>(xs: &[T]) -> (usize, &T) {
    xs.iter()
        .enumerate()
        .min_by(|(_i, a), (_j, b)| a.partial_cmp(b).unwrap())
        .unwrap()
}

/// Returns a vector containing the indices of true elements
pub fn find_indices(xs: &[bool]) -> Vec<usize> {
    xs.iter()
        .enumerate()
        .filter(|(_i, &v)| v)
        .map(|(i, _v)| i)
        .collect()
}