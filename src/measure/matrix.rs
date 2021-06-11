use super::Measurable;
use crate::Float;

pub struct DissimilarityMatrix<T: Float> {
    data: Vec<T>,
    n_elements: usize,
}

impl<'a, T: Float> DissimilarityMatrix<T> {
    pub fn from_points<F>(points: &[(T, T)], f: F) -> Self
    where
        F: Fn((T, T), (T, T)) -> T,
    {
        let n_elements = points.len();
        let mut data = vec![T::zero(); n_elements.pow(2)];

        for (i, &a) in points.iter().enumerate() {
            for (j, &b) in points.iter().enumerate() {
                data[i * n_elements + j] = f(a, b);
            }
        }

        Self { data, n_elements }
    }

    pub fn from_flat(data: &[T], n_elements: usize) -> Self {
        // Can we make this work without a copy?
        let data = data.to_owned();
        Self { data, n_elements }
    }
}

impl<'a, T: Float> Measurable<T> for DissimilarityMatrix<T> {
    #[inline]
    fn measure(&self, i: usize, j: usize) -> T {
        debug_assert!(i < self.n_elements && j < self.n_elements);
        unsafe { *self.data.get_unchecked(i * self.n_elements + j) }
    }

    fn num_elements(&self) -> usize {
        self.n_elements
    }
}
