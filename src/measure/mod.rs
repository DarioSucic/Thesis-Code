use crate::Float;

pub mod matrix;

pub trait Measurable<T: Float> {
    /// Measure the dissimilarity between two elements in the collection
    ///
    /// Note that the measure does not have to be symmetric. i.e `measure(i, j) != measure(j, i)` is possible
    fn measure(&self, i: usize, j: usize) -> T;

    /// Return the number of elements in the collection
    fn num_elements(&self) -> usize;
}
