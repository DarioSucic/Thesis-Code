use super::*;

pub struct PAM;

impl<T: Float> Solver<T> for PAM {
    fn fit(d: &impl Measurable<T>, k: usize) -> Vec<usize> {
        fit(d, k)
    }
}

fn fit<T: Float>(d: &impl Measurable<T>, k: usize) -> Vec<usize> {
    let mut medoid_indices = vec![0; k];
    let mut previous = vec![0; k];

    build(d, &mut medoid_indices);

    const MAX_ITER: usize = 1000;

    for _iteration in 0..MAX_ITER {
        previous.copy_from_slice(&medoid_indices);
        swap(d, &mut medoid_indices);

        if medoid_indices == previous {
            break;
        }
    }

    medoid_indices
}

fn build<T: Float>(d: &impl Measurable<T>, medoid_indices: &mut [usize]) {
    let num_elements = d.num_elements();
    let num_medoids = medoid_indices.len();

    for k in 0..num_medoids {
        let mut min_distance = T::infinity();
        let mut best = 0;

        for i in 0..num_elements {
            let mut total = T::zero();

            for j in 0..num_elements {
                let mut cost = d.measure(i, j);

                for &medoid_idx in &medoid_indices[0..k] {
                    let current = d.measure(medoid_idx, j);
                    if current < cost {
                        cost = current;
                    }
                }

                total = total + cost;
            }

            if total < min_distance {
                min_distance = total;
                best = i;
            }
        }

        medoid_indices[k] = best;
    }
}

fn swap<T: Float>(d: &impl Measurable<T>, medoid_indices: &mut [usize]) {
    let num_elements = d.num_elements();
    let num_medoids = medoid_indices.len();

    let mut min_distance = T::infinity();
    let mut best = 0;
    let mut medoid_to_swap = 0;

    for k in 0..num_medoids {
        for i in 0..num_elements {
            let mut total = T::zero();

            for j in 0..num_elements {
                let mut cost = d.measure(i, j);

                for medoid in 0..num_medoids {

                    // Optimization opportunity:
                    // Instead of performing this check, we can keep an array
                    // of the current medoids in the `k` loop above.
                    if medoid == k {
                        continue;
                    }

                    let medoid_idx = medoid_indices[medoid];
                    let current = d.measure(medoid_idx, j);
                    if current < cost {
                        cost = current;
                    }
                }

                total = total + cost;
            }

            if total < min_distance {
                min_distance = total;
                best = i;
                medoid_to_swap = k;
            }
        }
    }

    medoid_indices[medoid_to_swap] = best;
}
