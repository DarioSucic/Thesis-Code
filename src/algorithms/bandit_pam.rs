use super::*;

pub struct BanditPAM;

impl<T: BoundedNum> Solver<T> for BanditPAM {
    fn fit(d: &impl Measurable<T>, k: usize) -> Vec<usize> {
        fit(d, k)
    }
}

fn fit<T: BoundedNum>(d: &impl Measurable<T>, k: usize) -> Vec<usize> {
    // PERF: Improve
    let mut medoid_indices = vec![0; k];
    let mut previous = vec![0; k];

    build(d, &mut medoid_indices);

    const MAX_ITER: usize = 1000;

    for _iteration in 0..MAX_ITER {
        previous.copy_from_slice(&medoid_indices);
        swap(d, &mut medoid_indices);

        if medoid_indices.iter().eq(&previous) {
            break;
        }
    }

    medoid_indices
}

fn build<T: BoundedNum>(d: &impl Measurable<T>, medoid_indices: &mut [usize]) {
    let num_points = d.num_elements();
    let num_medoids = medoid_indices.len();

    for k in 0..num_medoids {
        let mut min_distance = T::max_value();
        let mut best = 0;

        for i in 0..num_points {
            let mut total = T::zero();

            for j in 0..num_points {
                let mut cost = d.measure(i, j);

                for &medoid_idx in &medoid_indices[0..k] {
                    let mcost = d.measure(medoid_idx, j);
                    if mcost < cost {
                        cost = mcost;
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

fn swap<T: BoundedNum>(d: &impl Measurable<T>, medoid_indices: &mut [usize]) {
    let num_points = d.num_elements();
    let num_medoids = medoid_indices.len();

    let mut min_distance = T::max_value();
    let mut best = 0;
    let mut medoid_to_swap = 0;

    for k in 0..num_medoids {
        for i in 0..num_points {
            let mut total = T::zero();

            for j in 0..num_points {
                let mut cost = d.measure(i, j);

                for medoid in 0..k {
                    if medoid == k {
                        continue;
                    }

                    let mcost = d.measure(medoid_indices[medoid], j);
                    if mcost < cost {
                        cost = mcost;
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
