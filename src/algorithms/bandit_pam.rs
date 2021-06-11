use std::{thread::current, vec};

use super::*;
use rand::prelude::*;

pub struct BanditPAM;

const BUILD_CONFIDENCE: usize = 1000;
const SWAP_CONFIDENCE: usize = 10_000;
const PRECISION: f64 = 0.001;
const BATCH_SIZE: usize = 100;
const MAX_ITER: usize = 1000;

impl<T: Float> Solver<T> for BanditPAM {
    fn fit(d: &impl Measurable<T>, k: usize) -> Vec<usize> {
        fit(d, k)
    }
}

fn fit<T: Float>(d: &impl Measurable<T>, k: usize) -> Vec<usize> {
    let mut medoid_indices = vec![0; k];
    let mut medoid_mat = vec![0; d.num_elements() * k];

    build(d, &mut medoid_indices, &mut medoid_mat);

    medoid_indices
}

fn build<T: Float>(d: &impl Measurable<T>, medoid_indices: &mut [usize], medoid_mat: &mut [usize]) {
    let num_elements = d.num_elements();
    let num_medoids = medoid_indices.len();

    let p = BUILD_CONFIDENCE * num_elements;
    let mut use_absolute = true;

    let mut estimates = vec![T::zero(); num_elements];
    let mut best_distances = vec![T::infinity(); num_elements];
    let mut sigma = vec![T::zero(); num_elements];
    let mut candidates = vec![true; num_elements];
    let mut lcbs = vec![T::zero(); num_elements];
    let mut ucbs = vec![T::zero(); num_elements];
    let mut t_samples = vec![0_usize; num_elements];
    let mut exact_mask = vec![false; num_elements];

    for k in 0..num_medoids {
        let mut step_count = 0;
        candidates.fill(true);
        t_samples.fill(0);
        exact_mask.fill(false);
        estimates.fill(T::zero());

        build_sigma(d, &mut best_distances, &mut sigma, use_absolute);

        // WARNING: LOGIC DIFFERENCE
        while candidates.iter().any(|&x| !x) {
            // TODO: Allocation
            let compute_exactly = t_samples
                .iter()
                .zip(&exact_mask)
                .map(|(&ts, &em)| ((ts + BATCH_SIZE) >= num_elements) != em)
                .collect::<Vec<bool>>();

            // Performance: O(n) for no reason
            if compute_exactly.contains(&true) {
                // PATTERN
                let targets = compute_exactly
                    .iter()
                    .enumerate()
                    .filter(|(_, &v)| v == true)
                    .map(|(i, _)| i)
                    .collect::<Vec<usize>>();

                let result = build_target(d, &targets, num_elements, &best_distances, use_absolute);

                for (&i, &r) in targets.iter().zip(&result) {
                    estimates[i] = r;
                    ucbs[i] = r;
                    lcbs[i] = r;
                    exact_mask[i] = true;
                    t_samples[i] += num_elements;
                    candidates[i] = false;
                }
            }

            // Optimization: Could move right after for-loop above?
            if candidates.iter().all(|&x| x) {
                break;
            }

            // PATTERN
            let targets = candidates
                .iter()
                .enumerate()
                .filter(|(_, &v)| v == true)
                .map(|(i, _)| i)
                .collect::<Vec<usize>>();

            let result = build_target(d, &targets, num_elements, &best_distances, use_absolute);

            let bs = T::from(BATCH_SIZE).unwrap();
            for (&i, &r) in targets.iter().zip(&result) {
                let ts = T::from(t_samples[i]).unwrap();
                estimates[i] = ts * estimates[i] + (r * bs) / (bs + ts);
            }

            for &i in &targets {
                t_samples[i] += BATCH_SIZE;
            }

            let adjust = T::from(p).unwrap().ln();

            for &i in &targets {
                let cb_delta = sigma[i] * (adjust / T::from(t_samples[i]).unwrap()).sqrt();
                ucbs[i] = estimates[i] + cb_delta;
                lcbs[i] = estimates[i] - cb_delta;
            }

            for i in 0..num_elements {
                let ucbs_min = *ucbs
                    .iter()
                    .min_by(|a, b| a.partial_cmp(b).unwrap())
                    .unwrap();
                candidates[i] = (lcbs[i] < ucbs_min) && !exact_mask[i];
            }

            step_count += 1;
        }

        let (lcbs_index_min, _lcbs_min) = lcbs
            .iter()
            .enumerate()
            .min_by(|(_i, a), (_j, b)| a.partial_cmp(b).unwrap())
            .unwrap();

        let medoid = lcbs_index_min;
        medoid_indices[k] = medoid;
        medoid_mat[k * num_elements] = medoid;

        for i in 0..num_elements {
            let cost = d.measure(i, medoid);
            if cost < best_distances[i] {
                best_distances[i] = cost;
            }
        }

        use_absolute = false;
        // Logging?
    }
}

fn stddev<T: Float>(xs: &[T]) -> T {
    let n = T::from(xs.len()).unwrap();

    let sum = xs.iter().fold(T::zero(), |acc, &x| acc + x);
    let mu = sum / n;

    (xs.iter().fold(T::zero(), |acc, &x| acc + (x - mu).powi(2)) / n).sqrt()
}

fn build_sigma<T: Float>(
    d: &impl Measurable<T>,
    best_distances: &mut Vec<T>,
    sigma: &mut Vec<T>,
    use_absolute: bool,
) {
    let n = d.num_elements();

    // Draw `batch_size` elements from [0, n-1] without replacement
    // Currently untested, but should work?
    let mut range = (0..n).collect::<Vec<usize>>();
    let (random_indices, _) = range.partial_shuffle(&mut thread_rng(), BATCH_SIZE);

    // Optimization: if batch_size > n, this overallocates
    let mut sample = vec![T::zero(); BATCH_SIZE];

    // Warning: Probably breaks if batch_size > n

    for i in 0..n {
        for j in 0..BATCH_SIZE {
            let random_idx = random_indices[j];
            let cost = d.measure(i, random_idx);

            // Optimization: Can the compiler reason about this?
            if use_absolute {
                sample[j] = cost;
            } else {
                let current_best = best_distances[random_idx];
                sample[j] = cost.min(current_best) - current_best;
            }
        }

        sigma[i] = stddev(&sample);
    }
}

fn build_target<T: Float>(
    d: &impl Measurable<T>,
    targets: &Vec<usize>,
    num_elements: usize,
    best_distances: &Vec<T>,
    use_absolute: bool,
) -> Vec<T> {
    let n = d.num_elements();

    let mut estimates = vec![T::zero(); n];

    // Draw `batch_size` elements from [0, n-1] without replacement
    // Currently untested, but should work?
    let mut range = (0..n).collect::<Vec<usize>>();
    let (random_indices, _) = range.partial_shuffle(&mut thread_rng(), BATCH_SIZE);

    for i in 0..n {
        let target_idx = targets[i];
        let mut total = T::zero();

        for j in 0..n {
            let random_idx = random_indices[j];
            let cost = d.measure(random_idx, target_idx);

            if use_absolute {
                total = total + cost;
            } else {
                let current_best = best_distances[random_idx];
                total = total + cost.min(current_best) - current_best;
            }
        }

        estimates[i] = total / T::from(BATCH_SIZE).unwrap();
    }

    estimates
}

fn swap<T: Float>(d: &impl Measurable<T>, medoid_indices: &mut [usize], assignments: &mut [usize]) {
    let num_elements = d.num_elements();
    let num_medoids = medoid_indices.len();
    let p = num_elements * num_medoids * SWAP_CONFIDENCE;

    let mut sigma = vec![T::zero(); num_elements * num_medoids];

    let mut best_distances = vec![T::infinity(); num_elements];
    let mut second_distances = vec![T::infinity(); num_elements];
    let mut estimates = vec![T::zero(); num_elements];
    let mut candidates = vec![true; num_elements];
    let mut lcbs = vec![T::zero(); num_elements];
    let mut ucbs = vec![T::zero(); num_elements];
    let mut t_samples = vec![0_usize; num_elements];
    let mut exact_mask = vec![false; num_elements];

    let mut iter = 0;
    let mut swap_performed = true;

    while swap_performed && iter < MAX_ITER {
        iter += 1;

        calc_best_distances_swap(
            d,
            medoid_indices,
            &mut best_distances,
            &mut second_distances,
            assignments,
        );

        swap_sigma(
            d,
            &mut sigma,
            &best_distances,
            &second_distances,
            assignments,
            num_medoids,
        );

        candidates.fill(true);
        exact_mask.fill(false);
        estimates.fill(T::zero());
        t_samples.fill(0);

        while candidates.iter().any(|&x| !x) {
            calc_best_distances_swap(
                d,
                medoid_indices,
                &mut best_distances,
                &mut second_distances,
                assignments,
            );

            let compute_exactly = t_samples
                .iter()
                .zip(&exact_mask)
                .map(|(&ts, &em)| ((ts + BATCH_SIZE) >= num_elements) != em);

            let targets = compute_exactly
                .enumerate()
                .filter(|&(_i, ce)| ce)
                .map(|(i, _ce)| i)
                .collect::<Vec<usize>>();

            if !targets.is_empty() {
                let result = swap_target(
                    d,
                    medoid_indices,
                    &targets,
                    num_elements,
                    &best_distances,
                    &second_distances,
                    assignments,
                );

                for (&i, &r) in targets.iter().zip(&result) {
                    estimates[i] = r;
                    ucbs[i] = r;
                    lcbs[i] = r;
                    exact_mask[i] = true;
                    t_samples[i] += num_elements;
                }

                for i in 0..num_elements {
                    let ucbs_min = *ucbs
                        .iter()
                        .min_by(|a, b| a.partial_cmp(b).unwrap())
                        .unwrap();
                    candidates[i] = (lcbs[i] < ucbs_min) && !exact_mask[i];
                }
            }

            
        }
    }
}

fn swap_target<T: Float>(
    d: &impl Measurable<T>,
    medoid_indices: &[usize],
    targets: &[usize],
    batch_size: usize,
    best_distances: &[T],
    second_distances: &[T],
    assignments: &[usize],
) -> Vec<T> {
    let num_elements = d.num_elements();
    let num_medoids = medoid_indices.len();

    let mut estimates = vec![T::zero(); targets.len()];

    let mut range = (0..num_elements).collect::<Vec<usize>>();
    let (random_indices, _) = range.partial_shuffle(&mut thread_rng(), batch_size);

    for i in 0..targets.len() {
        let mut total = T::zero();

        let n = targets[i] / num_medoids;
        let k = targets[i] % num_medoids;

        for j in 0..batch_size {
            let random_idx = random_indices[j];
            let cost = d.measure(n, random_idx);

            if k == assignments[random_idx] {
                total = total + cost.min(second_distances[random_idx]);
            } else {
                total = total + cost.min(best_distances[random_idx]);
            }

            total = total - best_distances[random_idx];
        }

        estimates[i] = total / T::from(random_indices.len()).unwrap();
    }

    estimates
}

fn calc_best_distances_swap<T: Float>(
    d: &impl Measurable<T>,
    medoid_indices: &[usize],
    best_distances: &mut [T],
    second_distances: &mut [T],
    assignments: &mut [usize],
) {
    let num_elements = d.num_elements();
    let num_medoids = medoid_indices.len();

    for i in 0..num_elements {
        let mut best = T::infinity();
        let mut second = T::infinity();

        for k in 0..num_medoids {
            let cost = d.measure(medoid_indices[k], i);

            if cost < best {
                assignments[i] = k;
                second = best;
                best = cost;
            } else if cost < second {
                second = cost;
            }
        }

        best_distances[i] = best;
        second_distances[i] = second;
    }
}

fn swap_sigma<T: Float>(
    d: &impl Measurable<T>,
    sigma: &mut [T],
    best_distances: &[T],
    second_distances: &[T],
    assignments: &[usize],
    num_medoids: usize,
) {
    let num_elements = d.num_elements();

    let mut range = (0..num_elements).collect::<Vec<usize>>();
    let (random_indices, _) = range.partial_shuffle(&mut thread_rng(), BATCH_SIZE);

    let mut sample = vec![T::zero(); BATCH_SIZE];

    for i in 0..num_elements * num_medoids {
        let n = i / num_medoids;
        let k = i % num_medoids;

        for j in 0..BATCH_SIZE {
            let random_idx = random_indices[j];
            let cost = d.measure(n, random_idx);

            if k == assignments[random_idx] {
                sample[j] = cost.min(second_distances[random_idx]);
            } else {
                sample[j] = cost.min(best_distances[random_idx]);
            }

            sample[j] = sample[j] - best_distances[random_idx];
        }

        sigma[k * num_elements + n] = stddev(&sample);
    }
}
