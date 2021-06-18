use std::fmt::Debug;

pub trait Float : num_traits::Float + Debug {}

impl Float for f64 {}
impl Float for f32 {}

pub mod algorithms;
pub mod measure;
pub mod utils;

#[cfg(test)]
mod tests {
    use std::fs::File;
    use std::io::{self, BufRead, BufReader};

    fn load_points(filename: &str) -> io::Result<Vec<(f64, f64)>> {
        let mut points = vec![];

        let file = File::open(filename)?;
        let reader = BufReader::new(file);

        for line in reader.lines() {
            let line = line?;
            let mut parts = line.split_ascii_whitespace();
            let a = parts.next().unwrap().parse::<f64>().unwrap();
            let b = parts.next().unwrap().parse::<f64>().unwrap();
            points.push((a, b));
        }

        Ok(points)
    }

    fn l2_norm(a: (f64, f64), b: (f64, f64)) -> f64 {
        let dx = a.0 - b.0;
        let dy = a.1 - b.1;
        (dx * dx + dy * dy).sqrt()
    }

    fn calc_loss(medoids: &[usize], points: &[(f64, f64)]) -> f64 {
        let dist_to_closest_medoid = |&point| {
            medoids
                .iter()
                .map(|&medoid| l2_norm(point, points[medoid]))
                .reduce(f64::min)
                .unwrap_or(0.0)
        };

        points.iter().map(dist_to_closest_medoid).sum()
    }

    #[test]
    fn measure_losses() {
        use crate::algorithms::{BanditPAM, Solver, PAM};
        use crate::measure::matrix::DissimilarityMatrix;

        let points = load_points("C:/Users/dario/thesis/code/BanditPAM/test_data").unwrap();

        let d = DissimilarityMatrix::from_points(&points, l2_norm);
        println!("Last point: {:?}", points.last().unwrap());

        println!("\n----------------------------------------------------------------------\n");

        macro_rules! test_algorithms {
            [$( $alg:ident ),*] => {
                $(
                    let mut medoids = $alg::fit(&d, 3);
                    medoids.sort();
                    println!("{:16} :: Medoids: {:3?},  Loss: {:16}", stringify!($alg), medoids, calc_loss(&medoids, &points));
                )*
            }
        }

        test_algorithms![PAM, BanditPAM];
        // test_algorithms![PAM];

        println!("\n----------------------------------------------------------------------\n");
    }
}
