use std::{
    fs::File,
    io::{self, BufRead, BufReader},
    time::Duration,
};

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use pam::{
    algorithms::{BanditPAM, Solver, PAM},
    measure::{matrix::DissimilarityMatrix, Measurable},
};

// Move this at some point
fn l2_norm(a: (f64, f64), b: (f64, f64)) -> f64 {
    ((a.0 - b.0).powi(2) + (a.1 - b.1).powi(2)).sqrt()
}

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

pub fn criterion_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("algorithms");
    group.warm_up_time(Duration::from_secs_f64(0.5));
    group.measurement_time(Duration::from_secs_f64(2.0));
    group.sample_size(10);

    let sizes = [
        ("tiny", 2, 4),
        ("small", 4, 32),
        ("medium", 8, 256),
        // ("large" , 16, 2048),
        // ("huge"  , 32, 4096)
    ];

    macro_rules! bench_algorithms {
        [$( $alg:ident ),*] => {
            $(
                for &(size_name, num_clusters, num_elements) in &sizes {
                    let points = load_points(&format!("benches/test_data/{}", size_name))
                        .expect("Failed to load points");
                    let d = DissimilarityMatrix::from_points(&points, l2_norm);
                    let benchmark_id = BenchmarkId::from_parameter(&format!(
                        "{}/{}(k={},n={})",
                        stringify!($alg), size_name, num_clusters, num_elements
                    ));
                    group.bench_function(benchmark_id, |b| {
                        b.iter(|| black_box($alg::fit(&d, num_clusters)))
                    });
                }
            )*
        }
    }

    bench_algorithms![PAM, BanditPAM];
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
