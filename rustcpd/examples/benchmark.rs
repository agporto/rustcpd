use rustcpd::{
    AtlasConfig, AtlasRegistration, DMatrix, DeformableConfig, DeformableRegistration, EmConfig,
    RigidConfig, RigidRegistration, gaussian_kernel,
};
use std::{env, hint::black_box, time::Instant};

fn cloud(count: usize) -> DMatrix<f64> {
    DMatrix::from_fn(count, 3, |i, j| {
        let z = i as f64 + 1.0;
        match j {
            0 => (z * 0.037).sin() * 1.7 + 0.0001 * z,
            1 => (z * 0.023).cos() * 0.9,
            _ => (z * 0.011).sin() * (z * 0.007).cos(),
        }
    })
}
fn rms(a: &DMatrix<f64>, b: &DMatrix<f64>) -> f64 {
    (a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).powi(2))
        .sum::<f64>()
        / a.len() as f64)
        .sqrt()
}
fn median(mut values: Vec<f64>) -> f64 {
    values.sort_by(f64::total_cmp);
    values[values.len() / 2]
}

fn main() {
    let args: Vec<_> = env::args().collect();
    let method = args.get(1).map(String::as_str).unwrap_or("rigid");
    let n = args.get(2).and_then(|v| v.parse().ok()).unwrap_or(300);
    let iterations = args.get(3).and_then(|v| v.parse().ok()).unwrap_or(10);
    let repeats = args.get(4).and_then(|v| v.parse().ok()).unwrap_or(7);
    let y = cloud(n);
    let mut times = Vec::with_capacity(repeats);
    let mut error = 0.0;
    let single_precision = method.ends_with("_f32");
    let method = method.strip_suffix("_f32").unwrap_or(method);
    let sparse = method.ends_with("_sparse");
    let method = method.strip_suffix("_sparse").unwrap_or(method);
    for _ in 0..repeats {
        let start = Instant::now();
        match method {
            "rigid" => {
                let angle: f64 = 0.12;
                let r = DMatrix::from_row_slice(
                    3,
                    3,
                    &[
                        angle.cos(),
                        -angle.sin(),
                        0.,
                        angle.sin(),
                        angle.cos(),
                        0.,
                        0.,
                        0.,
                        1.,
                    ],
                );
                let x = DMatrix::from_fn(n, 3, |i, j| {
                    1.04 * (0..3).map(|q| y[(i, q)] * r[(q, j)]).sum::<f64>()
                        + [0.12, -0.08, 0.04][j]
                });
                let result = RigidRegistration::new(
                    &x,
                    &y,
                    RigidConfig {
                        em: EmConfig {
                            max_iterations: iterations,
                            tolerance: 0.,
                            k: sparse.then_some(10),
                            single_precision,
                            ..Default::default()
                        },
                        scale: true,
                        normalize: false,
                    },
                )
                .unwrap()
                .register()
                .unwrap();
                error = rms(&result.points, &x);
                black_box(result);
            }
            "deformable" | "deformable_lowrank" => {
                let g = gaussian_kernel(&y, &y, 1.5).unwrap();
                let w =
                    DMatrix::from_fn(n, 3, |i, j| 0.003 * ((i * 3 + j + 1) as f64 * 0.041).sin());
                let x = &y + g * w;
                let result = DeformableRegistration::new(
                    &x,
                    &y,
                    DeformableConfig {
                        em: EmConfig {
                            max_iterations: iterations,
                            tolerance: 0.,
                            k: sparse.then_some(10),
                            single_precision,
                            ..Default::default()
                        },
                        alpha: 2.,
                        beta: 1.5,
                        low_rank: (method == "deformable_lowrank").then_some(300),
                        ..Default::default()
                    },
                )
                .unwrap()
                .register()
                .unwrap();
                error = rms(&result.points, &x);
                black_box(result);
            }
            "atlas" => {
                let rank = 12;
                let modes = DMatrix::from_fn(y.len(), rank, |i, k| {
                    0.01 * ((i + 1 + k * 7) as f64 * 0.019).sin()
                });
                let coefficients: Vec<_> = (0..rank)
                    .map(|k| 0.2 * ((k + 1) as f64 * 0.7).sin())
                    .collect();
                let x = DMatrix::from_fn(n, 3, |i, j| {
                    y[(i, j)]
                        + (0..rank)
                            .map(|k| modes[(i * 3 + j, k)] * coefficients[k])
                            .sum::<f64>()
                });
                let result = AtlasRegistration::new(
                    &x,
                    &y,
                    &modes,
                    AtlasConfig {
                        em: EmConfig {
                            max_iterations: iterations,
                            tolerance: 0.,
                            k: sparse.then_some(10),
                            single_precision,
                            ..Default::default()
                        },
                        eigenvalues: (0..rank).map(|k| 1. / (k + 1) as f64).collect(),
                        lambda_regularization: 0.1,
                        optimize_similarity: false,
                        ..Default::default()
                    },
                )
                .unwrap()
                .register()
                .unwrap();
                error = rms(&result.points, &x);
                black_box(result);
            }
            _ => panic!("method must be rigid, deformable, or atlas"),
        };
        times.push(start.elapsed().as_secs_f64());
    }
    println!("method,n,iterations,repeats,median_seconds,rms_error");
    println!(
        "{method},{n},{iterations},{repeats},{:.9},{:.12e}",
        median(times),
        error
    );
}
