//! Timing harness for pose-marginalized initialization.
//! Run: cargo run --release --example pose_bench [n_points] [repeats]

use rustcpd::{DMatrix, PoseMarginalizedConfig};
use std::time::Instant;

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

fn median(mut v: Vec<f64>) -> f64 {
    v.sort_by(f64::total_cmp);
    v[v.len() / 2]
}

fn main() {
    let args: Vec<_> = std::env::args().collect();
    let n = args.get(1).and_then(|v| v.parse().ok()).unwrap_or(2000);
    let repeats = args.get(2).and_then(|v| v.parse().ok()).unwrap_or(5);
    let single = args.get(3).map(|v| v == "f32").unwrap_or(false);

    let source = cloud(n);
    let rank = 12;
    let modes = DMatrix::from_fn(source.len(), rank, |i, k| {
        0.01 * ((i + 1 + k * 7) as f64 * 0.019).sin()
    });
    let coefficients: Vec<f64> = (0..rank)
        .map(|k| 0.2 * ((k + 1) as f64 * 0.7).sin())
        .collect();
    let angle = 0.6_f64;
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
    let deformed = DMatrix::from_fn(n, 3, |i, j| {
        source[(i, j)]
            + (0..rank)
                .map(|k| modes[(i * 3 + j, k)] * coefficients[k])
                .sum::<f64>()
    });
    let target = DMatrix::from_fn(n, 3, |i, j| {
        1.2 * (0..3).map(|q| deformed[(i, q)] * r[(q, j)]).sum::<f64>() + [0.5, -0.3, 0.2][j]
    });
    let eigenvalues: Vec<f64> = (0..rank).map(|k| 1.0 / (k + 1) as f64).collect();

    let config = PoseMarginalizedConfig {
        single_precision: single,
        ..Default::default()
    };
    // Warm up.
    let _ = config
        .initialize(&source, &target, &modes, &eigenvalues)
        .unwrap();
    let mut times = Vec::new();
    for _ in 0..repeats {
        let start = Instant::now();
        let out = config
            .initialize(&source, &target, &modes, &eigenvalues)
            .unwrap();
        times.push(start.elapsed().as_secs_f64());
        std::hint::black_box(out);
    }
    println!(
        "pose_init n={n} single={single} median={:.4}s ({} hypotheses)",
        median(times),
        config.rotation_count
    );
}
