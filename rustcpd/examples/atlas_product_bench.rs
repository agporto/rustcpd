//! Paired timings of the original atlas products and the faer candidates.
//! cargo run --release --example atlas_product_bench -- [rows] [rank] [pairs]
//! Set RAYON_NUM_THREADS explicitly when comparing machines.

use faer::{Accum, Par, linalg::matmul::matmul, mat::MatMut, mat::MatRef};
use rustcpd::DMatrix;
use std::{hint::black_box, time::Instant};

fn multiply(a: &DMatrix<f64>, b: &DMatrix<f64>, parallel: bool) -> DMatrix<f64> {
    let mut out = DMatrix::zeros(a.ncols(), b.ncols());
    matmul(
        MatMut::from_column_major_slice_mut(out.as_mut_slice(), a.ncols(), b.ncols()),
        Accum::Replace,
        MatRef::from_column_major_slice(a.as_slice(), a.nrows(), a.ncols()).transpose(),
        MatRef::from_column_major_slice(b.as_slice(), b.nrows(), b.ncols()),
        1.0,
        if parallel { Par::rayon(0) } else { Par::Seq },
    );
    out
}

fn median(values: &mut [f64]) -> f64 {
    values.sort_by(f64::total_cmp);
    values[values.len() / 2]
}

fn main() {
    let args: Vec<_> = std::env::args().collect();
    let rows = args.get(1).map(|s| s.parse().unwrap()).unwrap_or(6000);
    let rank = args.get(2).map(|s| s.parse().unwrap()).unwrap_or(64);
    let pairs = args.get(3).map(|s| s.parse().unwrap()).unwrap_or(7);
    let modes = DMatrix::from_fn(rows, rank, |i, j| {
        (2.0 / rows as f64).sqrt()
            * (std::f64::consts::PI * (i as f64 + 0.5) * (j + 1) as f64 / rows as f64).cos()
    });
    let weighted = DMatrix::from_fn(rows, rank, |i, j| {
        let weight = if i / 3 % 5 == 0 {
            1e-8
        } else {
            0.1 + (i / 3 % 29) as f64 / 17.0
        };
        modes[(i, j)] * weight
    });
    let residual = DMatrix::from_fn(rows, 1, |i, _| 0.1 * (i as f64 * 0.17).sin());
    for (operation, a, b) in [("gram", &modes, &weighted), ("rhs", &weighted, &residual)] {
        let original = a.tr_mul(b);
        for parallel in [false, true] {
            let actual = multiply(a, b, parallel);
            let relative_error = (&actual - &original).norm() / original.norm().max(1e-300);
            let serial = multiply(a, b, false);
            let bitwise_parallel = actual
                .iter()
                .zip(serial.iter())
                .all(|(a, b)| a.to_bits() == b.to_bits());
            assert!(
                relative_error < 1e-12,
                "matrix discrepancy: {relative_error}"
            );
            assert!(bitwise_parallel, "faer changes with parallel scheduling");
            let time_once = |optimized: bool, count: usize| {
                let start = Instant::now();
                for _ in 0..count {
                    let value = if optimized {
                        multiply(black_box(a), black_box(b), parallel)
                    } else {
                        black_box(a).tr_mul(black_box(b))
                    };
                    black_box(value);
                }
                start.elapsed().as_secs_f64() / count as f64
            };
            let warm = time_once(false, 1).max(1e-6);
            black_box(time_once(true, 1));
            let count = (0.015 / warm).ceil().clamp(1.0, 2000.0) as usize;
            let mut baseline_times = Vec::new();
            let mut candidate_times = Vec::new();
            for pair in 0..pairs {
                if pair % 2 == 0 {
                    baseline_times.push(time_once(false, count));
                    candidate_times.push(time_once(true, count));
                } else {
                    candidate_times.push(time_once(true, count));
                    baseline_times.push(time_once(false, count));
                }
            }
            let baseline = median(&mut baseline_times);
            let candidate = median(&mut candidate_times);
            println!(
                "{{\"operation\":\"{operation}\",\"rows\":{rows},\"rank\":{rank},\"parallel\":{parallel},\"original_s\":{baseline:.12e},\"faer_s\":{candidate:.12e},\"speedup\":{:.8},\"relative_error\":{relative_error:.8e},\"bitwise_parallel\":{bitwise_parallel},\"pairs\":{pairs}}}",
                baseline / candidate
            );
        }
    }
}
