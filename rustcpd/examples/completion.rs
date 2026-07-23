//! Shape completion + per-point uncertainty from a partial observation.
//!
//! Run: `cargo run --release --features completion --example completion`

use rustcpd::{AtlasConfig, AtlasRegistration, DMatrix, EmConfig, PosteriorOptions};

fn model(m: usize, k: usize) -> (DMatrix<f64>, DMatrix<f64>, Vec<f64>) {
    let mean = DMatrix::from_fn(m, 3, |i, j| {
        let z = (i + 1) as f64;
        match j {
            0 => (z * 0.3).sin() * 1.5,
            1 => (z * 0.2).cos(),
            _ => (z * 0.13).sin(),
        }
    });
    let mut modes = DMatrix::from_fn(m * 3, k, |i, c| {
        ((i + 1 + c * 5) as f64 * 0.17).sin() + 0.3 * ((i + 2 * c + 1) as f64 * 0.09).cos()
    });
    for c in 0..k {
        for prev in 0..c {
            let proj = (0..m * 3)
                .map(|i| modes[(i, c)] * modes[(i, prev)])
                .sum::<f64>();
            for i in 0..m * 3 {
                modes[(i, c)] -= proj * modes[(i, prev)];
            }
        }
        let norm = (0..m * 3)
            .map(|i| modes[(i, c)].powi(2))
            .sum::<f64>()
            .sqrt();
        for i in 0..m * 3 {
            modes[(i, c)] /= norm;
        }
    }
    let eigenvalues = (0..k).map(|c| 0.6 / (c + 1) as f64).collect();
    (mean, modes, eigenvalues)
}

fn main() -> Result<(), rustcpd::Error> {
    let (m, k) = (80, 5);
    let (mean, modes, eigenvalues) = model(m, k);
    let truth = [0.4, -0.3, 0.2, -0.12, 0.08];

    // True shape, posed into the target frame.
    let deformed = DMatrix::from_fn(m, 3, |i, j| {
        mean[(i, j)]
            + (0..k)
                .map(|c| modes[(i * 3 + j, c)] * truth[c])
                .sum::<f64>()
    });
    let angle = 0.25_f64;
    let r = DMatrix::from_row_slice(
        3,
        3,
        &[
            angle.cos(),
            -angle.sin(),
            0.0,
            angle.sin(),
            angle.cos(),
            0.0,
            0.0,
            0.0,
            1.0,
        ],
    );
    let full = DMatrix::from_fn(m, 3, |i, j| {
        1.15 * (0..3).map(|q| deformed[(i, q)] * r[(q, j)]).sum::<f64>() + [0.3, -0.2, 0.1][j]
    });

    // Observe only the first 55 of 80 points.
    let observed = 55;
    let partial = DMatrix::from_fn(observed, 3, |i, j| full[(i, j)]);

    let fit = AtlasRegistration::new(
        &partial,
        &mean,
        &modes,
        AtlasConfig {
            em: EmConfig {
                tolerance: 1e-8,
                max_iterations: 200,
                ..Default::default()
            },
            eigenvalues: eigenvalues.clone(),
            lambda_regularization: 1.0,
            optimize_similarity: true,
            ..Default::default()
        },
    )?
    .register()?;

    let posterior = fit.posterior(
        &partial,
        &mean,
        &modes,
        &eigenvalues,
        &PosteriorOptions {
            completeness: Some(observed as f64 / m as f64),
            ..Default::default()
        },
    )?;

    let completed = posterior.predict();
    let held_rms = ((observed..m)
        .map(|i| {
            (0..3)
                .map(|j| (completed[(i, j)] - full[(i, j)]).powi(2))
                .sum::<f64>()
        })
        .sum::<f64>()
        / (3 * (m - observed)) as f64)
        .sqrt();
    println!("held-out completion rms = {held_rms:.3e}");

    let variance = posterior.predictive_variance();
    let obs_sd = (variance[..observed].iter().sum::<f64>() / observed as f64).sqrt();
    let miss_sd = (variance[observed..].iter().sum::<f64>() / (m - observed) as f64).sqrt();
    println!("mean predictive sd: observed {obs_sd:.3e}, missing {miss_sd:.3e}");

    // Ensemble of plausible completions (deterministic in the seed).
    let ensemble = posterior.sample_shapes(200, 0)?;
    println!("drew {} plausible completions", ensemble.len());
    Ok(())
}
