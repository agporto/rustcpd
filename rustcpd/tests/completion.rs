//! Validation of the shape-completion analytic core (feature `completion`).
#![cfg(feature = "completion")]

use rustcpd::{
    AtlasConfig, AtlasRegistration, DMatrix, EmConfig, PosteriorOptions, complete_shape,
};

/// A small deterministic linear shape model: mean, modes (M·D × k), eigenvalues.
fn model(m: usize, d: usize, k: usize) -> (DMatrix<f64>, DMatrix<f64>, Vec<f64>) {
    let mean = DMatrix::from_fn(m, d, |i, j| {
        let z = (i + 1) as f64;
        match j {
            0 => (z * 0.3).sin() * 1.5,
            1 => (z * 0.2).cos(),
            _ => (z * 0.13).sin() * (z * 0.07).cos(),
        }
    });
    // Modes: deterministic, orthonormalized columns, then scaled.
    let mut modes = DMatrix::from_fn(m * d, k, |i, c| {
        ((i + 1 + c * 5) as f64 * 0.17).sin() + 0.3 * ((i + 2 * c + 1) as f64 * 0.09).cos()
    });
    for c in 0..k {
        for prev in 0..c {
            let proj = (0..m * d)
                .map(|i| modes[(i, c)] * modes[(i, prev)])
                .sum::<f64>();
            for i in 0..m * d {
                modes[(i, c)] -= proj * modes[(i, prev)];
            }
        }
        let norm = (0..m * d)
            .map(|i| modes[(i, c)].powi(2))
            .sum::<f64>()
            .sqrt();
        for i in 0..m * d {
            modes[(i, c)] /= norm;
        }
    }
    let eigenvalues: Vec<f64> = (0..k).map(|c| 0.6 / (c + 1) as f64).collect();
    (mean, modes, eigenvalues)
}

/// Reference posterior over `b` by Gaussian conditioning on the observed
/// coordinate block — an independent route from the precision form in
/// `complete_shape` (they agree by the Woodbury identity).
fn reference_posterior(
    modes: &DMatrix<f64>,
    eigenvalues: &[f64],
    observed_points: &[usize],
    residual_full: &DMatrix<f64>, // (M, D) obs − mean, meaningful on observed
    d: usize,
    sigma2: f64,
) -> (Vec<f64>, DMatrix<f64>) {
    let k = modes.ncols();
    let rows: Vec<usize> = observed_points
        .iter()
        .flat_map(|&p| (0..d).map(move |j| p * d + j))
        .collect();
    let n = rows.len();
    let u_obs = DMatrix::from_fn(n, k, |r, c| modes[(rows[r], c)]);
    let r_obs = DMatrix::from_fn(n, 1, |r, _| {
        let flat = rows[r];
        residual_full[(flat / d, flat % d)]
    });
    let lambda = DMatrix::from_fn(k, k, |a, b| if a == b { eigenvalues[a] } else { 0.0 });
    // Cov(y_obs) = U Λ Uᵀ + σ² I
    let mut cov_y = &u_obs * &lambda * u_obs.transpose();
    for i in 0..n {
        cov_y[(i, i)] += sigma2;
    }
    let cov_y_inv = cov_y.try_inverse().expect("cov invertible");
    let gain = &lambda * u_obs.transpose() * &cov_y_inv; // k×n
    let b_mean: Vec<f64> = (&gain * &r_obs).column(0).iter().copied().collect();
    let sigma_b = &lambda - &gain * &u_obs * &lambda;
    (b_mean, sigma_b)
}

#[test]
fn posterior_matches_gaussian_conditioning_reference() {
    let (m, d, k) = (12, 3, 4);
    let (mean, modes, eigenvalues) = model(m, d, k);
    let truth = [0.4, -0.25, 0.15, -0.1];
    // Full shape from the true coefficients.
    let shape = DMatrix::from_fn(m, d, |i, j| {
        mean[(i, j)]
            + (0..k)
                .map(|c| modes[(i * d + j, c)] * truth[c])
                .sum::<f64>()
    });
    let observed: Vec<usize> = (0..8).collect(); // first 8 points observed
    let residual = DMatrix::from_fn(m, d, |i, j| shape[(i, j)] - mean[(i, j)]);
    let mut weight = vec![0.0; m];
    for &p in &observed {
        weight[p] = 1.0;
    }
    let sigma2 = 0.01;
    let identity = DMatrix::identity(d, d);

    let posterior = complete_shape(
        &mean,
        &modes,
        &eigenvalues,
        &residual,
        &weight,
        sigma2,
        &identity,
        1.0,
        &[0.0, 0.0, 0.0],
        1.0,
        false,
    )
    .unwrap();

    let (ref_mean, ref_cov) =
        reference_posterior(&modes, &eigenvalues, &observed, &residual, d, sigma2);

    for (a, b) in posterior.coefficient_mean().iter().zip(&ref_mean) {
        assert!((a - b).abs() < 1e-9, "b_mean {a} vs {b}");
    }
    let cov = posterior.coefficient_covariance();
    for a in 0..k {
        for b in 0..k {
            assert!(
                (cov[(a, b)] - ref_cov[(a, b)]).abs() < 1e-9,
                "Σ_b[{a},{b}] {} vs {}",
                cov[(a, b)],
                ref_cov[(a, b)]
            );
        }
    }
}

#[test]
fn predictive_covariance_uses_row_vector_pushforward() {
    let (mean, modes, eigenvalues) = model(8, 3, 3);
    let residual = DMatrix::zeros(8, 3);
    let weight = vec![1.0; 8];
    let identity = DMatrix::identity(3, 3);
    let model_posterior = complete_shape(
        &mean,
        &modes,
        &eigenvalues,
        &residual,
        &weight,
        0.01,
        &identity,
        1.0,
        &[0.0; 3],
        1.0,
        false,
    )
    .unwrap();

    let angle = 0.41_f64;
    let rotation = DMatrix::from_row_slice(
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
    let scale = 1.7;
    let rotated = complete_shape(
        &mean,
        &modes,
        &eigenvalues,
        &residual,
        &weight,
        0.01,
        &rotation,
        scale,
        &[0.0; 3],
        1.0,
        false,
    )
    .unwrap();

    let model_covariance = model_posterior.predictive_covariance(0);
    let expected = scale * scale * (rotation.transpose() * model_covariance * &rotation);
    let actual = rotated.predictive_covariance(0);
    assert!(
        (&actual - &expected).norm() < 1e-12,
        "covariance pushforward error={}",
        (&actual - &expected).norm()
    );
}

#[test]
fn completion_reduces_error_and_variance_in_observed_region() {
    let (m, d, k) = (16, 3, 4);
    let (mean, modes, eigenvalues) = model(m, d, k);
    let truth = [0.5, -0.3, 0.2, -0.15];
    let shape = DMatrix::from_fn(m, d, |i, j| {
        mean[(i, j)]
            + (0..k)
                .map(|c| modes[(i * d + j, c)] * truth[c])
                .sum::<f64>()
    });
    let observed: Vec<usize> = (0..10).collect();
    let residual = DMatrix::from_fn(m, d, |i, j| shape[(i, j)] - mean[(i, j)]);
    let mut weight = vec![0.0; m];
    for &p in &observed {
        weight[p] = 1.0;
    }
    let identity = DMatrix::identity(d, d);
    let posterior = complete_shape(
        &mean,
        &modes,
        &eigenvalues,
        &residual,
        &weight,
        1e-4,
        &identity,
        1.0,
        &[0.0, 0.0, 0.0],
        1.0,
        false,
    )
    .unwrap();

    let completed = posterior.predict(); // pose identity → model frame
    // Completed shape is close to the true shape everywhere, including the
    // 6 unobserved points, because the modes couple the regions.
    let missing_err: f64 = (10..m)
        .map(|i| {
            (0..d)
                .map(|j| (completed[(i, j)] - shape[(i, j)]).powi(2))
                .sum::<f64>()
        })
        .sum::<f64>()
        .sqrt();
    let missing_baseline: f64 = (10..m)
        .map(|i| {
            (0..d)
                .map(|j| (mean[(i, j)] - shape[(i, j)]).powi(2))
                .sum::<f64>()
        })
        .sum::<f64>()
        .sqrt();
    assert!(
        missing_err < 0.25 * missing_baseline,
        "completion missing-region error {missing_err} vs mean-shape baseline {missing_baseline}"
    );

    // Predictive variance is smaller on observed points than on missing ones.
    let var = posterior.predictive_variance();
    let observed_mean_var: f64 =
        observed.iter().map(|&i| var[i]).sum::<f64>() / observed.len() as f64;
    let missing_mean_var: f64 = (10..m).map(|i| var[i]).sum::<f64>() / (m - 10) as f64;
    assert!(
        observed_mean_var < missing_mean_var,
        "observed var {observed_mean_var} should be < missing var {missing_mean_var}"
    );
}

#[test]
fn atlas_posterior_end_to_end_recovers_and_completes() {
    // Full model, synthesize a target from known coefficients + pose, register
    // on a PARTIAL target (a contiguous subset of points), then complete.
    let (m, d, k) = (60, 3, 5);
    let (mean, modes, eigenvalues) = model(m, d, k);
    let truth = [0.4, -0.3, 0.2, -0.12, 0.08];
    let deformed = DMatrix::from_fn(m, d, |i, j| {
        mean[(i, j)]
            + (0..k)
                .map(|c| modes[(i * d + j, c)] * truth[c])
                .sum::<f64>()
    });
    let angle = 0.2_f64;
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
    let (scale, t) = (1.1, [0.3, -0.2, 0.1]);
    let full_target = DMatrix::from_fn(m, d, |i, j| {
        scale * (0..d).map(|q| deformed[(i, q)] * r[(q, j)]).sum::<f64>() + t[j]
    });
    // Partial observation: keep the first 40 of 60 target points.
    let observed_count = 40;
    let partial = DMatrix::from_fn(observed_count, d, |i, j| full_target[(i, j)]);

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
    )
    .unwrap()
    .register()
    .unwrap();

    let posterior = fit
        .posterior(
            &partial,
            &mean,
            &modes,
            &eigenvalues,
            &PosteriorOptions {
                completeness: Some(observed_count as f64 / m as f64),
                ..Default::default()
            },
        )
        .unwrap();

    // The completed shape (target frame) predicts the held-out points far
    // better than the posed mean shape would.
    let completed = posterior.predict();
    let posed_mean = DMatrix::from_fn(m, d, |i, j| {
        fit.scale
            * (0..d)
                .map(|q| mean[(i, q)] * fit.rotation[(q, j)])
                .sum::<f64>()
            + fit.translation[j]
    });
    let held = observed_count..m;
    let completion_err: f64 = held
        .clone()
        .map(|i| {
            (0..d)
                .map(|j| (completed[(i, j)] - full_target[(i, j)]).powi(2))
                .sum::<f64>()
        })
        .sum::<f64>()
        .sqrt();
    let mean_err: f64 = held
        .map(|i| {
            (0..d)
                .map(|j| (posed_mean[(i, j)] - full_target[(i, j)]).powi(2))
                .sum::<f64>()
        })
        .sum::<f64>()
        .sqrt();
    assert!(
        completion_err < 0.5 * mean_err,
        "held-out completion error {completion_err} vs posed-mean baseline {mean_err}"
    );
}

#[test]
fn complete_shape_validates_inputs() {
    let (mean, modes, eigenvalues) = model(8, 3, 3);
    let residual = DMatrix::zeros(8, 3);
    let weight = vec![1.0; 8];
    let identity = DMatrix::identity(3, 3);
    // Bad sigma_eff2.
    assert!(
        complete_shape(
            &mean,
            &modes,
            &eigenvalues,
            &residual,
            &weight,
            -1.0,
            &identity,
            1.0,
            &[0.0; 3],
            1.0,
            false
        )
        .is_err()
    );
    // Bad temperature.
    assert!(
        complete_shape(
            &mean,
            &modes,
            &eigenvalues,
            &residual,
            &weight,
            1.0,
            &identity,
            1.0,
            &[0.0; 3],
            0.0,
            false
        )
        .is_err()
    );
    // Wrong weight length.
    assert!(
        complete_shape(
            &mean,
            &modes,
            &eigenvalues,
            &residual,
            &[1.0; 3],
            1.0,
            &identity,
            1.0,
            &[0.0; 3],
            1.0,
            false
        )
        .is_err()
    );
}

#[test]
fn sample_moments_match_analytic_posterior() {
    let (m, d, k) = (12, 3, 4);
    let (mean, modes, eigenvalues) = model(m, d, k);
    let truth = [0.4, -0.25, 0.15, -0.1];
    let shape = DMatrix::from_fn(m, d, |i, j| {
        mean[(i, j)]
            + (0..k)
                .map(|c| modes[(i * d + j, c)] * truth[c])
                .sum::<f64>()
    });
    let residual = DMatrix::from_fn(m, d, |i, j| shape[(i, j)] - mean[(i, j)]);
    let mut weight = vec![0.0; m];
    for w in weight.iter_mut().take(8) {
        *w = 1.0;
    }
    let identity = DMatrix::identity(d, d);
    let posterior = complete_shape(
        &mean,
        &modes,
        &eigenvalues,
        &residual,
        &weight,
        0.02,
        &identity,
        1.0,
        &[0.0; 3],
        1.0,
        false,
    )
    .unwrap();

    let n = 40000;
    let samples = posterior.sample_coefficients(n, 12345).unwrap();
    // Empirical mean → posterior mean.
    let mut emp_mean = vec![0.0; k];
    for s in &samples {
        for (a, value) in emp_mean.iter_mut().enumerate() {
            *value += s[a] / n as f64;
        }
    }
    for (a, &emp) in emp_mean.iter().enumerate() {
        assert!(
            (emp - posterior.coefficient_mean()[a]).abs() < 5e-3,
            "mean[{a}] emp {emp} vs {}",
            posterior.coefficient_mean()[a]
        );
    }
    // Empirical covariance → Σ_b.
    let cov = posterior.coefficient_covariance();
    for a in 0..k {
        for b in 0..k {
            let emp: f64 = samples
                .iter()
                .map(|s| (s[a] - emp_mean[a]) * (s[b] - emp_mean[b]))
                .sum::<f64>()
                / n as f64;
            assert!(
                (emp - cov[(a, b)]).abs() < 5e-3,
                "cov[{a},{b}] emp {emp} vs {}",
                cov[(a, b)]
            );
        }
    }
}

#[test]
fn sampling_is_deterministic() {
    let (m, d, k) = (10, 3, 3);
    let (mean, modes, eigenvalues) = model(m, d, k);
    let residual = DMatrix::from_fn(m, d, |i, _| 0.01 * (i as f64).sin());
    let weight = vec![1.0; m];
    let identity = DMatrix::identity(d, d);
    let posterior = complete_shape(
        &mean,
        &modes,
        &eigenvalues,
        &residual,
        &weight,
        0.05,
        &identity,
        1.0,
        &[0.0; 3],
        1.0,
        false,
    )
    .unwrap();
    let a = posterior.sample_shapes(50, 777).unwrap();
    let b = posterior.sample_shapes(50, 777).unwrap();
    for (sa, sb) in a.iter().zip(&b) {
        assert!(
            sa.iter().zip(sb.iter()).all(|(x, y)| x == y),
            "sampling not deterministic"
        );
    }
    // A different seed gives a different ensemble.
    let c = posterior.sample_shapes(50, 778).unwrap();
    assert!(a[0].iter().zip(c[0].iter()).any(|(x, y)| x != y));
}

#[test]
fn mixture_sampling_draws_from_all_components() {
    use rustcpd::mixture_sample_shapes;
    let (m, d, k) = (8, 3, 3);
    let (mean, modes, eigenvalues) = model(m, d, k);
    let weight = vec![1.0; m];
    let identity = DMatrix::identity(d, d);
    // Two posteriors with clearly different residuals (→ different completions).
    let ra = DMatrix::from_fn(m, d, |_, _| 0.5);
    let rb = DMatrix::from_fn(m, d, |_, _| -0.5);
    let pa = complete_shape(
        &mean,
        &modes,
        &eigenvalues,
        &ra,
        &weight,
        0.01,
        &identity,
        1.0,
        &[0.0; 3],
        1.0,
        false,
    )
    .unwrap();
    let pb = complete_shape(
        &mean,
        &modes,
        &eigenvalues,
        &rb,
        &weight,
        0.01,
        &identity,
        1.0,
        &[0.0; 3],
        1.0,
        false,
    )
    .unwrap();
    let shapes = mixture_sample_shapes(&[(1.0, &pa), (1.0, &pb)], 200, 42).unwrap();
    assert_eq!(shapes.len(), 200);
    // Deterministic.
    let again = mixture_sample_shapes(&[(1.0, &pa), (1.0, &pb)], 200, 42).unwrap();
    assert!(
        shapes
            .iter()
            .zip(&again)
            .all(|(x, y)| x.iter().zip(y.iter()).all(|(a, b)| a == b))
    );
    // Both modes appear: mean of first coord spans between the two component means.
    let overall: f64 = shapes.iter().map(|s| s[(0, 0)]).sum::<f64>() / shapes.len() as f64;
    let mean_a = pa.predict()[(0, 0)];
    let mean_b = pb.predict()[(0, 0)];
    let lo = mean_a.min(mean_b);
    let hi = mean_a.max(mean_b);
    assert!(
        overall > lo && overall < hi,
        "mixture mean {overall} not between {lo} and {hi}"
    );
}

#[test]
fn discrepancy_inflates_variance_for_out_of_model_data() {
    let (m, d, k) = (16, 3, 4);
    let (mean, modes, eigenvalues) = model(m, d, k);
    let truth = [0.4, -0.25, 0.15, -0.1];
    // Residual that lies exactly in the model span.
    let in_model = DMatrix::from_fn(m, d, |i, j| {
        (0..k)
            .map(|c| modes[(i * d + j, c)] * truth[c])
            .sum::<f64>()
    });
    let weight = vec![1.0; m];
    let identity = DMatrix::identity(d, d);
    let clean = complete_shape(
        &mean,
        &modes,
        &eigenvalues,
        &in_model,
        &weight,
        1e-6,
        &identity,
        1.0,
        &[0.0; 3],
        1.0,
        true,
    )
    .unwrap();
    // In-span data → negligible discrepancy.
    assert!(
        clean.discrepancy_variance() < 1e-3,
        "clean δ²={}",
        clean.discrepancy_variance()
    );

    // Add a signal outside the 4-mode span.
    let contaminated = DMatrix::from_fn(m, d, |i, j| {
        in_model[(i, j)] + 0.3 * ((i * d + j) as f64 * 1.7).sin()
    });
    let dirty = complete_shape(
        &mean,
        &modes,
        &eigenvalues,
        &contaminated,
        &weight,
        1e-6,
        &identity,
        1.0,
        &[0.0; 3],
        1.0,
        true,
    )
    .unwrap();
    assert!(
        dirty.discrepancy_variance() > clean.discrepancy_variance() + 1e-3,
        "dirty δ² {} should exceed clean {}",
        dirty.discrepancy_variance(),
        clean.discrepancy_variance()
    );
    // Inflation propagates into the predictive variance.
    assert!(dirty.predictive_variance()[0] > clean.predictive_variance()[0]);

    // Disabling discrepancy zeroes it and does not inflate.
    let off = complete_shape(
        &mean,
        &modes,
        &eigenvalues,
        &contaminated,
        &weight,
        1e-6,
        &identity,
        1.0,
        &[0.0; 3],
        1.0,
        false,
    )
    .unwrap();
    assert_eq!(off.discrepancy_variance(), 0.0);
    assert!(off.predictive_variance()[0] < dirty.predictive_variance()[0]);
}
