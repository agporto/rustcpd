use rustcpd::{
    AtlasConfig, AtlasRegistration, AtlasResult, DMatrix, EmConfig, PoseMarginalizedConfig,
};

fn rod(n: usize) -> DMatrix<f64> {
    DMatrix::from_fn(n, 3, |i, j| {
        let z = i as f64 / (n - 1) as f64;
        let radius = 0.1 + 0.6 * z;
        match j {
            0 => 4.0 * z,
            1 => radius * (7.0 * z).sin(),
            _ => radius * (7.0 * z).cos(),
        }
    })
}

fn rotation() -> DMatrix<f64> {
    let a = 0.3_f64;
    DMatrix::from_row_slice(
        3,
        3,
        &[a.cos(), -a.sin(), 0.0, a.sin(), a.cos(), 0.0, 0.0, 0.0, 1.0],
    )
}

fn fragment(source: &DMatrix<f64>) -> DMatrix<f64> {
    let r = rotation();
    DMatrix::from_fn(20, 3, |i, j| {
        (0..3).map(|k| source[(i, k)] * r[(k, j)]).sum::<f64>() + [0.5, -0.3, 0.2][j]
    })
}

fn config(iterations: usize) -> AtlasConfig {
    AtlasConfig {
        em: EmConfig {
            max_iterations: iterations,
            tolerance: 0.0,
            outlier_weight: 0.05,
            parallel: false,
            ..Default::default()
        },
        eigenvalues: vec![1.0],
        normalize: true,
        with_scale: false,
        adaptive_mixing: Some(0.1),
        ..Default::default()
    }
}

fn fit(x: &DMatrix<f64>, y: &DMatrix<f64>, modes: &DMatrix<f64>, cfg: AtlasConfig) -> AtlasResult {
    AtlasRegistration::new(x, y, modes, cfg)
        .unwrap()
        .register()
        .unwrap()
}

#[test]
fn split_em_preserves_state_across_normalized_and_raw_frames() {
    let y = rod(60);
    let x = fragment(&y);
    let modes = DMatrix::from_fn(y.len(), 1, |i, _| 0.01 * (i as f64 * 0.1).sin());
    let mut initial = config(12);
    initial.initial_rotation = Some(rotation());
    initial.initial_translation = Some(vec![0.5, -0.3, 0.2]);
    initial.em.sigma2 = Some(0.25);
    let full = fit(&x, &y, &modes, initial.clone());
    initial.em.max_iterations = 4;
    let first = fit(&x, &y, &modes, initial);
    for normalize in [true, false] {
        let continued = fit(
            &x,
            &y,
            &modes,
            AtlasConfig {
                initial_state: Some(first.state()),
                normalize,
                ..config(8)
            },
        );
        assert!((&continued.points - &full.points).amax() < 1e-8);
        assert!((continued.sigma2 - full.sigma2).abs() < 1e-10);
        for (a, b) in continued
            .mixing_weights
            .unwrap()
            .iter()
            .zip(full.mixing_weights.as_ref().unwrap())
        {
            assert!((a - b).abs() < 1e-8);
        }
        assert_eq!(continued.outlier_density, full.outlier_density);
    }
}

#[test]
fn state_maps_equal_length_permutations_and_new_sampling() {
    let y = rod(60);
    let modes = DMatrix::zeros(y.len(), 1);
    let original = fit(&fragment(&y), &y, &modes, config(3));
    let state = original.state();
    let permutation: Vec<usize> = (0..60).rev().collect();
    let permuted = DMatrix::from_fn(60, 3, |i, j| y[(permutation[i], j)]);
    let continued = fit(
        &fragment(&y),
        &permuted,
        &modes,
        AtlasConfig {
            initial_state: Some(state.clone()),
            ..config(0)
        },
    );
    let pi = continued.mixing_weights.unwrap();
    for (i, &index) in permutation.iter().enumerate() {
        assert!((pi[i] - state.mixing_weights.as_ref().unwrap()[index]).abs() < 1e-12);
    }
    // The same map supports a denser model and always returns normalized mass.
    let dense = rod(120);
    let weights = state.mixing_weights_on(&dense).unwrap().unwrap();
    assert_eq!(weights.len(), 120);
    assert!((weights.iter().sum::<f64>() - 1.0).abs() < 1e-12);
    assert!(weights.iter().all(|p| p.is_finite() && *p >= 0.0));
    let mut sparse_state = state;
    let mut sparse_weights = vec![0.0; 60];
    sparse_weights[0] = 1.0;
    sparse_state.mixing_weights = Some(sparse_weights.clone());
    sparse_weights.reverse();
    assert_eq!(
        sparse_state.mixing_weights_on(&permuted).unwrap().unwrap(),
        sparse_weights
    );
}

#[test]
fn state_supports_more_modes_and_fixed_mixing() {
    let y = rod(60);
    let x = fragment(&y);
    let initial = fit(&x, &y, &DMatrix::zeros(y.len(), 1), config(2));
    let resumed = fit(
        &x,
        &y,
        &DMatrix::zeros(y.len(), 2),
        AtlasConfig {
            initial_state: Some(initial.state()),
            adaptive_mixing: None,
            eigenvalues: vec![1.0, 0.5],
            ..config(1)
        },
    );
    assert_eq!(resumed.coefficients, vec![0.0, 0.0]);
    assert_eq!(resumed.mixing_weights, initial.mixing_weights);
}

#[test]
fn invalid_or_conflicting_states_are_rejected() {
    let y = rod(20);
    let modes = DMatrix::zeros(y.len(), 1);
    let state = fit(&y, &y, &modes, config(1)).state();
    let rejected = |state| {
        AtlasRegistration::new(
            &y,
            &y,
            &modes,
            AtlasConfig {
                initial_state: Some(state),
                ..config(1)
            },
        )
        .is_err()
    };
    let mut bad = state.clone();
    bad.sigma2 = f64::NAN;
    assert!(rejected(bad));
    let mut bad = state.clone();
    bad.mixing_reference = None;
    assert!(rejected(bad));
    let mut bad = state.clone();
    bad.mixing_weights = Some(vec![1.0; 20]);
    assert!(rejected(bad));
    let mut bad = state.clone();
    bad.coefficients = vec![0.0, 0.0];
    assert!(rejected(bad));
    assert!(
        AtlasRegistration::new(
            &y,
            &y,
            &modes,
            AtlasConfig {
                initial_state: Some(state),
                initial_rotation: Some(rotation()),
                ..config(1)
            }
        )
        .is_err()
    );
}

#[test]
fn pose_state_keeps_a_fragment_in_place_during_final_registration() {
    let y = rod(60);
    let x = fragment(&y);
    let modes = DMatrix::zeros(y.len(), 1);
    let pose = PoseMarginalizedConfig {
        rotation_count: 9,
        coarse_source_count: 60,
        coarse_target_count: 20,
        coarse_rank: 1,
        coarse_iterations: 10,
        refine_count: 6,
        refine_target_count: 20,
        refine_iterations: 30,
        with_scale: false,
        translation_anchor_count: 6,
        adaptive_mixing: Some(0.1),
        parallel: false,
        ..Default::default()
    }
    .initialize(&y, &x, &modes, &[1.0])
    .unwrap();
    let continued = fit(
        &x,
        &y,
        &modes,
        AtlasConfig {
            initial_state: Some(pose.state()),
            ..config(100)
        },
    );
    let error = ((0..20)
        .map(|i| {
            (0..3)
                .map(|j| (continued.points[(i, j)] - x[(i, j)]).powi(2))
                .sum::<f64>()
        })
        .sum::<f64>()
        / 60.0)
        .sqrt();
    assert!(error < 0.1, "continued fragment RMS = {error}");
    // The original failure: even an exact pose is erased by a broad restart.
    let reheated = fit(
        &x,
        &y,
        &modes,
        AtlasConfig {
            initial_rotation: Some(rotation()),
            initial_translation: Some(vec![0.5, -0.3, 0.2]),
            adaptive_mixing: Some(1.0),
            ..config(100)
        },
    );
    let old_error = ((0..20)
        .map(|i| {
            (0..3)
                .map(|j| (reheated.points[(i, j)] - x[(i, j)]).powi(2))
                .sum::<f64>()
        })
        .sum::<f64>()
        / 60.0)
        .sqrt();
    assert!(
        old_error > 0.15,
        "fixture must exercise the broad-restart failure: {old_error}"
    );
}

#[cfg(feature = "completion")]
#[test]
fn completion_uses_fitted_variance_and_mixing_after_resampling() {
    use rustcpd::PosteriorOptions;
    let y = rod(60);
    let x = fragment(&y);
    let modes = DMatrix::from_fn(y.len(), 1, |i, _| if i % 3 == 1 { 0.1 } else { 0.0 });
    let original = fit(&x, &y, &modes, config(2));
    let options = PosteriorOptions {
        visibility_floor: 0.0,
        estimate_discrepancy: false,
        ..Default::default()
    };
    let p = original
        .posterior(&x, &y, &modes, &[1.0], &options)
        .unwrap();
    assert_eq!(p.noise_variance(), original.sigma2);
    let mut different = original.clone();
    different.mixing_weights.as_mut().unwrap().reverse();
    let q = different
        .posterior(&x, &y, &modes, &[1.0], &options)
        .unwrap();
    assert!((p.coefficient_mean()[0] - q.coefficient_mean()[0]).abs() > 1e-5);
    // Reordering a model of equal size must reorder, not discard, occupancies.
    let permutation: Vec<usize> = (0..60).rev().collect();
    let permuted = DMatrix::from_fn(60, 3, |i, j| y[(permutation[i], j)]);
    let p2 = original
        .posterior(&x, &permuted, &modes, &[1.0], &options)
        .unwrap();
    assert!((p.coefficient_mean()[0] - p2.coefficient_mean()[0]).abs() < 1e-9);
    assert!((p.coefficient_covariance() - p2.coefficient_covariance()).amax() < 1e-9);
    let dense = rod(120);
    let dense_modes = DMatrix::from_fn(dense.len(), 1, |i, _| if i % 3 == 1 { 0.1 } else { 0.0 });
    assert!(
        original
            .posterior(&x, &dense, &dense_modes, &[1.0], &options)
            .is_ok()
    );
}
