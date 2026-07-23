use approx::assert_abs_diff_eq;
use rustcpd::{
    AffineConfig, AffineRegistration, AtlasConfig, AtlasRegistration, AtlasResult, Constraint,
    DMatrix, DeformableConfig, DeformableRegistration, EmConfig, LowRankMethod,
    PoseMarginalizedConfig, RigidConfig, RigidRegistration, correspondences, gaussian_kernel,
    initialize_sigma2,
};

fn cloud(count: usize) -> DMatrix<f64> {
    DMatrix::from_fn(count, 3, |i, j| {
        let z = i as f64 + 1.0;
        match j {
            0 => (z * 0.37).sin() * 1.7 + 0.01 * z,
            1 => (z * 0.23).cos() * 0.9,
            _ => (z * 0.11).sin() * (z * 0.07).cos(),
        }
    })
}

#[test]
fn atlas_similarity_uses_direct_row_vector_rotation() {
    let angle = 0.37_f64;
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
    let source = cloud(8);
    let scale = 1.2;
    let translation = vec![0.3, -0.2, 0.1];
    let result = AtlasResult {
        points: DMatrix::zeros(0, 3),
        coefficients: Vec::new(),
        rotation: rotation.clone(),
        scale,
        translation: translation.clone(),
        sigma2: 1.0,
        iterations: 0,
        difference: 0.0,
        negative_log_likelihood: 0.0,
    };
    let transformed = result.apply_similarity(&source).unwrap();
    let expected = DMatrix::from_fn(source.nrows(), 3, |i, j| {
        scale
            * (0..3)
                .map(|q| source[(i, q)] * rotation[(q, j)])
                .sum::<f64>()
            + translation[j]
    });
    assert!(rms(&transformed, &expected) < 1e-14);

    let recovered = DMatrix::from_fn(source.nrows(), 3, |i, j| {
        (0..3)
            .map(|q| (transformed[(i, q)] - translation[q]) * rotation[(j, q)])
            .sum::<f64>()
            / scale
    });
    assert!(rms(&recovered, &source) < 1e-14);
}

#[test]
fn pose_initialization_returns_a_valid_similarity() {
    let y = cloud(36);
    let angle: f64 = 0.35;
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
    let target = DMatrix::from_fn(y.nrows(), 3, |i, j| {
        1.1 * (0..3).map(|q| y[(i, q)] * rotation[(q, j)]).sum::<f64>() + [0.2, -0.1, 0.05][j]
    });
    let modes = DMatrix::zeros(y.len(), 1);
    let config = PoseMarginalizedConfig {
        rotation_count: 9,
        coarse_source_count: 36,
        coarse_target_count: 36,
        coarse_rank: 1,
        coarse_iterations: 5,
        refine_count: 3,
        refine_target_count: 36,
        refine_iterations: 12,
        parallel: false,
        ..Default::default()
    };
    let result = config.initialize(&y, &target, &modes, &[1.0]).unwrap();
    let points = DMatrix::from_fn(y.nrows(), 3, |i, j| {
        result.scale
            * (0..3)
                .map(|q| y[(i, q)] * result.rotation[(q, j)])
                .sum::<f64>()
            + result.translation[j]
    });
    assert!(
        rms(&points, &target) < 0.08,
        "rms={}",
        rms(&points, &target)
    );
    assert_eq!(result.hypotheses_evaluated, 9);
    assert_eq!(result.hypotheses_refined, 3);
    assert!(
        is_proper_rotation(&result.rotation),
        "returned pose rotation is not a proper rotation"
    );
}

fn is_proper_rotation(r: &DMatrix<f64>) -> bool {
    let identity = r.transpose() * r;
    let orthonormal = (0..3).all(|i| {
        (0..3).all(|j| {
            let expected = if i == j { 1.0 } else { 0.0 };
            (identity[(i, j)] - expected).abs() < 1e-9
        })
    });
    orthonormal && (r.determinant() - 1.0).abs() < 1e-9
}

#[test]
fn pose_initialization_with_screening_funnel_recovers() {
    // Exercise the screen -> survivor -> refine funnel: many rotations are
    // screened with a short EM, only a few survivors are refined. Previous
    // pose tests disabled the funnel (survivor_count > rotation_count).
    let y = cloud(40);
    let angle: f64 = 0.4;
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
    let target = DMatrix::from_fn(y.nrows(), 3, |i, j| {
        1.05 * (0..3).map(|q| y[(i, q)] * rotation[(q, j)]).sum::<f64>() + [0.15, -0.1, 0.05][j]
    });
    let modes = DMatrix::zeros(y.len(), 1);
    let config = PoseMarginalizedConfig {
        rotation_count: 40,
        coarse_source_count: 40,
        coarse_target_count: 40,
        coarse_rank: 1,
        coarse_iterations: 8,
        coarse_screen_iterations: 3, // screen short...
        coarse_survivor_count: 8,    // ...keep only 8 survivors (< 40): funnel ON
        refine_count: 6,
        refine_target_count: 40,
        refine_iterations: 20,
        parallel: false,
        ..Default::default()
    };
    let result = config.initialize(&y, &target, &modes, &[1.0]).unwrap();
    assert_eq!(result.hypotheses_evaluated, 40);
    assert!(result.hypotheses_refined <= 8);
    assert!(is_proper_rotation(&result.rotation));
    let points = DMatrix::from_fn(y.nrows(), 3, |i, j| {
        result.scale
            * (0..3)
                .map(|q| y[(i, q)] * result.rotation[(q, j)])
                .sum::<f64>()
            + result.translation[j]
    });
    assert!(rms(&points, &target) < 0.1, "rms={}", rms(&points, &target));
}

fn rms(a: &DMatrix<f64>, b: &DMatrix<f64>) -> f64 {
    (a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).powi(2))
        .sum::<f64>()
        / a.len() as f64)
        .sqrt()
}

#[test]
fn sigma2_matches_pairwise_definition() {
    let x = cloud(9);
    let y = cloud(7);
    let mut direct = 0.0;
    for i in 0..x.nrows() {
        for j in 0..y.nrows() {
            for q in 0..3 {
                direct += (x[(i, q)] - y[(j, q)]).powi(2);
            }
        }
    }
    direct /= (3 * x.nrows() * y.nrows()) as f64;
    assert_abs_diff_eq!(initialize_sigma2(&x, &y).unwrap(), direct, epsilon = 1e-12);
}

#[test]
fn rigid_recovers_known_similarity() {
    let y = cloud(60);
    let angle: f64 = 0.16;
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
    let scale = 1.06;
    let t = [0.18, -0.12, 0.08];
    let x = DMatrix::from_fn(y.nrows(), 3, |i, j| {
        scale * (0..3).map(|q| y[(i, q)] * r[(q, j)]).sum::<f64>() + t[j]
    });
    let result = RigidRegistration::new(
        &x,
        &y,
        RigidConfig {
            em: EmConfig {
                tolerance: 1e-8,
                max_iterations: 200,
                k: None,
                ..Default::default()
            },
            scale: true,
            normalize: false,
        },
    )
    .unwrap()
    .register()
    .unwrap();
    assert!(
        rms(&result.points, &x) < 1e-8,
        "rms={}",
        rms(&result.points, &x)
    );
    assert_abs_diff_eq!(result.scale, scale, epsilon = 1e-8);
}

#[test]
fn rigid_normalize_recovers_similarity_in_original_frame() {
    // Clouds in large, offset physical units. `normalize` conditions the
    // fit internally and must return the transform in the ORIGINAL frame.
    let base = cloud(60);
    let y = DMatrix::from_fn(60, 3, |i, j| {
        base[(i, j)] * 800.0 + [50_000.0, -30_000.0, 9000.0][j]
    });
    let angle: f64 = 0.21;
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
    let scale = 1.15;
    let t = [1200.0, -800.0, 400.0];
    let x = DMatrix::from_fn(y.nrows(), 3, |i, j| {
        scale * (0..3).map(|q| y[(i, q)] * r[(q, j)]).sum::<f64>() + t[j]
    });
    let result = RigidRegistration::new(
        &x,
        &y,
        RigidConfig {
            em: EmConfig {
                tolerance: 1e-10,
                max_iterations: 300,
                ..Default::default()
            },
            scale: true,
            normalize: true,
        },
    )
    .unwrap()
    .register()
    .unwrap();
    // Recovers the known transform in the original coordinate frame.
    assert!(
        rms(&result.points, &x) / rms(&x, &y) < 1e-6,
        "relative rms={}",
        rms(&result.points, &x) / rms(&x, &y)
    );
    assert_abs_diff_eq!(result.scale, scale, epsilon = 1e-5);
    // The returned (R, scale, t) reproduce the returned points from raw y.
    let reapplied = DMatrix::from_fn(y.nrows(), 3, |i, j| {
        result.scale
            * (0..3)
                .map(|q| y[(i, q)] * result.rotation[(q, j)])
                .sum::<f64>()
            + result.translation[j]
    });
    assert!(
        rms(&reapplied, &result.points) / rms(&x, &y) < 1e-9,
        "transform/points inconsistent: {}",
        rms(&reapplied, &result.points) / rms(&x, &y)
    );
}

#[test]
fn sparse_full_neighborhood_matches_dense() {
    let y = cloud(32);
    let x = DMatrix::from_fn(32, 3, |i, j| y[(i, j)] + [0.04, -0.02, 0.01][j]);
    let base = EmConfig {
        tolerance: 0.0,
        max_iterations: 6,
        k: None,
        ..Default::default()
    };
    let dense = RigidRegistration::new(
        &x,
        &y,
        RigidConfig {
            em: base.clone(),
            scale: true,
            normalize: false,
        },
    )
    .unwrap()
    .register()
    .unwrap();
    let sparse = RigidRegistration::new(
        &x,
        &y,
        RigidConfig {
            em: EmConfig {
                k: Some(x.nrows()),
                ..base
            },
            scale: true,
            normalize: false,
        },
    )
    .unwrap()
    .register()
    .unwrap();
    assert!(
        rms(&dense.points, &sparse.points) < 1e-11,
        "rms={}",
        rms(&dense.points, &sparse.points)
    );
}

#[test]
fn dense_parallel_and_serial_paths_match() {
    let y = cloud(80);
    let x = DMatrix::from_fn(80, 3, |i, j| y[(i, j)] + [0.04, -0.02, 0.01][j]);
    let run = |parallel| {
        RigidRegistration::new(
            &x,
            &y,
            RigidConfig {
                em: EmConfig {
                    tolerance: 0.0,
                    max_iterations: 8,
                    k: None,
                    parallel,
                    ..Default::default()
                },
                scale: true,
                normalize: false,
            },
        )
        .unwrap()
        .register()
        .unwrap()
    };
    let serial = run(false);
    let parallel = run(true);
    assert_abs_diff_eq!(serial.sigma2, parallel.sigma2, epsilon = 1e-15);
    assert_abs_diff_eq!(serial.objective, parallel.objective, epsilon = 1e-13);
    assert!(rms(&serial.points, &parallel.points) < 1e-15);
}

#[test]
fn affine_recovers_known_transform() {
    let y = cloud(60);
    let b = DMatrix::from_row_slice(
        3,
        3,
        &[1.03, 0.04, 0.0, -0.025, 0.98, 0.01, 0.0, 0.02, 1.01],
    );
    let t = [0.08, -0.05, 0.03];
    let x = DMatrix::from_fn(y.nrows(), 3, |i, j| {
        (0..3).map(|q| y[(i, q)] * b[(q, j)]).sum::<f64>() + t[j]
    });
    let result = AffineRegistration::new(
        &x,
        &y,
        AffineConfig {
            em: EmConfig {
                tolerance: 1e-8,
                max_iterations: 250,
                k: None,
                ..Default::default()
            },
            normalize: false,
        },
    )
    .unwrap()
    .register()
    .unwrap();
    assert!(
        rms(&result.points, &x) < 1e-7,
        "rms={}",
        rms(&result.points, &x)
    );
    assert!(
        (&result.transform - &b).norm() < 1e-6,
        "transform error={}",
        (&result.transform - &b).norm()
    );
    assert!(
        (&result.transform - b.transpose()).norm() > 1e-2,
        "regressed to transposed transform"
    );
}

#[test]
fn deformable_matches_generated_kernel_warp() {
    let y = cloud(30);
    let beta = 1.5;
    let g = gaussian_kernel(&y, &y, beta).unwrap();
    let w = DMatrix::from_fn(30, 3, |i, j| 0.004 * ((i * 3 + j + 1) as f64 * 0.41).sin());
    let x = &y + g * w;
    let result = DeformableRegistration::new(
        &x,
        &y,
        DeformableConfig {
            em: EmConfig {
                tolerance: 1e-7,
                max_iterations: 200,
                k: None,
                ..Default::default()
            },
            alpha: 2.0,
            beta,
            low_rank: None,
            ..Default::default()
        },
    )
    .unwrap()
    .register()
    .unwrap();
    assert!(
        rms(&result.points, &x) < 1e-6,
        "rms={}",
        rms(&result.points, &x)
    );
}

#[test]
fn atlas_recovers_coefficients_without_similarity() {
    let y = cloud(40);
    let rank = 3;
    let mut modes = DMatrix::from_fn(y.len(), rank, |i, k| {
        ((i + 1 + k * 7) as f64 * 0.19).sin() + 0.3 * ((i + 3 * k + 2) as f64 * 0.11).cos()
    });
    for k in 0..rank {
        for previous in 0..k {
            let projection = (0..modes.nrows())
                .map(|i| modes[(i, k)] * modes[(i, previous)])
                .sum::<f64>();
            for i in 0..modes.nrows() {
                modes[(i, k)] -= projection * modes[(i, previous)];
            }
        }
        let norm = (0..modes.nrows())
            .map(|i| modes[(i, k)].powi(2))
            .sum::<f64>()
            .sqrt();
        let column_scale = [0.5, 0.4, 0.3][k];
        for i in 0..modes.nrows() {
            modes[(i, k)] *= column_scale / norm;
        }
    }
    let truth = [0.35, -0.2, 0.15];
    let x = DMatrix::from_fn(40, 3, |i, j| {
        y[(i, j)]
            + (0..rank)
                .map(|k| modes[(i * 3 + j, k)] * truth[k])
                .sum::<f64>()
    });
    let result = AtlasRegistration::new(
        &x,
        &y,
        &modes,
        AtlasConfig {
            em: EmConfig {
                tolerance: 1e-5,
                max_iterations: 200,
                k: None,
                ..Default::default()
            },
            eigenvalues: vec![0.5, 0.3, 0.2],
            lambda_regularization: 0.001,
            optimize_similarity: false,
            ..Default::default()
        },
    )
    .unwrap()
    .register()
    .unwrap();
    assert!(
        rms(&result.points, &x) < 1e-5,
        "rms={}",
        rms(&result.points, &x)
    );
    for (actual, expected) in result.coefficients.iter().zip(truth) {
        assert_abs_diff_eq!(*actual, expected, epsilon = 2e-3);
    }
}

#[test]
fn atlas_adaptive_sparse_matches_dense() {
    // The adaptive sparse E-step stays dense while sigma2 is large and only
    // switches to the k-NN sparse E-step once the variance is small enough
    // that truncation is accurate. So it must recover the same coefficients
    // as the exact dense fit — unlike a fixed-k sparse fit from iteration 1,
    // which is biased while the mixture is still broad.
    let y = cloud(80);
    let rank = 3;
    let mut modes = DMatrix::from_fn(y.len(), rank, |i, k| {
        ((i + 1 + k * 7) as f64 * 0.19).sin() + 0.3 * ((i + 3 * k + 2) as f64 * 0.11).cos()
    });
    for k in 0..rank {
        for previous in 0..k {
            let projection = (0..modes.nrows())
                .map(|i| modes[(i, k)] * modes[(i, previous)])
                .sum::<f64>();
            for i in 0..modes.nrows() {
                modes[(i, k)] -= projection * modes[(i, previous)];
            }
        }
        let norm = (0..modes.nrows())
            .map(|i| modes[(i, k)].powi(2))
            .sum::<f64>()
            .sqrt();
        for i in 0..modes.nrows() {
            modes[(i, k)] *= [0.5, 0.4, 0.3][k] / norm;
        }
    }
    let truth = [0.3, -0.18, 0.12];
    let x = DMatrix::from_fn(80, 3, |i, j| {
        y[(i, j)]
            + (0..rank)
                .map(|k| modes[(i * 3 + j, k)] * truth[k])
                .sum::<f64>()
    });
    let run = |k, kdtree_radius_scale| {
        AtlasRegistration::new(
            &x,
            &y,
            &modes,
            AtlasConfig {
                em: EmConfig {
                    tolerance: 1e-6,
                    max_iterations: 200,
                    k,
                    ..Default::default()
                },
                eigenvalues: vec![0.5, 0.3, 0.2],
                lambda_regularization: 0.001,
                optimize_similarity: false,
                kdtree_radius_scale,
                ..Default::default()
            },
        )
        .unwrap()
        .register()
        .unwrap()
    };
    let dense = run(None, None);
    let adaptive = run(Some(20), Some(10.0)); // dense until sigma2<tau^2, then k=20 sparse
    for (a, dcoef) in adaptive.coefficients.iter().zip(&dense.coefficients) {
        assert_abs_diff_eq!(*a, *dcoef, epsilon = 1e-3);
    }
    for (a, truth) in adaptive.coefficients.iter().zip(truth) {
        assert_abs_diff_eq!(*a, truth, epsilon = 3e-3);
    }
}

#[test]
fn rigid_without_scale_recovers_pure_rotation_translation() {
    let y = cloud(60);
    let angle: f64 = 0.2;
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
    let t = [0.25, -0.1, 0.4];
    let x = DMatrix::from_fn(y.nrows(), 3, |i, j| {
        (0..3).map(|q| y[(i, q)] * r[(q, j)]).sum::<f64>() + t[j]
    });
    let result = RigidRegistration::new(
        &x,
        &y,
        RigidConfig {
            em: EmConfig {
                tolerance: 1e-10,
                max_iterations: 300,
                k: None,
                ..Default::default()
            },
            scale: false,
            normalize: false,
        },
    )
    .unwrap()
    .register()
    .unwrap();
    assert_abs_diff_eq!(result.scale, 1.0, epsilon = 0.0);
    assert!(
        rms(&result.points, &x) < 1e-8,
        "rms={}",
        rms(&result.points, &x)
    );
    // With a perfect fit the M-step variance must collapse toward zero.
    assert!(result.sigma2 < 1e-9, "sigma2={}", result.sigma2);
}

#[test]
fn rigid_recovers_2d_similarity() {
    let y = DMatrix::from_fn(50, 2, |i, j| {
        let z = i as f64 + 1.0;
        if j == 0 {
            (z * 0.31).sin() * 1.3 + 0.01 * z
        } else {
            (z * 0.17).cos() * 0.8
        }
    });
    let angle: f64 = 0.4;
    let mut x = DMatrix::<f64>::zeros(y.nrows(), 2);
    for i in 0..y.nrows() {
        x[(i, 0)] = 1.1 * (y[(i, 0)] * angle.cos() + y[(i, 1)] * angle.sin()) + 0.3;
        x[(i, 1)] = 1.1 * (-y[(i, 0)] * angle.sin() + y[(i, 1)] * angle.cos()) - 0.2;
    }
    for k in [None, Some(50)] {
        let result = RigidRegistration::new(
            &x,
            &y,
            RigidConfig {
                em: EmConfig {
                    tolerance: 1e-10,
                    max_iterations: 300,
                    k,
                    ..Default::default()
                },
                scale: true,
                normalize: false,
            },
        )
        .unwrap()
        .register()
        .unwrap();
        assert!(
            rms(&result.points, &x) < 1e-7,
            "k={:?} rms={}",
            k,
            rms(&result.points, &x)
        );
    }
}

#[test]
fn sparse_brute_force_matches_dense_above_three_dimensions() {
    let y = DMatrix::from_fn(40, 4, |i, j| {
        let z = i as f64 + 1.0;
        (z * 0.19 * (j + 1) as f64).sin() + 0.4 * (z * 0.07 * (j + 2) as f64).cos()
    });
    let x = DMatrix::from_fn(40, 4, |i, j| y[(i, j)] + [0.05, -0.03, 0.02, 0.01][j]);
    let base = EmConfig {
        tolerance: 0.0,
        max_iterations: 6,
        ..Default::default()
    };
    let dense = AffineRegistration::new(
        &x,
        &y,
        AffineConfig {
            em: base.clone(),
            normalize: false,
        },
    )
    .unwrap()
    .register()
    .unwrap();
    let sparse = AffineRegistration::new(
        &x,
        &y,
        AffineConfig {
            em: EmConfig {
                k: Some(x.nrows()),
                ..base
            },
            normalize: false,
        },
    )
    .unwrap()
    .register()
    .unwrap();
    assert!(
        rms(&dense.points, &sparse.points) < 1e-11,
        "rms={}",
        rms(&dense.points, &sparse.points)
    );
}

#[test]
fn rigid_with_outlier_weight_ignores_contaminated_targets() {
    let y = cloud(60);
    let angle: f64 = 0.1;
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
    let clean = DMatrix::from_fn(y.nrows(), 3, |i, j| {
        (0..3).map(|q| y[(i, q)] * r[(q, j)]).sum::<f64>() + [0.1, -0.05, 0.02][j]
    });
    // Append gross outliers far from the shape.
    let x = DMatrix::from_fn(y.nrows() + 6, 3, |i, j| {
        if i < y.nrows() {
            clean[(i, j)]
        } else {
            25.0 + (i * 3 + j) as f64
        }
    });
    let result = RigidRegistration::new(
        &x,
        &y,
        RigidConfig {
            em: EmConfig {
                tolerance: 1e-10,
                max_iterations: 300,
                outlier_weight: 0.2,
                ..Default::default()
            },
            scale: false,
            normalize: false,
        },
    )
    .unwrap()
    .register()
    .unwrap();
    assert!(
        rms(&result.points, &clean) < 1e-4,
        "rms={}",
        rms(&result.points, &clean)
    );
}

#[test]
fn low_rank_deformable_tracks_full_rank_solution() {
    let y = cloud(90);
    let beta = 2.0;
    let g = gaussian_kernel(&y, &y, beta).unwrap();
    let w = DMatrix::from_fn(90, 3, |i, j| 0.004 * ((i * 3 + j + 1) as f64 * 0.41).sin());
    let x = &y + g * w;
    let run = |low_rank| {
        DeformableRegistration::new(
            &x,
            &y,
            DeformableConfig {
                em: EmConfig {
                    tolerance: 1e-9,
                    max_iterations: 150,
                    ..Default::default()
                },
                alpha: 2.0,
                beta,
                low_rank,
                ..Default::default()
            },
        )
        .unwrap()
        .register()
        .unwrap()
    };
    let full = run(None);
    // A near-complete eigenbasis must reproduce the full-rank result to
    // approximation accuracy (the kernel's trailing eigenvalues sit at
    // numerical-noise level, so exact agreement is not expected).
    let near_full = run(Some(85));
    assert!(
        rms(&near_full.points, &full.points) < 1e-4,
        "rms={}",
        rms(&near_full.points, &full.points)
    );
    // A truncated basis should still register to approximation accuracy.
    let truncated = run(Some(30));
    assert!(
        rms(&truncated.points, &x) < 5e-2,
        "rms={}",
        rms(&truncated.points, &x)
    );
}

#[test]
fn pivoted_cholesky_matches_eigen_low_rank() {
    let y = cloud(120);
    let beta = 2.0;
    let g = gaussian_kernel(&y, &y, beta).unwrap();
    let w = DMatrix::from_fn(120, 3, |i, j| 0.004 * ((i * 3 + j + 1) as f64 * 0.41).sin());
    let x = &y + g * w;
    let run = |method| {
        DeformableRegistration::new(
            &x,
            &y,
            DeformableConfig {
                em: EmConfig {
                    tolerance: 1e-9,
                    max_iterations: 150,
                    ..Default::default()
                },
                alpha: 2.0,
                beta,
                low_rank: Some(60),
                low_rank_method: method,
                ..Default::default()
            },
        )
        .unwrap()
        .register()
        .unwrap()
    };
    let eigen = run(LowRankMethod::Eigen);
    let cholesky = run(LowRankMethod::PivotedCholesky);
    // Both build a rank-60 approximation of the same kernel; the pivoted
    // Cholesky factor, re-orthonormalized, must track the eigen basis to
    // approximation accuracy on the registered points.
    assert!(
        rms(&cholesky.points, &eigen.points) < 1e-3,
        "rms={}",
        rms(&cholesky.points, &eigen.points)
    );
}

#[test]
fn sparse_knn_truncated_tracks_dense() {
    // A genuine k-NN E-step (k far below N) must still recover the same
    // registration as the exact dense solve when each point's true match
    // is among its nearest neighbors. The existing sparse tests all use
    // k == N (every neighbor kept), so this is the first to exercise the
    // truncation / neighbor-selection path with edges actually dropped.
    let y = cloud(200);
    let x = DMatrix::from_fn(200, 3, |i, j| y[(i, j)] + [0.05, -0.03, 0.02][j]);
    let base = EmConfig {
        tolerance: 1e-9,
        max_iterations: 60,
        ..Default::default()
    };
    let run = |k| {
        RigidRegistration::new(
            &x,
            &y,
            RigidConfig {
                em: EmConfig { k, ..base.clone() },
                scale: true,
                normalize: false,
            },
        )
        .unwrap()
        .register()
        .unwrap()
    };
    let dense = run(None);
    let sparse = run(Some(10)); // 10 of 200 neighbors -> real truncation
    assert!(
        rms(&sparse.points, &x) < 1e-3,
        "sparse did not recover the transform: rms_vs_target={}",
        rms(&sparse.points, &x)
    );
    assert!(
        rms(&sparse.points, &dense.points) < 1e-3,
        "sparse diverged from dense: rms={}",
        rms(&sparse.points, &dense.points)
    );
}

#[test]
fn pivoted_cholesky_truncated_is_finite_and_deterministic() {
    // At an aggressively truncated rank the greedy pivoted-Cholesky
    // subspace is a *different* (non-optimal) rank-r approximation than
    // the top-r eigenbasis; the two are only guaranteed to agree once the
    // rank captures the warp (see `pivoted_cholesky_matches_eigen_low_rank`),
    // and low-rank truncation of a small fit is a numerically touchy
    // regime. So here we pin the two invariants that must hold regardless:
    // the truncated path stays finite, and it is bit-for-bit deterministic
    // between the parallel and serial builds.
    let y = cloud(120);
    let beta = 2.0;
    let g = gaussian_kernel(&y, &y, beta).unwrap();
    let w = DMatrix::from_fn(120, 3, |i, j| 0.01 * ((i * 3 + j + 1) as f64 * 0.41).sin());
    let x = &y + g * w;
    let run = |parallel| {
        DeformableRegistration::new(
            &x,
            &y,
            DeformableConfig {
                em: EmConfig {
                    tolerance: 1e-9,
                    max_iterations: 40,
                    parallel,
                    ..Default::default()
                },
                beta,
                low_rank: Some(30),
                low_rank_method: LowRankMethod::PivotedCholesky,
                ..Default::default()
            },
        )
        .unwrap()
        .register()
        .unwrap()
    };
    let parallel = run(true);
    let serial = run(false);
    assert!(
        parallel.points.iter().all(|v| v.is_finite()),
        "truncated pivoted Cholesky produced non-finite points"
    );
    assert!(
        rms(&parallel.points, &serial.points) < 1e-12,
        "pivoted Cholesky not deterministic across parallel/serial: rms={}",
        rms(&parallel.points, &serial.points)
    );
}

#[test]
fn pivoted_cholesky_handles_rank_deficient_kernel() {
    // Many coincident points make the kernel rank-deficient; the greedy
    // pivot must stop early (residual diagonal falls to the noise floor)
    // and still return a finite registration rather than dividing by a
    // zero pivot. Request far more rank than the kernel actually has.
    let mut y = cloud(60);
    for i in 30..60 {
        for j in 0..3 {
            y[(i, j)] = y[(i - 30, j)]; // duplicate the first 30 points
        }
    }
    let x = DMatrix::from_fn(60, 3, |i, j| y[(i, j)] + [0.02, -0.01, 0.015][j]);
    let result = DeformableRegistration::new(
        &x,
        &y,
        DeformableConfig {
            em: EmConfig {
                tolerance: 1e-8,
                max_iterations: 50,
                ..Default::default()
            },
            low_rank: Some(55), // > the ~30 effective rank
            low_rank_method: LowRankMethod::PivotedCholesky,
            ..Default::default()
        },
    )
    .unwrap()
    .register()
    .unwrap();
    assert!(
        result.points.iter().all(|v| v.is_finite()),
        "rank-deficient pivoted Cholesky produced non-finite points"
    );
    // The greedy pivot must actually have stopped early: 30 duplicated
    // points leave an effective rank of ~30, so the kept factor cannot
    // approach the requested 55 columns. A regression that kept pivoting
    // on roundoff columns would still be finite but fail this cap.
    let kept = result
        .low_rank_factor()
        .expect("low-rank fit must expose its factor")
        .ncols();
    assert!(
        kept <= 35,
        "early break did not trigger: kept {kept} columns"
    );
}

#[test]
fn rigid_degenerate_source_stays_finite() {
    // All-coincident source: the posterior-weighted spread `ypy` is zero,
    // so the scale estimate is unidentifiable. The result must stay finite
    // (scale left unchanged) rather than forming 0/0 = NaN.
    let y = DMatrix::from_fn(20, 3, |_, j| [1.0, -2.0, 0.5][j]);
    let x = cloud(20);
    let result = RigidRegistration::new(
        &x,
        &y,
        RigidConfig {
            em: EmConfig {
                tolerance: 1e-8,
                max_iterations: 20,
                ..Default::default()
            },
            scale: true,
            normalize: false,
        },
    )
    .unwrap()
    .register()
    .unwrap();
    // With a coincident source `ypy` is pure rounding noise, so the scale
    // must be HELD at its initial 1.0 — not merely finite. (The earlier
    // `> MIN_POSITIVE` guard let noise-level `ypy` through and produced
    // garbage finite scales here.)
    assert_eq!(result.scale, 1.0, "scale was estimated from noise");
    assert!(
        result.points.iter().all(|v| v.is_finite()),
        "degenerate rigid source produced non-finite points"
    );
    assert!(
        result.translation.iter().all(|v| v.is_finite()),
        "degenerate rigid source produced non-finite translation"
    );
}

#[test]
fn deformable_normalize_warps_in_original_frame() {
    // A smooth warp on clouds in large physical units. With `normalize`,
    // beta is interpreted in the internal unit-scale frame, and `points`
    // plus `transform` must come back in the original coordinates.
    let base = cloud(90);
    let y = DMatrix::from_fn(90, 3, |i, j| {
        base[(i, j)] * 500.0 + [20_000.0, 8000.0, -5000.0][j]
    });
    let gy = gaussian_kernel(&y, &y, 500.0 * 2.0).unwrap();
    let w = DMatrix::from_fn(90, 3, |i, j| 0.002 * ((i * 3 + j + 1) as f64 * 0.41).sin());
    let x = &y + gy * w;
    let result = DeformableRegistration::new(
        &x,
        &y,
        DeformableConfig {
            em: EmConfig {
                tolerance: 1e-10,
                max_iterations: 200,
                ..Default::default()
            },
            beta: 2.0,
            low_rank: None,
            normalize: true,
            ..Default::default()
        },
    )
    .unwrap()
    .register()
    .unwrap();
    assert!(result.points.iter().all(|v| v.is_finite()));
    // Registered to the target in the original frame.
    assert!(
        rms(&result.points, &x) / rms(&x, &y) < 0.2,
        "relative residual={}",
        rms(&result.points, &x) / rms(&x, &y)
    );
    // transform() on the training source reproduces the fitted points,
    // confirming the normalization round-trips through the warp.
    let retransformed = result.transform(&y).unwrap();
    assert!(
        rms(&retransformed, &result.points) / rms(&x, &y) < 1e-9,
        "transform inconsistent with points: {}",
        rms(&retransformed, &result.points) / rms(&x, &y)
    );
}

#[test]
fn affine_normalize_recovers_transform_in_original_frame() {
    let base = cloud(70);
    let y = DMatrix::from_fn(70, 3, |i, j| {
        base[(i, j)] * 600.0 + [40_000.0, -20_000.0, 7000.0][j]
    });
    // A mild affine (like `affine_recovers_known_transform`) so CPD's
    // correspondences lock exactly; the point here is that `normalize`
    // recovers it on large-coordinate data, not CPD's affine capacity.
    let b = DMatrix::from_row_slice(
        3,
        3,
        &[1.03, 0.04, 0.0, -0.025, 0.98, 0.01, 0.0, 0.02, 1.01],
    );
    let t = [500.0, -300.0, 150.0];
    let x = DMatrix::from_fn(70, 3, |i, j| {
        (0..3).map(|q| y[(i, q)] * b[(q, j)]).sum::<f64>() + t[j]
    });
    let result = AffineRegistration::new(
        &x,
        &y,
        AffineConfig {
            em: EmConfig {
                tolerance: 1e-10,
                max_iterations: 300,
                ..Default::default()
            },
            normalize: true,
        },
    )
    .unwrap()
    .register()
    .unwrap();
    assert!(
        rms(&result.points, &x) / rms(&x, &y) < 1e-5,
        "relative rms={}",
        rms(&result.points, &x) / rms(&x, &y)
    );
    // Returned (B, t) reproduce the points from raw y.
    let reapplied = DMatrix::from_fn(70, 3, |i, j| {
        (0..3)
            .map(|q| y[(i, q)] * result.transform[(q, j)])
            .sum::<f64>()
            + result.translation[j]
    });
    assert!(
        rms(&reapplied, &result.points) / rms(&x, &y) < 1e-9,
        "transform/points inconsistent: {}",
        rms(&reapplied, &result.points) / rms(&x, &y)
    );
    assert!(
        (&result.transform - &b).norm() < 1e-5,
        "transform error={}",
        (&result.transform - &b).norm()
    );
}

#[test]
fn register_with_callback_observes_and_stops_early() {
    let y = cloud(40);
    let x = DMatrix::from_fn(40, 3, |i, j| y[(i, j)] + [0.05, -0.03, 0.02][j]);
    // Observe: the callback sees monotonically advancing iterations and a
    // points snapshot of the right shape.
    let mut seen = Vec::new();
    let observed = RigidRegistration::new(
        &x,
        &y,
        RigidConfig {
            em: EmConfig {
                tolerance: 1e-10,
                max_iterations: 50,
                ..Default::default()
            },
            scale: true,
            normalize: false,
        },
    )
    .unwrap()
    .register_with(|state| {
        assert_eq!(state.points.shape(), (40, 3));
        seen.push((state.iteration, state.sigma2));
        true
    })
    .unwrap();
    assert_eq!(seen.len(), observed.iterations);
    assert_eq!(seen.first().unwrap().0, 1);
    assert_eq!(seen.last().unwrap().0, observed.iterations);

    // Early stop: returning false after 3 iterations halts there.
    let stopped = RigidRegistration::new(
        &x,
        &y,
        RigidConfig {
            em: EmConfig {
                tolerance: 0.0,
                max_iterations: 50,
                ..Default::default()
            },
            scale: true,
            normalize: false,
        },
    )
    .unwrap()
    .register_with(|state| state.iteration < 3)
    .unwrap();
    assert_eq!(stopped.iterations, 3);
}

#[test]
fn constrained_deformable_pins_landmarks() {
    let y = cloud(50);
    let x = DMatrix::from_fn(50, 3, |i, j| y[(i, j)] + [0.3, -0.2, 0.1][j]);
    let result = DeformableRegistration::new(
        &x,
        &y,
        DeformableConfig {
            em: EmConfig {
                tolerance: 1e-8,
                max_iterations: 200,
                ..Default::default()
            },
            alpha: 2.0,
            beta: 2.0,
            low_rank: None,
            constraints: vec![
                Constraint {
                    source: 0,
                    target: 0,
                },
                Constraint {
                    source: 25,
                    target: 25,
                },
            ],
            constraint_error: 1e-8,
            low_rank_method: LowRankMethod::Eigen,
            pivoted_cholesky_tolerance: 0.0,
            normalize: false,
        },
    )
    .unwrap()
    .register()
    .unwrap();
    for source in [0usize, 25] {
        let distance: f64 = (0..3)
            .map(|j| (result.points[(source, j)] - x[(source, j)]).powi(2))
            .sum::<f64>()
            .sqrt();
        assert!(distance < 1e-6, "source {source} distance {distance}");
    }
}

#[test]
fn low_rank_exposes_factors_and_skips_dense_kernel() {
    let y = cloud(80);
    let g = gaussian_kernel(&y, &y, 2.0).unwrap();
    let w = DMatrix::from_fn(80, 3, |i, j| 0.004 * ((i * 3 + j + 1) as f64 * 0.41).sin());
    let x = &y + g * w;
    let cfg = |low_rank| DeformableConfig {
        em: EmConfig {
            tolerance: 1e-9,
            max_iterations: 100,
            ..Default::default()
        },
        beta: 2.0,
        low_rank,
        ..Default::default()
    };
    // Low-rank fit: no dense kernel, factors present, and Q·diag(Λ)·Qᵀ
    // reconstructs the true kernel to approximation accuracy.
    let low = DeformableRegistration::new(&x, &y, cfg(Some(40)))
        .unwrap()
        .register()
        .unwrap();
    assert_eq!(
        low.kernel.shape(),
        (0, 0),
        "low-rank fit kept a dense kernel"
    );
    let (q, lambda) = low.low_rank_spectrum().expect("low-rank spectrum missing");
    assert_eq!(q.ncols(), lambda.len());
    // Q has orthonormal columns.
    let qtq = q.tr_mul(&q);
    for i in 0..qtq.nrows() {
        for j in 0..qtq.ncols() {
            let expected = if i == j { 1.0 } else { 0.0 };
            // 1e-6 rather than tighter: columns whose Gram eigenvalue sits
            // just above the eps floor get scaled by 1/sqrt(s), amplifying
            // rounding; the bound must hold across platforms.
            assert!((qtq[(i, j)] - expected).abs() < 1e-6);
        }
    }

    // Full-rank fit: dense kernel present, no factors.
    let full = DeformableRegistration::new(&x, &y, cfg(None))
        .unwrap()
        .register()
        .unwrap();
    assert_eq!(full.kernel.shape(), (80, 80));
    assert!(full.low_rank_factor().is_none());
    assert!(full.low_rank_basis().is_none());
    assert!(full.low_rank_eigenvalues().is_none());
}

#[test]
fn pivoted_cholesky_tolerance_limits_rank() {
    let y = cloud(100);
    let x = DMatrix::from_fn(100, 3, |i, j| y[(i, j)] + [0.02, -0.01, 0.015][j]);
    let run = |tol| {
        DeformableRegistration::new(
            &x,
            &y,
            DeformableConfig {
                em: EmConfig {
                    tolerance: 1e-8,
                    max_iterations: 30,
                    ..Default::default()
                },
                beta: 2.0,
                low_rank: Some(90),
                low_rank_method: LowRankMethod::PivotedCholesky,
                pivoted_cholesky_tolerance: tol,
                ..Default::default()
            },
        )
        .unwrap()
        .register()
        .unwrap()
    };
    let tight = run(0.0);
    let loose = run(0.05); // stop pivoting once residual diagonal <= 0.05
    let tight_rank = tight.low_rank_factor().unwrap().ncols();
    let loose_rank = loose.low_rank_factor().unwrap().ncols();
    assert!(
        loose_rank < tight_rank,
        "a larger tolerance should keep fewer pivots: loose={loose_rank}, tight={tight_rank}"
    );
}

#[test]
fn constraints_accumulate_distinct_targets_and_dedup() {
    let y = cloud(40);
    let x = cloud(40);
    // Pin source 0 to two DISTINCT targets (5 and 17): the accumulated
    // constraint should draw it toward their mean, not to either alone.
    let make = |constraints| {
        DeformableRegistration::new(
            &x,
            &y,
            DeformableConfig {
                em: EmConfig {
                    tolerance: 1e-9,
                    max_iterations: 300,
                    ..Default::default()
                },
                beta: 2.0,
                low_rank: None,
                constraints,
                constraint_error: 1e-10,
                ..Default::default()
            },
        )
        .unwrap()
        .register()
        .unwrap()
    };
    let two_targets = make(vec![
        Constraint {
            source: 0,
            target: 5,
        },
        Constraint {
            source: 0,
            target: 17,
        },
    ]);
    let mean_target: Vec<f64> = (0..3).map(|j| 0.5 * (x[(5, j)] + x[(17, j)])).collect();
    let to_mean: f64 = (0..3)
        .map(|j| (two_targets.points[(0, j)] - mean_target[j]).powi(2))
        .sum::<f64>()
        .sqrt();
    assert!(
        to_mean < 1e-4,
        "source 0 not pinned to target mean: {to_mean}"
    );

    // Duplicate identical pairs must be deduplicated: pinning (0->5) twice
    // behaves exactly like pinning it once (mass 1, not 2).
    let once = make(vec![Constraint {
        source: 0,
        target: 5,
    }]);
    let twice = make(vec![
        Constraint {
            source: 0,
            target: 5,
        },
        Constraint {
            source: 0,
            target: 5,
        },
    ]);
    assert!(
        rms(&once.points, &twice.points) < 1e-12,
        "duplicate constraint pair changed the result: rms={}",
        rms(&once.points, &twice.points)
    );
}

#[test]
fn invalid_inputs_are_rejected() {
    use rustcpd::Error;
    let y = cloud(10);
    let empty = DMatrix::<f64>::zeros(0, 3);
    assert_eq!(
        RigidRegistration::new(&empty, &y, RigidConfig::default()).err(),
        Some(Error::EmptyPointCloud)
    );
    let two_d = DMatrix::<f64>::zeros(10, 2);
    assert_eq!(
        RigidRegistration::new(&two_d, &y, RigidConfig::default()).err(),
        Some(Error::DimensionMismatch)
    );
    let mut non_finite = y.clone();
    non_finite[(0, 0)] = f64::NAN;
    assert_eq!(
        RigidRegistration::new(&non_finite, &y, RigidConfig::default()).err(),
        Some(Error::NonFiniteInput)
    );
    let bad_weight = RigidConfig {
        em: EmConfig {
            outlier_weight: 1.0,
            ..Default::default()
        },
        ..Default::default()
    };
    assert_eq!(
        RigidRegistration::new(&y, &y, bad_weight).err(),
        Some(Error::InvalidOutlierWeight)
    );
    let out_of_bounds = DeformableConfig {
        constraints: vec![Constraint {
            source: 99,
            target: 0,
        }],
        ..Default::default()
    };
    assert_eq!(
        DeformableRegistration::new(&y, &y, out_of_bounds).err(),
        Some(Error::ConstraintOutOfBounds)
    );
    // A rank-0 low-rank request is degenerate and must be rejected up
    // front (the eigen path would silently freeze the points otherwise).
    let rank_zero = DeformableConfig {
        low_rank: Some(0),
        ..Default::default()
    };
    assert_eq!(
        DeformableRegistration::new(&y, &y, rank_zero).err(),
        Some(Error::PositiveParameter("low_rank"))
    );
    let bad_tolerance = DeformableConfig {
        pivoted_cholesky_tolerance: 1.5,
        ..Default::default()
    };
    assert_eq!(
        DeformableRegistration::new(&y, &y, bad_tolerance).err(),
        Some(Error::PositiveParameter("pivoted_cholesky_tolerance"))
    );
}

#[test]
fn atlas_invalid_initializers_are_rejected() {
    use rustcpd::Error;

    let mean = cloud(12);
    let target = mean.clone();
    let modes = DMatrix::from_fn(mean.len(), 1, |row, _| ((row + 1) as f64 * 0.17).sin());
    let config = || AtlasConfig {
        eigenvalues: vec![1.0],
        ..Default::default()
    };

    let mut bad = config();
    bad.initial_scale = 0.0;
    assert_eq!(
        AtlasRegistration::new(&target, &mean, &modes, bad).err(),
        Some(Error::PositiveParameter("initial_scale"))
    );

    let mut bad = config();
    bad.initial_scale = f64::NAN;
    assert_eq!(
        AtlasRegistration::new(&target, &mean, &modes, bad).err(),
        Some(Error::PositiveParameter("initial_scale"))
    );

    let mut bad = config();
    bad.initial_coefficients = Some(vec![f64::NAN]);
    assert_eq!(
        AtlasRegistration::new(&target, &mean, &modes, bad).err(),
        Some(Error::NonFiniteInput)
    );

    let mut bad = config();
    bad.initial_translation = Some(vec![f64::NAN, 0.0, 0.0]);
    assert_eq!(
        AtlasRegistration::new(&target, &mean, &modes, bad).err(),
        Some(Error::NonFiniteInput)
    );

    let mut bad_rotation = DMatrix::identity(3, 3);
    bad_rotation[(0, 0)] = f64::NAN;
    let mut bad = config();
    bad.initial_rotation = Some(bad_rotation);
    assert_eq!(
        AtlasRegistration::new(&target, &mean, &modes, bad).err(),
        Some(Error::NonFiniteInput)
    );

    let mut bad_modes = modes.clone();
    bad_modes[(0, 0)] = f64::NAN;
    assert_eq!(
        AtlasRegistration::new(&target, &mean, &bad_modes, config()).err(),
        Some(Error::NonFiniteInput)
    );
}

#[test]
fn atlas_degenerate_inputs_keep_variance_and_scale_finite() {
    let collapsed = DMatrix::from_fn(12, 3, |_, j| [2.0, -3.0, 1.0][j]);
    let modes = DMatrix::from_fn(collapsed.len(), 1, |row, _| ((row + 1) as f64 * 0.19).sin());
    let collapsed_result = AtlasRegistration::new(
        &collapsed,
        &collapsed,
        &modes,
        AtlasConfig {
            em: EmConfig {
                sigma2: Some(1.0),
                tolerance: 0.0,
                max_iterations: 2,
                ..Default::default()
            },
            eigenvalues: vec![1.0],
            optimize_similarity: false,
            with_scale: false,
            ..Default::default()
        },
    )
    .unwrap()
    .register()
    .unwrap();
    assert!(collapsed_result.sigma2.is_finite());
    assert!(collapsed_result.sigma2 > 0.0);

    let target = cloud(20);
    let mean = DMatrix::from_fn(20, 3, |_, j| [1.0, -2.0, 0.5][j]);
    let zero_modes = DMatrix::zeros(mean.len(), 1);
    let scale_result = AtlasRegistration::new(
        &target,
        &mean,
        &zero_modes,
        AtlasConfig {
            em: EmConfig {
                sigma2: Some(1.0),
                tolerance: 0.0,
                max_iterations: 5,
                ..Default::default()
            },
            eigenvalues: vec![1.0],
            lambda_regularization: 1.0,
            optimize_similarity: true,
            with_scale: true,
            initial_scale: 1.0,
            ..Default::default()
        },
    )
    .unwrap()
    .register()
    .unwrap();
    assert_eq!(scale_result.scale, 1.0);
    assert!(scale_result.points.iter().all(|value| value.is_finite()));
    assert!(
        scale_result
            .translation
            .iter()
            .all(|value| value.is_finite())
    );
}

#[test]
fn atlas_normalized_sigma2_returns_in_original_frame() {
    let mean = cloud(40);
    let modes = DMatrix::from_fn(mean.len(), 1, |row, _| {
        0.05 * ((row + 1) as f64 * 0.23).sin()
    });
    let target = DMatrix::from_fn(40, 3, |i, j| mean[(i, j)] + 0.4 * modes[(i * 3 + j, 0)]);
    let run = |target: &DMatrix<f64>, mean: &DMatrix<f64>, modes: &DMatrix<f64>| {
        AtlasRegistration::new(
            target,
            mean,
            modes,
            AtlasConfig {
                em: EmConfig {
                    tolerance: 0.0,
                    max_iterations: 10,
                    ..Default::default()
                },
                eigenvalues: vec![1.0],
                lambda_regularization: 1.0,
                normalize: true,
                optimize_similarity: false,
                with_scale: false,
                ..Default::default()
            },
        )
        .unwrap()
        .register()
        .unwrap()
    };
    let base = run(&target, &mean, &modes);

    let coordinate_scale = 250.0;
    let offset = [40_000.0, -20_000.0, 7_000.0];
    let scaled_mean = DMatrix::from_fn(40, 3, |i, j| coordinate_scale * mean[(i, j)] + offset[j]);
    let scaled_target =
        DMatrix::from_fn(40, 3, |i, j| coordinate_scale * target[(i, j)] + offset[j]);
    let scaled_modes = &modes * coordinate_scale;
    let scaled = run(&scaled_target, &scaled_mean, &scaled_modes);
    let expected = base.sigma2 * coordinate_scale * coordinate_scale;
    assert!(
        (scaled.sigma2 - expected).abs() <= 1e-10 * expected.max(1.0),
        "sigma2 {} vs expected {expected}",
        scaled.sigma2
    );
}

#[test]
fn single_precision_estep_is_close_and_deterministic() {
    let y = cloud(120);
    let angle: f64 = 0.14;
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
    let x = DMatrix::from_fn(y.nrows(), 3, |i, j| {
        1.05 * (0..3).map(|q| y[(i, q)] * r[(q, j)]).sum::<f64>() + [0.15, -0.1, 0.05][j]
    });
    let run = |single_precision: bool, parallel: bool| {
        RigidRegistration::new(
            &x,
            &y,
            RigidConfig {
                em: EmConfig {
                    tolerance: 1e-8,
                    max_iterations: 200,
                    parallel,
                    single_precision,
                    ..Default::default()
                },
                scale: true,
                normalize: false,
            },
        )
        .unwrap()
        .register()
        .unwrap()
    };
    let double = run(false, true);
    let single = run(true, true);
    // Same registration to well beyond f32 resolution of the workload.
    assert!(
        rms(&single.points, &double.points) < 1e-5,
        "rms={}",
        rms(&single.points, &double.points)
    );
    assert!(rms(&single.points, &x) < 1e-4);
    // Parallel and serial single-precision runs stay bitwise identical.
    let serial = run(true, false);
    assert!(
        single
            .points
            .iter()
            .zip(serial.points.iter())
            .all(|(a, b)| a == b),
        "single-precision parallel/serial mismatch"
    );
}

#[test]
fn deformable_transform_reproduces_training_points() {
    let y = cloud(40);
    let beta = 1.5;
    let g = gaussian_kernel(&y, &y, beta).unwrap();
    let w = DMatrix::from_fn(40, 3, |i, j| 0.004 * ((i * 3 + j + 1) as f64 * 0.41).sin());
    let x = &y + g * w;
    let result = DeformableRegistration::new(
        &x,
        &y,
        DeformableConfig {
            em: EmConfig {
                tolerance: 1e-8,
                max_iterations: 200,
                ..Default::default()
            },
            alpha: 2.0,
            beta,
            low_rank: None,
            ..Default::default()
        },
    )
    .unwrap()
    .register()
    .unwrap();
    // Evaluating the field at the training points must reproduce `points`.
    let reapplied = result.transform(&y).unwrap();
    assert!(
        rms(&reapplied, &result.points) < 1e-12,
        "rms={}",
        rms(&reapplied, &result.points)
    );
    // A denser set of points (here a midpoint-refined copy) warps smoothly:
    // midpoints of the deformed field stay near the deformed midpoints of y.
    let midpoints = DMatrix::from_fn(39, 3, |i, j| 0.5 * (y[(i, j)] + y[(i + 1, j)]));
    let warped_midpoints = result.transform(&midpoints).unwrap();
    let deformed_midpoints = DMatrix::from_fn(39, 3, |i, j| {
        0.5 * (result.points[(i, j)] + result.points[(i + 1, j)])
    });
    assert!(
        rms(&warped_midpoints, &deformed_midpoints) < 5e-3,
        "rms={}",
        rms(&warped_midpoints, &deformed_midpoints)
    );
}

#[test]
fn deformable_transform_rejects_wrong_dimension() {
    let y = cloud(20);
    let x = DMatrix::from_fn(20, 3, |i, j| y[(i, j)] + 0.01);
    let result = DeformableRegistration::new(&x, &y, DeformableConfig::default())
        .unwrap()
        .register()
        .unwrap();
    assert!(result.transform(&DMatrix::<f64>::zeros(5, 2)).is_err());
}

#[test]
fn correspondences_recover_identity_matching() {
    let y = cloud(30);
    // Target is a lightly translated copy in the same point order, so after
    // registration source i should correspond to target i.
    let x = DMatrix::from_fn(30, 3, |i, j| y[(i, j)] + [0.03, -0.02, 0.01][j]);
    let result = RigidRegistration::new(
        &x,
        &y,
        RigidConfig {
            em: EmConfig {
                tolerance: 1e-10,
                max_iterations: 200,
                ..Default::default()
            },
            scale: false,
            normalize: false,
        },
    )
    .unwrap()
    .register()
    .unwrap();
    let matching = correspondences(&x, &result.points, result.sigma2, 0.0).unwrap();
    assert_eq!(matching.matches.len(), 30);
    let correct = matching
        .matches
        .iter()
        .enumerate()
        .filter(|&(i, &j)| i == j)
        .count();
    assert!(correct >= 28, "only {correct}/30 matched to themselves");
    // Each column of the posterior sums (with the outlier mass) to one.
    for j in 0..30 {
        let column_sum: f64 = (0..30).map(|i| matching.posterior[(i, j)]).sum();
        assert!(column_sum <= 1.0 + 1e-9, "column {j} sum {column_sum}");
    }
    assert!(
        matching
            .probability
            .iter()
            .all(|&p| (0.0..=1.0).contains(&p))
    );
}

#[test]
fn correspondences_validate_inputs() {
    let y = cloud(10);
    assert!(correspondences(&y, &y, -1.0, 0.0).is_err());
    assert!(correspondences(&y, &y, 1.0, 1.0).is_err());
    assert!(correspondences(&y, &DMatrix::<f64>::zeros(10, 2), 1.0, 0.0).is_err());
}

#[test]
fn atlas_reconstruct_reproduces_points_and_upsamples() {
    // Build an orthonormal mode set at full resolution, register on a
    // subsample, then reconstruct at full resolution.
    let full = cloud(60);
    let rank = 3;
    let mut modes_full = DMatrix::from_fn(full.len(), rank, |i, k| {
        ((i + 1 + k * 7) as f64 * 0.19).sin() + 0.3 * ((i + 3 * k + 2) as f64 * 0.11).cos()
    });
    for k in 0..rank {
        for previous in 0..k {
            let projection = (0..modes_full.nrows())
                .map(|i| modes_full[(i, k)] * modes_full[(i, previous)])
                .sum::<f64>();
            for i in 0..modes_full.nrows() {
                modes_full[(i, k)] -= projection * modes_full[(i, previous)];
            }
        }
        let norm = (0..modes_full.nrows())
            .map(|i| modes_full[(i, k)].powi(2))
            .sum::<f64>()
            .sqrt();
        let column_scale = [0.5, 0.4, 0.3][k];
        for i in 0..modes_full.nrows() {
            modes_full[(i, k)] *= column_scale / norm;
        }
    }
    let truth = [0.3, -0.18, 0.12];
    // Target built from a rotated+scaled+translated deformed full model.
    let angle: f64 = 0.25;
    let deformed_full = DMatrix::from_fn(full.nrows(), 3, |i, j| {
        full[(i, j)]
            + (0..rank)
                .map(|k| modes_full[(i * 3 + j, k)] * truth[k])
                .sum::<f64>()
    });
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
    let target = DMatrix::from_fn(full.nrows(), 3, |i, j| {
        1.2 * (0..3)
            .map(|q| deformed_full[(i, q)] * r[(q, j)])
            .sum::<f64>()
            + [0.3, -0.2, 0.1][j]
    });

    for normalize in [false, true] {
        let result = AtlasRegistration::new(
            &target,
            &full,
            &modes_full,
            AtlasConfig {
                em: EmConfig {
                    tolerance: 1e-9,
                    max_iterations: 300,
                    ..Default::default()
                },
                eigenvalues: vec![0.5, 0.4, 0.3],
                lambda_regularization: 1e-4,
                normalize,
                optimize_similarity: true,
                ..Default::default()
            },
        )
        .unwrap()
        .register()
        .unwrap();
        // reconstruct with the SAME mean/modes must reproduce points exactly.
        let rebuilt = result.reconstruct(&full, &modes_full).unwrap();
        assert!(
            rms(&rebuilt, &result.points) < 1e-9,
            "normalize={normalize} rms={}",
            rms(&rebuilt, &result.points)
        );
        // apply_similarity on the deformed model equals reconstruct.
        let deformed = DMatrix::from_fn(full.nrows(), 3, |i, j| {
            full[(i, j)]
                + (0..rank)
                    .map(|k| modes_full[(i * 3 + j, k)] * result.coefficients[k])
                    .sum::<f64>()
        });
        let via_similarity = result.apply_similarity(&deformed).unwrap();
        assert!(rms(&via_similarity, &result.points) < 1e-9);
    }
}

#[test]
fn atlas_reconstruct_validates_shapes() {
    let y = cloud(30);
    let modes = DMatrix::from_fn(y.len(), 2, |i, k| ((i + k + 1) as f64 * 0.2).sin());
    let result = AtlasRegistration::new(
        &y,
        &y,
        &modes,
        AtlasConfig {
            em: EmConfig {
                max_iterations: 5,
                ..Default::default()
            },
            eigenvalues: vec![0.5, 0.3],
            optimize_similarity: false,
            ..Default::default()
        },
    )
    .unwrap()
    .register()
    .unwrap();
    // Wrong mode count for the fitted coefficients.
    let bad_modes = DMatrix::from_fn(y.len(), 3, |_, _| 0.0);
    assert!(result.reconstruct(&y, &bad_modes).is_err());
    // Wrong dimensionality.
    assert!(
        result
            .apply_similarity(&DMatrix::<f64>::zeros(5, 2))
            .is_err()
    );
}
