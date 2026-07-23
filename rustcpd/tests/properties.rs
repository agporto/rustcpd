//! Property-based invariants: relationships that must hold for *any* valid
//! input, checked across many randomized cases.

use proptest::prelude::*;
use rustcpd::{
    Constraint, DMatrix, DeformableConfig, DeformableRegistration, EmConfig, RigidConfig,
    RigidRegistration, correspondences, gaussian_kernel, initialize_sigma2,
};

/// A cloud of `count` points in `dims` dimensions drawn from a bounded box,
/// deterministic in the sampled coordinates.
fn cloud_strategy(count: usize, dims: usize) -> impl Strategy<Value = DMatrix<f64>> {
    prop::collection::vec(-5.0f64..5.0, count * dims)
        .prop_map(move |values| DMatrix::from_row_slice(count, dims, &values))
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(48))]

    /// Rigid registration always yields an orthonormal rotation with
    /// determinant +1 (a proper rotation, never a reflection).
    #[test]
    fn rigid_rotation_is_a_proper_rotation(
        source in cloud_strategy(40, 3),
        offset in prop::array::uniform3(-2.0f64..2.0),
    ) {
        let target = DMatrix::from_fn(source.nrows(), 3, |i, j| source[(i, j)] + offset[j]);
        let result = RigidRegistration::new(
            &target,
            &source,
            RigidConfig {
                em: EmConfig { max_iterations: 30, ..Default::default() },
                scale: true,
                normalize: false,            },
        )
        .unwrap()
        .register()
        .unwrap();
        let r = &result.rotation;
        let gram = r.transpose() * r;
        for i in 0..3 {
            for j in 0..3 {
                let expected = if i == j { 1.0 } else { 0.0 };
                prop_assert!((gram[(i, j)] - expected).abs() < 1e-9);
            }
        }
        prop_assert!((r.determinant() - 1.0).abs() < 1e-9);
        prop_assert!(result.scale > 0.0);
    }

    /// Posterior columns are proper (sub-)distributions: every entry in
    /// [0, 1], each column summing with the outlier mass to at most one.
    #[test]
    fn posterior_is_a_valid_distribution(
        source in cloud_strategy(30, 3),
        weight in 0.0f64..0.6,
    ) {
        let target = DMatrix::from_fn(source.nrows(), 3, |i, j| source[(i, j)] + 0.05);
        let sigma2 = initialize_sigma2(&target, &source).unwrap();
        let matching = correspondences(&target, &source, sigma2, weight).unwrap();
        for value in matching.posterior.iter() {
            prop_assert!((0.0..=1.0).contains(value));
        }
        for j in 0..target.nrows() {
            let column_sum: f64 = (0..source.nrows()).map(|i| matching.posterior[(i, j)]).sum();
            prop_assert!(column_sum <= 1.0 + 1e-9, "column {j} sum {column_sum}");
        }
        for &p in &matching.probability {
            prop_assert!((0.0..=1.0).contains(&p));
        }
        prop_assert_eq!(matching.matches.len(), source.nrows());
    }

    /// `sigma2` initialization is symmetric and strictly positive.
    #[test]
    fn initialize_sigma2_is_symmetric_and_positive(
        a in cloud_strategy(20, 3),
        b in cloud_strategy(15, 3),
    ) {
        let ab = initialize_sigma2(&a, &b).unwrap();
        let ba = initialize_sigma2(&b, &a).unwrap();
        prop_assert!(ab > 0.0);
        prop_assert!((ab - ba).abs() < 1e-9 * ab.max(1.0));
    }

    /// The Gaussian kernel is symmetric with unit diagonal and entries in
    /// (0, 1] for any cloud and bandwidth.
    #[test]
    fn gaussian_kernel_is_symmetric_with_unit_diagonal(
        y in cloud_strategy(25, 3),
        beta in 0.2f64..4.0,
    ) {
        let g = gaussian_kernel(&y, &y, beta).unwrap();
        for i in 0..y.nrows() {
            prop_assert!((g[(i, i)] - 1.0).abs() < 1e-12);
            for j in 0..y.nrows() {
                // Far-apart points may underflow the Gaussian to exactly 0.
                prop_assert!(g[(i, j)] >= 0.0 && g[(i, j)] <= 1.0 + 1e-12);
                prop_assert!((g[(i, j)] - g[(j, i)]).abs() < 1e-12);
            }
        }
    }

    /// The deformable field evaluated at its own training points exactly
    /// reproduces the registered output, for any target perturbation.
    #[test]
    fn deformable_transform_round_trips(
        source in cloud_strategy(30, 3),
        shift in prop::array::uniform3(-0.5f64..0.5),
    ) {
        let target = DMatrix::from_fn(source.nrows(), 3, |i, j| source[(i, j)] + shift[j]);
        let result = DeformableRegistration::new(
            &target,
            &source,
            DeformableConfig {
                em: EmConfig { max_iterations: 25, ..Default::default() },
                low_rank: None,
                ..Default::default()
            },
        )
        .unwrap()
        .register()
        .unwrap();
        let reapplied = result.transform(&source).unwrap();
        for (a, b) in reapplied.iter().zip(result.points.iter()) {
            prop_assert!((a - b).abs() < 1e-9);
        }
    }

    /// Parallel and serial execution are bitwise-identical for any inputs.
    #[test]
    fn parallel_matches_serial_bitwise(
        source in cloud_strategy(50, 3),
        offset in prop::array::uniform3(-1.0f64..1.0),
    ) {
        let target = DMatrix::from_fn(source.nrows(), 3, |i, j| source[(i, j)] + offset[j]);
        let run = |parallel| {
            RigidRegistration::new(
                &target,
                &source,
                RigidConfig {
                    em: EmConfig {
                        tolerance: 0.0,
                        max_iterations: 12,
                        parallel,
                        ..Default::default()
                    },
                    scale: true,
                    normalize: false,                },
            )
            .unwrap()
            .register()
            .unwrap()
        };
        let serial = run(false);
        let parallel = run(true);
        for (a, b) in serial.points.iter().zip(parallel.points.iter()) {
            prop_assert_eq!(a, b);
        }
    }

    /// Constrained deformable registration pins each constrained source
    /// point to its target, whatever the rest of the clouds look like.
    #[test]
    fn constraints_are_honored(
        source in cloud_strategy(40, 3),
        shift in prop::array::uniform3(-0.6f64..0.6),
    ) {
        let target = DMatrix::from_fn(source.nrows(), 3, |i, j| source[(i, j)] + shift[j]);
        let constraints = vec![
            Constraint { source: 0, target: 0 },
            Constraint { source: 20, target: 20 },
        ];
        let result = DeformableRegistration::new(
            &target,
            &source,
            DeformableConfig {
                em: EmConfig { max_iterations: 60, ..Default::default() },
                low_rank: None,
                constraints,
                constraint_error: 1e-8,
                ..Default::default()
            },
        )
        .unwrap()
        .register()
        .unwrap();
        for &s in &[0usize, 20] {
            let distance: f64 = (0..3)
                .map(|j| (result.points[(s, j)] - target[(s, j)]).powi(2))
                .sum::<f64>()
                .sqrt();
            prop_assert!(distance < 1e-3, "source {s} distance {distance}");
        }
    }
}
