//! Prints high-precision registration outputs for fixed workloads so that
//! optimization work can be checked for numerical parity run-over-run.

use rustcpd::{
    AffineConfig, AffineRegistration, AtlasConfig, AtlasRegistration, Constraint, DMatrix,
    DeformableConfig, DeformableRegistration, EmConfig, RigidConfig, RigidRegistration,
    gaussian_kernel,
};

fn cloud(count: usize, d: usize) -> DMatrix<f64> {
    DMatrix::from_fn(count, d, |i, j| {
        let z = i as f64 + 1.0;
        match j {
            0 => (z * 0.37).sin() * 1.7 + 0.01 * z,
            1 => (z * 0.23).cos() * 0.9,
            _ => (z * 0.11).sin() * (z * 0.07).cos(),
        }
    })
}

fn dump(label: &str, values: impl IntoIterator<Item = f64>) {
    for (index, value) in values.into_iter().enumerate() {
        println!("{label}[{index}] {value:.17e}");
    }
}

fn sample(points: &DMatrix<f64>) -> Vec<f64> {
    let mut values = Vec::new();
    let step = (points.nrows() / 7).max(1);
    for i in (0..points.nrows()).step_by(step) {
        for j in 0..points.ncols() {
            values.push(points[(i, j)]);
        }
    }
    values
}

fn main() {
    let parallel = std::env::args().nth(1).as_deref() != Some("serial");
    let em = |k: Option<usize>| EmConfig {
        tolerance: 0.0,
        max_iterations: 12,
        k,
        parallel,
        ..Default::default()
    };

    // Rigid, dense and sparse.
    let y = cloud(400, 3);
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
    let x = DMatrix::from_fn(400, 3, |i, j| {
        1.04 * (0..3).map(|q| y[(i, q)] * r[(q, j)]).sum::<f64>() + [0.12, -0.08, 0.04][j]
    });
    for (name, k) in [("rigid_dense", None), ("rigid_sparse", Some(8))] {
        let result = RigidRegistration::new(
            &x,
            &y,
            RigidConfig {
                em: em(k),
                scale: true,
                normalize: false,
            },
        )
        .unwrap()
        .register()
        .unwrap();
        dump(name, [result.sigma2, result.scale, result.objective]);
        dump(name, result.translation.iter().copied());
        dump(name, sample(&result.points));
    }
    // Rigid without scale (outlier weight active).
    let no_scale = RigidRegistration::new(
        &x,
        &y,
        RigidConfig {
            em: EmConfig {
                outlier_weight: 0.1,
                ..em(None)
            },
            scale: false,
            normalize: false,
        },
    )
    .unwrap()
    .register()
    .unwrap();
    dump("rigid_no_scale", [no_scale.sigma2, no_scale.objective]);
    dump("rigid_no_scale", sample(&no_scale.points));

    // Affine.
    let b = DMatrix::from_row_slice(
        3,
        3,
        &[1.03, 0.04, 0.0, -0.025, 0.98, 0.01, 0.0, 0.02, 1.01],
    );
    let xa = DMatrix::from_fn(400, 3, |i, j| {
        (0..3).map(|q| y[(i, q)] * b[(q, j)]).sum::<f64>() + [0.08, -0.05, 0.03][j]
    });
    let affine = AffineRegistration::new(
        &xa,
        &y,
        AffineConfig {
            em: em(None),
            normalize: false,
        },
    )
    .unwrap()
    .register()
    .unwrap();
    dump("affine", [affine.sigma2]);
    dump("affine", sample(&affine.points));

    // Deformable: full rank, low rank, constrained, sparse.
    let yd = cloud(180, 3);
    let g = gaussian_kernel(&yd, &yd, 1.5).unwrap();
    let w = DMatrix::from_fn(180, 3, |i, j| {
        0.003 * ((i * 3 + j + 1) as f64 * 0.041).sin()
    });
    let xd = &yd + g * w;
    let deformable = |low_rank: Option<usize>, constraints: Vec<Constraint>, k: Option<usize>| {
        DeformableRegistration::new(
            &xd,
            &yd,
            DeformableConfig {
                em: em(k),
                alpha: 2.0,
                beta: 1.5,
                low_rank,
                constraints,
                constraint_error: 1e-8,
                ..Default::default()
            },
        )
        .unwrap()
        .register()
        .unwrap()
    };
    let full = deformable(None, Vec::new(), None);
    dump("deformable_full", [full.sigma2]);
    dump("deformable_full", sample(&full.points));
    let low = deformable(Some(60), Vec::new(), None);
    dump("deformable_low_rank", [low.sigma2]);
    dump("deformable_low_rank", sample(&low.points));
    let constrained = deformable(
        None,
        vec![
            Constraint {
                source: 0,
                target: 0,
            },
            Constraint {
                source: 90,
                target: 90,
            },
        ],
        None,
    );
    dump("deformable_constrained", [constrained.sigma2]);
    dump("deformable_constrained", sample(&constrained.points));
    let sparse = deformable(None, Vec::new(), Some(8));
    dump("deformable_sparse", [sparse.sigma2]);
    dump("deformable_sparse", sample(&sparse.points));

    // Atlas with similarity optimization and normalization.
    let ya = cloud(300, 3);
    let rank = 8;
    let modes = DMatrix::from_fn(ya.len(), rank, |i, k| {
        0.01 * ((i + 1 + k * 7) as f64 * 0.019).sin()
    });
    let coefficients: Vec<f64> = (0..rank)
        .map(|k| 0.2 * ((k + 1) as f64 * 0.7).sin())
        .collect();
    let xa2 = DMatrix::from_fn(300, 3, |i, j| {
        ya[(i, j)]
            + (0..rank)
                .map(|k| modes[(i * 3 + j, k)] * coefficients[k])
                .sum::<f64>()
    });
    let atlas = AtlasRegistration::new(
        &xa2,
        &ya,
        &modes,
        AtlasConfig {
            em: em(None),
            eigenvalues: (0..rank).map(|k| 1. / (k + 1) as f64).collect(),
            lambda_regularization: 0.1,
            normalize: true,
            optimize_similarity: true,
            ..Default::default()
        },
    )
    .unwrap()
    .register()
    .unwrap();
    dump("atlas", [atlas.sigma2, atlas.scale]);
    dump("atlas", atlas.coefficients.iter().copied());
    dump("atlas", sample(&atlas.points));
}
