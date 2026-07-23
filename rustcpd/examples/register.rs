//! Minimal end-to-end example: build two synthetic 3-D clouds related by a
//! known similarity transform, recover it with rigid CPD, then fit a
//! deformable warp and evaluate it at new points.
//!
//! Run with: `cargo run --release --example register`

use rustcpd::{
    DMatrix, DeformableConfig, DeformableRegistration, RigidConfig, RigidRegistration,
    correspondences, gaussian_kernel,
};

fn synthetic_cloud(count: usize) -> DMatrix<f64> {
    DMatrix::from_fn(count, 3, |i, j| {
        let z = i as f64 + 1.0;
        match j {
            0 => (z * 0.37).sin() * 1.7 + 0.01 * z,
            1 => (z * 0.23).cos() * 0.9,
            _ => (z * 0.11).sin() * (z * 0.07).cos(),
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

fn main() -> Result<(), rustcpd::Error> {
    let source = synthetic_cloud(200);

    // --- Rigid: recover a known rotation + scale + translation ----------
    let angle: f64 = 0.3;
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
    let target = DMatrix::from_fn(source.nrows(), 3, |i, j| {
        1.15 * (0..3)
            .map(|q| source[(i, q)] * rotation[(q, j)])
            .sum::<f64>()
            + [0.4, -0.2, 0.1][j]
    });

    let rigid = RigidRegistration::new(&target, &source, RigidConfig::default())?.register()?;
    println!(
        "rigid: recovered scale = {:.4} (truth 1.15), fit rms = {:.2e}, {} iters",
        rigid.scale,
        rms(&rigid.points, &target),
        rigid.iterations
    );

    // Soft correspondences: which target point each source point matched.
    let matching = correspondences(&target, &rigid.points, rigid.sigma2, 0.0)?;
    let self_matched = matching
        .matches
        .iter()
        .enumerate()
        .filter(|&(i, &j)| i == j)
        .count();
    println!(
        "correspondences: {self_matched}/{} source points matched to their true target",
        source.nrows()
    );

    // --- Deformable: fit a smooth warp, then apply it to new points -----
    let beta = 2.0;
    let g = gaussian_kernel(&source, &source, beta)?;
    let control = DMatrix::from_fn(source.nrows(), 3, |i, j| {
        0.02 * ((i * 3 + j + 1) as f64 * 0.41).sin()
    });
    let warped_target = &source + g * control;

    let deformable = DeformableRegistration::new(
        &warped_target,
        &source,
        DeformableConfig {
            beta,
            low_rank: None,
            ..Default::default()
        },
    )?
    .register()?;
    println!(
        "deformable: fit rms = {:.2e}, {} iters",
        rms(&deformable.points, &warped_target),
        deformable.iterations
    );

    // Evaluate the learned field at brand-new points (here, cloud midpoints).
    let new_points = DMatrix::from_fn(source.nrows() - 1, 3, |i, j| {
        0.5 * (source[(i, j)] + source[(i + 1, j)])
    });
    let warped_new = deformable.transform(&new_points)?;
    println!(
        "transform: warped {} new points not seen during registration",
        warped_new.nrows()
    );

    Ok(())
}
