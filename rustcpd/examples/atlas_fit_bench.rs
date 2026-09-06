//! Reproducible full-fit timings and complete numerical snapshots.
//! Build the identical harness against both revisions, then alternate binaries.
//! Arguments: scenario rank points repetitions serial|parallel [snapshot_path]

use rustcpd::{
    AtlasConfig, AtlasRegistration, AtlasResult, DMatrix, EmConfig, PoseMarginalizedConfig,
};
use std::{
    fs::File,
    hint::black_box,
    io::{BufWriter, Write},
    time::Instant,
};

fn median(values: &mut [f64]) -> f64 {
    values.sort_by(f64::total_cmp);
    values[values.len() / 2]
}

fn dump(writer: &mut impl Write, label: &str, values: impl IntoIterator<Item = f64>) {
    for (i, value) in values.into_iter().enumerate() {
        writeln!(writer, "{label}[{i}] {value:.17e}").unwrap();
    }
}

fn dump_fit(writer: &mut impl Write, prefix: &str, fit: &AtlasResult) {
    dump(
        writer,
        &format!("{prefix}.points"),
        fit.points.iter().copied(),
    );
    dump(
        writer,
        &format!("{prefix}.coefficients"),
        fit.coefficients.iter().copied(),
    );
    dump(
        writer,
        &format!("{prefix}.rotation"),
        fit.rotation.iter().copied(),
    );
    dump(
        writer,
        &format!("{prefix}.translation"),
        fit.translation.iter().copied(),
    );
    dump(writer, &format!("{prefix}.sigma2"), [fit.sigma2]);
    dump(writer, &format!("{prefix}.scale"), [fit.scale]);
    dump(
        writer,
        &format!("{prefix}.iterations"),
        [fit.iterations as f64],
    );
    dump(
        writer,
        &format!("{prefix}.objective"),
        [fit.negative_log_likelihood],
    );
    dump(
        writer,
        &format!("{prefix}.mixing"),
        fit.mixing_weights.iter().flatten().copied(),
    );
}

fn main() {
    let args: Vec<_> = std::env::args().collect();
    let scenario = args.get(1).map(String::as_str).unwrap_or("fragment");
    let rank: usize = args.get(2).map(|s| s.parse().unwrap()).unwrap_or(64);
    let m: usize = args.get(3).map(|s| s.parse().unwrap()).unwrap_or(2000);
    let repeats: usize = args.get(4).map(|s| s.parse().unwrap()).unwrap_or(3);
    let parallel = args.get(5).is_some_and(|s| s == "parallel");
    let mut snapshot = args
        .get(6)
        .map(|p| BufWriter::new(File::create(p).unwrap()));
    let fragment =
        scenario.contains("fragment") || scenario.contains("stress") || scenario.contains("pose");
    let stress = scenario.contains("stress");
    let pose_search = scenario.contains("pose");
    let normalized = !scenario.contains("raw");
    let shift = if scenario.contains("offset") {
        10000.0
    } else {
        0.0
    };
    // Asymmetric surface with a deterministic vertex order, not a training dataset.
    let mean = DMatrix::from_fn(m, 3, |i, j| {
        let z = 2.0 * (i as f64 + 0.5) / m as f64 - 1.0;
        let theta = i as f64 * 2.399_963_229_728_653;
        let radius = (1.0 - z * z).sqrt() * (1.0 + 0.2 * z);
        shift
            + match j {
                0 => radius * theta.cos() * (1.0 + 0.09 * (3.0 * theta).sin()),
                1 => 0.7 * radius * theta.sin(),
                _ => 1.6 * z + 0.05 * theta.cos(),
            }
    });
    // Orthonormal DCT columns, so the mode count is a genuine model rank.
    let mut modes = DMatrix::from_fn(m * 3, rank, |i, a| {
        (2.0 / (3 * m) as f64).sqrt()
            * (std::f64::consts::PI * (i as f64 + 0.5) * (a + 1) as f64 / (3 * m) as f64).cos()
    });
    if stress && rank > 1 {
        for i in 0..m * 3 {
            modes[(i, 1)] = modes[(i, 0)] + 1e-7 * modes[(i, 1)];
        }
    }
    let eigenvalues: Vec<f64> = (0..rank)
        .map(|a| 0.05 * m as f64 / ((a + 1) as f64).powf(if stress { 2.0 } else { 1.2 }))
        .collect();
    let b: Vec<f64> = eigenvalues
        .iter()
        .enumerate()
        .map(|(a, e)| 0.2 * e.sqrt() * ((a + 1) as f64 * 0.7).sin())
        .collect();
    let angle = if pose_search { 0.6_f64 } else { 0.12_f64 };
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
    let translation = vec![0.13, -0.09, 0.06];
    let true_scale = if fragment { 1.0 } else { 1.03 };
    let full = DMatrix::from_fn(m, 3, |i, j| {
        true_scale
            * (0..3)
                .map(|q| {
                    let deformed =
                        mean[(i, q)] + (0..rank).map(|a| modes[(3 * i + q, a)] * b[a]).sum::<f64>();
                    deformed * rotation[(q, j)]
                })
                .sum::<f64>()
            + translation[j]
    });
    let n = if fragment {
        (m * 35 / 100).max(20).min(m)
    } else {
        m
    };
    let x = DMatrix::from_fn(n, 3, |i, j| {
        full[(i, j)] + 0.001 * ((i * 3 + j + 1) as f64 * 1.77).sin()
    });
    let landmarks: Vec<(usize, Vec<f64>)> = if scenario.contains("landmark") {
        [0, n / 3, 2 * n / 3]
            .into_iter()
            .map(|i| (i, (0..3).map(|j| x[(i, j)]).collect()))
            .collect()
    } else {
        Vec::new()
    };
    let converged = scenario.contains("converged");
    let config = AtlasConfig {
        em: EmConfig {
            sigma2: if scenario.contains("cold") {
                None
            } else {
                Some(0.003)
            },
            max_iterations: if converged { 150 } else { 12 },
            tolerance: if converged { 1e-5 } else { 0.0 },
            outlier_weight: 0.05,
            k: scenario.contains("sparse").then_some(32),
            parallel,
            ..Default::default()
        },
        eigenvalues: eigenvalues.clone(),
        normalize: normalized,
        with_scale: !fragment,
        initial_rotation: Some(rotation.clone()),
        initial_translation: Some(translation),
        initial_scale: true_scale,
        initial_coefficients: Some(b.iter().map(|v| 0.8 * v).collect()),
        adaptive_mixing: fragment.then_some(0.1),
        lambda_regularization: if scenario.contains("weak") { 1e-8 } else { 0.1 },
        landmarks: landmarks.clone(),
        landmark_sigma: (!landmarks.is_empty()).then_some(if stress { 1e-5 } else { 0.01 }),
        ..Default::default()
    };
    let run = || {
        let mut cfg = config.clone();
        let mut pose = None;
        if pose_search {
            let initial = PoseMarginalizedConfig {
                rotation_count: 33,
                coarse_source_count: 400,
                coarse_target_count: 300,
                coarse_rank: 12.min(rank),
                coarse_iterations: 8,
                coarse_screen_iterations: 4,
                coarse_survivor_count: 12,
                refine_count: 6,
                refine_target_count: 600,
                refine_iterations: 15,
                with_scale: false,
                translation_anchor_count: 6,
                adaptive_mixing: Some(0.1),
                landmarks: landmarks.clone(),
                landmark_sigma: (!landmarks.is_empty()).then_some(0.01),
                refine_landmark_sigma: (!landmarks.is_empty()).then_some(0.01),
                parallel,
                ..Default::default()
            }
            .initialize(&mean, &x, &modes, &eigenvalues)
            .unwrap();
            cfg.initial_state = Some(initial.state());
            cfg.initial_coefficients = None;
            cfg.initial_rotation = None;
            cfg.initial_translation = None;
            cfg.initial_scale = 1.0;
            cfg.em.sigma2 = None;
            pose = Some(initial);
        }
        let fit = AtlasRegistration::new(&x, &mean, &modes, cfg)
            .unwrap()
            .register()
            .unwrap();
        (fit, pose)
    };
    black_box(run()); // Untimed warm-up including thread-pool initialization.
    let mut times = Vec::new();
    let mut last = None;
    for _ in 0..repeats {
        let start = Instant::now();
        let value = run();
        times.push(start.elapsed().as_secs_f64());
        last = Some(black_box(value));
    }
    let (fit, pose) = last.unwrap();
    if let Some(w) = &mut snapshot {
        dump_fit(w, "fit", &fit);
        if let Some(pose) = &pose {
            dump(w, "pose.coefficients", pose.coefficients.iter().copied());
            dump(w, "pose.rotation", pose.rotation.iter().copied());
            dump(w, "pose.translation", pose.translation.iter().copied());
            dump(w, "pose.sigma2", [pose.sigma2]);
            dump(w, "pose.score", [pose.score]);
            dump(w, "pose.margin", [pose.score_margin]);
            dump(
                w,
                "pose.diagnostics",
                [
                    pose.hypotheses_refined as f64,
                    pose.distinct_hypotheses as f64,
                    pose.winner_support as f64,
                ],
            );
            dump(
                w,
                "pose.mixing",
                pose.mixing_weights.iter().flatten().copied(),
            );
        } else {
            // Compare trajectories at several identical iteration budgets.
            for budget in [1, 2, 4, 8] {
                let mut cfg = config.clone();
                cfg.em.max_iterations = budget;
                cfg.em.tolerance = 0.0;
                let intermediate = AtlasRegistration::new(&x, &mean, &modes, cfg)
                    .unwrap()
                    .register()
                    .unwrap();
                dump_fit(w, &format!("iteration_{budget}"), &intermediate);
            }
        }
        // Exercise explicit state transfer in a different normalization frame.
        let mut cfg = config.clone();
        cfg.initial_state = Some(fit.state());
        cfg.initial_coefficients = None;
        cfg.initial_rotation = None;
        cfg.initial_translation = None;
        cfg.initial_scale = 1.0;
        cfg.em.sigma2 = None;
        cfg.em.max_iterations = 4;
        cfg.normalize = !normalized;
        let continuation = AtlasRegistration::new(&x, &mean, &modes, cfg)
            .unwrap()
            .register()
            .unwrap();
        dump_fit(w, "continued", &continuation);
    }
    #[cfg(feature = "completion")]
    if let Some(w) = &mut snapshot {
        let post = fit
            .posterior(
                &x,
                &mean,
                &modes,
                &eigenvalues,
                &rustcpd::PosteriorOptions {
                    visibility_floor: 0.0,
                    prior_temperature: config.lambda_regularization,
                    estimate_discrepancy: false,
                    ..Default::default()
                },
            )
            .unwrap();
        dump(w, "posterior.mean", post.coefficient_mean().iter().copied());
        dump(
            w,
            "posterior.covariance",
            post.coefficient_covariance().iter().copied(),
        );
        dump(w, "posterior.prediction", post.predict().iter().copied());
        dump(w, "posterior.variance", post.predictive_variance());
        dump(
            w,
            "posterior.samples",
            post.sample_coefficients(3, 19)
                .unwrap()
                .into_iter()
                .flatten(),
        );
    }
    if let Some(w) = &mut snapshot {
        w.flush().unwrap();
    }
    let seconds = median(&mut times);
    println!(
        "{{\"scenario\":\"{scenario}\",\"points\":{m},\"targets\":{n},\"rank\":{rank},\"parallel\":{parallel},\"median_s\":{seconds:.12e},\"iterations\":{},\"repetitions\":{repeats},\"sigma2\":{:.12e}}}",
        fit.iterations, fit.sigma2
    );
}
