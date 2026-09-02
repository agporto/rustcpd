use kiddo::{KdTree, SquaredEuclidean};
use nalgebra::DMatrix;
use rayon::prelude::*;

use crate::fastexp::{exp_non_positive, exp_non_positive_f32};
use crate::{Error, Result};

/// Shared expectation-maximization settings used by every registration.
#[derive(Clone, Debug)]
pub struct EmConfig {
    /// Initial Gaussian variance. `None` estimates it from the data via
    /// [`initialize_sigma2`].
    pub sigma2: Option<f64>,
    /// Maximum number of EM iterations.
    pub max_iterations: usize,
    /// Convergence threshold; see the crate-level documentation for the
    /// per-algorithm convergence criterion it applies to.
    pub tolerance: f64,
    /// Uniform-outlier mixture weight in `[0, 1)`. `0` disables the
    /// outlier component.
    pub outlier_weight: f64,
    /// `Some(k)` replaces the exact dense E-step with a source-to-target
    /// k-nearest-neighbor approximation. `None` is exact.
    pub k: Option<usize>,
    /// Use the crate's single Rayon pool for the E-step. Results are
    /// identical to serial execution; see the crate-level documentation.
    pub parallel: bool,
    /// Evaluate the dense E-step's distances and exponentials in `f32`
    /// while still accumulating every statistic in `f64`. Roughly halves
    /// the dense E-step cost at ~1e-7 relative accuracy, and lets the
    /// truncated E-step prune at single-precision resolution. Applies to
    /// 2-D/3-D dense registrations; other paths stay in `f64`. Results
    /// remain deterministic and independent of thread count.
    pub single_precision: bool,
}

impl Default for EmConfig {
    fn default() -> Self {
        Self {
            sigma2: None,
            max_iterations: 100,
            tolerance: 1e-3,
            outlier_weight: 0.0,
            k: None,
            parallel: true,
            single_precision: false,
        }
    }
}

impl EmConfig {
    pub(crate) fn validate(&self) -> Result<()> {
        if self.sigma2.is_some_and(|x| !x.is_finite() || x <= 0.0) {
            return Err(Error::PositiveParameter("sigma2"));
        }
        if !self.tolerance.is_finite() || self.tolerance < 0.0 {
            return Err(Error::PositiveParameter("tolerance"));
        }
        if !self.outlier_weight.is_finite() || !(0.0..1.0).contains(&self.outlier_weight) {
            return Err(Error::InvalidOutlierWeight);
        }
        Ok(())
    }
}

/// Sufficient statistics of the CPD posterior for one E-step.
#[derive(Clone, Debug)]
pub struct PosteriorStats {
    /// Full `M×N` posterior matrix, populated only when explicitly
    /// requested; the registrations work from the fused statistics below.
    pub p: Option<DMatrix<f64>>,
    /// `Pᵀ·1` — per-target posterior mass (length `N`).
    pub pt1: Vec<f64>,
    /// `P·1` — per-source posterior mass (length `M`).
    pub p1: Vec<f64>,
    /// `P·X` — posterior-weighted target coordinates (`M×D`).
    pub px: DMatrix<f64>,
    /// Total posterior mass `1ᵀP1`.
    pub np: f64,
    /// Negative log-likelihood of the current mixture.
    pub negative_log_likelihood: f64,
}

enum NeighborIndex {
    TwoDimensional(KdTree<f64, 2>),
    ThreeDimensional(KdTree<f64, 3>),
    BruteForce,
}

pub(crate) struct SparseIndex {
    keep: usize,
    index: NeighborIndex,
}

impl SparseIndex {
    pub(crate) fn new(x: &DMatrix<f64>, k: usize) -> Self {
        let keep = k.clamp(1, x.nrows());
        let index = match x.ncols() {
            2 => NeighborIndex::TwoDimensional(build_kd_tree::<2>(x)),
            3 => NeighborIndex::ThreeDimensional(build_kd_tree::<3>(x)),
            _ => NeighborIndex::BruteForce,
        };
        Self { keep, index }
    }
}

/// An absolute, extent-scaled lower bound for `sigma2`, matching the
/// deformable and atlas paths. Keeps the variance strictly positive (so
/// later E-steps never divide by zero) even when `tolerance == 0`, while
/// scaling with the target cloud so it never dominates a well-posed fit.
pub(crate) fn variance_floor(x: &DMatrix<f64>, d: usize) -> f64 {
    let n = x.nrows();
    if n == 0 || d == 0 {
        return f64::MIN_POSITIVE;
    }
    let mut sum_sq = 0.0;
    for j in 0..x.ncols() {
        let mean = (0..n).map(|i| x[(i, j)]).sum::<f64>() / n as f64;
        for i in 0..n {
            let centered = x[(i, j)] - mean;
            sum_sq += centered * centered;
        }
    }
    (f64::EPSILON * sum_sq / (n * d) as f64).max(f64::MIN_POSITIVE)
}

/// Shared normalization frame taken from the target cloud: its per-column
/// centroid and the RMS radius about that centroid (floored so it is
/// strictly positive). Applying the same frame to both clouds conditions
/// the fit into a unit-scale regime while preserving the relative geometry
/// — so length-scale parameters like the deformable `beta` become
/// meaningful regardless of the raw coordinate units.
pub(crate) fn cloud_frame(x: &DMatrix<f64>) -> (Vec<f64>, f64) {
    let (n, d) = (x.nrows(), x.ncols());
    let mut centroid = vec![0.0; d];
    for (j, c) in centroid.iter_mut().enumerate() {
        *c = (0..n).map(|i| x[(i, j)]).sum::<f64>() / n as f64;
    }
    let scale = ((0..n)
        .map(|i| {
            (0..d)
                .map(|j| (x[(i, j)] - centroid[j]).powi(2))
                .sum::<f64>()
        })
        .sum::<f64>()
        / n as f64)
        .sqrt()
        .max(f64::EPSILON);
    (centroid, scale)
}

/// Apply a normalization frame: `(p - centroid) / scale`.
pub(crate) fn normalized_cloud(p: &DMatrix<f64>, centroid: &[f64], scale: f64) -> DMatrix<f64> {
    DMatrix::from_fn(p.nrows(), p.ncols(), |i, j| {
        (p[(i, j)] - centroid[j]) / scale
    })
}

/// Snapshot of a registration passed to a per-iteration callback. The
/// callback returns `true` to continue or `false` to stop early. When the
/// fit is normalized, these values are in the internal normalized frame.
#[non_exhaustive]
pub struct IterationState<'a> {
    /// Iterations completed so far (1 on the first callback).
    pub iteration: usize,
    /// Current Gaussian variance.
    pub sigma2: f64,
    /// Current value of the convergence criterion.
    pub difference: f64,
    /// Current transformed source points (`M×D`).
    pub points: &'a DMatrix<f64>,
}

/// Shared conditioning frame for a registration: each cloud is centered on
/// its OWN centroid (translation is always free, so this costs nothing and
/// keeps both clouds at the origin even when they start far apart), and
/// both are divided by a single shared scale taken from the target (so an
/// isotropic-scale contract like rigid's is preserved — a common scale
/// cannot introduce a spurious relative scale). Returns
/// `(target_centroid, source_centroid, shared_scale)`.
pub(crate) fn shared_frame(x: &DMatrix<f64>, y: &DMatrix<f64>) -> (Vec<f64>, Vec<f64>, f64) {
    let (target_centroid, scale) = cloud_frame(x);
    let (source_centroid, _) = cloud_frame(y);
    (target_centroid, source_centroid, scale)
}

pub(crate) fn validate_clouds(x: &DMatrix<f64>, y: &DMatrix<f64>) -> Result<()> {
    if x.nrows() == 0 || y.nrows() == 0 || x.ncols() == 0 || y.ncols() == 0 {
        return Err(Error::EmptyPointCloud);
    }
    if x.ncols() != y.ncols() {
        return Err(Error::DimensionMismatch);
    }
    if !x.iter().chain(y.iter()).all(|v| v.is_finite()) {
        return Err(Error::NonFiniteInput);
    }
    Ok(())
}

/// Mean squared distance between all pairs of points in `x` and `y`,
/// divided by the dimensionality — the standard CPD `sigma2` initializer.
///
/// Computed in `O((N+M)·D)` through the identity
/// `E‖x−y‖² = var(x) + var(y) + ‖mean(x)−mean(y)‖²`.
pub fn initialize_sigma2(x: &DMatrix<f64>, y: &DMatrix<f64>) -> Result<f64> {
    validate_clouds(x, y)?;
    let d = x.ncols();
    let mut mx = vec![0.0; d];
    let mut my = vec![0.0; d];
    for i in 0..x.nrows() {
        for j in 0..d {
            mx[j] += x[(i, j)];
        }
    }
    for i in 0..y.nrows() {
        for j in 0..d {
            my[j] += y[(i, j)];
        }
    }
    for j in 0..d {
        mx[j] /= x.nrows() as f64;
        my[j] /= y.nrows() as f64;
    }
    let mut value = 0.0;
    for i in 0..x.nrows() {
        for j in 0..d {
            value += (x[(i, j)] - mx[j]).powi(2) / x.nrows() as f64;
        }
    }
    for i in 0..y.nrows() {
        for j in 0..d {
            value += (y[(i, j)] - my[j]).powi(2) / y.nrows() as f64;
        }
    }
    for j in 0..d {
        value += (mx[j] - my[j]).powi(2);
    }
    Ok((value / d as f64).max(f64::MIN_POSITIVE))
}

/// Dense Gaussian kernel `G[i, j] = exp(−‖xᵢ − yⱼ‖² / (2β²))`.
pub fn gaussian_kernel(x: &DMatrix<f64>, y: &DMatrix<f64>, beta: f64) -> Result<DMatrix<f64>> {
    gaussian_kernel_with_parallel(x, y, beta, true)
}

pub(crate) fn gaussian_kernel_with_parallel(
    x: &DMatrix<f64>,
    y: &DMatrix<f64>,
    beta: f64,
    parallel: bool,
) -> Result<DMatrix<f64>> {
    if !beta.is_finite() || beta <= 0.0 {
        return Err(Error::PositiveParameter("beta"));
    }
    if x.ncols() != y.ncols() {
        return Err(Error::DimensionMismatch);
    }
    // When both arguments are the same matrix the kernel is exactly
    // symmetric — (xᵢ−yⱼ) and (xⱼ−yᵢ) square to identical values — so
    // computing the lower triangle and mirroring halves the exp calls.
    let symmetric = std::ptr::eq(x, y);
    let mut g = DMatrix::zeros(x.nrows(), y.nrows());
    let scale = -0.5 / (beta * beta);
    let fill_column = |j: usize, column: &mut [f64]| {
        let start = if symmetric { j } else { 0 };
        for (i, value) in column.iter_mut().enumerate().skip(start) {
            let mut d2 = 0.0;
            for q in 0..x.ncols() {
                d2 += (x[(i, q)] - y[(j, q)]).powi(2);
            }
            *value = scale * d2;
        }
        exp_non_positive(&mut column[start..]);
    };
    if parallel && g.len() >= 4096 {
        g.as_mut_slice()
            .par_chunks_mut(x.nrows())
            .enumerate()
            .for_each(|(j, column)| fill_column(j, column));
    } else {
        g.as_mut_slice()
            .chunks_mut(x.nrows())
            .enumerate()
            .for_each(|(j, column)| fill_column(j, column));
    }
    if symmetric {
        // Mirror the computed lower triangle into the upper triangle.
        let m = x.nrows();
        let data = g.as_mut_slice();
        for j in 1..m {
            for i in 0..j {
                data[j * m + i] = data[i * m + j];
            }
        }
    }
    Ok(g)
}

/// Soft point-to-point correspondences recovered from a fitted
/// registration — the CPD posterior (responsibility) matrix plus, for
/// convenience, the single best target match for each source point.
#[derive(Clone, Debug)]
pub struct Correspondences {
    /// For each source point (row of the aligned cloud), the index of the
    /// target point with the highest posterior responsibility. Length `M`.
    pub matches: Vec<usize>,
    /// The posterior responsibility of each best match, in `[0, 1]`. Low
    /// values flag source points the model explains poorly (e.g. matched
    /// only by the uniform-outlier component). Length `M`.
    pub probability: Vec<f64>,
    /// The full `M×N` posterior matrix `P`, where `P[i, j]` is the
    /// responsibility of aligned source point `i` for target point `j`.
    /// Each column is normalized over sources (plus the outlier mass), the
    /// standard CPD convention.
    pub posterior: DMatrix<f64>,
}

/// Compute soft correspondences between a fixed `target` cloud and an
/// `aligned_source` cloud (typically a registration result's `points`)
/// under an isotropic Gaussian mixture with the given variance.
///
/// This is one CPD E-step exposed directly: use it after any registration
/// to read off which target point each source point maps to and how
/// confidently. `sigma2` is normally the `sigma2` field of the result;
/// smaller values sharpen the matches.
pub fn correspondences(
    target: &DMatrix<f64>,
    aligned_source: &DMatrix<f64>,
    sigma2: f64,
    outlier_weight: f64,
) -> Result<Correspondences> {
    validate_clouds(target, aligned_source)?;
    if !sigma2.is_finite() || sigma2 <= 0.0 {
        return Err(Error::PositiveParameter("sigma2"));
    }
    if !outlier_weight.is_finite() || !(0.0..1.0).contains(&outlier_weight) {
        return Err(Error::InvalidOutlierWeight);
    }
    let config = EmConfig {
        outlier_weight,
        k: None,
        ..EmConfig::default()
    };
    let stats = posterior_stats(target, aligned_source, sigma2, &config, true, None);
    let posterior = stats.p.expect("store_p requested");
    let (m, n) = (posterior.nrows(), posterior.ncols());
    let mut matches = vec![0usize; m];
    let mut probability = vec![0.0; m];
    for i in 0..m {
        let mut best_index = 0;
        let mut best_value = f64::NEG_INFINITY;
        for j in 0..n {
            let value = posterior[(i, j)];
            if value > best_value {
                best_value = value;
                best_index = j;
            }
        }
        matches[i] = best_index;
        probability[i] = best_value;
    }
    Ok(Correspondences {
        matches,
        probability,
        posterior,
    })
}

pub(crate) fn posterior_stats(
    x: &DMatrix<f64>,
    ty: &DMatrix<f64>,
    sigma2: f64,
    config: &EmConfig,
    store_p: bool,
    sparse_index: Option<&SparseIndex>,
) -> PosteriorStats {
    posterior_stats_weighted(x, ty, sigma2, config, store_p, sparse_index, None)
}

/// E-step with optional per-source mixing weights.
///
/// `source_weights`, when given, holds one strictly positive factor per
/// source point, expressed *relative to uniform*: `M·π_m` where `π_m` are
/// the mixing proportions (so a vector of ones reproduces classic CPD's
/// `1/M`). Each Gaussian kernel is multiplied by its factor before the
/// per-target normalization, and the uniform-outlier constant is left
/// untouched because it is already stated relative to `1/M`. Every path —
/// dense, single-precision, truncated, and k-NN sparse — applies the same
/// factors, so the choice of path never changes the answer beyond
/// rounding. The truncation radius widens by `2σ²·ln(max/min)` of the
/// weights so no non-negligible weighted term is pruned.
pub(crate) fn posterior_stats_weighted(
    x: &DMatrix<f64>,
    ty: &DMatrix<f64>,
    sigma2: f64,
    config: &EmConfig,
    store_p: bool,
    sparse_index: Option<&SparseIndex>,
    source_weights: Option<&[f64]>,
) -> PosteriorStats {
    debug_assert!(source_weights.is_none_or(|w| w.len() == ty.nrows()));
    if config.k.is_none() {
        // Once σ² is small relative to the cloud extent, most Gaussian
        // terms fall below f64 resolution and an anchored range search
        // beats streaming over every pair. The switch is a pure
        // performance heuristic: both paths agree to rounding, and the
        // decision depends only on the inputs, never on thread count.
        let d = x.ncols();
        if matches!(d, 2 | 3)
            && ty.nrows() >= 64
            && x.nrows() >= 64
            && truncation_margin(sigma2, ty.nrows(), config.single_precision)
                < 0.25 * bounding_box_diagonal_squared(ty)
        {
            let attempt = match d {
                2 => posterior_stats_truncated::<2>(
                    x,
                    ty,
                    sigma2,
                    config.outlier_weight,
                    store_p,
                    config.parallel,
                    config.single_precision,
                    source_weights,
                ),
                _ => posterior_stats_truncated::<3>(
                    x,
                    ty,
                    sigma2,
                    config.outlier_weight,
                    store_p,
                    config.parallel,
                    config.single_precision,
                    source_weights,
                ),
            };
            if let Some(stats) = attempt {
                return stats;
            }
        }
        if config.single_precision && matches!(d, 2 | 3) {
            return posterior_stats_dense_f32(
                x,
                ty,
                sigma2,
                config.outlier_weight,
                store_p,
                config.parallel,
                source_weights,
            );
        }
        return posterior_stats_dense(
            x,
            ty,
            sigma2,
            config.outlier_weight,
            store_p,
            config.parallel,
            source_weights,
        );
    }
    let index = sparse_index.expect("sparse posterior requires a prebuilt index");
    posterior_stats_sparse(
        x,
        ty,
        sigma2,
        config.outlier_weight,
        index,
        store_p,
        config.parallel,
        source_weights,
    )
}

/// `ln(max / min)` of a strictly positive weight vector, or `None` when the
/// weights are unusable for a truncation bound (non-finite or non-positive).
fn weight_log_ratio(weights: &[f64]) -> Option<f64> {
    let (mut low, mut high) = (f64::INFINITY, 0.0_f64);
    for &w in weights {
        if !w.is_finite() || w <= 0.0 {
            return None;
        }
        low = low.min(w);
        high = high.max(w);
    }
    if low == f64::INFINITY {
        return Some(0.0);
    }
    Some((high / low).ln())
}

fn build_kd_tree<const K: usize>(x: &DMatrix<f64>) -> KdTree<f64, K> {
    let mut tree = KdTree::new();
    for target in 0..x.nrows() {
        let point = std::array::from_fn(|dimension| x[(target, dimension)]);
        tree.add(&point, target as u64);
    }
    tree
}

fn kd_edges<const K: usize>(
    tree: &KdTree<f64, K>,
    ty: &DMatrix<f64>,
    keep: usize,
    parallel: bool,
) -> Vec<Vec<(usize, f64)>> {
    let query = |source: usize| {
        let point = std::array::from_fn(|dimension| ty[(source, dimension)]);
        tree.nearest_n::<SquaredEuclidean>(&point, keep)
            .into_iter()
            .map(|neighbor| (neighbor.item as usize, neighbor.distance))
            .collect()
    };
    if parallel {
        (0..ty.nrows()).into_par_iter().map(query).collect()
    } else {
        (0..ty.nrows()).map(query).collect()
    }
}

fn brute_edges(
    x: &DMatrix<f64>,
    ty: &DMatrix<f64>,
    keep: usize,
    parallel: bool,
) -> Vec<Vec<(usize, f64)>> {
    let query = |source: usize| {
        let mut values: Vec<_> = (0..x.nrows())
            .map(|target| {
                let squared: f64 = (0..x.ncols())
                    .map(|dimension| (x[(target, dimension)] - ty[(source, dimension)]).powi(2))
                    .sum();
                (target, squared)
            })
            .collect();
        if keep < values.len() {
            values.select_nth_unstable_by(keep, |a, b| a.1.total_cmp(&b.1));
            values.truncate(keep);
        }
        values
    };
    if parallel {
        (0..ty.nrows()).into_par_iter().map(query).collect()
    } else {
        (0..ty.nrows()).map(query).collect()
    }
}

fn posterior_stats_sparse(
    x: &DMatrix<f64>,
    ty: &DMatrix<f64>,
    sigma2: f64,
    w: f64,
    sparse_index: &SparseIndex,
    store_p: bool,
    parallel: bool,
    weights: Option<&[f64]>,
) -> PosteriorStats {
    let (n, m, d) = (x.nrows(), ty.nrows(), x.ncols());
    let mut edges = match &sparse_index.index {
        NeighborIndex::TwoDimensional(tree) => kd_edges::<2>(tree, ty, sparse_index.keep, parallel),
        NeighborIndex::ThreeDimensional(tree) => {
            kd_edges::<3>(tree, ty, sparse_index.keep, parallel)
        }
        NeighborIndex::BruteForce => brute_edges(x, ty, sparse_index.keep, parallel),
    };
    let outlier = (2.0 * std::f64::consts::PI * sigma2).powf(d as f64 / 2.0) * w / (1.0 - w)
        * m as f64
        / n as f64;
    let mut denominator = vec![outlier; n];
    for (source, row) in edges.iter_mut().enumerate() {
        let factor = weights.map_or(1.0, |w| w[source]);
        for (target, squared) in row {
            *squared = (-*squared / (2.0 * sigma2)).exp() * factor;
            denominator[*target] += *squared;
        }
    }
    for value in &mut denominator {
        *value = value.max(f64::MIN_POSITIVE);
    }
    let mut p = store_p.then(|| DMatrix::zeros(m, n));
    let mut pt1 = vec![0.0; n];
    let mut p1 = vec![0.0; m];
    let mut px = DMatrix::zeros(m, d);
    for (source, row) in edges.into_iter().enumerate() {
        for (target, value) in row {
            let value = value / denominator[target];
            if let Some(ref mut matrix) = p {
                matrix[(source, target)] = value;
            }
            pt1[target] += value;
            p1[source] += value;
            for dimension in 0..d {
                px[(source, dimension)] += value * x[(target, dimension)];
            }
        }
    }
    let np = p1.iter().sum();
    let nll = 0.5 * n as f64 * d as f64 * (2.0 * std::f64::consts::PI * sigma2).ln()
        - denominator.iter().map(|value| value.ln()).sum::<f64>();
    PosteriorStats {
        p,
        pt1,
        p1,
        px,
        np,
        negative_log_likelihood: nll,
    }
}

/// Partial sufficient statistics for one contiguous block of target
/// columns of the dense posterior.
struct DenseBlock {
    start: usize,
    pt1: Vec<f64>,
    p1: Vec<f64>,
    /// Row-major `M×D` accumulation of `P·X` for this block's targets.
    px: Vec<f64>,
    log_denominator: f64,
    /// `M` posterior values per target, only when the caller wants `P`.
    columns: Option<Vec<f64>>,
}

/// Fixed partition of the `N` target columns. The block structure depends
/// only on the problem size — never on the thread count — so parallel and
/// serial execution accumulate identical partial sums in an identical
/// order and produce bitwise-identical results.
fn dense_block_ranges(n: usize) -> Vec<(usize, usize)> {
    let blocks = n.div_ceil(64).clamp(1, 64);
    let len = n.div_ceil(blocks);
    (0..blocks)
        .map(|block| (block * len, ((block + 1) * len).min(n)))
        .filter(|(start, end)| start < end)
        .collect()
}

/// Streaming dense E-step: each block computes its posterior columns in a
/// scratch buffer of length `M` and folds them straight into the fused
/// statistics, so the full `M×N` posterior is never materialized unless
/// explicitly requested via `store_p`.
fn posterior_stats_dense(
    x: &DMatrix<f64>,
    ty: &DMatrix<f64>,
    sigma2: f64,
    w: f64,
    store_p: bool,
    parallel: bool,
    weights: Option<&[f64]>,
) -> PosteriorStats {
    let (n, m, d) = (x.nrows(), ty.nrows(), x.ncols());
    let outlier = (2.0 * std::f64::consts::PI * sigma2).powf(d as f64 / 2.0) * w / (1.0 - w)
        * m as f64
        / n as f64;
    let source_squared: Vec<f64> = (0..m)
        .map(|i| (0..d).map(|q| ty[(i, q)].powi(2)).sum::<f64>())
        .collect();
    let ty_data = ty.as_slice();
    let process = |&(start, end): &(usize, usize)| -> DenseBlock {
        let mut block = DenseBlock {
            start,
            pt1: Vec::with_capacity(end - start),
            p1: vec![0.0; m],
            px: vec![0.0; m * d],
            log_denominator: 0.0,
            columns: store_p.then(|| Vec::with_capacity((end - start) * m)),
        };
        let mut buffer = vec![0.0; m];
        // The 2-D/3-D specializations unroll the coordinate loops with
        // the exact accumulation order of the generic pass, so all three
        // paths are bitwise-identical; only code generation differs.
        match d {
            2 => dense_block_pass::<2>(
                &mut block,
                &mut buffer,
                x,
                ty_data,
                &source_squared,
                sigma2,
                outlier,
                (start, end),
                weights,
            ),
            3 => dense_block_pass::<3>(
                &mut block,
                &mut buffer,
                x,
                ty_data,
                &source_squared,
                sigma2,
                outlier,
                (start, end),
                weights,
            ),
            _ => dense_block_pass_dynamic(
                &mut block,
                &mut buffer,
                x,
                ty_data,
                &source_squared,
                sigma2,
                outlier,
                (start, end),
                weights,
            ),
        }
        block
    };
    let ranges = dense_block_ranges(n);
    let blocks: Vec<DenseBlock> = if parallel && n * m >= 4096 {
        ranges.par_iter().map(process).collect()
    } else {
        ranges.iter().map(process).collect()
    };
    combine_blocks(blocks, n, m, d, sigma2, store_p)
}

/// Multiply each kernel value by its source's mixing factor (no-op when
/// `weights` is `None`). Applied after the exponential and before the
/// per-target normalization, identically on every E-step path.
#[inline(always)]
fn apply_source_weights(buffer: &mut [f64], weights: Option<&[f64]>) {
    if let Some(weights) = weights {
        for (value, &factor) in buffer.iter_mut().zip(weights) {
            *value *= factor;
        }
    }
}

/// Sum with eight independent accumulators in a fixed order: breaks the
/// serial floating-point dependency chain (so the loop vectorizes) while
/// remaining deterministic across runs and thread counts.
#[inline(always)]
fn unrolled_sum(values: &[f64]) -> f64 {
    let mut partial = [0.0f64; 8];
    let chunks = values.chunks_exact(8);
    let remainder = chunks.remainder();
    for chunk in chunks {
        for (accumulator, &value) in partial.iter_mut().zip(chunk) {
            *accumulator += value;
        }
    }
    let mut total = ((partial[0] + partial[1]) + (partial[2] + partial[3]))
        + ((partial[4] + partial[5]) + (partial[6] + partial[7]));
    for &value in remainder {
        total += value;
    }
    total
}

/// [`unrolled_sum`] over `f32` values, accumulated in `f64`.
#[inline(always)]
fn unrolled_sum_f32(values: &[f32]) -> f64 {
    let mut partial = [0.0f64; 8];
    let chunks = values.chunks_exact(8);
    let remainder = chunks.remainder();
    for chunk in chunks {
        for (accumulator, &value) in partial.iter_mut().zip(chunk) {
            *accumulator += value as f64;
        }
    }
    let mut total = ((partial[0] + partial[1]) + (partial[2] + partial[3]))
        + ((partial[4] + partial[5]) + (partial[6] + partial[7]));
    for &value in remainder {
        total += value as f64;
    }
    total
}

/// One block of dense posterior columns with the dimensionality known at
/// compile time: the dot product, squared distance, and `P·X` update all
/// unroll, and every accumulation happens in the same order as the
/// dynamic pass so results stay bitwise-identical.
#[allow(clippy::too_many_arguments)]
fn dense_block_pass<const D: usize>(
    block: &mut DenseBlock,
    buffer: &mut [f64],
    x: &DMatrix<f64>,
    ty_data: &[f64],
    source_squared: &[f64],
    sigma2: f64,
    outlier: f64,
    (start, end): (usize, usize),
    weights: Option<&[f64]>,
) {
    let m = buffer.len();
    let columns: [&[f64]; D] = std::array::from_fn(|q| &ty_data[q * m..(q + 1) * m]);
    for target in start..end {
        let coordinates: [f64; D] = std::array::from_fn(|q| x[(target, q)]);
        let mut target_squared = 0.0;
        for &coordinate in &coordinates {
            target_squared += coordinate * coordinate;
        }
        for (i, value) in buffer.iter_mut().enumerate() {
            let mut dot = 0.0;
            for q in 0..D {
                dot += columns[q][i] * coordinates[q];
            }
            let squared = (source_squared[i] - 2.0 * dot + target_squared).max(0.0);
            *value = -squared / (2.0 * sigma2);
        }
        exp_non_positive(buffer);
        apply_source_weights(buffer, weights);
        let sum = unrolled_sum(buffer);
        let denominator = (outlier + sum).max(f64::MIN_POSITIVE);
        let inverse = denominator.recip();
        for (i, value) in buffer.iter_mut().enumerate() {
            *value *= inverse;
            block.p1[i] += *value;
            let row = &mut block.px[i * D..(i + 1) * D];
            for (slot, &coordinate) in row.iter_mut().zip(&coordinates) {
                *slot += *value * coordinate;
            }
        }
        // Σ value/denominator without a second serial reduction.
        block.pt1.push(sum * inverse);
        block.log_denominator += denominator.ln();
        if let Some(columns_store) = &mut block.columns {
            columns_store.extend_from_slice(buffer);
        }
    }
}

/// Single-precision variant of the streaming dense E-step (2-D/3-D):
/// distances and exponentials are evaluated in `f32`, every accumulated
/// statistic stays in `f64`. Same fixed block structure and combine
/// order, so results are deterministic and thread-count independent.
fn posterior_stats_dense_f32(
    x: &DMatrix<f64>,
    ty: &DMatrix<f64>,
    sigma2: f64,
    w: f64,
    store_p: bool,
    parallel: bool,
    weights: Option<&[f64]>,
) -> PosteriorStats {
    let (n, m, d) = (x.nrows(), ty.nrows(), x.ncols());
    let outlier = (2.0 * std::f64::consts::PI * sigma2).powf(d as f64 / 2.0) * w / (1.0 - w)
        * m as f64
        / n as f64;
    let ty32: Vec<f32> = ty.as_slice().iter().map(|&value| value as f32).collect();
    let weights32: Option<Vec<f32>> =
        weights.map(|values| values.iter().map(|&value| value as f32).collect());
    let weights32 = weights32.as_deref();
    let source_squared: Vec<f32> = (0..m)
        .map(|i| (0..d).map(|q| ty32[q * m + i] * ty32[q * m + i]).sum())
        .collect();
    let process = |&(start, end): &(usize, usize)| -> DenseBlock {
        let mut block = DenseBlock {
            start,
            pt1: Vec::with_capacity(end - start),
            p1: vec![0.0; m],
            px: vec![0.0; m * d],
            log_denominator: 0.0,
            columns: store_p.then(|| Vec::with_capacity((end - start) * m)),
        };
        let mut buffer = vec![0.0f32; m];
        match d {
            2 => dense_block_pass_f32::<2>(
                &mut block,
                &mut buffer,
                x,
                &ty32,
                &source_squared,
                sigma2,
                outlier,
                (start, end),
                weights32,
            ),
            _ => dense_block_pass_f32::<3>(
                &mut block,
                &mut buffer,
                x,
                &ty32,
                &source_squared,
                sigma2,
                outlier,
                (start, end),
                weights32,
            ),
        }
        block
    };
    let ranges = dense_block_ranges(n);
    let blocks: Vec<DenseBlock> = if parallel && n * m >= 4096 {
        ranges.par_iter().map(process).collect()
    } else {
        ranges.iter().map(process).collect()
    };
    combine_blocks(blocks, n, m, d, sigma2, store_p)
}

/// One block of single-precision dense posterior columns; see
/// [`posterior_stats_dense_f32`].
#[allow(clippy::too_many_arguments)]
fn dense_block_pass_f32<const D: usize>(
    block: &mut DenseBlock,
    buffer: &mut [f32],
    x: &DMatrix<f64>,
    ty32: &[f32],
    source_squared: &[f32],
    sigma2: f64,
    outlier: f64,
    (start, end): (usize, usize),
    weights: Option<&[f32]>,
) {
    let m = buffer.len();
    let columns: [&[f32]; D] = std::array::from_fn(|q| &ty32[q * m..(q + 1) * m]);
    let inverse_variance = (-0.5 / sigma2) as f32;
    for target in start..end {
        let coordinates: [f64; D] = std::array::from_fn(|q| x[(target, q)]);
        let coordinates32: [f32; D] = std::array::from_fn(|q| coordinates[q] as f32);
        let mut target_squared = 0.0f32;
        for &coordinate in &coordinates32 {
            target_squared += coordinate * coordinate;
        }
        for (i, value) in buffer.iter_mut().enumerate() {
            let mut dot = 0.0f32;
            for q in 0..D {
                dot += columns[q][i] * coordinates32[q];
            }
            let squared = (source_squared[i] - 2.0 * dot + target_squared).max(0.0);
            *value = squared * inverse_variance;
        }
        exp_non_positive_f32(buffer);
        if let Some(weights) = weights {
            for (value, &factor) in buffer.iter_mut().zip(weights) {
                *value *= factor;
            }
        }
        let sum = unrolled_sum_f32(buffer);
        let denominator = (outlier + sum).max(f64::MIN_POSITIVE);
        let inverse = denominator.recip();
        for (i, &value) in buffer.iter().enumerate() {
            let normalized = value as f64 * inverse;
            block.p1[i] += normalized;
            let row = &mut block.px[i * D..(i + 1) * D];
            for (slot, &coordinate) in row.iter_mut().zip(&coordinates) {
                *slot += normalized * coordinate;
            }
        }
        block.pt1.push(sum * inverse);
        block.log_denominator += denominator.ln();
        if let Some(store) = &mut block.columns {
            for &value in buffer.iter() {
                store.push(value as f64 * inverse);
            }
        }
    }
}

/// Dimension-generic fallback of [`dense_block_pass`] for `D > 3`.
#[allow(clippy::too_many_arguments)]
fn dense_block_pass_dynamic(
    block: &mut DenseBlock,
    buffer: &mut [f64],
    x: &DMatrix<f64>,
    ty_data: &[f64],
    source_squared: &[f64],
    sigma2: f64,
    outlier: f64,
    (start, end): (usize, usize),
    weights: Option<&[f64]>,
) {
    let m = buffer.len();
    let dims = ty_data.len() / m;
    let mut coordinates = vec![0.0; dims];
    for target in start..end {
        let mut target_squared = 0.0;
        for (q, coordinate) in coordinates.iter_mut().enumerate() {
            *coordinate = x[(target, q)];
            target_squared += *coordinate * *coordinate;
        }
        // buffer = ty · x_target, accumulated over contiguous columns.
        for (q, &coordinate) in coordinates.iter().enumerate() {
            let column = &ty_data[q * m..(q + 1) * m];
            if q == 0 {
                for (value, &source) in buffer.iter_mut().zip(column) {
                    *value = source * coordinate;
                }
            } else {
                for (value, &source) in buffer.iter_mut().zip(column) {
                    *value += source * coordinate;
                }
            }
        }
        for (value, &source_norm) in buffer.iter_mut().zip(source_squared) {
            let squared = (source_norm - 2.0 * *value + target_squared).max(0.0);
            *value = -squared / (2.0 * sigma2);
        }
        exp_non_positive(buffer);
        apply_source_weights(buffer, weights);
        let sum = unrolled_sum(buffer);
        let denominator = (outlier + sum).max(f64::MIN_POSITIVE);
        let inverse = denominator.recip();
        for (i, value) in buffer.iter_mut().enumerate() {
            *value *= inverse;
            block.p1[i] += *value;
            for (q, &coordinate) in coordinates.iter().enumerate() {
                block.px[i * dims + q] += *value * coordinate;
            }
        }
        block.pt1.push(sum * inverse);
        block.log_denominator += denominator.ln();
        if let Some(columns_store) = &mut block.columns {
            columns_store.extend_from_slice(buffer);
        }
    }
}

/// Fold per-block partial statistics in fixed block order (bitwise
/// deterministic regardless of how the blocks were computed).
fn combine_blocks(
    blocks: Vec<DenseBlock>,
    n: usize,
    m: usize,
    d: usize,
    sigma2: f64,
    store_p: bool,
) -> PosteriorStats {
    let mut pt1 = Vec::with_capacity(n);
    let mut p1 = vec![0.0; m];
    let mut px_rows = vec![0.0; m * d];
    let mut nll = 0.5 * n as f64 * d as f64 * (2.0 * std::f64::consts::PI * sigma2).ln();
    let mut p = store_p.then(|| DMatrix::zeros(m, n));
    for block in blocks {
        pt1.extend_from_slice(&block.pt1);
        for (total, value) in p1.iter_mut().zip(&block.p1) {
            *total += value;
        }
        for (total, value) in px_rows.iter_mut().zip(&block.px) {
            *total += value;
        }
        nll -= block.log_denominator;
        if let (Some(matrix), Some(columns)) = (&mut p, &block.columns) {
            for (offset, column) in columns.chunks_exact(m).enumerate() {
                let target = block.start + offset;
                for (i, &value) in column.iter().enumerate() {
                    matrix[(i, target)] = value;
                }
            }
        }
    }
    let np = p1.iter().sum();
    let px = DMatrix::from_fn(m, d, |i, q| px_rows[i * d + q]);
    PosteriorStats {
        p,
        pt1,
        p1,
        px,
        np,
        negative_log_likelihood: nll,
    }
}

/// `2·σ²·ln(1/ε)` with `ε` chosen so that the *sum* of all truncated
/// terms stays below one unit in the last place of any posterior
/// denominator: a source farther than this squared-distance margin beyond
/// a target's nearest source contributes less than `1e−16/M` of the
/// dominant term and is invisible at `f64` resolution.
fn truncation_margin(sigma2: f64, m: usize, single_precision: bool) -> f64 {
    // ~16 significant digits protected in f64 mode, ~7 in f32 mode: when
    // the caller opts into single precision the invisible-term threshold
    // rises, the search radius shrinks, and pruning starts earlier.
    let digits = if single_precision { 7.0 } else { 16.0 };
    2.0 * sigma2 * (digits * std::f64::consts::LN_10 + (m as f64).ln())
}

/// Squared diagonal of the bounding box of `points`.
fn bounding_box_diagonal_squared(points: &DMatrix<f64>) -> f64 {
    (0..points.ncols())
        .map(|q| {
            let column = points.column(q);
            let (mut low, mut high) = (f64::INFINITY, f64::NEG_INFINITY);
            for &value in column.iter() {
                low = low.min(value);
                high = high.max(value);
            }
            (high - low).powi(2)
        })
        .sum()
}

/// Exact-at-`f64` truncated dense E-step for 2-D/3-D clouds.
///
/// For each target the nearest source distance `d₀²` anchors a search
/// radius `d₀² + truncation_margin`; every source outside it is provably
/// below double-precision resolution in that target's posterior column,
/// so dropping it leaves the statistics identical to the dense path up to
/// rounding. As `σ²` shrinks during EM the active set collapses and the
/// per-iteration cost falls from `O(N·M)` toward `O(N·log M)`. All work —
/// queries, accumulation, and the ordered combine — scales with the
/// surviving edges, never with `N·M`.
///
/// Returns `None` when a deterministic probe of evenly spaced targets
/// finds the active set still too large for range queries to beat the
/// streaming dense path.
fn posterior_stats_truncated<const K: usize>(
    x: &DMatrix<f64>,
    ty: &DMatrix<f64>,
    sigma2: f64,
    w: f64,
    store_p: bool,
    parallel: bool,
    single_precision: bool,
    weights: Option<&[f64]>,
) -> Option<PosteriorStats> {
    let (n, m, d) = (x.nrows(), ty.nrows(), x.ncols());
    let outlier = (2.0 * std::f64::consts::PI * sigma2).powf(d as f64 / 2.0) * w / (1.0 - w)
        * m as f64
        / n as f64;
    // With mixing weights the dominant term of a column may belong to a
    // low-weight source while a heavier source sits farther away, so the
    // "invisible beyond the margin" bound must absorb the weight range.
    // Degenerate weights (non-positive / non-finite) make no such bound
    // possible; fall back to the streaming dense path.
    let weight_margin = match weights {
        Some(values) => 2.0 * sigma2 * weight_log_ratio(values)?,
        None => 0.0,
    };
    let tree = build_kd_tree::<K>(ty);
    let margin = truncation_margin(sigma2, m, single_precision) + weight_margin;
    // Probe 32 evenly spaced targets; if the mean active fraction is
    // above ~15%, per-hit traversal overhead outweighs the skipped work.
    let probes = 32.min(n);
    let mut sampled = 0usize;
    for index in 0..probes {
        let target = index * n / probes;
        let point: [f64; K] = std::array::from_fn(|q| x[(target, q)]);
        let radius = tree.nearest_one::<SquaredEuclidean>(&point).distance + margin;
        sampled += tree
            .within_unsorted::<SquaredEuclidean>(&point, radius)
            .len();
    }
    if sampled as f64 > 0.15 * (probes * m) as f64 {
        return None;
    }
    struct TruncatedBlock {
        pt1: Vec<f64>,
        log_denominator: f64,
        /// Surviving posterior entries as (source, target, value) edges,
        /// grouped by target in query order.
        edges: Vec<(u32, u32, f64)>,
    }
    let process = |&(start, end): &(usize, usize)| -> TruncatedBlock {
        let mut block = TruncatedBlock {
            pt1: Vec::with_capacity(end - start),
            log_denominator: 0.0,
            edges: Vec::new(),
        };
        let mut values = Vec::new();
        for target in start..end {
            let point: [f64; K] = std::array::from_fn(|q| x[(target, q)]);
            let nearest = tree.nearest_one::<SquaredEuclidean>(&point);
            let radius = nearest.distance + margin;
            let first_edge = block.edges.len();
            values.clear();
            for hit in tree.within_unsorted::<SquaredEuclidean>(&point, radius) {
                block.edges.push((hit.item as u32, target as u32, 0.0));
                values.push(-hit.distance / (2.0 * sigma2));
            }
            exp_non_positive(&mut values);
            if let Some(weights) = weights {
                for (edge, value) in block.edges[first_edge..].iter().zip(values.iter_mut()) {
                    *value *= weights[edge.0 as usize];
                }
            }
            let mut denominator = outlier;
            for &value in values.iter() {
                denominator += value;
            }
            let denominator = denominator.max(f64::MIN_POSITIVE);
            let mut mass = 0.0;
            for (edge, &value) in block.edges[first_edge..].iter_mut().zip(values.iter()) {
                let normalized = value / denominator;
                edge.2 = normalized;
                mass += normalized;
            }
            block.pt1.push(mass);
            block.log_denominator += denominator.ln();
        }
        block
    };
    let ranges = dense_block_ranges(n);
    let blocks: Vec<TruncatedBlock> = if parallel && n * m >= 4096 {
        ranges.par_iter().map(process).collect()
    } else {
        ranges.iter().map(process).collect()
    };
    let mut pt1 = Vec::with_capacity(n);
    let mut p1 = vec![0.0; m];
    let mut px_rows = vec![0.0; m * d];
    let mut nll = 0.5 * n as f64 * d as f64 * (2.0 * std::f64::consts::PI * sigma2).ln();
    let mut p = store_p.then(|| DMatrix::zeros(m, n));
    for block in blocks {
        pt1.extend_from_slice(&block.pt1);
        nll -= block.log_denominator;
        for &(source, target, value) in &block.edges {
            let (source, target) = (source as usize, target as usize);
            p1[source] += value;
            for q in 0..d {
                px_rows[source * d + q] += value * x[(target, q)];
            }
            if let Some(matrix) = &mut p {
                matrix[(source, target)] = value;
            }
        }
    }
    let np = p1.iter().sum();
    let px = DMatrix::from_fn(m, d, |i, q| px_rows[i * d + q]);
    Some(PosteriorStats {
        p,
        pt1,
        p1,
        px,
        np,
        negative_log_likelihood: nll,
    })
}

pub(crate) fn weighted_mean(points: &DMatrix<f64>, weights: &[f64], total: f64) -> Vec<f64> {
    let mut mean = vec![0.0; points.ncols()];
    for i in 0..points.nrows() {
        for j in 0..points.ncols() {
            mean[j] += weights[i] * points[(i, j)];
        }
    }
    for value in &mut mean {
        *value /= total;
    }
    mean
}

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_slice_close(actual: &[f64], expected: &[f64], tolerance: f64, label: &str) {
        assert_eq!(actual.len(), expected.len(), "{label} length");
        for (index, (&actual, &expected)) in actual.iter().zip(expected).enumerate() {
            let allowed = tolerance * actual.abs().max(expected.abs()).max(1.0);
            assert!(
                (actual - expected).abs() <= allowed,
                "{label}[{index}] {actual} vs {expected}"
            );
        }
    }

    fn truncated_matches_dense<const D: usize>() {
        let count = 96;
        let x = DMatrix::from_fn(count, D, |i, j| {
            let z = i as f64;
            match j {
                0 => 1.25 * z,
                1 => (0.17 * z).sin(),
                _ => (0.11 * z).cos(),
            }
        });
        let ty = DMatrix::from_fn(count, D, |i, j| x[(i, j)] + 0.005 * (j + 1) as f64);
        let sigma2 = 0.002;
        let outlier_weight = 0.1;
        let dense = posterior_stats_dense(&x, &ty, sigma2, outlier_weight, true, false, None);
        let truncated = posterior_stats_truncated::<D>(
            &x,
            &ty,
            sigma2,
            outlier_weight,
            true,
            false,
            false,
            None,
        )
        .expect("test cloud must select the truncated path");

        const TOLERANCE: f64 = 1e-10;
        const NLL_TOLERANCE: f64 = 1e-8;
        assert_slice_close(&truncated.pt1, &dense.pt1, TOLERANCE, "pt1");
        assert_slice_close(&truncated.p1, &dense.p1, TOLERANCE, "p1");
        assert!((truncated.np - dense.np).abs() <= TOLERANCE);
        let nll_difference =
            (truncated.negative_log_likelihood - dense.negative_log_likelihood).abs();
        assert!(
            nll_difference <= NLL_TOLERANCE,
            "negative log likelihood differs by {nll_difference}"
        );
        assert_slice_close(
            truncated.px.as_slice(),
            dense.px.as_slice(),
            TOLERANCE,
            "px",
        );
        let truncated_p = truncated.p.expect("posterior requested");
        let dense_p = dense.p.expect("posterior requested");
        assert_slice_close(truncated_p.as_slice(), dense_p.as_slice(), TOLERANCE, "p");
    }

    #[test]
    fn truncated_estep_matches_dense_in_two_dimensions() {
        truncated_matches_dense::<2>();
    }

    #[test]
    fn truncated_estep_matches_dense_in_three_dimensions() {
        truncated_matches_dense::<3>();
    }

    /// Per-source mixing weights must be applied identically on every
    /// E-step path: dense f64, dense f32 (to single precision), truncated,
    /// and k-NN sparse (with k = M so it is exact).
    #[test]
    fn weighted_estep_paths_agree() {
        let count = 96;
        let x = DMatrix::from_fn(count, 3, |i, j| {
            let z = i as f64;
            match j {
                0 => 1.25 * z,
                1 => (0.17 * z).sin(),
                _ => (0.11 * z).cos(),
            }
        });
        let ty = DMatrix::from_fn(count, 3, |i, j| x[(i, j)] + 0.005 * (j + 1) as f64);
        let sigma2 = 0.002;
        let outlier_weight = 0.1;
        // Weights spanning two orders of magnitude, mean ≈ 1.
        let weights: Vec<f64> = (0..count)
            .map(|i| 0.05 + 2.0 * ((0.3 * i as f64).sin().abs()))
            .collect();
        let dense =
            posterior_stats_dense(&x, &ty, sigma2, outlier_weight, true, false, Some(&weights));
        // Weights redistribute mass but never create or destroy it: the
        // per-target inlier mass still sums to Np, and heavier sources
        // must end up with more mass than uniform weighting gives them.
        let uniform = posterior_stats_dense(&x, &ty, sigma2, outlier_weight, true, false, None);
        assert!((dense.pt1.iter().sum::<f64>() - dense.np).abs() < 1e-9);
        let heaviest = (0..count)
            .max_by(|&a, &b| weights[a].total_cmp(&weights[b]))
            .unwrap();
        assert!(dense.p1[heaviest] >= uniform.p1[heaviest]);

        let truncated = posterior_stats_truncated::<3>(
            &x,
            &ty,
            sigma2,
            outlier_weight,
            true,
            false,
            false,
            Some(&weights),
        )
        .expect("test cloud must select the truncated path");
        assert_slice_close(&truncated.pt1, &dense.pt1, 1e-10, "pt1");
        assert_slice_close(&truncated.p1, &dense.p1, 1e-10, "p1");
        assert_slice_close(truncated.px.as_slice(), dense.px.as_slice(), 1e-10, "px");
        assert!(
            (truncated.negative_log_likelihood - dense.negative_log_likelihood).abs() <= 1e-8
        );

        let single = posterior_stats_dense_f32(
            &x,
            &ty,
            sigma2,
            outlier_weight,
            false,
            false,
            Some(&weights),
        );
        assert_slice_close(&single.p1, &dense.p1, 1e-4, "p1 (f32)");
        assert_slice_close(&single.pt1, &dense.pt1, 1e-4, "pt1 (f32)");

        let index = SparseIndex::new(&x, count);
        let sparse = posterior_stats_sparse(
            &x,
            &ty,
            sigma2,
            outlier_weight,
            &index,
            false,
            false,
            Some(&weights),
        );
        assert_slice_close(&sparse.p1, &dense.p1, 1e-10, "p1 (sparse)");
        assert_slice_close(&sparse.pt1, &dense.pt1, 1e-10, "pt1 (sparse)");
        assert_slice_close(sparse.px.as_slice(), dense.px.as_slice(), 1e-10, "px (sparse)");

        // A vector of ones is exactly classic CPD.
        let ones = vec![1.0; count];
        let unit = posterior_stats_dense(&x, &ty, sigma2, outlier_weight, false, false, Some(&ones));
        assert_slice_close(&unit.p1, &uniform.p1, 1e-15, "p1 (unit weights)");
    }
}
