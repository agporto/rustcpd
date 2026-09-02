//! Python bindings for the `rustcpd` crate.
//!
//! Compiled against the CPython stable ABI (`abi3-py39`), so one wheel per
//! platform serves every CPython ≥ 3.9. All registrations release the GIL
//! while the Rust core runs.

use cpd::DMatrix;
use numpy::ndarray::Array2;
use numpy::ndarray::ArrayView2;
use numpy::{AllowTypeChange, PyArray1, PyArray2, PyArrayLike2, ToPyArray};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;
use rustcpd as cpd;

fn matrix_from(view: ArrayView2<'_, f64>) -> DMatrix<f64> {
    DMatrix::from_fn(view.nrows(), view.ncols(), |i, j| view[(i, j)])
}

fn matrix_to(py: Python<'_>, matrix: &DMatrix<f64>) -> Py<PyArray2<f64>> {
    Array2::from_shape_fn((matrix.nrows(), matrix.ncols()), |(i, j)| matrix[(i, j)])
        .to_pyarray(py)
        .unbind()
}

fn vector_to(py: Python<'_>, values: &[f64]) -> Py<PyArray1<f64>> {
    PyArray1::from_slice(py, values).unbind()
}

fn error(err: cpd::Error) -> PyErr {
    PyValueError::new_err(err.to_string())
}

/// Invoke a Python per-iteration callback with the current state, returning
/// whether to continue. The callback receives a dict
/// `{"iteration", "sigma2", "difference", "points"}`; `None` (no explicit
/// return) continues, a bool continues/stops, and any other return type is
/// a `TypeError` — silently ignoring e.g. `return 0` would be a trap.
fn invoke_callback(
    py: Python<'_>,
    callback: &Py<PyAny>,
    state: &cpd::IterationState,
) -> PyResult<bool> {
    let dict = PyDict::new(py);
    dict.set_item("iteration", state.iteration)?;
    dict.set_item("sigma2", state.sigma2)?;
    dict.set_item("difference", state.difference)?;
    dict.set_item("points", matrix_to(py, state.points))?;
    let ret = callback.call1(py, (dict,))?;
    if ret.is_none(py) {
        return Ok(true);
    }
    ret.extract::<bool>(py).map_err(|_| {
        pyo3::exceptions::PyTypeError::new_err(
            "registration callback must return None or a bool (False stops the iteration)",
        )
    })
}

#[allow(clippy::too_many_arguments)]
fn em_config(
    sigma2: Option<f64>,
    max_iterations: usize,
    tolerance: f64,
    outlier_weight: f64,
    k: Option<usize>,
    parallel: bool,
    single_precision: bool,
) -> cpd::EmConfig {
    cpd::EmConfig {
        sigma2,
        max_iterations,
        tolerance,
        outlier_weight,
        k,
        parallel,
        single_precision,
    }
}

/// Result of a rigid registration.
#[pyclass(frozen, module = "rustcpd")]
pub struct RigidResult {
    /// Transformed source points `s·Y·R + t`, shape `(M, D)`.
    #[pyo3(get)]
    pub points: Py<PyArray2<f64>>,
    /// Rotation in row-vector convention (`y @ rotation`), shape `(D, D)`.
    #[pyo3(get)]
    pub rotation: Py<PyArray2<f64>>,
    /// Translation vector, shape `(D,)`.
    #[pyo3(get)]
    pub translation: Py<PyArray1<f64>>,
    /// Estimated isotropic scale (1.0 when disabled).
    #[pyo3(get)]
    pub scale: f64,
    /// Final Gaussian variance.
    #[pyo3(get)]
    pub sigma2: f64,
    /// EM iterations performed.
    #[pyo3(get)]
    pub iterations: usize,
    /// Final EM objective value.
    #[pyo3(get)]
    pub objective: f64,
    /// Final convergence-criterion value.
    #[pyo3(get)]
    pub difference: f64,
}

/// Result of an affine registration.
#[pyclass(frozen, module = "rustcpd")]
pub struct AffineResult {
    /// Transformed source points `Y·B + t`, shape `(M, D)`.
    #[pyo3(get)]
    pub points: Py<PyArray2<f64>>,
    /// Affine matrix in row-vector convention (`y @ transform`).
    #[pyo3(get)]
    pub transform: Py<PyArray2<f64>>,
    /// Translation vector, shape `(D,)`.
    #[pyo3(get)]
    pub translation: Py<PyArray1<f64>>,
    /// Final Gaussian variance.
    #[pyo3(get)]
    pub sigma2: f64,
    /// EM iterations performed.
    #[pyo3(get)]
    pub iterations: usize,
    /// Final EM objective value.
    #[pyo3(get)]
    pub objective: f64,
    /// Final convergence-criterion value.
    #[pyo3(get)]
    pub difference: f64,
}

/// Result of a deformable registration.
#[pyclass(frozen, module = "rustcpd")]
pub struct DeformableResult {
    /// Deformed source points `Y + G·W`, shape `(M, D)`.
    #[pyo3(get)]
    pub points: Py<PyArray2<f64>>,
    /// Deformation coefficients `W`, shape `(M, D)`.
    #[pyo3(get)]
    pub weights: Py<PyArray2<f64>>,
    /// Dense Gaussian kernel `G`, shape `(M, M)` — populated only for a
    /// full-rank fit (`low_rank=None`). For a low-rank fit it is an empty
    /// `(0, 0)` array; use `low_rank_basis` / `low_rank_eigenvalues`.
    #[pyo3(get)]
    pub kernel: Py<PyArray2<f64>>,
    /// The moving source cloud `Y`, shape `(M, D)`.
    #[pyo3(get)]
    pub source: Py<PyArray2<f64>>,
    /// Gaussian kernel width `beta` used to fit the deformation.
    #[pyo3(get)]
    pub beta: f64,
    /// Final Gaussian variance.
    #[pyo3(get)]
    pub sigma2: f64,
    /// EM iterations performed.
    #[pyo3(get)]
    pub iterations: usize,
    /// Final convergence-criterion value.
    #[pyo3(get)]
    pub difference: f64,
    // Retained in native form so `transform` avoids re-parsing NumPy.
    source_matrix: DMatrix<f64>,
    weights_matrix: DMatrix<f64>,
    // Low-rank factor `L` (`G ≈ L Lᵀ`) for a low-rank fit; the `(Q, Λ)`
    // form is derived on demand by the `low_rank_basis` /
    // `low_rank_eigenvalues` getters below.
    low_rank_factor: Option<DMatrix<f64>>,
    // Normalization frame `(target_centroid, source_centroid, scale)` when
    // the fit was normalized, so `transform` maps new points through the
    // same conditioning.
    normalization: Option<(Vec<f64>, Vec<f64>, f64)>,
}

#[pymethods]
impl DeformableResult {
    /// Orthonormal low-rank basis `Q`, shape `(M, r)`, or `None` for a
    /// full-rank fit. Computed on demand from the stored factor, with
    /// `G ≈ Q @ diag(low_rank_eigenvalues) @ Q.T`.
    #[getter]
    fn low_rank_basis(&self, py: Python<'_>) -> Option<Py<PyArray2<f64>>> {
        self.low_rank_factor
            .as_ref()
            .and_then(cpd::low_rank_spectrum)
            .map(|(q, _)| matrix_to(py, &q))
    }

    /// Low-rank eigenvalues `Λ`, shape `(r,)`, or `None` for a full-rank
    /// fit. Computed on demand from the stored factor.
    #[getter]
    fn low_rank_eigenvalues(&self, py: Python<'_>) -> Option<Py<PyArray1<f64>>> {
        self.low_rank_factor
            .as_ref()
            .and_then(cpd::low_rank_spectrum)
            .map(|(_, s)| vector_to(py, &s))
    }

    /// Evaluate the learned deformation at arbitrary points `z`
    /// (shape `(P, D)`), returning `z + G(z, Y) @ W`.
    ///
    /// The registration warps only the points it was fit on; this applies
    /// the same continuous displacement field to any points in the source
    /// frame — a full-resolution mesh, landmarks, a grid — so a coarse
    /// registration can drive a dense warp.
    fn transform(
        &self,
        py: Python<'_>,
        z: PyArrayLike2<'_, f64, AllowTypeChange>,
    ) -> PyResult<Py<PyArray2<f64>>> {
        let z = matrix_from(z.as_array());
        let normalization =
            self.normalization
                .as_ref()
                .map(|(target_centroid, source_centroid, scale)| {
                    (
                        target_centroid.as_slice(),
                        source_centroid.as_slice(),
                        *scale,
                    )
                });
        let warped = py
            .detach(|| {
                cpd::apply_deformation(
                    &self.source_matrix,
                    &self.weights_matrix,
                    self.beta,
                    normalization,
                    &z,
                )
            })
            .map_err(error)?;
        Ok(matrix_to(py, &warped))
    }
}

/// Result of a statistical-shape-model (atlas) registration.
#[pyclass(frozen, module = "rustcpd")]
pub struct AtlasResult {
    /// Deformed-and-transformed model points, shape `(M, D)`.
    #[pyo3(get)]
    pub points: Py<PyArray2<f64>>,
    /// Estimated shape coefficients, shape `(rank,)`.
    #[pyo3(get)]
    pub coefficients: Py<PyArray1<f64>>,
    /// Rotation in row-vector convention (`y @ rotation`), shape `(D, D)`.
    #[pyo3(get)]
    pub rotation: Py<PyArray2<f64>>,
    /// Estimated isotropic scale (1.0 when disabled).
    #[pyo3(get)]
    pub scale: f64,
    /// Translation vector, shape `(D,)`.
    #[pyo3(get)]
    pub translation: Py<PyArray1<f64>>,
    /// Final Gaussian variance.
    #[pyo3(get)]
    pub sigma2: f64,
    /// EM iterations performed.
    #[pyo3(get)]
    pub iterations: usize,
    /// Final convergence-criterion value.
    #[pyo3(get)]
    pub difference: f64,
    /// Pointwise RMS of the anchored-landmark residuals `sqrt(sum||r_l||^2 / K)`
    /// (original target frame); NaN when no landmarks were supplied. Under
    /// isotropic per-coordinate noise `tau` its noise floor is `sqrt(D)*tau`
    /// (`sqrt(3)*tau` in 3D), not `tau`; for a scale-free check form the reduced
    /// chi-square `(landmark_rms / (sqrt(D)*tau))**2`, which is ~1 at the noise
    /// floor.
    #[pyo3(get)]
    pub landmark_rms: f64,
    /// Final per-source mixing proportions (shape `(M,)`, summing to 1) when
    /// `adaptive_mixing` was set; `None` for classic uniform mixing. Small
    /// values mark model points the target did not support (for a partial
    /// target, the unobserved part of the model).
    #[pyo3(get)]
    pub mixing_weights: Option<Py<PyArray1<f64>>>,
    // Retained in native form for reconstruct / apply_similarity.
    coefficients_vec: Vec<f64>,
    rotation_matrix: DMatrix<f64>,
    translation_vec: Vec<f64>,
}

#[pymethods]
impl AtlasResult {
    /// Reconstruct the fitted shape at full resolution:
    /// `similarity(mean + modes @ b)` in the target frame.
    ///
    /// The atlas can be *registered* on a subsample; pass the
    /// full-resolution `mean` (shape `(P, D)`) and `modes` (shape
    /// `(P*D, rank)`, point-major) here to rebuild the dense fitted model.
    /// The number of modes must match the fitted coefficients.
    fn reconstruct(
        &self,
        py: Python<'_>,
        mean: PyArrayLike2<'_, f64, AllowTypeChange>,
        modes: PyArrayLike2<'_, f64, AllowTypeChange>,
    ) -> PyResult<Py<PyArray2<f64>>> {
        let mean = matrix_from(mean.as_array());
        let modes = matrix_from(modes.as_array());
        let inner = self.core();
        let rebuilt = py
            .detach(|| inner.reconstruct(&mean, &modes))
            .map_err(error)?;
        Ok(matrix_to(py, &rebuilt))
    }

    /// Apply only the fitted similarity transform (rotation, scale,
    /// translation) to arbitrary points `z` (shape `(P, D)`), returning
    /// `scale * (z @ rotation) + translation`. Ignores the shape deformation.
    fn apply_similarity(
        &self,
        py: Python<'_>,
        z: PyArrayLike2<'_, f64, AllowTypeChange>,
    ) -> PyResult<Py<PyArray2<f64>>> {
        let z = matrix_from(z.as_array());
        let inner = self.core();
        let out = py.detach(|| inner.apply_similarity(&z)).map_err(error)?;
        Ok(matrix_to(py, &out))
    }

    /// Build a shape posterior from this fit for completion and calibrated
    /// per-point uncertainty. `target` is the observed (possibly partial)
    /// cloud, `mean`/`modes`/`eigenvalues` the model (may be denser than
    /// what was registered). Visibility is inferred from the fitted
    /// correspondence, optionally anchored by `completeness ∈ (0, 1]`.
    #[pyo3(signature = (target, mean, modes, eigenvalues, *, completeness = None,
        visibility_floor = 1e-6, prior_temperature = 1.0, outlier_weight = 0.0,
        estimate_discrepancy = true))]
    #[allow(clippy::too_many_arguments)]
    fn posterior(
        &self,
        py: Python<'_>,
        target: PyArrayLike2<'_, f64, AllowTypeChange>,
        mean: PyArrayLike2<'_, f64, AllowTypeChange>,
        modes: PyArrayLike2<'_, f64, AllowTypeChange>,
        eigenvalues: Vec<f64>,
        completeness: Option<f64>,
        visibility_floor: f64,
        prior_temperature: f64,
        outlier_weight: f64,
        estimate_discrepancy: bool,
    ) -> PyResult<ShapePosterior> {
        let target = matrix_from(target.as_array());
        let mean = matrix_from(mean.as_array());
        let modes = matrix_from(modes.as_array());
        let options = cpd::PosteriorOptions {
            completeness,
            visibility_floor,
            prior_temperature,
            outlier_weight,
            estimate_discrepancy,
        };
        let inner = self.core();
        let posterior = py
            .detach(|| inner.posterior(&target, &mean, &modes, &eigenvalues, &options))
            .map_err(error)?;
        Ok(ShapePosterior { inner: posterior })
    }
}

/// Gaussian posterior over shape coefficients for completion and per-point
/// uncertainty. Build with `AtlasResult.posterior(...)` or `complete_shape`.
#[pyclass(frozen, module = "rustcpd")]
pub struct ShapePosterior {
    inner: cpd::ShapePosterior,
}

#[pymethods]
impl ShapePosterior {
    /// Posterior mean shape coefficients, shape `(k,)`.
    #[getter]
    fn coefficient_mean(&self, py: Python<'_>) -> Py<PyArray1<f64>> {
        vector_to(py, self.inner.coefficient_mean())
    }

    /// Posterior covariance of the coefficients, shape `(k, k)`.
    #[getter]
    fn coefficient_covariance(&self, py: Python<'_>) -> Py<PyArray2<f64>> {
        matrix_to(py, self.inner.coefficient_covariance())
    }

    /// Completed shape in the target frame, shape `(M, D)`.
    fn predict(&self, py: Python<'_>) -> Py<PyArray2<f64>> {
        matrix_to(py, &self.inner.predict())
    }

    /// Completed shape in the model frame (before pose), shape `(M, D)`.
    fn predict_model_frame(&self, py: Python<'_>) -> Py<PyArray2<f64>> {
        matrix_to(py, &self.inner.predict_model_frame())
    }

    /// Per-point total predictive variance (target frame), shape `(M,)`.
    /// The confidence map: small where the fragment constrains the shape.
    /// Includes the discrepancy variance when it was estimated.
    fn predictive_variance(&self, py: Python<'_>) -> Py<PyArray1<f64>> {
        vector_to(py, &self.inner.predictive_variance())
    }

    /// Full `D×D` predictive covariance at model point `i` (target frame).
    fn predictive_covariance(&self, py: Python<'_>, i: usize) -> Py<PyArray2<f64>> {
        matrix_to(py, &self.inner.predictive_covariance(i))
    }

    /// Assumed observation-noise variance in the model frame.
    #[getter]
    fn noise_variance(&self) -> f64 {
        self.inner.noise_variance()
    }

    /// Estimated model-inadequacy (discrepancy) variance: the observed
    /// residual the model could not explain. `0.0` when disabled or when
    /// the model fits the observed region within noise. Large values flag
    /// out-of-distribution fragments.
    #[getter]
    fn discrepancy_variance(&self) -> f64 {
        self.inner.discrepancy_variance()
    }

    /// Draw `count` coefficient vectors from the posterior, shape
    /// `(count, k)`. Deterministic in `seed`.
    fn sample_coefficients(
        &self,
        py: Python<'_>,
        count: usize,
        seed: u64,
    ) -> PyResult<Py<PyArray2<f64>>> {
        let samples = py
            .detach(|| self.inner.sample_coefficients(count, seed))
            .map_err(error)?;
        let k = self.inner.coefficient_mean().len();
        let flat: Vec<f64> = samples.into_iter().flatten().collect();
        let array = Array2::from_shape_fn((count, k), |(i, j)| flat[i * k + j]);
        Ok(array.to_pyarray(py).unbind())
    }

    /// Draw `count` completed shapes (target frame), as a list of `(M, D)`
    /// arrays. Deterministic in `seed`.
    fn sample_shapes(
        &self,
        py: Python<'_>,
        count: usize,
        seed: u64,
    ) -> PyResult<Vec<Py<PyArray2<f64>>>> {
        let shapes = py
            .detach(|| self.inner.sample_shapes(count, seed))
            .map_err(error)?;
        Ok(shapes.iter().map(|s| matrix_to(py, s)).collect())
    }
}

impl AtlasResult {
    /// Rebuild a minimal core result carrying only the fields the
    /// transform helpers read.
    fn core(&self) -> cpd::AtlasResult {
        cpd::AtlasResult {
            points: DMatrix::zeros(0, 0),
            coefficients: self.coefficients_vec.clone(),
            rotation: self.rotation_matrix.clone(),
            scale: self.scale,
            translation: self.translation_vec.clone(),
            sigma2: self.sigma2,
            iterations: self.iterations,
            difference: self.difference,
            negative_log_likelihood: f64::INFINITY,
            landmark_rms: f64::NAN,
            mixing_weights: None,
        }
    }
}

/// Best pose hypothesis from the pose-marginalized initializer.
#[pyclass(frozen, module = "rustcpd")]
pub struct PoseInitialization {
    /// Shape coefficients of the winning hypothesis, shape `(rank,)`.
    #[pyo3(get)]
    pub coefficients: Py<PyArray1<f64>>,
    /// Rotation of the winning hypothesis (`y @ rotation`).
    #[pyo3(get)]
    pub rotation: Py<PyArray2<f64>>,
    /// Scale of the winning hypothesis.
    #[pyo3(get)]
    pub scale: f64,
    /// Translation of the winning hypothesis, shape `(3,)`.
    #[pyo3(get)]
    pub translation: Py<PyArray1<f64>>,
    /// Negative-log-posterior score of the winner (lower is better).
    /// Unnormalized: it only ranks hypotheses *within a single run* (and
    /// includes the keypoint penalty when landmarks are set). Not comparable
    /// across runs, across a different keypoint count, or a different
    /// `landmark_sigma` / `landmark_weight`. For a cross-fit confidence signal
    /// use the atlas `sigma2` with `calibration.PoseConfidenceCalibrator`.
    #[pyo3(get)]
    pub score: f64,
    /// Score gap to the runner-up hypothesis. Same within-run caveat as `score`.
    #[pyo3(get)]
    pub score_margin: f64,
    /// Shannon entropy of the refined-hypothesis posterior.
    #[pyo3(get)]
    pub posterior_entropy: f64,
    /// Effective number of competitive hypotheses.
    #[pyo3(get)]
    pub effective_hypotheses: f64,
    /// Rotation hypotheses evaluated in the coarse pass.
    #[pyo3(get)]
    pub hypotheses_evaluated: usize,
    /// Hypotheses carried through refinement (before merging).
    #[pyo3(get)]
    pub hypotheses_refined: usize,
    /// Distinct solutions among the refined hypotheses after merging starts
    /// that converged to the same fit (`merge_tolerance`).
    #[pyo3(get)]
    pub distinct_hypotheses: usize,
    /// Refined starts that converged to the winning solution. Several starts
    /// agreeing is evidence *for* the winner, not ambiguity.
    #[pyo3(get)]
    pub winner_support: usize,
    /// Translation anchors actually used per rotation (1 = centroid only).
    #[pyo3(get)]
    pub translation_anchors_used: usize,
}

/// Rigid (rotation + translation + optional scale) registration of the
/// moving `source` cloud onto the fixed `target` cloud (2-D or 3-D).
///
/// `normalize=True` conditions the fit in an internal unit-scale frame and
/// maps the result back to the original coordinates (recommended for
/// clouds in large physical units). `callback`, if given, is called after
/// each EM iteration with a dict `{"iteration", "sigma2", "difference",
/// "points"}`; return `False` to stop early, `True` or `None` to continue
/// (with `normalize=True` the state is in the internal normalized frame).
#[pyfunction]
#[pyo3(signature = (target, source, *, scale = true, normalize = false,
    sigma2 = None, max_iterations = 100, tolerance = 1e-3, outlier_weight = 0.0,
    k = None, parallel = true, single_precision = false, callback = None))]
#[allow(clippy::too_many_arguments)]
fn register_rigid(
    py: Python<'_>,
    target: PyArrayLike2<'_, f64, AllowTypeChange>,
    source: PyArrayLike2<'_, f64, AllowTypeChange>,
    scale: bool,
    normalize: bool,
    sigma2: Option<f64>,
    max_iterations: usize,
    tolerance: f64,
    outlier_weight: f64,
    k: Option<usize>,
    parallel: bool,
    single_precision: bool,
    callback: Option<Py<PyAny>>,
) -> PyResult<RigidResult> {
    let x = matrix_from(target.as_array());
    let y = matrix_from(source.as_array());
    let config = cpd::RigidConfig {
        em: em_config(
            sigma2,
            max_iterations,
            tolerance,
            outlier_weight,
            k,
            parallel,
            single_precision,
        ),
        scale,
        normalize,
    };
    let result = match callback {
        Some(callback) => {
            // Release the GIL for the whole run (so other Python threads
            // keep executing during the heavy Rust iterations) and
            // reacquire it only for the brief callback invocations.
            let mut callback_error: Option<PyErr> = None;
            let out = py
                .detach(|| {
                    cpd::RigidRegistration::new(&x, &y, config)?.register_with(|state| {
                        Python::attach(|py| match invoke_callback(py, &callback, state) {
                            Ok(cont) => cont,
                            Err(err) => {
                                callback_error = Some(err);
                                false
                            }
                        })
                    })
                })
                .map_err(error)?;
            if let Some(err) = callback_error {
                return Err(err);
            }
            out
        }
        None => py
            .detach(|| cpd::RigidRegistration::new(&x, &y, config)?.register())
            .map_err(error)?,
    };
    Ok(RigidResult {
        points: matrix_to(py, &result.points),
        rotation: matrix_to(py, &result.rotation),
        translation: vector_to(py, &result.translation),
        scale: result.scale,
        sigma2: result.sigma2,
        iterations: result.iterations,
        objective: result.objective,
        difference: result.difference,
    })
}

/// Affine registration of the moving `source` cloud onto the fixed
/// `target` cloud.
///
/// `normalize` and `callback` behave as in `register_rigid`.
#[pyfunction]
#[pyo3(signature = (target, source, *, normalize = false, sigma2 = None,
    max_iterations = 100, tolerance = 1e-3, outlier_weight = 0.0, k = None,
    parallel = true, single_precision = false, callback = None))]
#[allow(clippy::too_many_arguments)]
fn register_affine(
    py: Python<'_>,
    target: PyArrayLike2<'_, f64, AllowTypeChange>,
    source: PyArrayLike2<'_, f64, AllowTypeChange>,
    normalize: bool,
    sigma2: Option<f64>,
    max_iterations: usize,
    tolerance: f64,
    outlier_weight: f64,
    k: Option<usize>,
    parallel: bool,
    single_precision: bool,
    callback: Option<Py<PyAny>>,
) -> PyResult<AffineResult> {
    let x = matrix_from(target.as_array());
    let y = matrix_from(source.as_array());
    let config = cpd::AffineConfig {
        em: em_config(
            sigma2,
            max_iterations,
            tolerance,
            outlier_weight,
            k,
            parallel,
            single_precision,
        ),
        normalize,
    };
    let result = match callback {
        Some(callback) => {
            // See register_rigid: GIL released for the run, reacquired
            // only inside the callback.
            let mut callback_error: Option<PyErr> = None;
            let out = py
                .detach(|| {
                    cpd::AffineRegistration::new(&x, &y, config)?.register_with(|state| {
                        Python::attach(|py| match invoke_callback(py, &callback, state) {
                            Ok(cont) => cont,
                            Err(err) => {
                                callback_error = Some(err);
                                false
                            }
                        })
                    })
                })
                .map_err(error)?;
            if let Some(err) = callback_error {
                return Err(err);
            }
            out
        }
        None => py
            .detach(|| cpd::AffineRegistration::new(&x, &y, config)?.register())
            .map_err(error)?,
    };
    Ok(AffineResult {
        points: matrix_to(py, &result.points),
        transform: matrix_to(py, &result.transform),
        translation: vector_to(py, &result.translation),
        sigma2: result.sigma2,
        iterations: result.iterations,
        objective: result.objective,
        difference: result.difference,
    })
}

/// Deformable (non-rigid) registration of the moving `source` cloud onto
/// the fixed `target` cloud.
///
/// `low_rank` approximates the Gaussian kernel with a rank-`low_rank`
/// factor when it is smaller than the source size; pass `None` for the
/// exact full-rank solve. `low_rank_method` selects how that factor is
/// built: `"eigen"` (the default; a full symmetric eigendecomposition,
/// also accepted as `"full"`) or `"pivoted_cholesky"` (a greedy pivoted
/// incomplete Cholesky, much cheaper to build for large clouds; also
/// accepted as `"cholesky"`/`"pivoted"`). `constraints` pins source
/// points to target points as `[(source_index, target_index), ...]`.
///
/// `low_rank_tolerance` is the pivoted-Cholesky residual-diagonal
/// early-stop tolerance in `[0, 1)` (ignored by `"eigen"`).
///
/// A low-rank fit never *returns* the dense kernel: `.kernel` is empty and
/// the approximation is exposed as `.low_rank_basis` (Q) and
/// `.low_rank_eigenvalues` (Λ), with `G ≈ Q @ diag(Λ) @ Q.T`. Only
/// `"pivoted_cholesky"` also avoids *building* the dense kernel (it reads
/// kernel columns on demand, `O(M·rank)` memory); `"eigen"` still forms it
/// temporarily for the decomposition.
///
/// `callback`, if given, is called after each EM iteration with a dict
/// `{"iteration", "sigma2", "difference", "points"}`; return `False` to
/// stop early, `True` or `None` to continue (any other return raises
/// `TypeError`). The GIL is released while the Rust core runs and
/// reacquired only for each callback invocation. With `normalize=True`
/// the callback state is in the internal normalized frame.
#[pyfunction]
#[pyo3(signature = (target, source, *, alpha = 2.0, beta = 2.0,
    low_rank = 300, low_rank_method = "eigen", low_rank_tolerance = 0.0,
    normalize = false, constraints = None, constraint_error = 1e-8,
    sigma2 = None, max_iterations = 100, tolerance = 1e-3,
    outlier_weight = 0.0, k = None, parallel = true, single_precision = false,
    callback = None))]
#[allow(clippy::too_many_arguments)]
fn register_deformable(
    py: Python<'_>,
    target: PyArrayLike2<'_, f64, AllowTypeChange>,
    source: PyArrayLike2<'_, f64, AllowTypeChange>,
    alpha: f64,
    beta: f64,
    low_rank: Option<usize>,
    low_rank_method: &str,
    low_rank_tolerance: f64,
    normalize: bool,
    constraints: Option<Vec<(usize, usize)>>,
    constraint_error: f64,
    sigma2: Option<f64>,
    max_iterations: usize,
    tolerance: f64,
    outlier_weight: f64,
    k: Option<usize>,
    parallel: bool,
    single_precision: bool,
    callback: Option<Py<PyAny>>,
) -> PyResult<DeformableResult> {
    let x = matrix_from(target.as_array());
    let y = matrix_from(source.as_array());
    let low_rank_method = match low_rank_method.to_ascii_lowercase().as_str() {
        "eigen" | "full" | "eig" => cpd::LowRankMethod::Eigen,
        "pivoted_cholesky" | "cholesky" | "pivoted" | "chol" => cpd::LowRankMethod::PivotedCholesky,
        other => {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "unknown low_rank_method {other:?}; expected \"eigen\" or \"pivoted_cholesky\""
            )));
        }
    };
    let config = cpd::DeformableConfig {
        em: em_config(
            sigma2,
            max_iterations,
            tolerance,
            outlier_weight,
            k,
            parallel,
            single_precision,
        ),
        alpha,
        beta,
        low_rank,
        low_rank_method,
        pivoted_cholesky_tolerance: low_rank_tolerance,
        normalize,
        constraints: constraints
            .unwrap_or_default()
            .into_iter()
            .map(|(source, target)| cpd::Constraint { source, target })
            .collect(),
        constraint_error,
    };
    let result = match callback {
        Some(callback) => {
            // See register_rigid: GIL released for the run, reacquired
            // only inside the callback.
            let mut callback_error: Option<PyErr> = None;
            let out = py
                .detach(|| {
                    cpd::DeformableRegistration::new(&x, &y, config)?.register_with(|state| {
                        Python::attach(|py| match invoke_callback(py, &callback, state) {
                            Ok(cont) => cont,
                            Err(err) => {
                                callback_error = Some(err);
                                false
                            }
                        })
                    })
                })
                .map_err(error)?;
            if let Some(err) = callback_error {
                return Err(err);
            }
            out
        }
        None => py
            .detach(|| cpd::DeformableRegistration::new(&x, &y, config)?.register())
            .map_err(error)?,
    };
    let normalization = result
        .normalization()
        .map(|(target_c, source_c, scale)| (target_c.to_vec(), source_c.to_vec(), scale));
    let low_rank_factor = result.low_rank_factor().cloned();
    Ok(DeformableResult {
        points: matrix_to(py, &result.points),
        weights: matrix_to(py, &result.weights),
        kernel: matrix_to(py, &result.kernel),
        source: matrix_to(py, &result.source),
        beta: result.beta,
        sigma2: result.sigma2,
        iterations: result.iterations,
        difference: result.difference,
        source_matrix: result.source,
        weights_matrix: result.weights,
        low_rank_factor,
        normalization,
    })
}

/// Statistical-shape-model (atlas) registration: fits `mean + modes @ b`
/// plus a similarity transform to the fixed `target` cloud.
///
/// `modes` has shape `(M*D, rank)` with rows in point-major order and
/// `eigenvalues` one positive value per mode.
///
/// For **partial targets** (the target is a fragment of the model) see
/// `scale_bounds` — a `(min, max)` interval that stops the closed-form
/// scale from shrinking the whole model into the fragment — and
/// `adaptive_mixing=alpha`, which re-estimates per-model-point mixing
/// proportions so points with no supporting data switch off (reported as
/// `result.mixing_weights`).
#[pyfunction]
#[pyo3(signature = (target, mean, modes, eigenvalues, *,
    lambda_regularization = 0.1, normalize = false,
    optimize_similarity = true, with_scale = true,
    kdtree_radius_scale = None,
    initial_coefficients = None, initial_rotation = None,
    initial_scale = 1.0, initial_translation = None, sigma2 = None,
    landmark_indices = None, landmark_targets = None, landmark_weight = 0.0,
    landmark_sigma = None, scale_bounds = None, adaptive_mixing = None,
    max_iterations = 100, tolerance = 1e-3, outlier_weight = 0.0,
    k = None, parallel = true, single_precision = false))]
#[allow(clippy::too_many_arguments)]
fn register_atlas(
    py: Python<'_>,
    target: PyArrayLike2<'_, f64, AllowTypeChange>,
    mean: PyArrayLike2<'_, f64, AllowTypeChange>,
    modes: PyArrayLike2<'_, f64, AllowTypeChange>,
    eigenvalues: Vec<f64>,
    lambda_regularization: f64,
    normalize: bool,
    optimize_similarity: bool,
    with_scale: bool,
    kdtree_radius_scale: Option<f64>,
    initial_coefficients: Option<Vec<f64>>,
    initial_rotation: Option<PyArrayLike2<'_, f64, AllowTypeChange>>,
    initial_scale: f64,
    initial_translation: Option<Vec<f64>>,
    sigma2: Option<f64>,
    landmark_indices: Option<Vec<usize>>,
    landmark_targets: Option<PyArrayLike2<'_, f64, AllowTypeChange>>,
    landmark_weight: f64,
    landmark_sigma: Option<f64>,
    scale_bounds: Option<(f64, f64)>,
    adaptive_mixing: Option<f64>,
    max_iterations: usize,
    tolerance: f64,
    outlier_weight: f64,
    k: Option<usize>,
    parallel: bool,
    single_precision: bool,
) -> PyResult<AtlasResult> {
    let x = matrix_from(target.as_array());
    let mean = matrix_from(mean.as_array());
    let modes = matrix_from(modes.as_array());
    let landmarks: Vec<(usize, Vec<f64>)> = match (landmark_indices, landmark_targets.as_ref()) {
        (Some(indices), Some(points)) => {
            let pts = matrix_from(points.as_array());
            if indices.len() != pts.nrows() {
                return Err(PyValueError::new_err(
                    "landmark_indices and landmark_targets must have matching lengths",
                ));
            }
            indices
                .into_iter()
                .enumerate()
                .map(|(row, index)| (index, (0..pts.ncols()).map(|j| pts[(row, j)]).collect()))
                .collect()
        }
        (None, None) => Vec::new(),
        _ => {
            return Err(PyValueError::new_err(
                "landmark_indices and landmark_targets must be provided together",
            ));
        }
    };
    let config = cpd::AtlasConfig {
        em: em_config(
            sigma2,
            max_iterations,
            tolerance,
            outlier_weight,
            k,
            parallel,
            single_precision,
        ),
        eigenvalues,
        lambda_regularization,
        normalize,
        optimize_similarity,
        with_scale,
        kdtree_radius_scale,
        initial_coefficients,
        initial_rotation: initial_rotation.as_ref().map(|r| matrix_from(r.as_array())),
        initial_scale,
        initial_translation,
        landmarks,
        landmark_weight,
        landmark_sigma,
        scale_bounds,
        adaptive_mixing,
    };
    let result = py
        .detach(|| cpd::AtlasRegistration::new(&x, &mean, &modes, config)?.register())
        .map_err(error)?;
    Ok(AtlasResult {
        points: matrix_to(py, &result.points),
        coefficients: vector_to(py, &result.coefficients),
        rotation: matrix_to(py, &result.rotation),
        scale: result.scale,
        translation: vector_to(py, &result.translation),
        sigma2: result.sigma2,
        iterations: result.iterations,
        difference: result.difference,
        landmark_rms: result.landmark_rms,
        mixing_weights: result.mixing_weights.as_deref().map(|pi| vector_to(py, pi)),
        coefficients_vec: result.coefficients,
        rotation_matrix: result.rotation,
        translation_vec: result.translation,
    })
}

/// Pose-marginalized initialization: sweeps a rotation lattice and
/// returns the best-scoring similarity transform and shape coefficients
/// for starting a full atlas registration (3-D only).
///
/// Set `with_scale=False` when `source` and `modes` were already pre-scaled
/// from an external physical-size estimate. Rotation and translation remain
/// optimized, while the residual isotropic scale is fixed at 1.0.
///
/// **Partial targets.** By default every rotation is seeded by placing the
/// model centroid on the target centroid, which is wrong for a fragment
/// (the proximal third of a femur is not centred on the bone). Set
/// `translation_anchor_count > 1` to also seed fragment-sized local
/// centroids of the model as translation anchors; this needs the scale
/// pinned down (`with_scale=False` or `scale_bounds=(lo, hi)`) and only
/// activates when the target is smaller than `anchor_completeness_threshold`
/// of the model, so complete targets are unaffected. `adaptive_mixing=alpha`
/// lets unobserved model points switch off, and `initial_sigma2` (in the
/// normalized frame, target RMS radius = 1; try 0.1-0.3) starts the annealing
/// at the fragment's own scale. `merge_tolerance` merges refined starts that
/// converged to the same fit before the ambiguity diagnostics are computed.
#[pyfunction]
#[pyo3(signature = (source, target, modes, eigenvalues, *,
    rotation_count = 193, coarse_source_count = 400,
    coarse_target_count = 400, coarse_rank = 12, coarse_iterations = 8,
    coarse_screen_iterations = 8, coarse_survivor_count = 193,
    coarse_score_mode = "trajectory".to_string(),
    refine_count = 12, refine_source_count = None,
    refine_target_count = 1600, refine_iterations = 30,
    lambda_regularization = 0.1, outlier_weight = 0.05,
    identity_prior_probability = 0.2,
    landmark_indices = None, landmark_targets = None, landmark_weight = 0.0,
    landmark_sigma = None,
    refine_landmark_weight = 0.0, refine_landmark_sigma = None,
    with_scale = true,
    translation_anchor_count = 1, anchor_completeness_threshold = 0.9,
    scale_bounds = None, adaptive_mixing = None, merge_tolerance = 0.02,
    initial_sigma2 = None,
    seed = 0, parallel = true,
    single_precision = false))]
#[allow(clippy::too_many_arguments)]
fn pose_initialize(
    py: Python<'_>,
    source: PyArrayLike2<'_, f64, AllowTypeChange>,
    target: PyArrayLike2<'_, f64, AllowTypeChange>,
    modes: PyArrayLike2<'_, f64, AllowTypeChange>,
    eigenvalues: Vec<f64>,
    rotation_count: usize,
    coarse_source_count: usize,
    coarse_target_count: usize,
    coarse_rank: usize,
    coarse_iterations: usize,
    coarse_screen_iterations: usize,
    coarse_survivor_count: usize,
    coarse_score_mode: String,
    refine_count: usize,
    refine_source_count: Option<usize>,
    refine_target_count: usize,
    refine_iterations: usize,
    lambda_regularization: f64,
    outlier_weight: f64,
    identity_prior_probability: f64,
    landmark_indices: Option<Vec<usize>>,
    landmark_targets: Option<PyArrayLike2<'_, f64, AllowTypeChange>>,
    landmark_weight: f64,
    landmark_sigma: Option<f64>,
    refine_landmark_weight: f64,
    refine_landmark_sigma: Option<f64>,
    with_scale: bool,
    translation_anchor_count: usize,
    anchor_completeness_threshold: f64,
    scale_bounds: Option<(f64, f64)>,
    adaptive_mixing: Option<f64>,
    merge_tolerance: f64,
    initial_sigma2: Option<f64>,
    seed: u64,
    parallel: bool,
    single_precision: bool,
) -> PyResult<PoseInitialization> {
    let source = matrix_from(source.as_array());
    let target = matrix_from(target.as_array());
    let modes = matrix_from(modes.as_array());
    let landmarks: Vec<(usize, Vec<f64>)> = match (landmark_indices, landmark_targets.as_ref()) {
        (Some(indices), Some(points)) => {
            let pts = matrix_from(points.as_array());
            if indices.len() != pts.nrows() {
                return Err(PyValueError::new_err(
                    "landmark_indices and landmark_targets must have matching lengths",
                ));
            }
            indices
                .into_iter()
                .enumerate()
                .map(|(row, index)| (index, (0..pts.ncols()).map(|j| pts[(row, j)]).collect()))
                .collect()
        }
        (None, None) => Vec::new(),
        _ => {
            return Err(PyValueError::new_err(
                "landmark_indices and landmark_targets must be provided together",
            ));
        }
    };
    let coarse_score_mode = match coarse_score_mode.as_str() {
        "trajectory" => cpd::PoseScoreMode::Trajectory,
        "final" => cpd::PoseScoreMode::Final,
        other => {
            return Err(PyValueError::new_err(format!(
                "coarse_score_mode must be 'trajectory' or 'final', got {other:?}"
            )));
        }
    };
    let config = cpd::PoseMarginalizedConfig {
        rotation_count,
        coarse_source_count,
        coarse_target_count,
        coarse_rank,
        coarse_iterations,
        coarse_screen_iterations,
        coarse_survivor_count,
        coarse_score_mode,
        refine_count,
        refine_source_count,
        refine_target_count,
        refine_iterations,
        lambda_regularization,
        outlier_weight,
        identity_prior_probability,
        landmarks,
        landmark_weight,
        landmark_sigma,
        refine_landmark_weight,
        refine_landmark_sigma,
        with_scale,
        seed,
        parallel,
        single_precision,
        translation_anchor_count,
        anchor_completeness_threshold,
        scale_bounds,
        adaptive_mixing,
        merge_tolerance,
        initial_sigma2,
    };
    let result = py
        .detach(|| config.initialize(&source, &target, &modes, &eigenvalues))
        .map_err(error)?;
    Ok(PoseInitialization {
        coefficients: vector_to(py, &result.coefficients),
        rotation: matrix_to(py, &result.rotation),
        scale: result.scale,
        translation: vector_to(py, &result.translation),
        score: result.score,
        score_margin: result.score_margin,
        posterior_entropy: result.posterior_entropy,
        effective_hypotheses: result.effective_hypotheses,
        hypotheses_evaluated: result.hypotheses_evaluated,
        hypotheses_refined: result.hypotheses_refined,
        distinct_hypotheses: result.distinct_hypotheses,
        winner_support: result.winner_support,
        translation_anchors_used: result.translation_anchors_used,
    })
}

/// Dense Gaussian kernel `G[i, j] = exp(-||x_i - y_j||^2 / (2 beta^2))`.
#[pyfunction]
fn gaussian_kernel(
    py: Python<'_>,
    x: PyArrayLike2<'_, f64, AllowTypeChange>,
    y: PyArrayLike2<'_, f64, AllowTypeChange>,
    beta: f64,
) -> PyResult<Py<PyArray2<f64>>> {
    let x = matrix_from(x.as_array());
    let y = matrix_from(y.as_array());
    let kernel = py
        .detach(|| cpd::gaussian_kernel(&x, &y, beta))
        .map_err(error)?;
    Ok(matrix_to(py, &kernel))
}

/// Standard CPD `sigma2` initializer: the mean squared distance between
/// all pairs of points, divided by the dimensionality.
#[pyfunction]
fn initialize_sigma2(
    x: PyArrayLike2<'_, f64, AllowTypeChange>,
    y: PyArrayLike2<'_, f64, AllowTypeChange>,
) -> PyResult<f64> {
    cpd::initialize_sigma2(&matrix_from(x.as_array()), &matrix_from(y.as_array())).map_err(error)
}

/// Soft point-to-point correspondences from a fitted registration.
#[pyclass(frozen, module = "rustcpd")]
pub struct Correspondences {
    /// For each aligned source point, the index of the best-matching
    /// target point, shape `(M,)` (`int64`).
    #[pyo3(get)]
    pub matches: Py<PyArray1<i64>>,
    /// Posterior responsibility of each best match in `[0, 1]`, shape
    /// `(M,)`. Low values flag poorly explained source points.
    #[pyo3(get)]
    pub probability: Py<PyArray1<f64>>,
    /// Full `M×N` posterior matrix `P`, where `P[i, j]` is the
    /// responsibility of aligned source point `i` for target point `j`.
    /// Columns are normalized over sources (CPD convention).
    #[pyo3(get)]
    pub posterior: Py<PyArray2<f64>>,
}

/// Soft correspondences between a fixed `target` cloud and an
/// `aligned_source` cloud (typically a registration result's `points`)
/// under an isotropic Gaussian mixture with variance `sigma2`.
///
/// Exposes one CPD E-step: read off which target point each source point
/// maps to and how confidently. `sigma2` is usually the result's `sigma2`;
/// smaller values sharpen the matches.
#[pyfunction]
#[pyo3(signature = (target, aligned_source, sigma2, *, outlier_weight = 0.0))]
fn correspondences(
    py: Python<'_>,
    target: PyArrayLike2<'_, f64, AllowTypeChange>,
    aligned_source: PyArrayLike2<'_, f64, AllowTypeChange>,
    sigma2: f64,
    outlier_weight: f64,
) -> PyResult<Correspondences> {
    let target = matrix_from(target.as_array());
    let source = matrix_from(aligned_source.as_array());
    let result = py
        .detach(|| cpd::correspondences(&target, &source, sigma2, outlier_weight))
        .map_err(error)?;
    let matches: Vec<i64> = result.matches.iter().map(|&i| i as i64).collect();
    Ok(Correspondences {
        matches: PyArray1::from_slice(py, &matches).unbind(),
        probability: vector_to(py, &result.probability),
        posterior: matrix_to(py, &result.posterior),
    })
}

/// Build a shape posterior from explicit model-frame inputs (the testable
/// core behind `AtlasResult.posterior`). `residual` is `(M, D)` — observed
/// model-frame position minus the mean, meaningful where `weight > 0`;
/// `weight` is the per-point soft observation weight (`0` = unobserved);
/// `sigma_eff2` is the model-frame noise variance; `(rotation, scale,
/// translation)` maps the model frame to the target frame.
#[pyfunction]
#[pyo3(signature = (mean, modes, eigenvalues, residual, weight, sigma_eff2,
    rotation, scale, translation, *, prior_temperature = 1.0,
    estimate_discrepancy = true))]
#[allow(clippy::too_many_arguments)]
fn complete_shape(
    mean: PyArrayLike2<'_, f64, AllowTypeChange>,
    modes: PyArrayLike2<'_, f64, AllowTypeChange>,
    eigenvalues: Vec<f64>,
    residual: PyArrayLike2<'_, f64, AllowTypeChange>,
    weight: Vec<f64>,
    sigma_eff2: f64,
    rotation: PyArrayLike2<'_, f64, AllowTypeChange>,
    scale: f64,
    translation: Vec<f64>,
    prior_temperature: f64,
    estimate_discrepancy: bool,
) -> PyResult<ShapePosterior> {
    let mean = matrix_from(mean.as_array());
    let modes = matrix_from(modes.as_array());
    let residual = matrix_from(residual.as_array());
    let rotation = matrix_from(rotation.as_array());
    let posterior = cpd::complete_shape(
        &mean,
        &modes,
        &eigenvalues,
        &residual,
        &weight,
        sigma_eff2,
        &rotation,
        scale,
        &translation,
        prior_temperature,
        estimate_discrepancy,
    )
    .map_err(error)?;
    Ok(ShapePosterior { inner: posterior })
}

/// Draw completed shapes from a *mixture* of posteriors — e.g. one per
/// competing pose hypothesis — turning pose ambiguity into multi-modal
/// predictive uncertainty. `components` is a list of `(weight, posterior)`
/// pairs (weights need not sum to one). Returns a list of `(M, D)` arrays,
/// deterministic in `seed`.
#[pyfunction]
fn mixture_sample_shapes(
    py: Python<'_>,
    components: Vec<(f64, Py<ShapePosterior>)>,
    count: usize,
    seed: u64,
) -> PyResult<Vec<Py<PyArray2<f64>>>> {
    // Borrow each posterior's inner Rust object for the call.
    let borrows: Vec<PyRef<'_, ShapePosterior>> =
        components.iter().map(|(_, p)| p.borrow(py)).collect();
    let refs: Vec<(f64, &cpd::ShapePosterior)> = components
        .iter()
        .zip(&borrows)
        .map(|((w, _), b)| (*w, &b.inner))
        .collect();
    let shapes = py
        .detach(|| cpd::mixture_sample_shapes(&refs, count, seed))
        .map_err(error)?;
    Ok(shapes.iter().map(|s| matrix_to(py, s)).collect())
}

#[pymodule]
fn _core(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<RigidResult>()?;
    module.add_class::<AffineResult>()?;
    module.add_class::<DeformableResult>()?;
    module.add_class::<AtlasResult>()?;
    module.add_class::<PoseInitialization>()?;
    module.add_class::<Correspondences>()?;
    module.add_class::<ShapePosterior>()?;
    module.add_function(wrap_pyfunction!(register_rigid, module)?)?;
    module.add_function(wrap_pyfunction!(register_affine, module)?)?;
    module.add_function(wrap_pyfunction!(register_deformable, module)?)?;
    module.add_function(wrap_pyfunction!(register_atlas, module)?)?;
    module.add_function(wrap_pyfunction!(pose_initialize, module)?)?;
    module.add_function(wrap_pyfunction!(gaussian_kernel, module)?)?;
    module.add_function(wrap_pyfunction!(initialize_sigma2, module)?)?;
    module.add_function(wrap_pyfunction!(correspondences, module)?)?;
    module.add_function(wrap_pyfunction!(complete_shape, module)?)?;
    module.add_function(wrap_pyfunction!(mixture_sample_shapes, module)?)?;
    module.add("__version__", env!("CARGO_PKG_VERSION"))?;
    Ok(())
}
