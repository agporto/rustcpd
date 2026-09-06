use kiddo::{KdTree, SquaredEuclidean};
use nalgebra::DMatrix;

use crate::em::validate_clouds;
use crate::{Error, Result};

/// A resumable atlas fit, expressed entirely in the original coordinate frames.
///
/// Pass to [`crate::AtlasConfig::initial_state`] to continue without restarting
/// the correspondence variance or mixture. Model/EM settings (scale constraints,
/// regularization, outlier weight and adaptive-mixing strength) remain the
/// responsibility of the receiving config. Keep them unchanged for a true
/// continuation. The receiving model may use more modes or a different sampling
/// of the same mean shape; additional coefficients start at zero.
#[derive(Clone, Debug)]
pub struct AtlasState {
    /// Shape coefficients, in the same basis as the receiving model.
    pub coefficients: Vec<f64>,
    /// Direct row-vector rotation (`y * rotation`).
    pub rotation: DMatrix<f64>,
    /// Similarity scale.
    pub scale: f64,
    /// Translation in original target coordinates.
    pub translation: Vec<f64>,
    /// Gaussian variance in squared original target-coordinate units.
    /// Converted automatically when the next registration normalizes its data.
    pub sigma2: f64,
    /// Uniform background density in original target-coordinate units.
    /// Preserved when normalization or target sample count changes, so the
    /// same outlier weight continues to describe the same observation model.
    pub outlier_density: f64,
    /// Per-source mixture probabilities, or `None` for uniform mixing.
    pub mixing_weights: Option<Vec<f64>>,
    /// Mean-shape coordinates on which `mixing_weights` were fitted, in the
    /// original model frame. Required when weights are present. Keeping the
    /// coordinates avoids silently misassigning weights to reordered vertices.
    pub mixing_reference: Option<DMatrix<f64>>,
}

impl AtlasState {
    pub(crate) fn validate(&self, dimensions: usize, rank: usize) -> Result<()> {
        if self.rotation.shape() != (dimensions, dimensions)
            || self.translation.len() != dimensions
            || self.coefficients.len() > rank
        {
            return Err(Error::InvalidShape("initial_state"));
        }
        if !self
            .rotation
            .iter()
            .chain(&self.translation)
            .chain(&self.coefficients)
            .all(|v| v.is_finite())
        {
            return Err(Error::NonFiniteInput);
        }
        if !self.scale.is_finite()
            || self.scale <= 0.0
            || !self.sigma2.is_finite()
            || self.sigma2 <= 0.0
            || !self.outlier_density.is_finite()
            || self.outlier_density <= 0.0
        {
            return Err(Error::PositiveParameter("initial_state scale/sigma2"));
        }
        if let Some(weights) = &self.mixing_weights {
            let reference = self
                .mixing_reference
                .as_ref()
                .ok_or(Error::InvalidShape("mixing_reference"))?;
            validate_clouds(reference, reference)?;
            if reference.ncols() != dimensions || weights.len() != reference.nrows() {
                return Err(Error::InvalidShape("mixing_weights"));
            }
            validate_mixing_weights(weights)?;
        }
        Ok(())
    }

    /// Transfer the mixture to another sampling of the same model mean.
    ///
    /// Uses nearest-neighbor interpolation of relative occupancies in the
    /// undeformed model frame, then normalizes on the destination vertices.
    /// This preserves uniform mixing and exact vertex permutations. It assumes
    /// comparable surface sampling; it is not a surface-area quadrature rule.
    pub fn mixing_weights_on(&self, mean: &DMatrix<f64>) -> Result<Option<Vec<f64>>> {
        self.validate(mean.ncols(), self.coefficients.len())?;
        match (&self.mixing_weights, &self.mixing_reference) {
            (Some(weights), Some(reference)) => {
                Ok(Some(MixingWeightMap::new(reference, mean)?.apply(weights)))
            }
            _ => Ok(None),
        }
    }
}

pub(crate) fn validate_mixing_weights(weights: &[f64]) -> Result<()> {
    if weights.is_empty()
        || weights.iter().any(|w| !w.is_finite() || *w < 0.0)
        || (weights.iter().sum::<f64>() - 1.0).abs() > 1e-6
    {
        return Err(Error::PositiveParameter(
            "mixing_weights (probabilities summing to one)",
        ));
    }
    Ok(())
}

/// A geometry-only map, reusable across all hypotheses at a stage boundary.
pub(crate) struct MixingWeightMap {
    indices: Vec<usize>,
    identity: bool,
    permutation: bool,
}

impl MixingWeightMap {
    pub(crate) fn new(reference: &DMatrix<f64>, destination: &DMatrix<f64>) -> Result<Self> {
        validate_clouds(reference, destination)?;
        let identity = reference == destination;
        let indices = if identity {
            (0..reference.nrows()).collect()
        } else {
            match reference.ncols() {
                2 => nearest_indices::<2>(reference, destination),
                3 => nearest_indices::<3>(reference, destination),
                _ => (0..destination.nrows())
                    .map(|i| {
                        (0..reference.nrows())
                            .min_by(|&a, &b| {
                                let distance = |j| {
                                    (0..reference.ncols())
                                        .map(|d| (destination[(i, d)] - reference[(j, d)]).powi(2))
                                        .sum::<f64>()
                                };
                                distance(a).total_cmp(&distance(b))
                            })
                            .unwrap()
                    })
                    .collect(),
            }
        };
        let mut seen = vec![false; reference.nrows()];
        let permutation = indices.len() == reference.nrows()
            && indices.iter().all(|&index| {
                let new = !seen[index];
                seen[index] = true;
                new
            });
        Ok(Self {
            indices,
            identity,
            permutation,
        })
    }

    pub(crate) fn apply(&self, weights: &[f64]) -> Vec<f64> {
        if self.identity {
            return weights.to_vec();
        }
        if self.permutation {
            return self.indices.iter().map(|&i| weights[i]).collect();
        }
        // Match the strictly positive relative-factor floor used by the E-step.
        // Even a destination consisting only of unsupported vertices remains
        // a valid (uniform, very weakly supported) mixture after normalization.
        let floor = 1e-9 / weights.len() as f64;
        let mut mapped: Vec<f64> = self
            .indices
            .iter()
            .map(|&i| weights[i].max(floor))
            .collect();
        let total: f64 = mapped.iter().sum();
        for weight in &mut mapped {
            *weight /= total;
        }
        mapped
    }
}

fn nearest_indices<const D: usize>(
    reference: &DMatrix<f64>,
    destination: &DMatrix<f64>,
) -> Vec<usize> {
    let mut tree: KdTree<f64, D> = KdTree::new();
    for i in 0..reference.nrows() {
        tree.add(&std::array::from_fn(|j| reference[(i, j)]), i as u64);
    }
    (0..destination.nrows())
        .map(|i| {
            tree.nearest_one::<SquaredEuclidean>(&std::array::from_fn(|j| destination[(i, j)]))
                .item as usize
        })
        .collect()
}

pub(crate) fn mixing_factors(pi: Option<&[f64]>) -> Option<Vec<f64>> {
    pi.map(|values| {
        let m = values.len() as f64;
        values.iter().map(|&p| (p * m).max(1e-9)).collect()
    })
}

/// Express a physical background density using the E-step's `w / N`
/// convention. The common rescaling of both mixture terms cancels from the
/// responsibilities. This also makes continuation across normalized/raw fits
/// invariant to the chosen working frame.
pub(crate) fn effective_outlier_weight(
    weight: f64,
    density: f64,
    targets: usize,
    working_scale: f64,
    dimensions: usize,
) -> f64 {
    if weight == 0.0 {
        return 0.0;
    }
    let log_odds = weight.ln() - (-weight).ln_1p()
        + density.ln()
        + (targets as f64).ln()
        + dimensions as f64 * working_scale.ln();
    let value = if log_odds >= 0.0 {
        1.0 / (1.0 + (-log_odds).exp())
    } else {
        let odds = log_odds.exp();
        odds / (1.0 + odds)
    };
    value.min(1.0 - f64::EPSILON)
}
