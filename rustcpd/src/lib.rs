//! Coherent Point Drift registration in Rust.
//!
//! Rigid, affine, deformable, constrained deformable, and
//! statistical-shape-model registration algorithms.
//!
//! # Determinism and parallelism
//!
//! The numerical core is deterministic. Parallel execution (the default,
//! [`EmConfig::parallel`] = `true`) partitions work into fixed blocks that
//! are combined in a fixed order, so results do not depend on thread
//! scheduling or thread count and are identical to serial execution. Set
//! [`EmConfig::parallel`] = `false` to avoid spawning worker threads
//! entirely.
//!
//! # Convergence criteria
//!
//! Each algorithm mirrors the convergence test of its reference
//! implementation: rigid uses the absolute change in the EM objective,
//! affine the relative change in the EM objective, deformable the absolute
//! change in `sigma2`, and atlas the maximum of the relative `sigma2`
//! change and the mean absolute coefficient change.

#![warn(missing_docs)]

mod affine;
mod atlas;
#[cfg(feature = "completion")]
mod completion;
mod deformable;
mod em;
mod error;
mod fastexp;
mod pose;
mod rigid;
mod solve;

pub use affine::{AffineConfig, AffineRegistration, AffineResult};
pub use atlas::{AtlasConfig, AtlasRegistration, AtlasResult};
#[cfg(feature = "completion")]
pub use completion::{PosteriorOptions, ShapePosterior, complete_shape, mixture_sample_shapes};
pub use deformable::{
    Constraint, DeformableConfig, DeformableRegistration, DeformableResult, LowRankMethod,
    apply_deformation, low_rank_spectrum,
};
pub use em::{
    Correspondences, EmConfig, IterationState, PosteriorStats, correspondences, gaussian_kernel,
    initialize_sigma2,
};
pub use error::{Error, Result};
pub use nalgebra::DMatrix;
pub use pose::{
    FRAGMENT_INITIAL_SIGMA2, PoseMarginalizedConfig, PoseMarginalizedInitialization, PoseScoreMode,
};
pub use rigid::{RigidConfig, RigidRegistration, RigidResult};
