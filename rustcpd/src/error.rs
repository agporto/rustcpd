use thiserror::Error;

/// Failure modes of registration construction and execution.
#[derive(Debug, Error, PartialEq)]
#[allow(missing_docs)] // the #[error] strings document each variant
pub enum Error {
    #[error("point clouds must be non-empty")]
    EmptyPointCloud,
    #[error("point clouds must have the same dimensionality")]
    DimensionMismatch,
    #[error("all coordinates must be finite")]
    NonFiniteInput,
    #[error("{0} must be finite and positive")]
    PositiveParameter(&'static str),
    #[error("outlier weight must be in [0, 1)")]
    InvalidOutlierWeight,
    #[error("rigid registration supports only 2D or 3D points")]
    UnsupportedRigidDimension,
    #[error("matrix dimensions are inconsistent: {0}")]
    InvalidShape(&'static str),
    #[error("constraint index is out of bounds")]
    ConstraintOutOfBounds,
    #[error("linear system is singular")]
    SingularSystem,
}

/// Crate-wide result alias.
pub type Result<T> = std::result::Result<T, Error>;
