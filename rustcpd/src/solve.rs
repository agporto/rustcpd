//! Thin bridge to `faer` for the large `M×M` factorizations of the
//! deformable path, with explicit parallelism control so
//! [`crate::EmConfig::parallel`] governs every thread the crate spawns.

use faer::Accum;
use faer::Par;
use faer::dyn_stack::{MemBuffer, MemStack};
use faer::linalg::evd::{self, ComputeEigenvectors};
use faer::linalg::lu::partial_pivoting::{factor, solve};
use faer::linalg::matmul::matmul;
use faer::mat::{MatMut, MatRef};
use nalgebra::DMatrix;

fn par(parallel: bool) -> Par {
    if parallel { Par::rayon(0) } else { Par::Seq }
}

/// Write `dst = opᵃ(a) · opᵇ(b)` using faer's matmul, where `opᵃ`/`opᵇ`
/// optionally transpose their operand. `dst` must be preallocated with the
/// result dimensions; no allocation happens here, so callers can reuse a
/// scratch buffer across iterations. faer's SIMD GEMM is used in place of
/// nalgebra's for the large `M×rank` products of the low-rank M-step.
pub(crate) fn matmul_into(
    dst: &mut DMatrix<f64>,
    a: &DMatrix<f64>,
    a_transposed: bool,
    b: &DMatrix<f64>,
    b_transposed: bool,
    parallel: bool,
) {
    let (m, n) = (dst.nrows(), dst.ncols());
    let a_ref = MatRef::from_column_major_slice(a.as_slice(), a.nrows(), a.ncols());
    let b_ref = MatRef::from_column_major_slice(b.as_slice(), b.nrows(), b.ncols());
    let a_op = if a_transposed {
        a_ref.transpose()
    } else {
        a_ref
    };
    let b_op = if b_transposed {
        b_ref.transpose()
    } else {
        b_ref
    };
    let dst_mut = MatMut::from_column_major_slice_mut(dst.as_mut_slice(), m, n);
    matmul(dst_mut, Accum::Replace, a_op, b_op, 1.0, par(parallel));
}

/// Solve `a · x = rhs` in place for a general square `a` through faer's
/// partial-pivoting LU. Returns `None` when the system is singular enough
/// to produce non-finite values.
pub(crate) fn lu_solve(a: DMatrix<f64>, rhs: DMatrix<f64>, parallel: bool) -> Option<DMatrix<f64>> {
    let (m, k) = (a.nrows(), rhs.ncols());
    let par = par(parallel);
    let mut lu = a;
    let mut perm = vec![0usize; m];
    let mut perm_inv = vec![0usize; m];
    let mut buffer = MemBuffer::new(factor::lu_in_place_scratch::<usize, f64>(
        m,
        m,
        par,
        Default::default(),
    ));
    let (_, perm_ref) = factor::lu_in_place(
        MatMut::from_column_major_slice_mut(lu.as_mut_slice(), m, m),
        &mut perm,
        &mut perm_inv,
        par,
        MemStack::new(&mut buffer),
        Default::default(),
    );
    let mut solution = rhs;
    let lu_ref = MatRef::from_column_major_slice(lu.as_slice(), m, m);
    let mut solve_buffer = MemBuffer::new(solve::solve_in_place_scratch::<usize, f64>(m, k, par));
    solve::solve_in_place(
        lu_ref,
        lu_ref,
        perm_ref,
        MatMut::from_column_major_slice_mut(solution.as_mut_slice(), m, k),
        par,
        MemStack::new(&mut solve_buffer),
    );
    solution
        .iter()
        .all(|value| value.is_finite())
        .then_some(solution)
}

/// Full symmetric eigendecomposition of `g`, returning the eigenvector
/// matrix and eigenvalues in faer's nondecreasing order.
pub(crate) fn symmetric_eigen(
    g: &DMatrix<f64>,
    parallel: bool,
) -> Option<(DMatrix<f64>, Vec<f64>)> {
    let m = g.nrows();
    let par = par(parallel);
    let mut s = faer::diag::Diag::<f64>::zeros(m);
    let mut u = faer::Mat::<f64>::zeros(m, m);
    let mut buffer = MemBuffer::new(evd::self_adjoint_evd_scratch::<f64>(
        m,
        ComputeEigenvectors::Yes,
        par,
        Default::default(),
    ));
    evd::self_adjoint_evd(
        MatRef::from_column_major_slice(g.as_slice(), m, m),
        s.as_mut(),
        Some(u.as_mut()),
        par,
        MemStack::new(&mut buffer),
        Default::default(),
    )
    .ok()?;
    let vectors = DMatrix::from_fn(m, m, |i, j| u[(i, j)]);
    let values = (0..m).map(|i| s[i]).collect();
    Some((vectors, values))
}
