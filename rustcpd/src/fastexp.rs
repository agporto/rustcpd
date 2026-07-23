//! Vectorizable elementwise `exp` for non-positive arguments.
//!
//! The Gaussian E-step and kernel construction spend most of their time in
//! `exp`. This module evaluates the Cephes rational approximation (the same
//! one used by many libm implementations; ≤ 2 ulp over the relevant range)
//! in a form LLVM auto-vectorizes. On x86-64 the vector path requires
//! AVX2+FMA and is selected at runtime with a scalar-libm fallback, so a
//! given machine always produces identical results run-over-run.
//!
//! Arguments are exponents of Gaussian weights, hence always ≤ 0. Inputs
//! below −706 return exactly 0 instead of a subnormal; those weights are
//! smaller than 1e−300 and contribute nothing to any accumulated statistic.

/// Cephes `exp` coefficients (Moshier, public domain).
const C1: f64 = 6.931_457_519_531_25e-1;
const C2: f64 = 1.428_606_820_309_417_2e-6;
const P0: f64 = 1.261_771_930_748_105_9e-4;
const P1: f64 = 3.029_944_077_074_419_6e-2;
const P2: f64 = 1.0; // Cephes P2 rounds to exactly 1.0 in f64.
const Q0: f64 = 3.001_985_051_386_644_6e-6;
const Q1: f64 = 2.524_483_403_496_841e-3;
const Q2: f64 = 2.272_655_482_081_550_3e-1;
const Q3: f64 = 2.0;

#[inline(always)]
fn exp_body(values: &mut [f64]) {
    for value in values {
        let x = *value;
        let n = f64::mul_add(std::f64::consts::LOG2_E, x, 0.5).floor();
        let r = f64::mul_add(-n, C2, f64::mul_add(-n, C1, x));
        let rr = r * r;
        let p = r * f64::mul_add(f64::mul_add(P0, rr, P1), rr, P2);
        let q = f64::mul_add(f64::mul_add(f64::mul_add(Q0, rr, Q1), rr, Q2), rr, Q3);
        let e = 1.0 + 2.0 * p / (q - p);
        let scale = f64::from_bits(((n as i64 + 1023) as u64) << 52);
        *value = if x < -706.0 { 0.0 } else { e * scale };
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
fn exp_avx2(values: &mut [f64]) {
    exp_body(values);
}

/// Replace every element `x ≤ 0` of `values` with `exp(x)`.
pub(crate) fn exp_non_positive(values: &mut [f64]) {
    #[cfg(target_arch = "x86_64")]
    {
        if std::arch::is_x86_feature_detected!("avx2") && std::arch::is_x86_feature_detected!("fma")
        {
            // SAFETY: the required target features were just detected.
            return unsafe { exp_avx2(values) };
        }
        for value in values {
            *value = value.exp();
        }
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        // FMA is baseline on aarch64 and most other modern targets, so the
        // polynomial evaluates efficiently without a runtime check.
        exp_body(values);
    }
}

/// Cephes single-precision `exp` coefficients (Moshier, public domain).
const F32_C1: f32 = 0.693_359_4;
const F32_C2: f32 = -2.121_944_4e-4;
const F32_P0: f32 = 1.987_569_1e-4;
const F32_P1: f32 = 1.398_2e-3;
const F32_P2: f32 = 8.333_452e-3;
const F32_P3: f32 = 4.166_579_6e-2;
const F32_P4: f32 = 1.666_666_6e-1;
const F32_P5: f32 = 5.000_000_3e-1;

#[inline(always)]
fn exp_body_f32(values: &mut [f32]) {
    for value in values {
        let x = *value;
        let n = f32::mul_add(std::f32::consts::LOG2_E, x, 0.5).floor();
        let r = f32::mul_add(-n, F32_C2, f32::mul_add(-n, F32_C1, x));
        let z = r * r;
        let mut p = F32_P0;
        p = f32::mul_add(p, r, F32_P1);
        p = f32::mul_add(p, r, F32_P2);
        p = f32::mul_add(p, r, F32_P3);
        p = f32::mul_add(p, r, F32_P4);
        p = f32::mul_add(p, r, F32_P5);
        let e = f32::mul_add(z, p, r) + 1.0;
        let scale = f32::from_bits(((n as i32 + 127) as u32) << 23);
        *value = if x < -87.0 { 0.0 } else { e * scale };
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
fn exp_avx2_f32(values: &mut [f32]) {
    exp_body_f32(values);
}

/// Replace every element `x ≤ 0` of `values` with `exp(x)` in single
/// precision (relative error ≤ ~2e-7; inputs below −87 return exactly 0).
pub(crate) fn exp_non_positive_f32(values: &mut [f32]) {
    #[cfg(target_arch = "x86_64")]
    {
        if std::arch::is_x86_feature_detected!("avx2") && std::arch::is_x86_feature_detected!("fma")
        {
            // SAFETY: the required target features were just detected.
            return unsafe { exp_avx2_f32(values) };
        }
        for value in values {
            *value = value.exp();
        }
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        exp_body_f32(values);
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn f32_matches_libm_to_a_few_ulp() {
        let mut worst = 0.0_f32;
        let mut x = -100.0_f32;
        while x <= 0.0 {
            let mut buffer = [x];
            super::exp_non_positive_f32(&mut buffer);
            let reference = x.exp();
            if reference > 1e-30 {
                worst = worst.max(((buffer[0] - reference) / reference).abs());
            }
            x += 1.7e-4;
        }
        assert!(worst <= 4.0 * f32::EPSILON, "worst relative error {worst}");
        let mut zero = [0.0_f32];
        super::exp_non_positive_f32(&mut zero);
        assert_eq!(zero[0], 1.0);
    }

    #[test]
    fn matches_libm_to_two_ulp() {
        let mut worst = 0.0_f64;
        let mut x = -750.0_f64;
        while x <= 0.0 {
            let mut buffer = [x];
            super::exp_non_positive(&mut buffer);
            let reference = x.exp();
            if reference > 1e-300 {
                worst = worst.max(((buffer[0] - reference) / reference).abs());
            } else {
                assert!(
                    buffer[0] == 0.0 || (buffer[0] - reference).abs() < 1e-300,
                    "x={x}"
                );
            }
            x += 1.03e-3;
        }
        assert!(worst <= 2.0 * f64::EPSILON, "worst relative error {worst}");
        let mut zero = [0.0];
        super::exp_non_positive(&mut zero);
        assert_eq!(zero[0], 1.0);
    }
}
