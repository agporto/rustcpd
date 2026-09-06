# Migrating from rustcpd 3.1 to 4.0

Version 4.0 preserves fitted state through fragment registration and makes
completion use the fitted observation model. The Rust crate and Python package
share the major version because the Rust completion options include a source
incompatibility. These changes are not a promise of numerically identical
completion results to 3.1.

## Rust: explicit outlier weights are now optional overrides

`PosteriorOptions::outlier_weight` changed from `f64` to `Option<f64>`.
Use `Some(w)` wherever existing code explicitly assigned `w`:

```rust
use rustcpd::PosteriorOptions; // requires the `completion` feature

let options = PosteriorOptions {
    outlier_weight: Some(0.05),
    ..PosteriorOptions::default()
};
```

`None` (the new default) inherits the fitted observation model;
`Some(0.0)` explicitly requests clean soft assignments. The additional fields
in atlas/pose configuration and result types can also require updates to
exhaustive struct literals and destructuring. Prefer configuration literals
with `..Default::default()` and result patterns with `..` where appropriate.

## Python: distinguish fitted assignments from clean assignments

Numeric `outlier_weight` arguments still work. Omitting the argument, or passing
`None`, now inherits the registration's outlier weight instead of assuming zero:

```python
# Use the fitted observation model (recommended for consistent completion).
posterior = fit.posterior(target, mean, modes, eigenvalues)

# Deliberately omit the uniform outlier-background component.
clean_posterior = fit.posterior(
    target, mean, modes, eigenvalues, outlier_weight=0.0
)
```

The second call restores only the clean-assignment choice. It does not undo the
other corrections: completion now retains fitted variance, adaptive mixture
weights, and the physical background-density convention across sampling and
normalization changes. Do not treat it as a general 3.1 compatibility switch.

## Prior temperature now means a prior precision multiplier

The legacy name `prior_temperature` is retained, but its documented meaning is
now implemented consistently: for shape eigenvalue `eigenvalues[j]`, the prior
precision contribution is `prior_temperature / eigenvalues[j]`. Larger values
therefore impose stronger shrinkage toward the mean shape. The default `1.0`
is unchanged.

To use the same prior strength as atlas registration, set
`prior_temperature` to the registration's `lambda_regularization`. Agreement
of posterior means additionally requires matching the conditioning information,
visibility filtering, pose, and observation model; matching this scalar alone
is not sufficient.

Version 3.1 used the reciprocal convention for this non-default parameter.
For a positive old value `t`, `1.0 / t` reproduces the old prior precision
contribution, not necessarily the full old completion. Revalidate results
rather than mechanically changing all temperatures.

## Continue a fit with state, not just a transform

Passing only rotation, translation, scale, and coefficients restarts variance
and mixture initialization. Use `initial_state=init.state` or `fit.state` to
carry fitted pose, shape, variance, mixture information, and background density
into the next atlas stage.

State transfers fitted quantities, not the entire configuration. Keep the
receiving `lambda_regularization`, `outlier_weight`, scale constraints,
`adaptive_mixing`, landmark settings, and intended EM options explicit. Do not
combine `initial_state` with individual pose/shape/variance initializers.

This example assumes the mean and target use comparable physical units, fixes
scale, and uses illustrative outlier/mixing values rather than universal defaults:

```python
import rustcpd


def complete_fragment(target, mean, modes, eigenvalues):
    fit_options = dict(
        lambda_regularization=1.0,
        outlier_weight=0.05,
        with_scale=False,
        adaptive_mixing=0.05,
    )
    # Note the argument order: pose_initialize takes source, then target.
    init = rustcpd.pose_initialize(
        mean, target, modes, eigenvalues,
        translation_anchor_count=8,
        **fit_options,
    )
    # register_atlas instead takes target, then model mean.
    fit = rustcpd.register_atlas(
        target, mean, modes, eigenvalues,
        initial_state=init.state,
        normalize=True,
        max_iterations=100,
        **fit_options,
    )
    posterior = fit.posterior(
        target, mean, modes, eigenvalues,
        outlier_weight=None,
        prior_temperature=fit_options["lambda_regularization"],
    )
    return fit, posterior
```

For data requiring scale optimization, use biologically justified
`scale_bounds=(minimum, maximum)` consistently in both stages instead of fixing
scale. Do not allow an unconstrained whole model to shrink into a small fragment.
Landmark-guided workflows must also pass the corresponding landmark information
and noise/weight settings to the receiving stage; these are not inferred from
`AtlasState`. In the pose search, refinement landmark settings use the
`refine_landmark_sigma` / `refine_landmark_weight` names.

State variance is stored in original target units and converted for normalized
registration. Mixture transfer handles reordered or differently sampled model
vertices. Shape coefficients still require the same model basis and mode order;
state transfer is not a conversion between unrelated statistical shape models.

## Diagnostics and reproducibility

Duplicate fitted solutions are merged before screening and refinement pruning.
`score_margin`, `posterior_entropy`, and `effective_hypotheses` describe distinct
solutions rather than counting repeated starts in the same basin as ambiguity.
`merge_tolerance=0.0` disables merging, even for exactly equal fits. Existing
thresholds or calibrators based on these diagnostics should be reassessed.

The pose search's `initial_sigma2` is an override for the first coarse pass in
normalized coordinates. Later stages retain the fitted variance rather than
resetting it. It is not an override to force one variance through every stage.

For a reproducible upgrade, retain the 3.1 environment and saved outputs, run
representative complete and partial specimens under 4.0, and compare fitted
pose, coefficients, completed shape, and uncertainty. Record the package version,
model/basis, sampling, coordinate units, and explicit configuration alongside
new outputs. Numerical regression tests do not replace specimen-level validation
or calibration for a particular biological dataset.
