"""Shape completion and per-point uncertainty from a partial observation.

Synthesizes a shape from a small linear model, hides part of it, registers
the atlas to the partial cloud, then completes the missing region and reads
off a per-point confidence map. Run: python examples/completion.py
"""

import numpy as np

import rustcpd as cpd


def model(m, k):
    z = (np.arange(m) + 1.0)[:, None]
    mean = np.column_stack(
        (np.sin(z[:, 0] * 0.3) * 1.5, np.cos(z[:, 0] * 0.2), np.sin(z[:, 0] * 0.13))
    )
    rows = np.arange(m * 3)[:, None]
    modes = np.sin((rows + 1 + 5 * np.arange(k)) * 0.17) + 0.3 * np.cos(
        (rows + 2 * np.arange(k) + 1) * 0.09
    )
    modes, _ = np.linalg.qr(modes)  # orthonormal columns
    eigenvalues = np.array([0.6 / (c + 1) for c in range(k)])
    return mean, modes, eigenvalues


def main():
    m, k = 80, 5
    mean, modes, eigenvalues = model(m, k)
    truth = np.array([0.4, -0.3, 0.2, -0.12, 0.08])

    # A true shape, posed into the target frame.
    deformed = mean + (modes @ truth).reshape(m, 3)
    angle = 0.25
    r = np.array(
        [[np.cos(angle), -np.sin(angle), 0.0],
         [np.sin(angle), np.cos(angle), 0.0], [0.0, 0.0, 1.0]]
    )
    full = 1.15 * (deformed @ r) + np.array([0.3, -0.2, 0.1])

    # Observe only the first 55 of 80 points (a partial scan).
    observed = 55
    partial = full[:observed]

    fit = cpd.register_atlas(
        partial, mean, modes, eigenvalues,
        lambda_regularization=1.0, optimize_similarity=True,
        tolerance=1e-8, max_iterations=200,
    )

    # Complete the shape and get the confidence map.
    post = fit.posterior(partial, mean, modes, eigenvalues, completeness=observed / m)
    completed = post.predict()               # (M, 3), target frame
    variance = post.predictive_variance()    # (M,), per-point total variance

    held = slice(observed, m)
    completion_rms = np.sqrt(np.mean((completed[held] - full[held]) ** 2))
    posed_mean = fit.scale * (mean @ fit.rotation) + fit.translation
    baseline_rms = np.sqrt(np.mean((posed_mean[held] - full[held]) ** 2))
    print(f"held-out completion rms = {completion_rms:.3e} "
          f"(posed-mean baseline {baseline_rms:.3e})")
    print(f"mean predictive sd: observed {np.sqrt(variance[:observed].mean()):.3e}, "
          f"missing {np.sqrt(variance[observed:].mean()):.3e}")

    # An ensemble of plausible completions (for bands / downstream stats).
    ensemble = post.sample_shapes(200, seed=0)
    spread = np.std([s[held] for s in ensemble], axis=0).mean()
    print(f"ensemble of {len(ensemble)} completions; mean held-out spread {spread:.3e}")


if __name__ == "__main__":
    main()
