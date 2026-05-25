"""Chapter 2 — QR-based Lyapunov-spectrum computation.

Implements the Benettin et al. (1980) QR algorithm for computing the
Lyapunov spectrum of a discrete time-varying linear system. Applies it to
the ring-of-oscillators system from Chapter 1, §1.4, in two regimes:

* **Damped** ($c = 0.2$): all Lyapunov exponents should be strictly
  negative — every direction in state space contracts.
* **Undamped** ($c = 0$, energy-preserving): all exponents should be at
  zero (within numerical noise) — the system is marginally stable.

For the autonomous (constant Jacobian) case used here, the Lyapunov
exponents reduce to the real parts of the eigenvalues of the *discrete-time*
Jacobian $e^{\\statemat \\Delta t}$, divided by $\\Delta t$. We verify this
identity numerically as a sanity check.

Output
------
``public/figures/ch02/lyapunov_spectrum.png`` (referenced from §2.3).

References
----------
Benettin, G., Galgani, L., Giorgilli, A., Strelcyn, J.-M. (1980).
*Lyapunov characteristic exponents for smooth dynamical systems and for
Hamiltonian systems; a method for computing all of them*. Meccanica 15(1).
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import expm

from companions._shared.plot_utils import (
    SSM_COLORS,
    apply_style,
    create_tufte_figure,
    save_figure,
    set_tufte_labels,
    set_tufte_title,
)

# Reuse the ring-Jacobian builder from Ch 1.
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from companions.ch01.jax.coupled_oscillators import build_ring_state_matrix  # noqa: E402

_REPO_ROOT = Path(__file__).resolve().parents[3]
_OUTPUT_PATH = _REPO_ROOT / "public" / "figures" / "ch02" / "lyapunov_spectrum"


def qr_lyapunov(jacobians: np.ndarray, n_steps: int) -> np.ndarray:
    """Compute the Lyapunov spectrum via the QR re-orthonormalization algorithm.

    For an autonomous system with constant per-step Jacobian
    ``J = jacobians[0]``, this is the autonomous Lyapunov spectrum and
    should agree with $\\log|\\text{eigvals}(J)|$ sorted by magnitude.

    Parameters
    ----------
    jacobians : ndarray of shape (T, N, N)
        Sequence of per-step Jacobian matrices. If T < n_steps, the sequence
        is reused cyclically.
    n_steps : int
        Total number of iterations.

    Returns
    -------
    ndarray of shape (N,)
        Sorted Lyapunov exponents in descending order.
    """
    if jacobians.ndim != 3 or jacobians.shape[1] != jacobians.shape[2]:
        raise ValueError(
            f"jacobians must have shape (T, N, N), got {jacobians.shape}"
        )
    if n_steps < 1:
        raise ValueError(f"n_steps must be positive, got {n_steps}")

    T, N, _ = jacobians.shape
    Q = np.eye(N)
    log_diag_sum = np.zeros(N)

    for t in range(n_steps):
        J_t = jacobians[t % T]
        M = J_t @ Q
        Q, R = np.linalg.qr(M)
        # Force the diagonal of R to be positive (sign convention; doesn't change Lyapunov values).
        signs = np.sign(np.diag(R))
        signs[signs == 0] = 1.0
        Q = Q * signs[np.newaxis, :]
        R = (signs[:, np.newaxis]) * R
        log_diag_sum += np.log(np.abs(np.diag(R)) + 1e-300)

    lyapunov = np.sort(log_diag_sum / n_steps)[::-1]
    return lyapunov


def autonomous_lyapunov_reference(A: np.ndarray, dt: float) -> np.ndarray:
    """Closed-form Lyapunov spectrum for the autonomous system $\\dot x = A x$.

    Discretized at step $dt$, the per-step Jacobian is $e^{A \\, dt}$ and
    its eigenvalues are $e^{\\lambda_i \\, dt}$. The Lyapunov exponents are
    $\\Re(\\lambda_i)$, sorted descending.
    """
    eigs = np.linalg.eigvals(A)
    return np.sort(eigs.real)[::-1]


def make_figure() -> plt.Figure:
    """Build the Lyapunov-spectrum comparison figure (damped vs undamped ring)."""
    apply_style()
    n = 8
    dt = 0.05
    n_steps = 2000

    A_damped = build_ring_state_matrix(n=n, k=4.0, c=0.2, kappa=1.0)
    A_undamped = build_ring_state_matrix(n=n, k=4.0, c=0.0, kappa=1.0)

    # Discrete-time Jacobian (one for all steps; autonomous case).
    J_damped = expm(A_damped * dt)
    J_undamped = expm(A_undamped * dt)

    spec_damped = qr_lyapunov(J_damped[np.newaxis, ...], n_steps) / dt
    spec_undamped = qr_lyapunov(J_undamped[np.newaxis, ...], n_steps) / dt
    ref_damped = autonomous_lyapunov_reference(A_damped, dt)
    ref_undamped = autonomous_lyapunov_reference(A_undamped, dt)

    fig, axes = create_tufte_figure(nrows=1, ncols=2, figsize=(11.0, 4.5))
    ax_d, ax_u = axes  # type: ignore[misc]

    idx = np.arange(1, 2 * n + 1)
    ax_d.scatter(idx, spec_damped, s=50, color=SSM_COLORS["accent"],
                 edgecolors="white", linewidths=0.8, zorder=3, label="QR algorithm")
    ax_d.scatter(idx, ref_damped, s=20, color=SSM_COLORS["alert"],
                 marker="x", zorder=4, label="reference $\\Re(\\lambda_i)$")
    ax_d.axhline(0.0, color=SSM_COLORS["baseline"], linewidth=0.8, linestyle="--")
    set_tufte_title(ax_d, "Damped ring (c=0.2): all $\\lambda_i < 0$")
    set_tufte_labels(ax_d, xlabel="index $i$", ylabel="Lyapunov exponent $\\lambda_i$")
    ax_d.legend(loc="lower left", frameon=False, fontsize=9)

    ax_u.scatter(idx, spec_undamped, s=50, color=SSM_COLORS["accent"],
                 edgecolors="white", linewidths=0.8, zorder=3, label="QR algorithm")
    ax_u.scatter(idx, ref_undamped, s=20, color=SSM_COLORS["alert"],
                 marker="x", zorder=4, label="reference $\\Re(\\lambda_i)$")
    ax_u.axhline(0.0, color=SSM_COLORS["baseline"], linewidth=0.8, linestyle="--")
    set_tufte_title(ax_u, "Undamped ring (c=0): all $\\lambda_i \\approx 0$")
    set_tufte_labels(ax_u, xlabel="index $i$", ylabel="Lyapunov exponent $\\lambda_i$")
    ax_u.set_ylim(-0.1, 0.1)
    ax_u.legend(loc="upper right", frameon=False, fontsize=9)

    fig.suptitle("Lyapunov spectrum via QR algorithm (n=8 ring of oscillators)",
                 fontsize=12, y=1.02)
    fig.tight_layout()
    return fig


def main() -> None:
    fig = make_figure()
    paths = save_figure(fig, _OUTPUT_PATH, formats=("png",))
    plt.close(fig)
    for p in paths:
        print(f"Wrote {p}")


if __name__ == "__main__":
    main()
