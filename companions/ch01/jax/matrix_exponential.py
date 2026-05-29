"""Chapter 1 — matrix-exponential convergence: truncated series vs scaling-and-squaring.

Compares the truncated power series $\\sum_{k=0}^K M^k / k!$ against
``scipy.linalg.expm`` (Pade approximants + scaling-and-squaring, the
production-quality algorithm). Plots the relative Frobenius-norm error
$\\|e^M_K - e^M_{\\text{scipy}}\\|_F / \\|e^M_{\\text{scipy}}\\|_F$ versus the
truncation order $K$, for two matrices of different spectral radii.

The headline observation: series convergence is *fast* when $\\|M\\|$ is small
(a few terms suffice) but *catastrophic* for large $\\|M\\|$ — which is exactly
why production code uses Pade-plus-scaling, not naive series truncation. This
motivates Chapter 4's split between methods that *look* like truncated series
(bilinear, explicit RK) and methods that *exponentiate* (ZOH, the
matrix-exponential-based S4 parameterization).

Idiomatic-JAX note (this companion is a NumPy→JAX teaching example)
------------------------------------------------------------------
A convergence sweep is the canonical place to meet two JAX idioms:

* **``lax.scan`` instead of a Python accumulation loop.** The partial sums
  $S_0, S_1, \\ldots, S_K$ form a recurrence ($S_k = S_{k-1} + M^k/k!$). NumPy
  recomputes each $S_K$ from scratch inside the sweep — O(K^2) work. ``lax.scan``
  threads the running term and running sum as *carry* and *emits every partial
  sum in one O(K) fused pass*.
* **``vmap`` instead of a list comprehension.** The per-order Frobenius error is
  a map over the stacked partial sums; ``jax.vmap`` vectorises it.

The whole pipeline is ``jit``-compiled. ``scipy.linalg.expm`` is kept as the
ground-truth reference (and as a contrast: the production algorithm we are *not*
reimplementing).

Output
------
Writes ``public/figures/ch01/matrix_exponential_convergence.png`` (auxiliary
figure for §1.2; available to exercise solutions).
"""

from __future__ import annotations

from functools import partial
from pathlib import Path

import jax

# Enable float64 before any jnp array exists; the convergence floor we plot sits
# at ~1e-16, far below float32's ~1e-7, so single precision would hide the story.
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy.linalg import expm  # noqa: E402

from companions._shared.plot_utils import (  # noqa: E402
    SSM_COLORS,
    apply_style,
    create_tufte_figure,
    save_figure,
    set_tufte_labels,
    set_tufte_title,
)

_REPO_ROOT = Path(__file__).resolve().parents[3]
_OUTPUT_PATH = _REPO_ROOT / "public" / "figures" / "ch01" / "matrix_exponential_convergence"


@partial(jax.jit, static_argnums=1)
def series_partial_sums(M: jnp.ndarray, k_max: int) -> jnp.ndarray:
    """All partial sums $S_0, \\ldots, S_{k_{\\max}}$ of $e^M$'s power series.

    Computed with a single ``jax.lax.scan``: the carry is ``(term, total)`` where
    ``term`` $= M^{k-1}/(k-1)!$ and ``total`` $= S_{k-1}$; each step multiplies in
    one more factor and emits the new partial sum. ``k_max`` is a static argument
    because it sets the scan length (and hence the output shape).

    NumPy contrast: the equivalent NumPy code recomputes the running product in a
    Python ``for`` loop and, for a *sweep* over many ``k_max``, repeats that work
    per order (O(K^2)). This emits every partial sum in one O(K) pass.

    Parameters
    ----------
    M : jnp.ndarray of shape (N, N)
        Input matrix.
    k_max : int
        Maximum series order (inclusive); must be non-negative.

    Returns
    -------
    jnp.ndarray of shape (k_max + 1, N, N)
        Stack of partial sums ``[S_0, S_1, ..., S_{k_max}]``.
    """
    n = M.shape[0]
    eye = jnp.eye(n, dtype=M.dtype)

    def step(carry: tuple[jnp.ndarray, jnp.ndarray], k: jnp.ndarray):
        term, total = carry
        term = term @ M / k  # M^k / k!  from  M^{k-1}/(k-1)!
        total = total + term
        return (term, total), total

    ks = jnp.arange(1, k_max + 1)
    _, tail = jax.lax.scan(step, (eye, eye), ks)  # tail = [S_1, ..., S_{k_max}]
    return jnp.concatenate([eye[None], tail], axis=0)  # prepend S_0 = I


def truncated_series(M: np.ndarray, K: int) -> jnp.ndarray:
    """Single partial sum $S_K = \\sum_{k=0}^{K} M^k / k!$.

    Thin validated wrapper over :func:`series_partial_sums` (returns its last row).

    Raises
    ------
    ValueError
        If ``K < 0`` or ``M`` is not square.
    """
    if K < 0:
        raise ValueError(f"truncation order K must be non-negative, got {K}")
    M = jnp.asarray(M)
    if M.ndim != 2 or M.shape[0] != M.shape[1]:
        raise ValueError(f"M must be a square matrix, got shape {M.shape}")
    return series_partial_sums(M, K)[K]


def relative_frobenius_error(approx: jnp.ndarray, reference: jnp.ndarray) -> float:
    """Relative Frobenius-norm error $\\|A - B\\|_F / \\|B\\|_F$."""
    ref_norm = float(jnp.linalg.norm(reference))
    if ref_norm == 0.0:
        raise ValueError("reference matrix has zero Frobenius norm; cannot normalize")
    return float(jnp.linalg.norm(approx - reference) / ref_norm)


def convergence_errors(M: np.ndarray, k_max: int) -> np.ndarray:
    """Relative Frobenius error of $S_0..S_{k_{\\max}}$ against ``scipy.expm(M)``.

    The per-order error map is a ``jax.vmap`` over the stacked partial sums — the
    idiomatic replacement for a Python ``[truncated_series(M, k) for k in ...]``
    list comprehension.
    """
    reference = jnp.asarray(expm(np.asarray(M)))
    partials = series_partial_sums(jnp.asarray(M), k_max)  # (k_max+1, N, N)
    ref_norm = jnp.linalg.norm(reference)
    errs = jax.vmap(lambda S: jnp.linalg.norm(S - reference) / ref_norm)(partials)
    return np.asarray(errs)


def make_figure() -> plt.Figure:
    """Build the convergence comparison figure for two test matrices."""
    apply_style()

    # Small matrix: spectral radius ~ 1, series converges quickly.
    A_small = np.array([[-0.5, 1.0], [-1.0, -0.5]])
    # Large matrix: spectral radius ~ 14, needs many terms.
    A_large = np.array([[-5.0, 10.0], [-10.0, -5.0]])

    k_max = 39
    orders = np.arange(0, k_max + 1)
    errors_small = convergence_errors(A_small, k_max)
    errors_large = convergence_errors(A_large, k_max)

    fig, ax = create_tufte_figure(nrows=1, ncols=1, figsize=(7.0, 4.5))
    ax.semilogy(
        orders,
        errors_small,
        color=SSM_COLORS["accent"],
        linewidth=1.6,
        marker="o",
        markersize=4,
        label=r"$\|M\| \approx 1$ (fast)",
    )
    ax.semilogy(
        orders,
        errors_large,
        color=SSM_COLORS["alert"],
        linewidth=1.6,
        marker="s",
        markersize=4,
        label=r"$\|M\| \approx 14$ (slow)",
    )
    ax.axhline(
        1e-14,
        color=SSM_COLORS["baseline"],
        linewidth=0.8,
        linestyle=":",
        label="machine epsilon (float64)",
    )
    set_tufte_title(ax, "Truncated-series error vs `scipy.linalg.expm`")
    set_tufte_labels(ax, xlabel="truncation order $K$", ylabel=r"relative Frobenius error")
    ax.legend(loc="upper right", frameon=False, fontsize=9)
    ax.set_xlim(0, int(orders[-1]))
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
