"""Chapter 1 — matrix-exponential convergence vs scaling-and-squaring.

Compares the truncated power series $\\sum_{k=0}^K M^k / k!$ against
``scipy.linalg.expm`` (which uses Pade approximants + scaling-and-squaring,
the production-quality algorithm). Plots the relative Frobenius-norm error
$\\|e^M_K - e^M_{\\text{scipy}}\\|_F / \\|e^M_{\\text{scipy}}\\|_F$ as a
function of the truncation order $K$, for two matrices of different
spectral radii.

The headline observation: convergence rate of the series is *fast* when
$\\|M\\|$ is small (a few terms suffice), but *catastrophic* for large
$\\|M\\|$ — which is exactly why production code uses Pade-plus-scaling,
not naive series truncation. This motivates Chapter 4's discussion of
why discretization methods that *look* like truncated series (the bilinear
transform; explicit Runge–Kutta) need very different stability analyses
from methods that *exponentiate* (ZOH; the matrix-exponential-based S4
parameterization).

Output
------
Writes ``public/figures/ch01/matrix_exponential_convergence.png`` (auxiliary
figure for §1.2; not referenced from the chapter prose but available to
exercise solutions).
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

_REPO_ROOT = Path(__file__).resolve().parents[3]
_OUTPUT_PATH = _REPO_ROOT / "public" / "figures" / "ch01" / "matrix_exponential_convergence"


def truncated_series(M: np.ndarray, K: int) -> np.ndarray:
    """Compute $\\sum_{k=0}^{K} M^k / k!$ by direct summation.

    Parameters
    ----------
    M : ndarray of shape (N, N)
        Input matrix.
    K : int
        Maximum series order (inclusive); must be non-negative.

    Returns
    -------
    ndarray of shape (N, N)
        Partial sum $I + M + M^2/2! + \\cdots + M^K/K!$.

    Raises
    ------
    ValueError
        If ``K < 0`` or ``M`` is not square.
    """
    if K < 0:
        raise ValueError(f"truncation order K must be non-negative, got {K}")
    if M.ndim != 2 or M.shape[0] != M.shape[1]:
        raise ValueError(f"M must be a square matrix, got shape {M.shape}")
    n = M.shape[0]
    total = np.eye(n, dtype=M.dtype)
    term = np.eye(n, dtype=M.dtype)
    for k in range(1, K + 1):
        term = term @ M / k
        total = total + term
    return total


def relative_frobenius_error(approx: np.ndarray, reference: np.ndarray) -> float:
    """Relative Frobenius-norm error $\\|A - B\\|_F / \\|B\\|_F$."""
    ref_norm = np.linalg.norm(reference, ord="fro")
    if ref_norm == 0.0:
        raise ValueError("reference matrix has zero Frobenius norm; cannot normalize")
    return float(np.linalg.norm(approx - reference, ord="fro") / ref_norm)


def make_figure() -> plt.Figure:
    """Build the convergence comparison figure for two test matrices."""
    apply_style()

    # Small matrix: 2x2 with spectral radius ~ 1; series converges quickly.
    A_small = np.array([[-0.5, 1.0], [-1.0, -0.5]])
    # Large matrix: 2x2 with spectral radius ~ 10; needs many terms.
    A_large = np.array([[-5.0, 10.0], [-10.0, -5.0]])

    orders = np.arange(0, 40)
    expm_small = expm(A_small)
    expm_large = expm(A_large)

    errors_small = [relative_frobenius_error(truncated_series(A_small, k), expm_small) for k in orders]
    errors_large = [relative_frobenius_error(truncated_series(A_large, k), expm_large) for k in orders]

    fig, ax = create_tufte_figure(nrows=1, ncols=1, figsize=(7.0, 4.5))
    ax.semilogy(orders, errors_small, color=SSM_COLORS["accent"], linewidth=1.6,
                marker="o", markersize=4, label=r"$\|M\| \approx 1$ (fast)")
    ax.semilogy(orders, errors_large, color=SSM_COLORS["alert"], linewidth=1.6,
                marker="s", markersize=4, label=r"$\|M\| \approx 14$ (slow)")
    ax.axhline(1e-14, color=SSM_COLORS["baseline"], linewidth=0.8, linestyle=":",
               label="machine epsilon (float64)")
    set_tufte_title(ax, "Truncated-series error vs `scipy.linalg.expm`")
    set_tufte_labels(ax, xlabel="truncation order $K$",
                     ylabel=r"relative Frobenius error")
    ax.legend(loc="upper right", frameon=False, fontsize=9)
    ax.set_xlim(0, orders[-1])
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
