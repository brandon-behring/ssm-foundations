"""Chapter 3 — condition number growth across structured matrix families.

Compares $\\kappa(A)$ (operator-norm condition number) as a function of
matrix size $N$ for three families:

1. **Random Gaussian** — entries iid $\\mathcal{N}(0, 1)$. The condition
   number grows roughly linearly in $N$ (Edelman's classical result on
   random matrix conditioning).
2. **Hilbert matrix** — $H_{ij} = 1/(i + j - 1)$. The textbook
   ill-conditioning example: $\\kappa(H_N)$ grows like $e^{3.5 N}$.
3. **HiPPO-LegS matrix** — a structured matrix with the property that
   $\\kappa$ stays bounded as $N$ grows. This is one reason HiPPO is the
   default initialization for S4-family SSMs.

Output
------
``public/figures/ch03/condition_number_growth.png`` (Figure 3.1).
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from companions._shared.plot_utils import (
    SSM_COLORS,
    apply_style,
    create_tufte_figure,
    save_figure,
    set_tufte_labels,
    set_tufte_title,
)

_REPO_ROOT = Path(__file__).resolve().parents[3]
_OUTPUT_PATH = _REPO_ROOT / "public" / "figures" / "ch03" / "condition_number_growth"


def hippo_legs(N: int) -> np.ndarray:
    """Construct the HiPPO-LegS matrix.

    Defined in Gu et al. (2020) HiPPO. The closed form is:
    A[i, j] = -sqrt((2i+1)(2j+1)) if i > j; -(i+1) if i == j; 0 if i < j.

    Parameters
    ----------
    N : int
        Matrix dimension.

    Returns
    -------
    ndarray of shape (N, N)
    """
    if N < 1:
        raise ValueError(f"N must be positive, got {N}")
    A = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            if i > j:
                A[i, j] = -np.sqrt((2 * i + 1) * (2 * j + 1))
            elif i == j:
                A[i, j] = -(i + 1)
    return A


def hilbert(N: int) -> np.ndarray:
    """Construct the N x N Hilbert matrix."""
    i, j = np.meshgrid(np.arange(N) + 1, np.arange(N) + 1, indexing="ij")
    return 1.0 / (i + j - 1)


def random_gaussian(N: int, seed: int = 0) -> np.ndarray:
    """Construct an N x N random Gaussian matrix with fixed seed."""
    rng = np.random.default_rng(seed)
    return rng.standard_normal((N, N))


def make_figure() -> plt.Figure:
    """Build the condition-number growth comparison figure."""
    apply_style()
    sizes = np.array([2, 4, 8, 16, 32, 64, 128, 256])

    kappa_gauss = np.array([np.linalg.cond(random_gaussian(N)) for N in sizes])
    # Hilbert blows up fast; restrict to small sizes to avoid overflow.
    sizes_hilbert = sizes[sizes <= 32]
    kappa_hilbert = np.array([np.linalg.cond(hilbert(N)) for N in sizes_hilbert])
    kappa_hippo = np.array([np.linalg.cond(hippo_legs(N)) for N in sizes])

    fig, ax = create_tufte_figure(nrows=1, ncols=1, figsize=(7.5, 5.0))
    ax.loglog(sizes, kappa_gauss, color=SSM_COLORS["accent"], linewidth=1.6,
              marker="o", markersize=5, label="random Gaussian")
    ax.loglog(sizes_hilbert, kappa_hilbert, color=SSM_COLORS["alert"], linewidth=1.6,
              marker="s", markersize=5, label="Hilbert matrix (ill-conditioned)")
    ax.loglog(sizes, kappa_hippo, color=SSM_COLORS["highlight"], linewidth=1.6,
              marker="^", markersize=5, label="HiPPO-LegS (well-conditioned)")
    ax.axhline(1e16, color=SSM_COLORS["baseline"], linewidth=0.8, linestyle=":",
               label="float64 numerical-singularity threshold")

    set_tufte_title(ax, "Condition number $\\kappa(A)$ vs matrix size $N$")
    set_tufte_labels(ax, xlabel="$N$", ylabel="$\\kappa(A)$ (log scale)")
    ax.legend(loc="upper left", frameon=False, fontsize=9)
    ax.set_xlim(sizes[0] * 0.7, sizes[-1] * 1.5)
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
