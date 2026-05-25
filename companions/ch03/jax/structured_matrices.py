"""Chapter 3 — visual comparison of four structured matrix families.

Constructs N=8 examples of each structure and displays them as heatmaps
(color encodes $\\log_{10}|A_{ij}|$ for visibility across magnitude
ranges). Shows the characteristic patterns:

* Toeplitz: constant along each diagonal
* Vandermonde: powers of node values increase across rows
* Cauchy: dense 1/(x_i - y_j) pattern
* 1-semiseparable: lower-triangular with rank-1 off-diagonal blocks
  (products of prefix factors)

Output
------
``public/figures/ch03/structured_matrices.png`` (Figure 3.2).
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from companions._shared.plot_utils import (
    apply_style,
    create_tufte_figure,
    save_figure,
    set_tufte_title,
)

_REPO_ROOT = Path(__file__).resolve().parents[3]
_OUTPUT_PATH = _REPO_ROOT / "public" / "figures" / "ch03" / "structured_matrices"


def toeplitz(first_col: np.ndarray, first_row: np.ndarray | None = None) -> np.ndarray:
    """Construct a Toeplitz matrix from its first column (and optional first row).

    If first_row is None, defaults to first_col[::-1] padded with zeros
    (giving a lower-triangular Toeplitz).
    """
    n = len(first_col)
    if first_row is None:
        first_row = np.zeros(n)
        first_row[0] = first_col[0]
    T = np.empty((n, n))
    for i in range(n):
        for j in range(n):
            T[i, j] = first_col[i - j] if i >= j else first_row[j - i]
    return T


def vandermonde(nodes: np.ndarray, ncols: int | None = None) -> np.ndarray:
    """Construct a Vandermonde matrix V[i, j] = nodes[i] ** j."""
    if ncols is None:
        ncols = len(nodes)
    return np.vander(nodes, ncols, increasing=True)


def cauchy(xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
    """Construct a Cauchy matrix C[i, j] = 1 / (xs[i] - ys[j])."""
    if np.any(xs[:, None] == ys[None, :]):
        raise ValueError("xs and ys must be disjoint (no zero denominators)")
    return 1.0 / (xs[:, None] - ys[None, :])


def one_semiseparable(factors: np.ndarray) -> np.ndarray:
    """Construct a 1-semiseparable lower-triangular matrix.

    Given factors $a_1, ..., a_{n-1}$, the matrix has $M_{ij} = a_{j+1}
    a_{j+2} \\cdots a_i$ for $i > j$ and $M_{ii} = 1$. This is the matrix
    form of the scalar recurrence $h_t = a_t h_{t-1} + b_t$.
    """
    n = len(factors) + 1
    M = np.eye(n)
    for j in range(n):
        for i in range(j + 1, n):
            M[i, j] = np.prod(factors[j:i])
    return M


def make_figure() -> plt.Figure:
    """Build the 2x2 grid of structured matrices as log-magnitude heatmaps."""
    apply_style()
    n = 8

    # Construct one example from each family.
    T = toeplitz(np.array([3.0, 1.0, 0.5, 0.25, 0.125, 0.0625, 0.03125, 0.015625]),
                 np.array([3.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]))
    V = vandermonde(np.linspace(0.5, 1.2, n))
    xs = np.linspace(0.1, 1.0, n)
    ys = np.linspace(1.5, 2.5, n)
    C = cauchy(xs, ys)
    factors = 0.9 * np.ones(n - 1)  # decaying recurrence; gives diagonal-dominant lower triangle
    S = one_semiseparable(factors)

    matrices = [
        (T, "Toeplitz (constant along diagonals)"),
        (V, "Vandermonde (powers across columns)"),
        (C, "Cauchy ($1/(x_i - y_j)$)"),
        (S, "1-semiseparable lower-triangular"),
    ]

    fig, axes = create_tufte_figure(nrows=2, ncols=2, figsize=(11.0, 9.0))
    for ax, (mat, title) in zip(axes.flat, matrices):  # type: ignore[union-attr]
        # log10 of |entry|; clip small values for visibility.
        log_mag = np.log10(np.abs(mat) + 1e-12)
        im = ax.imshow(log_mag, cmap="viridis", aspect="equal",
                       interpolation="nearest", vmin=-6, vmax=2)
        set_tufte_title(ax, title, fontsize=10)
        ax.set_xticks(range(n))
        ax.set_yticks(range(n))
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label=r"$\log_{10}|A_{ij}|$")

    fig.suptitle("Four structured matrix families (N=8)", fontsize=12, y=1.0)
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
