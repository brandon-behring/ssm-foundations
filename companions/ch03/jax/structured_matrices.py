"""Chapter 3 — visual comparison of four structured matrix families.

Constructs N=8 examples of each structure and displays them as heatmaps
(color encodes $\\log_{10}|A_{ij}|$ for visibility across magnitude
ranges). Shows the characteristic patterns:

* Toeplitz: constant along each diagonal
* Vandermonde: powers of node values increase across rows
* Cauchy: dense 1/(x_i - y_j) pattern
* 1-semiseparable: lower-triangular with rank-1 off-diagonal blocks
  (products of prefix factors)

Idiomatic-JAX note (this companion is a NumPy→JAX teaching example)
------------------------------------------------------------------
Each family is defined by an index formula, so the natural JAX idiom is
*vectorised construction on an index grid* rather than element-by-element
``for i: for j: A[i, j] = ...`` assignment:

* **Toeplitz** — one ``jnp.where(i >= j, first_col[|i-j|], first_row[|i-j|])``
  gather replaces the O(n²) branchy double loop.
* **1-semiseparable** — the entry $M_{ij} = a_{j+1}\\cdots a_i$ is the *ratio*
  $P_i / P_j$ of prefix products, so one ``jnp.cumprod`` (O(n)) plus a broadcast
  ratio on the strict-lower triangle replaces the O(n³) ``np.prod(factors[j:i])``
  per-entry loop.
* **Vandermonde / Cauchy** — ``jnp.vander`` and a single broadcast already
  vectorise; only ``np → jnp`` changes.

Output
------
``public/figures/ch03/structured_matrices.png`` (Figure 3.2).

Usage
-----
::

    PYTHONPATH=. python companions/ch03/jax/structured_matrices.py
"""

from __future__ import annotations

from pathlib import Path

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from companions._shared.plot_utils import (  # noqa: E402
    apply_style,
    create_tufte_figure,
    save_figure,
    set_tufte_title,
)

_REPO_ROOT = Path(__file__).resolve().parents[3]
_OUTPUT_PATH = _REPO_ROOT / "public" / "figures" / "ch03" / "structured_matrices"


def toeplitz(first_col: np.ndarray, first_row: np.ndarray | None = None) -> jnp.ndarray:
    """Construct a Toeplitz matrix from its first column (and optional first row).

    $T_{ij} = c_{i-j}$ for $i \\geq j$ and $r_{j-i}$ for $i < j$, built with a
    single ``jnp.where`` gather on the $|i-j|$ index grid. If ``first_row`` is
    None it defaults to $(c_0, 0, \\ldots, 0)$ (a lower-triangular Toeplitz).
    """
    first_col = jnp.asarray(first_col)
    n = first_col.shape[0]
    if first_row is None:
        first_row = jnp.zeros(n, dtype=first_col.dtype).at[0].set(first_col[0])
    else:
        first_row = jnp.asarray(first_row)
    i = jnp.arange(n)[:, None]
    j = jnp.arange(n)[None, :]
    d = jnp.abs(i - j)  # |i - j|; selects the diagonal index for both branches
    return jnp.where(i >= j, first_col[d], first_row[d])


def vandermonde(nodes: np.ndarray, ncols: int | None = None) -> jnp.ndarray:
    """Construct a Vandermonde matrix $V_{ij} = \\text{nodes}_i^{\\,j}$."""
    nodes = jnp.asarray(nodes)
    if ncols is None:
        ncols = nodes.shape[0]
    return jnp.vander(nodes, ncols, increasing=True)


def cauchy(xs: np.ndarray, ys: np.ndarray) -> jnp.ndarray:
    """Construct a Cauchy matrix $C_{ij} = 1/(x_i - y_j)$."""
    xs = jnp.asarray(xs)
    ys = jnp.asarray(ys)
    if bool(jnp.any(xs[:, None] == ys[None, :])):
        raise ValueError("xs and ys must be disjoint (no zero denominators)")
    return 1.0 / (xs[:, None] - ys[None, :])


def one_semiseparable(factors: np.ndarray) -> jnp.ndarray:
    """Construct a 1-semiseparable lower-triangular matrix.

    Given factors $a_1, \\ldots, a_{n-1}$, the matrix has $M_{ij} = a_{j+1}
    a_{j+2} \\cdots a_i$ for $i > j$ and $M_{ii} = 1$ — the matrix form of the
    scalar recurrence $h_t = a_t h_{t-1} + b_t$.

    Built from prefix products $P_m = a_1 \\cdots a_m$ (with $P_0 = 1$): then
    $M_{ij} = P_i / P_j$ on the strict-lower triangle. One ``jnp.cumprod`` plus a
    broadcast ratio replaces the per-entry ``np.prod(factors[j:i])`` double loop.
    """
    factors = jnp.asarray(factors)
    n = factors.shape[0] + 1
    prefix = jnp.concatenate([jnp.ones(1, dtype=factors.dtype), jnp.cumprod(factors)])
    i = jnp.arange(n)[:, None]
    j = jnp.arange(n)[None, :]
    ratio = prefix[i] / prefix[j]  # = a_{j+1} ... a_i  for i > j
    return jnp.where(i > j, ratio, 0.0) + jnp.eye(n)


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
        log_mag = np.log10(np.abs(np.asarray(mat)) + 1e-12)
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
