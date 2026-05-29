"""Chapter 3 — condition number growth across structured matrix families.

Compares $\\kappa(A)$ (operator-norm condition number) as a function of
matrix size $N$ for three families:

1. **Random Gaussian** — entries iid $\\mathcal{N}(0, 1)$. The condition
   number grows roughly linearly in $N$ (Edelman's classical result on
   random matrix conditioning).
2. **Hilbert matrix** — $H_{ij} = 1/(i + j - 1)$. The textbook
   ill-conditioning example: $\\kappa(H_N)$ grows like $e^{3.5 N}$.
3. **HiPPO-LegS matrix** — a structured matrix whose condition number grows only
   *polynomially* (empirically $\\kappa \\sim N^2$), far slower than the Hilbert
   matrix's exponential blow-up. This sub-exponential growth — not boundedness —
   is what makes HiPPO-LegS a stable default initialization for S4-family SSMs.

Idiomatic-JAX note (this companion is a NumPy→JAX teaching example)
------------------------------------------------------------------
* **``jnp.where`` on an index grid** builds the HiPPO-LegS matrix in one
  vectorised pass — the strict-lower, diagonal, and strict-upper cases become two
  masked ``where`` calls instead of the NumPy ``for i: for j:`` element loop.
* **The size sweep stays a Python loop.** ``κ`` is evaluated for a list of *different*
  matrix sizes $N$; ``jax.vmap`` cannot vectorise over ragged shapes, so the sweep
  is a comprehension while only the per-$N$ *construction* is vectorised. The
  per-matrix conditioning uses ``jnp.linalg.cond``.

Output
------
``public/figures/ch03/condition_number_growth.png`` (Figure 3.1).
"""

from __future__ import annotations

from pathlib import Path

import jax

# Enable float64 before any jnp array exists; κ(Hilbert) reaches ~1e16, so we
# need the full double-precision dynamic range to plot the growth curves.
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from companions._shared.plot_utils import (  # noqa: E402
    SSM_COLORS,
    apply_style,
    create_tufte_figure,
    save_figure,
    set_tufte_labels,
    set_tufte_title,
)

_REPO_ROOT = Path(__file__).resolve().parents[3]
_OUTPUT_PATH = _REPO_ROOT / "public" / "figures" / "ch03" / "condition_number_growth"


def hippo_legs(N: int) -> jnp.ndarray:
    """Construct the HiPPO-LegS matrix.

    Defined in Gu et al. (2020) HiPPO. The closed form is
    $A_{ij} = -\\sqrt{(2i+1)(2j+1)}$ if $i > j$; $-(i+1)$ if $i = j$; $0$ if
    $i < j$. Built with two ``jnp.where`` masks on the $(i, j)$ index grid.

    Parameters
    ----------
    N : int
        Matrix dimension; must be positive.

    Returns
    -------
    jnp.ndarray of shape (N, N)

    Raises
    ------
    ValueError
        If ``N < 1``.
    """
    if N < 1:
        raise ValueError(f"N must be positive, got {N}")
    i = jnp.arange(N)[:, None]
    j = jnp.arange(N)[None, :]
    off_diag = -jnp.sqrt((2.0 * i + 1.0) * (2.0 * j + 1.0))  # value for i > j
    diag = -(i + 1.0)  # value for i == j (broadcasts over columns)
    A = jnp.where(i > j, off_diag, 0.0)
    return jnp.where(i == j, diag, A)


def hilbert(N: int) -> jnp.ndarray:
    """Construct the N x N Hilbert matrix $H_{ij} = 1/(i + j - 1)$."""
    i, j = jnp.meshgrid(jnp.arange(N) + 1, jnp.arange(N) + 1, indexing="ij")
    return 1.0 / (i + j - 1)


def random_gaussian(N: int, seed: int = 0) -> jnp.ndarray:
    """Construct an N x N random Gaussian matrix with a fixed seed.

    Kept on NumPy's seeded ``default_rng`` for a reproducible reference matrix
    (``jax.random`` with an explicit PRNGKey would be the pure-JAX alternative;
    the conditioning story does not depend on the RNG choice).
    """
    rng = np.random.default_rng(seed)
    return jnp.asarray(rng.standard_normal((N, N)))


def make_figure() -> plt.Figure:
    """Build the condition-number growth comparison figure."""
    apply_style()
    sizes = np.array([2, 4, 8, 16, 32, 64, 128, 256])

    # Ragged-shape sweep: a Python comprehension over sizes (vmap can't help),
    # with jnp.linalg.cond doing the per-matrix conditioning.
    kappa_gauss = np.array([float(jnp.linalg.cond(random_gaussian(N))) for N in sizes])
    # Hilbert blows up fast; restrict to small sizes to avoid overflow.
    sizes_hilbert = sizes[sizes <= 32]
    kappa_hilbert = np.array([float(jnp.linalg.cond(hilbert(N))) for N in sizes_hilbert])
    kappa_hippo = np.array([float(jnp.linalg.cond(hippo_legs(N))) for N in sizes])

    fig, ax = create_tufte_figure(nrows=1, ncols=1, figsize=(7.5, 5.0))
    ax.loglog(sizes, kappa_gauss, color=SSM_COLORS["accent"], linewidth=1.6,
              marker="o", markersize=5, label="random Gaussian")
    ax.loglog(sizes_hilbert, kappa_hilbert, color=SSM_COLORS["alert"], linewidth=1.6,
              marker="s", markersize=5, label="Hilbert matrix (ill-conditioned)")
    ax.loglog(sizes, kappa_hippo, color=SSM_COLORS["highlight"], linewidth=1.6,
              marker="^", markersize=5, label=r"HiPPO-LegS ($\kappa \sim N^2$)")
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
