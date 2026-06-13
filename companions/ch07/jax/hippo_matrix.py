"""Chapter 7 — the HiPPO-LegS projection operator: matrix structure and spectrum.

Builds the HiPPO-LegS (scaled-Legendre) state matrix $\\statemat$ and input vector
$\\inputmat$ that the entire S4 / Mamba lineage inherits as initialization. The matrix
is the *projection-update operator*: it advances the coefficients of the best $N$-term
Legendre approximation of the input history $u(s),\\, s\\in[0,t]$.

Closed form (normalized scaled-Legendre basis)
----------------------------------------------
::

    A[i, j] = -sqrt((2i+1)(2j+1))   for i > j     (lower off-diagonal)
            = -(i + 1)              for i == j    (diagonal)
            = 0                     for i < j     (strictly upper)
    B[i]    =  sqrt(2i + 1)

Because $\\statemat$ is **lower-triangular**, its eigenvalues are exactly its diagonal,
$\\{-1, -2, \\ldots, -N\\}$ — all real and strictly negative, so the continuous HiPPO
dynamics are asymptotically stable (Chapter 2). This is *why* HiPPO initialization
avoids the vanishing/exploding dynamics of a random $\\statemat$.

JAX idiom (vs the reference port)
---------------------------------
The reference builds $\\statemat = T M T^{-1}$ with a similarity transform (a matmul +
an explicit ``inv``). Here we construct the *closed form directly* with a single
vectorized ``jnp.where`` over a ``meshgrid`` — no Python loop, no matrix inverse, and
``jit``-fusable to one kernel. The two routes are algebraically identical (see §7.3);
the closed form is the more transparent spelling for a teaching companion.

Port credit
-----------
HiPPO-LegS construction ported from
``post_transformers/experiments/jax/week04/s4_hippo.py`` (``make_hippo_legs``), which
in turn follows ``experiments/refs/s4/src/models/hippo/hippo.py`` (Gu et al., 2020,
arXiv:2008.07669) and the Annotated S4. We switch float32 -> float64 to match this
book's global ``jax_enable_x64`` (Chapter 4) and to keep the eigenvalue
computation accurate at larger $N$ (HiPPO-LegS is highly non-normal — §7.5 — so
its spectrum is sensitive in low precision).

Output
------
``public/figures/ch07/hippo_matrix_structure.png`` — heatmap of $\\statemat$ (N=8).
``public/figures/ch07/hippo_eigenvalues.png`` — spectrum of $\\statemat$ in the complex
plane (N=16), on the negative real axis.

Usage
-----
::

    PYTHONPATH=. python companions/ch07/jax/hippo_matrix.py
"""

from __future__ import annotations

from pathlib import Path

import jax

# Enable float64 before any jnp array is created (matches Chapter 4). HiPPO-LegS's
# §7.5 non-normality makes its spectrum sensitive, so the eigenvalue figure needs
# the extra precision once N grows past ~32.
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
_OUT_DIR = _REPO_ROOT / "public" / "figures" / "ch07"


# ---------------------------------------------------------------------------
# The HiPPO-LegS projection operator
# ---------------------------------------------------------------------------


def make_hippo_legs(n: int) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Construct the HiPPO-LegS matrices $\\statemat \\in \\R^{N\\times N}$, $\\inputmat \\in \\R^{N\\times 1}$.

    Built directly from the closed form (see module docstring) with a vectorized
    ``jnp.where`` over a ``meshgrid`` — the idiomatic-JAX spelling of "fill an
    $N\\times N$ matrix by an index rule," with no Python loop and no matrix inverse.

    Parameters
    ----------
    n : int
        State dimension $N$ (number of Legendre basis functions); must be >= 1.

    Returns
    -------
    A : jnp.ndarray, shape (n, n)
        Continuous-time state matrix. Lower-triangular; eigenvalues are exactly
        $-1, -2, \\ldots, -N$ (all in the open left half-plane).
    B : jnp.ndarray, shape (n, 1)
        Continuous-time input matrix, $B_i = \\sqrt{2i+1}$.

    Raises
    ------
    ValueError
        If ``n < 1``.
    """
    if n < 1:
        raise ValueError(f"state dimension n must be >= 1, got {n}")
    q = jnp.arange(n, dtype=jnp.float64)
    # row index i along axis 0, column index j along axis 1.
    i_idx, j_idx = jnp.meshgrid(q, q, indexing="ij")
    lower = jnp.sqrt((2.0 * i_idx + 1.0) * (2.0 * j_idx + 1.0))
    diag = i_idx + 1.0
    A = -jnp.where(i_idx > j_idx, lower, jnp.where(i_idx == j_idx, diag, 0.0))
    B = jnp.sqrt(2.0 * q + 1.0)[:, None]
    return A, B


def legs_eigenvalues(n: int) -> jnp.ndarray:
    """Eigenvalues of the HiPPO-LegS $\\statemat$ (numerically, via ``jnp.linalg.eigvals``).

    For the exact lower-triangular matrix these equal the diagonal $-1,\\ldots,-N$;
    computing them numerically is the regression check that the construction did not
    accidentally introduce an upper-triangular leak.
    """
    A, _ = make_hippo_legs(n)
    return jnp.linalg.eigvals(A)


# ---------------------------------------------------------------------------
# Figure 1: matrix structure
# ---------------------------------------------------------------------------


def make_matrix_structure_figure(n: int = 8) -> plt.Figure:
    """Heatmap of $\\statemat$ showing the lower-triangular projection-operator structure."""
    apply_style()
    A = np.asarray(make_hippo_legs(n)[0])

    fig, ax = create_tufte_figure(figsize=(5.6, 5.0))
    im = ax.imshow(A, cmap="RdBu", vmin=-np.max(np.abs(A)), vmax=np.max(np.abs(A)))
    set_tufte_title(ax, rf"HiPPO-LegS $A$ structure ($N={n}$)")
    set_tufte_labels(ax, xlabel=r"column $j$", ylabel=r"row $i$")
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label=r"$A_{ij}$")
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Figure 2: eigenvalue spectrum
# ---------------------------------------------------------------------------


def make_eigenvalue_figure(n: int = 16) -> plt.Figure:
    """Spectrum of $\\statemat$ in the complex plane — real, negative, at $-1,\\ldots,-N$."""
    apply_style()
    eigs = np.asarray(legs_eigenvalues(n))

    fig, ax = create_tufte_figure(figsize=(6.4, 4.6))
    ax.axvspan(-(n + 1.0), 0.0, alpha=0.08, color=SSM_COLORS["accent"])
    ax.axhline(0.0, color=SSM_COLORS["baseline"], linewidth=0.6)
    ax.axvline(0.0, color=SSM_COLORS["baseline"], linewidth=0.6)
    ax.scatter(
        eigs.real,
        eigs.imag,
        s=70,
        color=SSM_COLORS["accent"],
        edgecolors="white",
        linewidths=1.0,
        zorder=3,
    )
    ax.set_xlim(-(n + 1.0), 1.0)
    ax.set_ylim(-1.0, 1.0)
    set_tufte_title(ax, rf"HiPPO-LegS spectrum ($N={n}$): eigenvalues $= -1,\ldots,-N$")
    set_tufte_labels(ax, xlabel=r"$\operatorname{Re}(\lambda)$", ylabel=r"$\operatorname{Im}(\lambda)$")
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    _OUT_DIR.mkdir(parents=True, exist_ok=True)
    print("Chapter 7 — hippo_matrix.py")
    print("=" * 60)

    # Echo a few structural facts the chapter prose and tests rely on.
    for n in (4, 8, 16):
        eigs = np.sort(np.asarray(legs_eigenvalues(n)).real)
        print(f"  N={n:2d}: eigenvalues (sorted real) = {np.round(eigs, 3)}")
    A8 = np.asarray(make_hippo_legs(8)[0])
    print(f"  A[3,1] (N=8) = {A8[3, 1]:.6f}  (closed form -sqrt(7*3) = {-np.sqrt(7 * 3):.6f})")
    print(f"  upper-triangle max |A_ij| (should be 0): {np.max(np.abs(np.triu(A8, k=1))):.2e}")

    fig1 = make_matrix_structure_figure(n=8)
    for p in save_figure(fig1, _OUT_DIR / "hippo_matrix_structure", formats=("png",)):
        print(f"Wrote {p}")
    plt.close(fig1)

    fig2 = make_eigenvalue_figure(n=16)
    for p in save_figure(fig2, _OUT_DIR / "hippo_eigenvalues", formats=("png",)):
        print(f"Wrote {p}")
    plt.close(fig2)


if __name__ == "__main__":
    main()
