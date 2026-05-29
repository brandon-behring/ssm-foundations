"""Chapter 1 — eigenvalue spectrum of the ring-of-coupled-oscillators Jacobian.

Constructs the $2n \\times 2n$ state matrix $\\statemat$ for $n$ identical
damped oscillators arranged on a ring with nearest-neighbor coupling (stiffness
$\\kappa$), and visualizes its eigenvalues in the complex plane. The
circulant structure of the spatial coupling makes the eigenvalues fall into
two arcs corresponding to the standing-wave modes — see §1.4 of the chapter.

Idiomatic-JAX note (this companion is a NumPy→JAX teaching example)
------------------------------------------------------------------
Assembling a structured sparse matrix is the canonical place to meet JAX's
*functional update* idiom:

* **``jnp.zeros(...).at[rows, cols].add(vals)`` instead of a Python double loop
  of in-place ``A[i, j] = v`` assignments.** NumPy mutates the array element by
  element inside ``for i in range(n)``; JAX arrays are immutable, so we build the
  five families of nonzeros (position-derivative, spring+Laplacian diagonal,
  damping, and the two coupling off-diagonals) as flat ``(row, col, val)`` arrays
  and scatter them in a single vectorised pass. ``.add`` (scatter-add) is the
  natural "assemble from contributions" spelling and coincides with ``.set`` here
  because the ring (``n >= 3``) produces no duplicate ``(row, col)`` pairs.

The eigenvalues are then read off with ``jnp.linalg.eigvals`` (the JAX spelling of
``np.linalg.eigvals``).

Output
------
Writes ``public/figures/ch01/jacobian_eigenvalues.png`` (referenced from
``src/content/chapters/ch01-linear-odes.mdx`` §1.4).

Usage
-----
::

    PYTHONPATH=. python companions/ch01/jax/coupled_oscillators.py
"""

from __future__ import annotations

from pathlib import Path

import jax

# Enable float64 before any jnp array exists; eigenvalues of the lightly-damped
# ring sit close to the imaginary axis and single precision would blur the arcs.
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
_OUTPUT_PATH = _REPO_ROOT / "public" / "figures" / "ch01" / "jacobian_eigenvalues"


def build_ring_state_matrix(
    n: int,
    k: float = 4.0,
    c: float = 0.2,
    kappa: float = 1.0,
) -> jnp.ndarray:
    """Assemble the $2n \\times 2n$ state matrix for a ring of damped oscillators.

    State vector layout: $(q_1, \\dot q_1, q_2, \\dot q_2, \\ldots, q_n, \\dot q_n)$.
    Position rows have unit derivative; velocity rows pick up the spring,
    damping, and discrete-Laplacian coupling terms.

    The matrix is built with a single vectorised ``.at[rows, cols].add(vals)``
    scatter rather than a Python double loop of ``A[i, j] = ...`` assignments —
    see the module-level Idiomatic-JAX note.

    Parameters
    ----------
    n : int
        Number of oscillators in the ring; must be at least 3.
    k : float
        Spring stiffness; positive.
    c : float
        Damping coefficient; non-negative.
    kappa : float
        Nearest-neighbor coupling stiffness; non-negative.

    Returns
    -------
    jnp.ndarray of shape (2n, 2n)

    Raises
    ------
    ValueError
        If ``n < 3``, ``k <= 0``, ``c < 0``, or ``kappa < 0``.
    """
    if n < 3:
        raise ValueError(f"need at least 3 oscillators for a non-trivial ring, got n={n}")
    if k <= 0:
        raise ValueError(f"stiffness k must be positive, got {k}")
    if c < 0 or kappa < 0:
        raise ValueError(f"damping and coupling must be non-negative, got c={c}, kappa={kappa}")

    i = jnp.arange(n)
    q_idx = 2 * i  # position-component rows/cols
    v_idx = 2 * i + 1  # velocity-component rows/cols
    left = 2 * ((i - 1) % n)  # q-index of the left neighbor (cyclic)
    right = 2 * ((i + 1) % n)  # q-index of the right neighbor (cyclic)

    # Flatten every nonzero into one (row, col, val) scatter. The five blocks are
    # the five terms of dh/dt: dq/dt = v; and dv/dt = -(k + 2κ) q_i - c v_i
    # + κ q_{i-1} + κ q_{i+1}.
    rows = jnp.concatenate([q_idx, v_idx, v_idx, v_idx, v_idx])
    cols = jnp.concatenate([v_idx, q_idx, v_idx, left, right])
    vals = jnp.concatenate([
        jnp.ones(n),  # dq/dt = v
        jnp.full(n, -k - 2.0 * kappa),  # spring + Laplacian diagonal
        jnp.full(n, -c),  # damping
        jnp.full(n, kappa),  # +κ q_{i-1}
        jnp.full(n, kappa),  # +κ q_{i+1}
    ])
    return jnp.zeros((2 * n, 2 * n)).at[rows, cols].add(vals)


def make_figure() -> plt.Figure:
    """Build the eigenvalue-spectrum figure for n=8, k=4, c=0.2, kappa=1."""
    apply_style()
    A = build_ring_state_matrix(n=8)
    eigs = np.asarray(jnp.linalg.eigvals(A))

    fig, ax = create_tufte_figure(nrows=1, ncols=1, figsize=(6.0, 5.5))
    ax.axhline(0.0, color=SSM_COLORS["baseline"], linewidth=0.8, linestyle="-")
    ax.axvline(0.0, color=SSM_COLORS["baseline"], linewidth=0.8, linestyle="-")
    ax.scatter(eigs.real, eigs.imag, s=60, color=SSM_COLORS["accent"],
               edgecolors="white", linewidths=0.8, zorder=3,
               label="eigenvalues of $\\mathbf{A}$")
    # Highlight the imaginary-axis (Re=0) and left-half-plane regions visually.
    ax.axvspan(ax.get_xlim()[0], 0.0, alpha=0.05, color=SSM_COLORS["accent"])

    set_tufte_title(ax, "Eigenvalues of the ring-Jacobian (n=8)")
    set_tufte_labels(ax, xlabel=r"$\Re(\lambda)$", ylabel=r"$\Im(\lambda)$")
    ax.set_aspect("equal", adjustable="datalim")
    ax.legend(loc="upper right", frameon=False, fontsize=9)
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
