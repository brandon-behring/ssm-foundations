r"""Chapter 8 §8.6 — S5: one diagonal MIMO SSM run as a parallel associative scan.

S5 keeps a single *diagonal* state matrix (like S4D, §8.5) but makes it
**MIMO** — one shared state driving $H$ input/output channels via $B \in
\mathbb{C}^{P\times H}$, $C \in \mathbb{C}^{H\times P}$ — and abandons the
convolutional view in favour of a **parallel associative scan** over the linear
recurrence

.. math::

    h_k = \bar A \odot h_{k-1} + \bar B u_k .

The scan operates on $(\bar A, \bar B u_k)$ pairs with the associative operator

.. math::

    (a_1, b_1) \bullet (a_2, b_2) = (a_2 \odot a_1,\; a_2 \odot b_1 + b_2),

so :func:`s5_parallel_scan` computes exactly the same states as the sequential
recurrence (:func:`s5_sequential_scan`) — pinned equal in the tests — but with
**critical-path depth $\lceil \log_2 L \rceil$** instead of $L$ (Exercise 8.6).
This is the same primitive Chapter 9 (Mamba/SSD) makes *input-dependent*; here
$\bar A$ is fixed (LTI), so the chapter holds selectivity and channel-mixing back.

Diagonal modes use the S4D-Lin init (§8.5) for a self-contained case study;
production S5 initializes from the HiPPO-N (normal) matrix. As with S4D the $P$
modes are complex and the real output is reconstructed by $2\,\mathrm{Re}$.

.. note::

    PyTorch has no native parallel ``associative_scan``; the torch companion
    (``companions/ch08/torch/s5_sequential.py``) is sequential-only. The
    $O(\log L)$ depth advantage is a JAX/parallel-hardware story.

Port credit
-----------
Associative-scan structure follows
``post_transformers/experiments/jax/week06/s5_scan.py`` and Smith et al.,
*Simplified State Space Layers for Sequence Modeling* (S5, arXiv:2208.04933).

Output
------
``public/figures/ch08/s5-scan-depth.png`` — parallel vs sequential critical-path
depth (left) and the state-agreement check (right).

Usage
-----
::

    PYTHONPATH=. python companions/ch08/jax/s5_scan.py
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402

from companions.ch08.jax.s4d_kernel import make_s4d_lin  # noqa: E402

if TYPE_CHECKING:
    from matplotlib.figure import Figure

__all__ = [
    "s5_binary_operator",
    "s5_parallel_scan",
    "s5_sequential_scan",
    "discretize_s5",
    "s5_apply",
]

_REPO_ROOT = Path(__file__).resolve().parents[3]
_OUT_DIR = _REPO_ROOT / "public" / "figures" / "ch08"


def s5_binary_operator(
    left: tuple[jnp.ndarray, jnp.ndarray],
    right: tuple[jnp.ndarray, jnp.ndarray],
) -> tuple[jnp.ndarray, jnp.ndarray]:
    r"""The associative scan operator $(a_1,b_1)\bullet(a_2,b_2) = (a_2 a_1,\, a_2 b_1 + b_2)$.

    ``left`` is the earlier segment, ``right`` the later one; the result composes
    "apply left, then right" on the affine map $h \mapsto a h + b$ (Exercise 8.6).
    """
    a_i, bu_i = left
    a_j, bu_j = right
    return a_j * a_i, a_j * bu_i + bu_j


def s5_parallel_scan(Abar: jnp.ndarray, Bu: jnp.ndarray) -> jnp.ndarray:
    r"""States $h_k$ via :func:`jax.lax.associative_scan` — depth $\lceil\log_2 L\rceil$.

    Parameters
    ----------
    Abar : jnp.ndarray, shape (P,), complex
        Diagonal discrete state matrix $\bar A$.
    Bu : jnp.ndarray, shape (L, P), complex
        Per-step driving terms $\bar B u_k$.

    Returns
    -------
    hs : jnp.ndarray, shape (L, P), complex
        Inclusive-prefix states $h_0, \ldots, h_{L-1}$ (from $h_{-1} = 0$).
    """
    L = Bu.shape[0]
    A_elems = jnp.broadcast_to(Abar, (L,) + Abar.shape)
    _, hs = jax.lax.associative_scan(s5_binary_operator, (A_elems, Bu))
    return hs


def s5_sequential_scan(Abar: jnp.ndarray, Bu: jnp.ndarray) -> jnp.ndarray:
    r"""The same states $h_k$ via a sequential :func:`jax.lax.scan` (the O(L) reference)."""

    def step(h: jnp.ndarray, bu_k: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
        h_new = Abar * h + bu_k
        return h_new, h_new

    h0 = jnp.zeros(Bu.shape[1], dtype=Bu.dtype)
    _, hs = jax.lax.scan(step, h0, Bu)
    return hs


def discretize_s5(
    A: jnp.ndarray,
    B: jnp.ndarray,
    dt: jnp.ndarray | float,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    r"""Diagonal ZOH: $\bar A = e^{A\Delta}$, $\bar B = \frac{\bar A - 1}{A}\,B$."""
    if jnp.any(A == 0):
        raise ValueError(
            "diagonal modes A must be nonzero (a zero mode has no ZOH input (Abar-1)/A)"
        )
    Abar = jnp.exp(A * jnp.asarray(dt, dtype=A.dtype))
    Bbar = ((Abar - 1.0) / A)[:, None] * B
    return Abar, Bbar


def s5_apply(
    A: jnp.ndarray,
    B: jnp.ndarray,
    C: jnp.ndarray,
    dt: jnp.ndarray | float,
    u: jnp.ndarray,
    parallel: bool = True,
) -> jnp.ndarray:
    r"""Run the diagonal MIMO S5 SSM on input ``u`` (shape ``(L, H)``).

    Parameters
    ----------
    A : jnp.ndarray, shape (P,), complex
        Diagonal modes.
    B : jnp.ndarray, shape (P, H), complex
        Input matrix.
    C : jnp.ndarray, shape (H, P), complex
        Output matrix.
    dt : jnp.ndarray or float
        Discretization step.
    u : jnp.ndarray, shape (L, H), real
        Input sequence.
    parallel : bool
        Use the associative scan (True) or the sequential reference (False).

    Returns
    -------
    y : jnp.ndarray, shape (L, H), real
        Output sequence $y_k = 2\,\mathrm{Re}(C h_k)$.
    """
    Abar, Bbar = discretize_s5(A, B, dt)
    Bu = jnp.asarray(u, dtype=Bbar.dtype) @ Bbar.T  # (L, P)
    hs = s5_parallel_scan(Abar, Bu) if parallel else s5_sequential_scan(Abar, Bu)
    return 2.0 * jnp.real(hs @ C.T)


# ---------------------------------------------------------------------------
# Figure: parallel-scan depth advantage + state agreement
# ---------------------------------------------------------------------------


def _demo_system(n_modes: int = 8, h_dim: int = 4, seed: int = 0):
    import numpy as np

    rng = np.random.default_rng(seed)
    A = make_s4d_lin(n_modes)
    B = jnp.asarray(
        rng.standard_normal((n_modes, h_dim)) + 1j * rng.standard_normal((n_modes, h_dim))
    )
    C = jnp.asarray(
        rng.standard_normal((h_dim, n_modes)) + 1j * rng.standard_normal((h_dim, n_modes))
    )
    return A, B, C


def make_scan_figure() -> Figure:
    """Left: critical-path depth (parallel vs sequential). Right: state agreement."""
    import numpy as np

    from companions._shared.plot_utils import (
        SSM_COLORS,
        apply_style,
        create_tufte_figure,
        set_tufte_labels,
        set_tufte_title,
    )

    apply_style()
    A, B, C = _demo_system()
    Abar, Bbar = discretize_s5(A, B, 0.1)

    # Left: algorithmic critical-path depth vs sequence length.
    Ls = np.array([2**k for k in range(4, 15)])
    seq_depth = Ls.astype(float)
    par_depth = np.ceil(np.log2(Ls))

    # Right: numerical agreement parallel vs sequential, across L.
    Ls2 = [2**k for k in range(4, 13)]
    diffs = []
    for L in Ls2:
        z = jnp.linspace(0.0, 1.0, L)
        Bu = jnp.broadcast_to(jnp.sin(2 * jnp.pi * 3 * z)[:, None], (L, A.shape[0])).astype(
            Abar.dtype
        )
        hp = s5_parallel_scan(Abar, Bu)
        hs = s5_sequential_scan(Abar, Bu)
        diffs.append(float(jnp.max(jnp.abs(hp - hs))))

    fig, (ax1, ax2) = create_tufte_figure(ncols=2, figsize=(11.0, 4.2))
    ax1.loglog(Ls, seq_depth, "o-", color=SSM_COLORS["alert"], label=r"sequential: depth $= L$")
    ax1.loglog(
        Ls,
        par_depth,
        "s-",
        color=SSM_COLORS["accent"],
        label=r"parallel: depth $= \lceil\log_2 L\rceil$",
    )
    set_tufte_title(ax1, "Critical-path depth")
    set_tufte_labels(ax1, xlabel=r"sequence length $L$", ylabel="parallel steps")
    ax1.legend(loc="upper left", fontsize=8, frameon=False)

    ax2.semilogy(Ls2, diffs, "o-", color=SSM_COLORS["highlight"])
    ax2.axhline(1e-12, color=SSM_COLORS["baseline"], linewidth=0.6, linestyle="--")
    set_tufte_title(ax2, "Parallel vs sequential states agree")
    set_tufte_labels(
        ax2,
        xlabel=r"sequence length $L$",
        ylabel=r"$\max_k |h_k^{\mathrm{par}} - h_k^{\mathrm{seq}}|$",
    )
    fig.tight_layout()
    return fig


def main() -> None:
    import matplotlib.pyplot as plt
    import numpy as np

    from companions._shared.plot_utils import save_figure

    print("Chapter 8 — s5_scan.py")
    print("=" * 60)

    A, B, C = _demo_system()
    L = 256
    z = jnp.linspace(0.0, 1.0, L)
    u = jnp.stack([jnp.sin(2 * jnp.pi * (k + 1) * z) for k in range(B.shape[1])], axis=1)
    y_par = s5_apply(A, B, C, 0.1, u, parallel=True)
    y_seq = s5_apply(A, B, C, 0.1, u, parallel=False)
    agree = float(jnp.max(jnp.abs(y_par - y_seq)))
    print(f"  L={L}: max|y_parallel - y_sequential| = {agree:.3e}  (§8.6 scan-equivalence: ~0)")
    print(f"  parallel depth ceil(log2 {L}) = {int(np.ceil(np.log2(L)))} vs sequential {L}")

    _OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig = make_scan_figure()
    for p in save_figure(fig, _OUT_DIR / "s5-scan-depth", formats=("png",)):
        print(f"Wrote {p}")
    plt.close(fig)


if __name__ == "__main__":
    main()
