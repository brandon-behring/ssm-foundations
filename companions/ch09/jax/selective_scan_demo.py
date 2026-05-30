r"""Chapter 9 §9.3-9.4 — the selective scan is the only route, and why it needs a fused kernel.

§9.3. Because the transition $\bar A_t$ now varies per step (§9.2), the
convolution$\to$FFT path of Chapter 8 is gone: there is no single kernel to
transform. What survives is the §8.6 associative scan, fed the time-varying
pairs $(\bar A_t, \bar B_t u_t)$. This module pins that the selective associative
scan equals the sequential recurrence to machine precision across sequence
lengths, and that it retains the $\lceil\log_2 L\rceil$ critical-path depth — the
only parallel route left.

§9.4. The selective scan's state is $h_t \in \R^N$ *per channel*; materializing
every state for the backward pass costs $O(B\,L\,D\,N)$, an $N\times$ blow-up
over the $O(B\,L\,D)$ inputs/outputs. Mamba's fused kernel keeps the expanded
state in SRAM and **recomputes** it in the backward pass instead of writing it
to HBM (the same ``jax.checkpoint`` trick the Week-7 source applies to the
scan). This module plots the memory the recomputation avoids.

Outputs
-------
``public/figures/ch09/selective-scan.png`` — scan vs sequential agreement (left)
and parallel-vs-sequential depth (right).
``public/figures/ch09/memory-cost.png`` — materialized $B L D N$ vs fused $B L D$
state memory, and the $N\times$ ratio.

Usage
-----
::

    PYTHONPATH=. python companions/ch09/jax/selective_scan_demo.py
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402

from companions.ch09.jax.selective_ssm import (  # noqa: E402
    discretize_selective,
    selective_scan,
    selective_sequential,
    stable_A,
)

if TYPE_CHECKING:
    from matplotlib.figure import Figure

__all__ = ["scan_vs_sequential_residual", "materialized_vs_fused_bytes", "make_scan_figure"]

_REPO_ROOT = Path(__file__).resolve().parents[3]
_OUT_DIR = _REPO_ROOT / "public" / "figures" / "ch09"


def _time_varying_pairs(length: int, n: int, seed: int = 0):
    """Fixed reproducible time-varying (Abar, Bu) pairs of shape (L, N)."""
    import numpy as np

    rng = np.random.default_rng(seed)
    A = stable_A(jnp.asarray(rng.standard_normal(n)))
    delta = jax.nn.softplus(jnp.asarray(rng.standard_normal(length)))  # positive, varying
    B = jnp.asarray(rng.standard_normal((length, n)))
    u = jnp.asarray(rng.standard_normal(length))
    Abar, Bbar = discretize_selective(A, delta, B)
    return Abar, Bbar * u[:, None]


def scan_vs_sequential_residual(length: int, n: int = 8, seed: int = 0) -> float:
    r"""max$_k |h_k^\text{scan} - h_k^\text{seq}|$ for the time-varying selective scan."""
    Abar, Bu = _time_varying_pairs(length, n, seed)
    h_par = selective_scan(Abar, Bu)
    h_seq = selective_sequential(Abar, Bu)
    return float(jnp.max(jnp.abs(h_par - h_seq)))


def materialized_vs_fused_bytes(
    batch: int, length: int, d_inner: int, n_state: int, bytes_per: int = 4
) -> tuple[int, int]:
    r"""State memory: materialized $B L D N$ vs fused $B L D$ (elements $\times$ bytes).

    Materialized stores every per-step state $(B, L, D, N)$ for the backward pass;
    the fused/recompute kernel keeps only the $(B, L, D)$ outputs (plus a small
    $(B, D, N)$ running state, dropped here as lower order). The ratio is $N$.
    """
    materialized = batch * length * d_inner * n_state * bytes_per
    fused = batch * length * d_inner * bytes_per
    return materialized, fused


def make_scan_figure() -> Figure:
    """Left: scan == sequential across L. Right: parallel vs sequential depth."""
    import numpy as np

    from companions._shared.plot_utils import (
        SSM_COLORS,
        apply_style,
        create_tufte_figure,
        set_tufte_labels,
        set_tufte_title,
    )

    apply_style()
    lengths = [2**k for k in range(4, 13)]
    diffs = [scan_vs_sequential_residual(length) for length in lengths]
    seq_depth = np.array(lengths, dtype=float)
    par_depth = np.ceil(np.log2(lengths))

    fig, (ax1, ax2) = create_tufte_figure(ncols=2, figsize=(11.0, 4.2))

    ax1.semilogy(lengths, diffs, "o-", color=SSM_COLORS["highlight"])
    ax1.axhline(1e-12, color=SSM_COLORS["baseline"], linewidth=0.6, linestyle="--")
    ax1.set_xscale("log", base=2)
    set_tufte_title(ax1, "Selective scan == sequential")
    set_tufte_labels(
        ax1,
        xlabel=r"sequence length $L$",
        ylabel=r"$\max_k |h_k^{\mathrm{scan}} - h_k^{\mathrm{seq}}|$",
    )

    ax2.loglog(lengths, seq_depth, "o-", color=SSM_COLORS["alert"], label=r"sequential: $L$")
    ax2.loglog(lengths, par_depth, "s-", color=SSM_COLORS["accent"],
               label=r"selective scan: $\lceil\log_2 L\rceil$")
    set_tufte_title(ax2, "Depth: the only parallel route left")
    set_tufte_labels(ax2, xlabel=r"sequence length $L$", ylabel="parallel steps")
    ax2.legend(loc="upper left", fontsize=8, frameon=False)

    fig.tight_layout()
    return fig


def make_memory_figure() -> Figure:
    """Left: materialized vs fused state memory vs N. Right: the N-times ratio."""
    import numpy as np

    from companions._shared.plot_utils import (
        SSM_COLORS,
        apply_style,
        create_tufte_figure,
        set_tufte_labels,
        set_tufte_title,
    )

    apply_style()
    # Representative Mamba dims: batch 8, length 2048, d_inner 1024.
    batch, length, d_inner = 8, 2048, 1024
    n_states = np.array([2**k for k in range(0, 8)])  # 1 .. 128
    mats, fused = [], []
    for n in n_states:
        m, f = materialized_vs_fused_bytes(batch, length, d_inner, int(n))
        mats.append(m)
        fused.append(f)
    mats = np.array(mats, dtype=float)
    fused = np.array(fused, dtype=float)

    fig, (ax1, ax2) = create_tufte_figure(ncols=2, figsize=(11.0, 4.2))

    ax1.loglog(n_states, mats / 1e9, "o-", color=SSM_COLORS["alert"],
               label=r"materialized $B L D N$")
    ax1.loglog(n_states, fused / 1e9, "s-", color=SSM_COLORS["accent"],
               label=r"fused $B L D$")
    ax1.set_xscale("log", base=2)
    set_tufte_title(ax1, "State memory ($B{=}8, L{=}2048, D{=}1024$)")
    set_tufte_labels(ax1, xlabel="state dim $N$", ylabel="memory (GB, fp32)")
    ax1.legend(loc="upper left", fontsize=8, frameon=False)

    ax2.semilogx(n_states, mats / fused, "o-", color=SSM_COLORS["highlight"])
    ax2.set_xscale("log", base=2)
    set_tufte_title(ax2, r"Recomputation saves an $N\times$ factor")
    set_tufte_labels(ax2, xlabel="state dim $N$", ylabel="materialized / fused")

    fig.tight_layout()
    return fig


def main() -> None:
    import matplotlib.pyplot as plt

    from companions._shared.plot_utils import save_figure

    print("Chapter 9 — selective_scan_demo.py")
    print("=" * 60)

    for length in (256, 1024, 4096):
        resid = scan_vs_sequential_residual(length)
        print(f"  L={length}: max|h_scan - h_seq| = {resid:.3e}  (§9.3 equivalence: ~0)")

    batch, length, d_inner, n_state = 8, 2048, 1024, 16
    mat, fused = materialized_vs_fused_bytes(batch, length, d_inner, n_state)
    print(
        f"  memory @ (B={batch}, L={length}, D={d_inner}, N={n_state}): "
        f"materialized {mat / 1e9:.1f} GB vs fused {fused / 1e9:.3f} GB "
        f"= {mat // fused}x  (§9.4 recompute saving)"
    )

    _OUT_DIR.mkdir(parents=True, exist_ok=True)
    for name, fig in (("selective-scan", make_scan_figure()), ("memory-cost", make_memory_figure())):
        for p in save_figure(fig, _OUT_DIR / name, formats=("png",)):
            print(f"Wrote {p}")
        plt.close(fig)


if __name__ == "__main__":
    main()
