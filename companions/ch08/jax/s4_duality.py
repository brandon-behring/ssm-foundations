r"""Chapter 8 §8.3 — the convolution<->recurrence duality, visualized.

Builds an S4 system (HiPPO-LegS $A$, ZOH discretization), computes its convolution
kernel $K_k = C\bar A^k\bar B$, then runs the SSM both ways: the recurrent
``lax.scan`` (O(L) sequential) and the FFT convolution (O(L log L) parallel).
For zero initial state the two outputs are identical — the figure overlays them
and reports the residual (~$10^{-15}$ in float64).

Output
------
``public/figures/ch08/kernel-duality.png`` — left: the decaying kernel taps;
right: recurrent vs convolutional outputs overlaid.

Usage
-----
::

    PYTHONPATH=. python companions/ch08/jax/s4_duality.py
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import jax

if TYPE_CHECKING:
    from matplotlib.figure import Figure

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402

from companions.ch08.jax.s4_core import (  # noqa: E402
    discretize_zoh,
    make_hippo_legs,
    ssm_convolutional,
    ssm_kernel_naive,
    ssm_recurrent,
)

_REPO_ROOT = Path(__file__).resolve().parents[3]
_OUT_DIR = _REPO_ROOT / "public" / "figures" / "ch08"


def _demo(n: int = 16, dt: float = 0.1, L: int = 128):
    import numpy as np

    A, B = make_hippo_legs(n)
    Ab, Bb = discretize_zoh(A, B, dt)
    rng = np.random.default_rng(0)
    C = jnp.asarray(rng.standard_normal((1, n)))
    D = jnp.asarray(0.0)
    z = jnp.linspace(0.0, 1.0, L)
    u = jnp.sin(2.0 * jnp.pi * 3.0 * z) + 0.5 * jnp.cos(2.0 * jnp.pi * 7.0 * z)
    K = ssm_kernel_naive(Ab, Bb, C, L)
    y_rec = ssm_recurrent(Ab, Bb, C, D, u)
    y_conv = ssm_convolutional(K, D, u)
    return K, u, y_rec, y_conv


def make_duality_figure() -> Figure:
    import numpy as np

    from companions._shared.plot_utils import (
        SSM_COLORS,
        apply_style,
        create_tufte_figure,
        set_tufte_labels,
        set_tufte_title,
    )

    apply_style()
    K, _u, y_rec, y_conv = _demo()
    K, y_rec, y_conv = np.asarray(K), np.asarray(y_rec), np.asarray(y_conv)
    residual = float(np.max(np.abs(y_rec - y_conv)))
    taps = np.arange(K.shape[0])

    fig, (ax1, ax2) = create_tufte_figure(ncols=2, figsize=(11.0, 4.2))
    ax1.plot(taps, K, color=SSM_COLORS["accent"], linewidth=1.4)
    ax1.fill_between(taps, K, color=SSM_COLORS["accent"], alpha=0.12)
    set_tufte_title(ax1, r"SSM kernel $K_k = C\bar A^k \bar B$")
    set_tufte_labels(ax1, xlabel=r"tap $k$", ylabel=r"$K_k$")

    t = np.arange(y_rec.shape[0])
    ax2.plot(t, y_rec, color=SSM_COLORS["accent"], linewidth=2.4, label="recurrent (lax.scan)")
    ax2.plot(
        t,
        y_conv,
        color=SSM_COLORS["highlight"],
        linewidth=1.0,
        linestyle="--",
        label="convolutional (FFT)",
    )
    set_tufte_title(ax2, rf"Two views, one output  (max residual $= {residual:.1e}$)")
    set_tufte_labels(ax2, xlabel=r"time $k$", ylabel=r"$y_k$")
    ax2.legend(loc="upper right", fontsize=8, frameon=False)
    fig.tight_layout()
    return fig


def main() -> None:
    import matplotlib.pyplot as plt

    from companions._shared.plot_utils import save_figure

    print("Chapter 8 — s4_duality.py")
    print("=" * 60)
    K, _u, y_rec, y_conv = _demo()
    residual = float(jnp.max(jnp.abs(y_rec - y_conv)))
    print(f"  duality residual max|y_rec - y_conv| = {residual:.3e}  (§8.3: ~0)")
    print(f"  kernel decay: K[0]={float(K[0]):.4f} -> K[-1]={float(K[-1]):.2e}")

    _OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig = make_duality_figure()
    for p in save_figure(fig, _OUT_DIR / "kernel-duality", formats=("png",)):
        print(f"Wrote {p}")
    plt.close(fig)


if __name__ == "__main__":
    main()
