r"""Chapter 8 §8.4 — the naive kernel is $O(N^2 L)$ (why S4 needs the Cauchy trick).

Times :func:`s4_core.ssm_kernel_naive` across state dimensions $N$ at fixed
sequence length $L$. The naive kernel iterates $\bar A^k$ (an $N\times N$ matmul
per tap), so its cost grows like $N^2 L$ — the wall-clock tracks an $N^2$
reference line. S4's contribution (§8.4) is computing the *same* kernel in
$O(N \log^2 N)$ via the Cauchy/Woodbury structure of the DPLR matrix; that curve
is drawn for contrast but **not implemented** — this book's companions
deliberately use the transparent naive kernel (good for $N \le 64$).

Output
------
``public/figures/ch08/kernel-vs-n.png`` — measured kernel cost vs $N$, with
$O(N^2)$ and $O(N\log^2 N)$ reference curves.

Usage
-----
::

    PYTHONPATH=. python companions/ch08/jax/s4_kernel_cost.py
"""

from __future__ import annotations

import time
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
    ssm_kernel_naive,
)

_REPO_ROOT = Path(__file__).resolve().parents[3]
_OUT_DIR = _REPO_ROOT / "public" / "figures" / "ch08"


def time_kernel(n: int, L: int, repeats: int = 7) -> float:
    """Median wall-clock (seconds) of one ``ssm_kernel_naive`` call at state dim ``n``."""
    import numpy as np

    A, B = make_hippo_legs(n)
    Ab, Bb = discretize_zoh(A, B, 0.1)
    C = jnp.ones((1, n))
    kfn = jax.jit(lambda: ssm_kernel_naive(Ab, Bb, C, L))
    kfn().block_until_ready()  # warmup + compile (excluded from timing)
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        kfn().block_until_ready()
        times.append(time.perf_counter() - t0)
    return float(np.median(times))


def make_cost_figure(L: int = 512) -> Figure:
    import numpy as np

    from companions._shared.plot_utils import (
        SSM_COLORS,
        apply_style,
        create_tufte_figure,
        set_tufte_labels,
        set_tufte_title,
    )

    apply_style()
    Ns = np.array([8, 16, 32, 64, 128, 256])
    measured = np.array([time_kernel(int(n), L) for n in Ns])

    # Reference curves anchored to the measured time at the largest N.
    anchor = measured[-1]
    n2_ref = anchor * (Ns / Ns[-1]) ** 2
    nlog2_ref = anchor * (Ns * np.log2(Ns) ** 2) / (Ns[-1] * np.log2(Ns[-1]) ** 2)

    fig, ax = create_tufte_figure(figsize=(6.6, 4.6))
    ax.loglog(
        Ns, n2_ref, "--", color=SSM_COLORS["alert"], linewidth=1.2, label=r"$O(N^2 L)$ reference"
    )
    ax.loglog(
        Ns,
        nlog2_ref,
        ":",
        color=SSM_COLORS["baseline"],
        linewidth=1.2,
        label=r"$O(N \log^2 N)$ (Cauchy, not implemented)",
    )
    ax.loglog(
        Ns,
        measured,
        "o-",
        color=SSM_COLORS["accent"],
        linewidth=1.6,
        label="naive kernel (measured)",
    )
    set_tufte_title(ax, rf"Naive kernel cost grows as $N^2$  ($L={L}$)")
    set_tufte_labels(ax, xlabel=r"state dimension $N$", ylabel="wall-clock per kernel (s)")
    ax.legend(loc="upper left", fontsize=8, frameon=False)
    fig.tight_layout()
    return fig


def main() -> None:
    import matplotlib.pyplot as plt

    from companions._shared.plot_utils import save_figure

    print("Chapter 8 — s4_kernel_cost.py")
    print("=" * 60)
    L = 512
    for n in (16, 64, 256):
        t = time_kernel(n, L)
        print(f"  N={n:3d}, L={L}: {t * 1e3:.3f} ms/kernel")

    _OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig = make_cost_figure(L=L)
    for p in save_figure(fig, _OUT_DIR / "kernel-vs-n", formats=("png",)):
        print(f"Wrote {p}")
    plt.close(fig)


if __name__ == "__main__":
    main()
