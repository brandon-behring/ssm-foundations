r"""Chapter 8 §8.5 — S4D: the diagonal restriction of S4.

S4D drops S4's low-rank correction (§8.2) and keeps only a *diagonal* state
matrix $A = \mathrm{diag}(A_1,\ldots,A_M)$. The S4 Cauchy kernel then collapses to
a plain Vandermonde sum:

.. math::

    K_l = 2\,\mathrm{Re}\!\Big( \sum_{n} C_n\,\frac{\bar A_n - 1}{A_n}\,\bar A_n^{\,l} \Big),
    \qquad \bar A_n = e^{A_n \Delta}.

**S4D-Lin** initialization places the modes at $A_n = -\tfrac12 + i\pi n$. The
parameterization $A = -e^{\texttt{log\_A\_real}} + i\,\texttt{A\_imag}$ enforces
$\mathrm{Re}(A) < 0$ *by construction* — the real part is a negative exponential,
so no mode can leave the unit disk under ZOH no matter how training moves the
parameters. This is the structural cure for the Chapter 7 §7.5 conditioning
danger: S4 *hopes* the diagonalized basis stays well-conditioned; S4D *enforces*
stability and sidesteps the basis entirely.

The $M = N/2$ modes are complex; the factor $2\,\mathrm{Re}$ reconstructs the $N$
real states from $M$ conjugate pairs. Complex128 throughout (matching the book's
float64): the kernel is genuinely complex and the §7.5 story needs the precision.

Port credit
-----------
Vandermonde kernel and S4D-Lin init follow
``post_transformers/experiments/refs/s4/models/s4/s4d.py`` (Gu et al., *On the
Parameterization and Initialization of Diagonal State Space Models* — S4D,
arXiv:2206.11893), reduced to a functional JAX core (no ``nn.Module``, SISO).

Output
------
``public/figures/ch08/s4d-spectrum.png`` — S4D-Lin modes $-\tfrac12 + i\pi n$ in
the complex plane beside the full HiPPO-LegS spectrum $-1,\ldots,-N$.

Usage
-----
::

    PYTHONPATH=. python companions/ch08/jax/s4d_kernel.py
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402

if TYPE_CHECKING:
    from matplotlib.figure import Figure

__all__ = ["s4d_lin_params", "assemble_diagonal", "make_s4d_lin", "s4d_kernel"]

_REPO_ROOT = Path(__file__).resolve().parents[3]
_OUT_DIR = _REPO_ROOT / "public" / "figures" / "ch08"


def s4d_lin_params(n_modes: int) -> tuple[jnp.ndarray, jnp.ndarray]:
    r"""S4D-Lin parameters: ``log_A_real = log(1/2)`` and ``A_imag = pi*n``.

    Stored in the trainable parameterization (real part as a *log* magnitude) so
    that :func:`assemble_diagonal` yields $\mathrm{Re}(A) < 0$ for any real value
    the parameters take.

    Parameters
    ----------
    n_modes : int
        Number of complex modes $M = N/2$; must be >= 1.

    Returns
    -------
    log_A_real : jnp.ndarray, shape (n_modes,)
        ``log(0.5)`` repeated — so ``-exp(log_A_real) = -1/2``.
    A_imag : jnp.ndarray, shape (n_modes,)
        ``pi * [0, 1, ..., n_modes-1]``.

    Raises
    ------
    ValueError
        If ``n_modes < 1``.
    """
    if n_modes < 1:
        raise ValueError(f"n_modes must be >= 1, got {n_modes}")
    log_A_real = jnp.log(0.5 * jnp.ones(n_modes, dtype=jnp.float64))
    A_imag = jnp.pi * jnp.arange(n_modes, dtype=jnp.float64)
    return log_A_real, A_imag


def assemble_diagonal(log_A_real: jnp.ndarray, A_imag: jnp.ndarray) -> jnp.ndarray:
    r"""Assemble the complex diagonal $A = -e^{\texttt{log\_A\_real}} + i\,\texttt{A\_imag}$.

    The negative-exponential real part guarantees $\mathrm{Re}(A) < 0$ (the
    by-construction stability of §8.5), independent of the parameter values.
    """
    return -jnp.exp(log_A_real) + 1j * A_imag


def make_s4d_lin(n_modes: int) -> jnp.ndarray:
    r"""Convenience: the S4D-Lin diagonal $A_n = -\tfrac12 + i\pi n$ (shape ``(n_modes,)``)."""
    return assemble_diagonal(*s4d_lin_params(n_modes))


def s4d_kernel(
    A: jnp.ndarray,
    C: jnp.ndarray,
    dt: jnp.ndarray | float,
    L: int,
) -> jnp.ndarray:
    r"""S4D Vandermonde convolution kernel (real, length ``L``).

    .. math::

        K_l = 2\,\mathrm{Re}\Big( \sum_n C_n \frac{\bar A_n - 1}{A_n} \bar A_n^{\,l} \Big),
        \qquad \bar A_n = e^{A_n \Delta}.

    The input matrix $B_n = 1$ is folded into $\bar B_n = (\bar A_n - 1)/A_n$
    (the diagonal ZOH), so only $A$, $C$ are passed.

    Parameters
    ----------
    A : jnp.ndarray, shape (M,), complex
        Diagonal modes (e.g. from :func:`make_s4d_lin`).
    C : jnp.ndarray, shape (M,), complex
        Output weights, one per mode.
    dt : jnp.ndarray or float
        Discretization step $\Delta$.
    L : int
        Number of kernel taps.

    Returns
    -------
    K : jnp.ndarray, shape (L,), real
        The S4D convolution kernel.

    Raises
    ------
    ValueError
        If ``A`` and ``C`` differ in length, or ``L < 1``.
    """
    if A.shape != C.shape:
        raise ValueError(f"A and C must share shape, got {A.shape} vs {C.shape}")
    if L < 1:
        raise ValueError(f"L must be >= 1, got {L}")
    dtA = A * jnp.asarray(dt, dtype=A.dtype)
    Abar = jnp.exp(dtA)
    Ctilde = C * (Abar - 1.0) / A
    powers = jnp.exp(dtA[:, None] * jnp.arange(L)[None, :])  # (M, L) = Abar^l
    return 2.0 * jnp.real(jnp.sum(Ctilde[:, None] * powers, axis=0))


# ---------------------------------------------------------------------------
# Figure: S4D-Lin spectrum vs the full HiPPO-LegS spectrum
# ---------------------------------------------------------------------------


def make_spectrum_figure(n_modes: int = 16) -> Figure:
    """Scatter the S4D-Lin modes beside the HiPPO-LegS eigenvalues in $\\mathbb{C}$."""
    import numpy as np

    from companions._shared.plot_utils import (
        SSM_COLORS,
        apply_style,
        create_tufte_figure,
        set_tufte_labels,
        set_tufte_title,
    )
    from companions.ch08.jax.s4_core import make_hippo_legs

    apply_style()
    s4d = np.asarray(make_s4d_lin(n_modes))
    hippo = np.linalg.eigvals(np.asarray(make_hippo_legs(2 * n_modes)[0]))

    fig, ax = create_tufte_figure(figsize=(6.6, 5.0))
    ax.axvline(0.0, color=SSM_COLORS["baseline"], linewidth=0.6)
    ax.axhline(0.0, color=SSM_COLORS["baseline"], linewidth=0.6)
    ax.scatter(
        hippo.real,
        hippo.imag,
        s=55,
        color=SSM_COLORS["baseline"],
        edgecolors="white",
        linewidths=0.8,
        zorder=2,
        label=rf"HiPPO-LegS spectrum ($N={2 * n_modes}$): $-1,\ldots,-N$",
    )
    ax.scatter(
        s4d.real,
        s4d.imag,
        s=70,
        color=SSM_COLORS["accent"],
        edgecolors="white",
        linewidths=1.0,
        zorder=3,
        label=rf"S4D-Lin modes ($M={n_modes}$): $-0.5 + i\pi n$",
    )
    ax.set_xlim(-(2 * n_modes + 1.0), 1.5)
    set_tufte_title(ax, "S4D-Lin diagonal init vs the full HiPPO spectrum")
    set_tufte_labels(
        ax, xlabel=r"$\operatorname{Re}(\lambda)$", ylabel=r"$\operatorname{Im}(\lambda)$"
    )
    ax.legend(loc="upper left", fontsize=8, frameon=False)
    fig.tight_layout()
    return fig


def main() -> None:
    import matplotlib.pyplot as plt
    import numpy as np

    from companions._shared.plot_utils import save_figure

    print("Chapter 8 — s4d_kernel.py")
    print("=" * 60)

    n_modes = 16
    A = make_s4d_lin(n_modes)
    print(f"  S4D-Lin: max Re(A) = {float(np.max(A.real)):.4f}  (< 0 by construction)")
    rng = np.random.default_rng(0)
    C = jnp.asarray(rng.standard_normal(n_modes) + 1j * rng.standard_normal(n_modes))
    K = s4d_kernel(A, C, 0.1, 64)
    print(
        f"  kernel: K[0] = {float(K[0]):.4f}, max|K| = {float(np.max(np.abs(K))):.4f}, "
        f"K real dtype = {K.dtype}"
    )

    _OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig = make_spectrum_figure(n_modes=16)
    for p in save_figure(fig, _OUT_DIR / "s4d-spectrum", formats=("png",)):
        print(f"Wrote {p}")
    plt.close(fig)


if __name__ == "__main__":
    main()
