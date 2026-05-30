r"""Chapter 10 §10.4 — complex state: decay and oscillation in one head.

Mamba-1/2 (Chapter 9) use a *real* diagonal state matrix $A < 0$, so each mode
can only decay. Mamba-3's second change (alongside the integrator of
``discretization.py``) is a **complex** diagonal mode $A = \log\rho + i\theta$,
whose discrete transition

.. math::

    \alpha = e^{A\Delta} = \underbrace{e^{\Delta\log\rho}}_{\text{decay } \rho^{\Delta}}
             \cdot\; \underbrace{e^{i\theta\Delta}}_{\text{rotation}}

both *decays* (magnitude $|\alpha| = e^{\Delta\operatorname{Re}(A)} < 1$) and
*rotates* (phase advances $\theta\Delta$ per step). A single head is now a damped
oscillator — the spiral phase portrait of Chapters 1-2, not the pure exponential
decay of a real mode. This is what lets one head represent an oscillatory feature
(a frequency) instead of only a forgetting timescale.

The RoPE equivalence (the production trick)
-------------------------------------------
Mamba-3 never instantiates a complex dtype. It keeps a *real* 2-D state and
applies **rotary position embeddings (RoPE)** — i.e. 2-D rotations — to the input
and output projections $B, C$. The justification is the elementary isomorphism
between the complex numbers and $2\times 2$ rotation-scalings: representing
$z = a + bi$ as the real vector $[a, b]^\top$,

.. math::

    (\rho e^{i\theta})\, z \;\longleftrightarrow\;
    \rho \begin{pmatrix}\cos\theta & -\sin\theta\\ \sin\theta & \cos\theta\end{pmatrix}
    \begin{pmatrix}a\\ b\end{pmatrix}.

So the complex-scalar recurrence $x_k = \alpha x_{k-1} + b_k$ and the real 2-D
recurrence $s_k = \rho R(\theta)\, s_{k-1} + [\operatorname{Re} b_k,
\operatorname{Im} b_k]^\top$ produce identical trajectories. *Theorem
``ch10:rope-complex-equivalence``* (proved in §10.9 Exercise 10.5); pinned to
machine precision in ``tests/test_complex_state.py`` via
:func:`rope_equivalence_residual`.

Connection to Chapter 8
-----------------------
S4D (§8.5) already used *complex* modes — but it was linear time-*invariant*. The
novelty here is **complex + selective** (input-dependent $\Delta_t, B_t, C_t$):
Mamba-3 is, in the dynamical-systems reading, the synthesis of Chapter 8's complex
state and Chapter 9's selectivity.

Idiomatic-JAX note
------------------
:func:`complex_scalar_recurrence` is written as a closed form $x_k = \alpha^k x_0$
(homogeneous) rather than a Python loop — JAX broadcasts the power over the step
index in one expression. The driven recurrences in :func:`rope_equivalence_residual`
use ``lax.scan`` (the sequential primitive of Chapters 8-9), since a driven
time-varying recurrence has no closed form.

Port credit
-----------
``complex_scalar_recurrence`` is ported (float64, idiomatic) from
``post_transformers/experiments/jax/week09/mamba3.py``; the RoPE/complex
equivalence demonstration is added here for §10.4. Mamba-3: Lahoti et al.,
arXiv:2603.15569.

Usage
-----
::

    PYTHONPATH=. python companions/ch10/jax/complex_state.py
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import TYPE_CHECKING

import jax

# Complex modes need complex128; enable float64 before any jnp array is created.
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402

if TYPE_CHECKING:
    from matplotlib.figure import Figure

__all__ = [
    "complex_scalar_recurrence",
    "rope_matrix",
    "complex_to_real2",
    "rope_equivalence_residual",
    "decay_rate",
]

_REPO_ROOT = Path(__file__).resolve().parents[3]
_OUT_DIR = _REPO_ROOT / "public" / "figures" / "ch10"


def complex_scalar_recurrence(
    rho: float, theta: float, n_steps: int, x0: complex = 1.0 + 0.0j
) -> jnp.ndarray:
    r"""Homogeneous complex-scalar trajectory $x_k = (\rho e^{i\theta})^k x_0$.

    A single Mamba-3 complex mode with magnitude $\rho \in (0, 1]$ and angular
    frequency $\theta$ per step. Plotted in $(\operatorname{Re} x,
    \operatorname{Im} x)$ this is a logarithmic spiral with decay rate
    $\log\rho$ per step (the discrete analog of a damped oscillator).

    Parameters
    ----------
    rho : float
        Per-step magnitude. Must be in $(0, 1]$ for a non-growing mode.
    theta : float
        Angular frequency per step (radians).
    n_steps : int
        Number of steps; returns ``n_steps + 1`` points including $x_0$.
    x0 : complex, default 1+0j
        Initial state.

    Returns
    -------
    xs : jnp.ndarray, shape (n_steps + 1,), complex128
        Trajectory $x_0, x_1, \ldots, x_{n\_steps}$.

    Raises
    ------
    ValueError
        If ``rho`` is outside $(0, 1]$ or ``n_steps`` is negative.
    """
    if not 0.0 < rho <= 1.0:
        raise ValueError(f"rho must be in (0, 1], got {rho}")
    if n_steps < 0:
        raise ValueError(f"n_steps must be non-negative, got {n_steps}")
    alpha = rho * jnp.exp(1j * jnp.asarray(theta, dtype=jnp.float64))
    ks = jnp.arange(n_steps + 1)
    return jnp.asarray(x0, dtype=jnp.complex128) * alpha**ks


def rope_matrix(theta: float) -> jnp.ndarray:
    r"""The $2\times2$ rotation matrix $R(\theta)$ — the real form of $e^{i\theta}$.

    .. math::

        R(\theta) = \begin{pmatrix}\cos\theta & -\sin\theta\\
                                    \sin\theta & \cos\theta\end{pmatrix}.

    Applying $R(\theta)$ to $[a, b]^\top$ equals multiplying $a + bi$ by
    $e^{i\theta}$ — the algebraic identity Mamba-3 uses to realize a complex state
    with real-valued RoPE rotations.

    Parameters
    ----------
    theta : float
        Rotation angle (radians).

    Returns
    -------
    R : jnp.ndarray, shape (2, 2), float64.
    """
    c = jnp.cos(jnp.asarray(theta, dtype=jnp.float64))
    s = jnp.sin(jnp.asarray(theta, dtype=jnp.float64))
    return jnp.asarray([[c, -s], [s, c]])


def complex_to_real2(z: jnp.ndarray) -> jnp.ndarray:
    r"""Represent complex ``z`` as the real 2-vector stack $[\operatorname{Re} z,
    \operatorname{Im} z]$ along a new last axis.

    Parameters
    ----------
    z : jnp.ndarray
        Complex array of any shape.

    Returns
    -------
    s : jnp.ndarray, shape ``z.shape + (2,)``, float64.
    """
    return jnp.stack([jnp.real(z), jnp.imag(z)], axis=-1)


def rope_equivalence_residual(
    rho: float, theta: float, drive: jnp.ndarray
) -> float:
    r"""Max $|x^{\mathrm{complex}}_k - (s^{\mathrm{real}}_k[0] + i s^{\mathrm{real}}_k[1])|$.

    Runs the driven recurrence two ways and returns their maximum discrepancy:

    * **complex** : $x_k = (\rho e^{i\theta}) x_{k-1} + d_k$ in complex128;
    * **real 2-D RoPE** : $s_k = \rho R(\theta) s_{k-1} + [\operatorname{Re} d_k,
      \operatorname{Im} d_k]^\top$ in float64.

    The two are algebraically identical (`ch10:rope-complex-equivalence`), so the
    residual is at machine precision — the proof that RoPE realizes the complex
    state without a complex dtype.

    Parameters
    ----------
    rho : float
        Mode magnitude in $(0, 1]$.
    theta : float
        Angular frequency per step.
    drive : jnp.ndarray, shape (L,), complex
        Per-step complex driving term $d_k$.

    Returns
    -------
    residual : float
    """
    if not 0.0 < rho <= 1.0:
        raise ValueError(f"rho must be in (0, 1], got {rho}")
    if drive.ndim != 1:
        raise ValueError(f"drive must be 1-D (L,), got shape {drive.shape}")

    alpha = rho * jnp.exp(1j * jnp.asarray(theta, dtype=jnp.float64))
    drive = jnp.asarray(drive, dtype=jnp.complex128)

    def cstep(x, d):
        x_new = alpha * x + d
        return x_new, x_new

    _, xs_complex = jax.lax.scan(cstep, jnp.asarray(0.0 + 0.0j), drive)

    R = rho * rope_matrix(theta)  # (2, 2)
    drive_real = complex_to_real2(drive)  # (L, 2)

    def rstep(s, d2):
        s_new = R @ s + d2
        return s_new, s_new

    _, ss_real = jax.lax.scan(rstep, jnp.zeros(2), drive_real)  # (L, 2)

    recombined = ss_real[:, 0] + 1j * ss_real[:, 1]
    return float(jnp.max(jnp.abs(xs_complex - recombined)))


def decay_rate(xs: jnp.ndarray) -> float:
    r"""Empirical per-step decay rate $\log|x_{k+1}| - \log|x_k|$ (should equal $\log\rho$).

    Averaged over the trajectory (constant for a homogeneous mode). Used to pin
    the spiral's decay against the analytic $\log\rho$.

    Parameters
    ----------
    xs : jnp.ndarray, shape (L,), complex
        A homogeneous trajectory from :func:`complex_scalar_recurrence`.

    Returns
    -------
    rate : float
        Mean log-magnitude increment per step.
    """
    mags = jnp.abs(xs)
    if jnp.any(mags <= 0):
        raise ValueError("trajectory contains a zero/negative magnitude; cannot take log")
    log_mag = jnp.log(mags)
    return float(jnp.mean(jnp.diff(log_mag)))


# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------

_RHO = 0.95
_THETA = math.pi / 9.0  # 20 degrees per step
_N = 60


def make_spiral_figure() -> Figure:
    """The damped-oscillator spiral of one complex mode in the (Re, Im) plane."""
    import numpy as np

    from companions._shared.plot_utils import (
        SSM_COLORS,
        apply_style,
        create_tufte_figure,
        set_tufte_labels,
        set_tufte_title,
    )

    apply_style()
    xs = np.asarray(complex_scalar_recurrence(_RHO, _THETA, _N))
    t = np.arange(_N + 1)

    fig, (ax1, ax2) = create_tufte_figure(ncols=2, figsize=(11.0, 4.3))

    # Panel 1: the spiral.
    ax1.plot(xs.real, xs.imag, "-", color=SSM_COLORS["accent"], lw=1.0)
    ax1.scatter(xs.real, xs.imag, c=t, cmap="viridis", s=14, zorder=3)
    ax1.scatter([xs.real[0]], [xs.imag[0]], color=SSM_COLORS["alert"], s=40, zorder=4, label="$x_0$")
    ax1.axhline(0.0, color=SSM_COLORS["baseline"], lw=0.6)
    ax1.axvline(0.0, color=SSM_COLORS["baseline"], lw=0.6)
    set_tufte_title(ax1, rf"Complex mode $\rho={_RHO},\ \theta=20^\circ$: decay + rotation")
    set_tufte_labels(ax1, xlabel=r"$\operatorname{Re}\,x_k$", ylabel=r"$\operatorname{Im}\,x_k$")
    ax1.set_aspect("equal", adjustable="box")
    ax1.legend(loc="upper right", fontsize=8, frameon=False)

    # Panel 2: magnitude decay vs the analytic rho^k envelope.
    ax2.semilogy(t, np.abs(xs), "o-", color=SSM_COLORS["accent"], markersize=3, label=r"$|x_k|$")
    ax2.semilogy(t, _RHO**t, ":", color=SSM_COLORS["alert"], lw=1.2, label=r"$\rho^k$ envelope")
    set_tufte_title(ax2, r"Magnitude decays geometrically at rate $\log\rho$")
    set_tufte_labels(ax2, xlabel="step $k$", ylabel=r"$|x_k|$")
    ax2.legend(loc="upper right", fontsize=8, frameon=False)

    fig.tight_layout()
    return fig


def main() -> None:
    import matplotlib.pyplot as plt
    import numpy as np

    from companions._shared.plot_utils import save_figure

    print("Chapter 10 — complex_state.py")
    print("=" * 64)

    xs = complex_scalar_recurrence(_RHO, _THETA, _N)
    rate = decay_rate(xs)
    print(f"  complex mode rho={_RHO}, theta={_THETA:.4f} ({math.degrees(_THETA):.0f} deg/step)")
    print(f"    measured decay rate = {rate:.6f}  vs  log(rho) = {math.log(_RHO):.6f}")

    # RoPE <-> complex equivalence on a seeded complex drive.
    rng = np.random.default_rng(0)
    drive = jnp.asarray(rng.standard_normal(40) + 1j * rng.standard_normal(40))
    resid = rope_equivalence_residual(_RHO, _THETA, drive)
    print(f"  RoPE(2-D rotation) vs complex multiply: max|diff| = {resid:.3e}  (§10.4 equivalence: ~0)")

    _OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig = make_spiral_figure()
    for p in save_figure(fig, _OUT_DIR / "complex-spiral", formats=("png",)):
        print(f"Wrote {p}")
    plt.close(fig)


if __name__ == "__main__":
    main()
