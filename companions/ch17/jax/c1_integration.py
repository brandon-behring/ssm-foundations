r"""Chapter 17 §17.2 — the C1 atlas cell: exact-exponential vs symplectic vs RK4 on an SSM mode.

The C1 (symplectic-integrators) pilot asks whether trained SSM transitions have enough
Hamiltonian structure to justify a structure-preserving integrator over the exp-trapezoidal
default. This module builds the one *idealized atlas cell* that frames the question, composing
two shipped companions:

* Chapter 6's symplectic demo (``symplectic_demo``): Störmer–Verlet (symplectic, 2nd order) and
  RK4 (non-symplectic) integrators of the harmonic oscillator, with energy-drift tracking;
* Chapter 10's complex-mode SSM (``complex_scalar_recurrence``): a Mamba-3 mode with magnitude
  $\rho$ and angular frequency $\theta$.

The bridge: the harmonic oscillator $\dot q = p,\ \dot p = -q$ written as a complex state
$z = q + ip$ obeys $\dot z = -iz$ — a Mamba-3 complex mode at the **purely imaginary** eigenvalue
$\lambda = -i$. Its exact discrete transition is $e^{-i\,dt}$ with $|e^{-i\,dt}| = 1$, so the SSM's
native (exact-exponential) integrator conserves the mode energy $|z|^2 = q^2 + p^2 = 2H$ *exactly*.

**The integrated signature (NEW — neither ch06 nor ch10 produced it):** the long-horizon secular
energy drift per period of the three integrators applied to the *same* oscillator mode —
exact-exponential (the diagonal SSM's transition), symplectic Verlet, and non-symplectic RK4 — and
its scaling with the step size $dt$ (the atlas's second axis). The verdict the C1 pilot starts from:
a *diagonal* SSM's exact exponential already conserves the conservative-mode energy, so the
symplectic-integrator advantage over a generic integrator bites only **off the diagonal** (coupled
or input-dependent transitions, where the exponential is not applied exactly) — the empirical
question the pilot's symplectic atlas pursues on *trained* matrices.

No training anywhere: the mode is constructed; this is the reproducible atlas-cell template, not a
pilot result. The trained-model program is the C1 pilot's, in post_transformers.

Idiomatic-JAX / port credit
---------------------------
Greenfield composition. Integrators + harness reused verbatim from
``companions/ch06/jax/symplectic_demo.py``; the SSM complex mode from
``companions/ch10/jax/complex_state.py``. The energy-conservation comparison across the three
integrators on one oscillator mode is the new object.

Usage
-----
::

    PYTHONPATH=. python companions/ch17/jax/c1_integration.py
"""

from __future__ import annotations

from pathlib import Path

import jax

# Enable float64 before any jnp array (and before importing ch06/ch10, which import jnp at load).
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

from companions.ch06.jax import symplectic_demo as sym  # noqa: E402
from companions.ch10.jax.complex_state import complex_scalar_recurrence  # noqa: E402

__all__ = [
    "exact_exponential_energy",
    "verlet_energy",
    "rk4_energy",
    "secular_drift_per_period",
    "endpoint_drift_per_period",
    "energy_band",
    "atlas_cell",
]

_REPO_ROOT = Path(__file__).resolve().parents[3]
_OUT_DIR = _REPO_ROOT / "public" / "figures" / "ch17"

_TWO_PI = 2.0 * np.pi


def _n_steps(dt: float, periods: float) -> int:
    if dt <= 0.0:
        raise ValueError(f"dt must be > 0; got {dt}")
    if periods <= 0.0:
        raise ValueError(f"periods must be > 0; got {periods}")
    return int(round(periods * _TWO_PI / dt))


def exact_exponential_energy(dt: float, n_steps: int, q0: float = 1.0, p0: float = 0.0) -> np.ndarray:
    r"""Energy $H_k = \tfrac12|z_k|^2$ of the oscillator as a Mamba-3 mode (exact exponential).

    The mode $z = q + ip$ at eigenvalue $-i$ has exact discrete transition $e^{-i\,dt}$
    (magnitude 1), so this is the diagonal SSM's native integrator — energy-conserving by
    construction. Reuses Chapter 10's ``complex_scalar_recurrence`` ($\rho = 1$, $\theta = -dt$).
    """
    xs = complex_scalar_recurrence(1.0, -float(dt), n_steps, x0=complex(q0, p0))
    return 0.5 * np.asarray(jnp.abs(xs) ** 2)


def verlet_energy(dt: float, n_steps: int, q0: float = 1.0, p0: float = 0.0) -> np.ndarray:
    """Harmonic-oscillator energy trajectory under Störmer–Verlet (symplectic; reuses ch06)."""
    _, qs, ps = sym.simulate(sym.verlet_step, sym.harmonic_T_grad, sym.harmonic_V_grad,
                             q0, p0, dt, n_steps)
    return np.asarray(sym.harmonic_H(jnp.asarray(qs), jnp.asarray(ps)))


def rk4_energy(dt: float, n_steps: int, q0: float = 1.0, p0: float = 0.0) -> np.ndarray:
    """Harmonic-oscillator energy trajectory under RK4 (non-symplectic; reuses ch06)."""
    _, qs, ps = sym.simulate(sym.rk4_step_hamilton, sym.harmonic_T_grad, sym.harmonic_V_grad,
                             q0, p0, dt, n_steps)
    return np.asarray(sym.harmonic_H(jnp.asarray(qs), jnp.asarray(ps)))


def secular_drift_per_period(energy: np.ndarray, dt: float) -> float:
    r"""Secular (accumulating) energy trend per period — the least-squares slope of $H_k$ vs period.

    This is the quantity that separates a symplectic integrator from a non-symplectic one: a
    symplectic map conserves a *modified* energy, so $H_k$ oscillates within a fixed band with
    **zero** secular slope; a non-symplectic map's energy slope is nonzero (it accumulates). The
    linear fit is robust to the bounded oscillation — unlike a raw endpoint difference, which would
    sample the oscillation's phase (see :func:`endpoint_drift_per_period`, kept for the ch06 check).
    """
    energy = np.asarray(energy, dtype=float)
    if energy.ndim != 1 or energy.shape[0] < 2:
        raise ValueError("energy must be a 1-D trajectory of length >= 2")
    periods_axis = np.arange(energy.shape[0]) * dt / _TWO_PI
    return float(np.polyfit(periods_axis, energy, 1)[0])


def endpoint_drift_per_period(energy: np.ndarray, dt: float) -> float:
    r"""Endpoint energy change per period, $(H_{\mathrm{end}} - H_0)/\text{periods}$ — ch06's metric.

    Faithful to Chapter 6's ``rk4_drift_per_period`` (used to cross-check the reused RK4 path). For
    a small-oscillation integrator (RK4) it approximates the secular trend; for a large-oscillation
    one (Verlet) it would mislead — hence :func:`secular_drift_per_period` is the atlas headline.
    """
    energy = np.asarray(energy, dtype=float)
    if energy.ndim != 1 or energy.shape[0] < 2:
        raise ValueError("energy must be a 1-D trajectory of length >= 2")
    periods = (energy.shape[0] - 1) * dt / _TWO_PI
    return float((energy[-1] - energy[0]) / periods)


def energy_band(energy: np.ndarray) -> float:
    r"""Peak-to-peak energy oscillation $\max H - \min H$ (the modified-energy band amplitude)."""
    energy = np.asarray(energy, dtype=float)
    return float(np.max(energy) - np.min(energy))


def atlas_cell(dt: float, periods: float, q0: float = 1.0, p0: float = 0.0) -> dict[str, float]:
    r"""One atlas cell: secular drift + oscillation band of the three integrators on the SSM mode.

    Returns the integration signature — the secular energy drift per period (and the bounded
    oscillation band) for the exact-exponential SSM transition, symplectic Verlet, and RK4.
    """
    n = _n_steps(dt, periods)
    e_exp = exact_exponential_energy(dt, n, q0, p0)
    e_ver = verlet_energy(dt, n, q0, p0)
    e_rk4 = rk4_energy(dt, n, q0, p0)
    e0 = 0.5 * (q0 * q0 + p0 * p0)
    return {
        "E0": e0,
        "exact_exp_drift": secular_drift_per_period(e_exp, dt),
        "exact_exp_band": energy_band(e_exp),
        "verlet_drift": secular_drift_per_period(e_ver, dt),
        "verlet_band": energy_band(e_ver),
        "rk4_drift": secular_drift_per_period(e_rk4, dt),
        "rk4_band": energy_band(e_rk4),
    }


# ---------------------------------------------------------------------------
# Figure + measured numbers (§17.2).
# ---------------------------------------------------------------------------

_FIG_DT = 0.1
_FIG_PERIODS = 200
_FIG_DT_SWEEP = (0.05, 0.1, 0.2, 0.4)


def _make_figure() -> None:
    import matplotlib.pyplot as plt

    from companions._shared.plot_utils import (
        SSM_COLORS,
        apply_style,
        create_tufte_figure,
        save_figure,
        set_tufte_labels,
        set_tufte_title,
    )

    apply_style()
    n = _n_steps(_FIG_DT, _FIG_PERIODS)
    e_exp = exact_exponential_energy(_FIG_DT, n)
    e_ver = verlet_energy(_FIG_DT, n)
    e_rk4 = rk4_energy(_FIG_DT, n)
    e0 = 0.5
    periods_axis = np.arange(n + 1) * _FIG_DT / _TWO_PI

    cell = atlas_cell(_FIG_DT, _FIG_PERIODS)
    # Reduction cross-check: our reused RK4 path reproduces ch06's rk4_drift_per_period (endpoint
    # metric) exactly — confirms the integrator is ch06's, not a re-derivation.
    my_rk4_endpoint = endpoint_drift_per_period(e_rk4, _FIG_DT)
    ch06_rk4 = sym.rk4_drift_per_period(_FIG_DT, _FIG_PERIODS)

    print("Chapter 17 — c1_integration.py")
    print("=" * 64)
    print(f"  C1 atlas cell (harmonic-oscillator SSM mode, dt={_FIG_DT}, {_FIG_PERIODS} periods):")
    print(f"    {'integrator':<20}{'secular slope/period':>22}{'energy band':>16}")
    print(f"    {'exact-exp (SSM)':<20}{cell['exact_exp_drift']:>22.3e}{cell['exact_exp_band']:>16.3e}")
    print(f"    {'Verlet (symplectic)':<20}{cell['verlet_drift']:>22.3e}{cell['verlet_band']:>16.3e}")
    print(f"    {'RK4 (non-sympl.)':<20}{cell['rk4_drift']:>22.3e}{cell['rk4_band']:>16.3e}")
    print(f"    Verlet secular/RK4 secular = {abs(cell['verlet_drift']) / abs(cell['rk4_drift']):.3e} "
          f"(symplectic kills the secular trend; band is bounded)")
    print(f"    reduction: our RK4 endpoint drift {my_rk4_endpoint:.6e} == ch06.rk4_drift_per_period "
          f"{ch06_rk4:.6e}  (|diff| {abs(my_rk4_endpoint - ch06_rk4):.1e})")

    # dt-sweep (the atlas's second axis): secular drift vs step size.
    print(f"  dt-sweep secular drift/period:")
    sweep = {dt: atlas_cell(dt, _FIG_PERIODS) for dt in _FIG_DT_SWEEP}
    for dt in _FIG_DT_SWEEP:
        c = sweep[dt]
        print(f"    dt={dt:<5}: exact-exp {c['exact_exp_drift']:>10.2e}  "
              f"Verlet {c['verlet_drift']:>10.2e}  RK4 {c['rk4_drift']:>10.2e}")

    fig, axes = create_tufte_figure(1, 2, figsize=(11.0, 4.3))
    ax_e, ax_sweep = axes  # type: ignore[misc]

    # First few periods: the bounded oscillation each integrator carries (the trade-off Verlet pays
    # for zero secular drift). Over the full horizon Verlet's fast oscillation fills the axis.
    n_short = int(round(6 * _TWO_PI / _FIG_DT))
    sl = slice(0, n_short + 1)
    ax_e.plot(periods_axis[sl], (e_ver - e0)[sl], color=SSM_COLORS["accent"], lw=1.0,
              label="Verlet (symplectic)")
    ax_e.plot(periods_axis[sl], (e_rk4 - e0)[sl], color=SSM_COLORS["alert"], lw=1.0,
              label="RK4 (non-symplectic)")
    ax_e.plot(periods_axis[sl], (e_exp - e0)[sl], color=SSM_COLORS["highlight"], lw=1.4,
              label="exact-exp (diagonal SSM)")
    ax_e.axhline(0.0, color=SSM_COLORS["baseline"], lw=0.7, ls=":")
    set_tufte_title(ax_e, "Energy error, first 6 periods (bounded oscillation)")
    set_tufte_labels(ax_e, "periods", r"$H_k - H_0$")
    ax_e.legend(frameon=False, fontsize=8, loc="lower left")

    dts = np.asarray(_FIG_DT_SWEEP)
    rk4_drifts = np.abs([sweep[dt]["rk4_drift"] for dt in _FIG_DT_SWEEP])
    ver_drifts = np.abs([sweep[dt]["verlet_drift"] for dt in _FIG_DT_SWEEP]) + 1e-300
    exp_drifts = np.abs([sweep[dt]["exact_exp_drift"] for dt in _FIG_DT_SWEEP]) + 1e-300
    ax_sweep.loglog(dts, rk4_drifts, "o-", color=SSM_COLORS["alert"], ms=4, label="RK4 (secular)")
    ax_sweep.loglog(dts, ver_drifts, "s-", color=SSM_COLORS["accent"], ms=4, label="Verlet (secular)")
    ax_sweep.loglog(dts, exp_drifts, "D-", color=SSM_COLORS["highlight"], ms=4, label="exact-exp")
    set_tufte_title(ax_sweep, "Secular drift vs step size (the atlas axis)")
    set_tufte_labels(ax_sweep, r"step size $dt$", r"$|$secular drift / period$|$")
    ax_sweep.legend(frameon=False, fontsize=8, loc="lower right")

    fig.tight_layout()
    _OUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in save_figure(fig, _OUT_DIR / "c1-atlas-cell", formats=("png",)):
        print(f"  wrote {path.relative_to(_REPO_ROOT)}")
    plt.close(fig)


def main() -> None:
    _make_figure()


if __name__ == "__main__":
    main()
