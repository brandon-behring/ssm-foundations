r"""Chapter 10 §10.2-10.3 — discretization order and stability for Mamba-3.

Mamba-3 (Lahoti et al. 2026, arXiv:2603.15569; ICLR 2026 Oral) keeps Mamba-2's
selective SSD structure (Chapter 9) and changes the *integrator*: first-order
zero-order hold (ZOH) is replaced by the second-order **exponential-trapezoidal**
scheme of Chapter 4 §4.5. This module makes two numerical-analysis claims
concrete, both of which are the C1 symplectic-integrator pilot's empirical anchor:

* **§10.2 — order of accuracy.** ZOH is globally first-order; exp-trapezoidal is
  globally second-order. THE load-bearing subtlety: for *exponential* integrators
  the state transition $\alpha = e^{A\Delta}$ is computed exactly, so it is
  **identical** for ZOH and exp-trapezoidal. On a *homogeneous* system
  ($u \equiv 0$) both are therefore exact to machine precision and the order
  difference is **invisible**. The order-2 gain lives entirely in the input
  quadrature ($\beta, \gamma$), so it is measurable *only on a forced system*
  with non-constant input. :func:`order_sweep` measures global error on the
  forced damped oscillator; :func:`order_sweep` with ``forced=False`` shows the
  homogeneous blindness (both schemes hit roundoff). This redeems Chapter 4
  Exercise 4.3 ("setting $u \equiv 0$ makes ZOH exact; the slope is meaningless").

* **§10.3 — stability decouples from accuracy.** Because $\alpha = e^{A\Delta}$ is
  exact, $|\alpha| = e^{\operatorname{Re}(A)\Delta} < 1$ for $\operatorname{Re}(A)
  < 0$ at *any* step size: ZOH and exp-trapezoidal are unconditionally stable on
  the state transition, and stability is set by $A$ alone, not by the order of the
  quadrature. Bilinear (Tustin) is A-stable too but only *approximates*
  $e^{A\Delta}$ by its $(1,1)$-Padé form, so on stiff modes ($A\Delta \to -\infty$)
  its amplification factor tends to $-1$ (undamped) rather than $0$. Forward Euler
  is not even A-stable. :func:`amplification` and the figures make this contrast
  visible.

Idiomatic-JAX note (this companion is a NumPy->JAX teaching example)
------------------------------------------------------------------
The discretizers are pure elementwise functions of ``(A, dt)`` — no loop, no
state — so they read identically in NumPy and JAX; the only JAX-specific line is
``jax.config.update("jax_enable_x64", True)`` so that a slope-2 convergence study
is not masked by float32 roundoff (the order test needs ~10 decades of dynamic
range). The forced integration in :func:`integrate` is a genuine sequential
recurrence and uses ``lax.scan`` (the same primitive as Chapters 8-9), in
contrast to a Python ``for`` loop a NumPy reference would write.

Port credit
-----------
The three discretizers and the homogeneous LTE harness are ported from
``post_transformers/experiments/jax/week09/mamba3.py``; the forced-system order
sweep (:func:`order_sweep`, :func:`forced_exact`) and the corrected second-order
bilinear are added here, because the predecessor measured order on the
homogeneous system (which, per the subtlety above, cannot see it). Mamba-3:
Lahoti et al., arXiv:2603.15569. Exponential integrators: Hochbruck & Ostermann,
*Acta Numerica* 2010.

Usage
-----
::

    PYTHONPATH=. python companions/ch10/jax/discretization.py
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import jax

# Enable float64 before any jnp array is created (matches Chapters 4, 7, 8, 9).
# The order sweep spans ~10 decades; float32 roundoff would flatten the slope.
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402

if TYPE_CHECKING:
    from matplotlib.figure import Figure

__all__ = [
    "discretize_zoh",
    "discretize_bilinear",
    "discretize_exp_trapezoidal",
    "forced_exact",
    "integrate",
    "global_error",
    "order_sweep",
    "amplification",
]

_REPO_ROOT = Path(__file__).resolve().parents[3]
_OUT_DIR = _REPO_ROOT / "public" / "figures" / "ch10"


# ---------------------------------------------------------------------------
# §10.2 — three scalar discretizations of  dx/dt = A x + u(t)
#
# Each returns coefficients of the one-step update. All share the exact state
# transition alpha = exp(A dt) EXCEPT bilinear, which uses the (1,1)-Pade form.
# They differ in how they approximate the forcing integral, which is where the
# order of accuracy lives.
# ---------------------------------------------------------------------------


def discretize_zoh(A: jnp.ndarray, dt: jnp.ndarray | float) -> tuple[jnp.ndarray, jnp.ndarray]:
    r"""Zero-order hold (first-order). ``x_k = alpha x_{k-1} + beta u_{k-1}``.

    Holds the input constant at the left endpoint over $[t_{k-1}, t_k)$. The
    forcing integral is then $\int_0^{\Delta} e^{A(\Delta - s)}\,ds = (e^{A\Delta}
    - 1)/A$, exact for piecewise-constant input but only first-order for smooth
    input. The state transition $\alpha = e^{A\Delta}$ is exact, so ZOH is exact
    on the homogeneous part — the source of the order-blindness on $u \equiv 0$.

    Parameters
    ----------
    A : jnp.ndarray
        Scalar (or broadcastable) continuous-time mode. Real or complex.
    dt : jnp.ndarray or float
        Step size $\Delta > 0$.

    Returns
    -------
    alpha, beta : jnp.ndarray
        ``x_k = alpha * x_{k-1} + beta * u_{k-1}``.
    """
    A = jnp.asarray(A)
    dt = jnp.asarray(dt, dtype=A.dtype if jnp.iscomplexobj(A) else jnp.float64)
    alpha = jnp.exp(A * dt)
    # L'Hopital: beta -> dt as A -> 0.
    safe_A = jnp.where(jnp.abs(A) < 1e-12, jnp.ones_like(A), A)
    beta = jnp.where(jnp.abs(A) < 1e-12, dt * jnp.ones_like(A), (alpha - 1.0) / safe_A)
    return alpha, beta


def discretize_bilinear(A: jnp.ndarray, dt: jnp.ndarray | float) -> tuple[jnp.ndarray, jnp.ndarray]:
    r"""Bilinear / Tustin (second-order, A-stable). ``x_k = alpha x_{k-1} + beta (u_{k-1}+u_k)``.

    Applies the trapezoidal rule to *both* sides of $x' = Ax + u$:

    .. math::

        \frac{x_k - x_{k-1}}{\Delta} = A\,\frac{x_k + x_{k-1}}{2}
            + \frac{u_{k-1} + u_k}{2},

    giving $\alpha = (1 + A\Delta/2)/(1 - A\Delta/2)$ (the $(1,1)$-Pade
    approximation of $e^{A\Delta}$) and a single weight
    $\beta = (\Delta/2)/(1 - A\Delta/2)$ on each endpoint. Second-order for smooth
    forcing and A-stable, but $\alpha$ only *approximates* the exponential, so on
    stiff modes ($A\Delta \to -\infty$) $\alpha \to -1$: stiff modes are not
    damped (see :func:`amplification`). This is the contrast §10.3 draws against
    the exponential schemes, whose $\alpha = e^{A\Delta} \to 0$.

    Returns
    -------
    alpha, beta : jnp.ndarray
        ``x_k = alpha * x_{k-1} + beta * (u_{k-1} + u_k)`` (beta weights both
        endpoints — that two-endpoint average is what makes it second-order).
    """
    A = jnp.asarray(A)
    dt = jnp.asarray(dt, dtype=A.dtype if jnp.iscomplexobj(A) else jnp.float64)
    half = dt / 2.0
    denom = 1.0 - half * A
    alpha = (1.0 + half * A) / denom
    beta = half / denom
    return alpha, beta


def discretize_exp_trapezoidal(
    A: jnp.ndarray, dt: jnp.ndarray | float, lam: float = 0.5
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    r"""Exponential-trapezoidal (second-order). ``x_k = alpha x_{k-1} + beta u_{k-1} + gamma u_k``.

    Mamba-3's second-order integrator (the exponential-integrator family of
    Chapter 4 §4.5, in trapezoidal-*quadrature* form). Keeps the *exact* exponential
    transition and approximates the forcing integral by the trapezoidal quadrature
    rule — the $\lambda$-weighted endpoint average of §10.2, not the linear-interpolant
    integration of §4.5's $\varphi$-family:

    .. math::

        x_k = e^{A\Delta} x_{k-1}
            + (1-\lambda)\,\Delta\, e^{A\Delta}\, u_{k-1}
            + \lambda\,\Delta\, u_k,

    i.e. $\alpha = e^{A\Delta}$, $\beta = (1-\lambda)\Delta\,\alpha$ (left
    endpoint, carried across the step so it picks up the full decay), $\gamma =
    \lambda\Delta$ (right endpoint, injected at the end so it sees no decay).
    $\lambda = \tfrac12$ is the symmetric trapezoid (global order 2); $\lambda = 1$
    degenerates to a shifted first-order ZOH. Mamba-3 makes $\lambda$
    input-dependent (a sigmoid of a learned projection); we take it constant here.

    Returns
    -------
    alpha, beta, gamma : jnp.ndarray
        The three-tuple (vs ZOH's two): the trapezoid reads *both* endpoints.
    """
    if not 0.0 <= lam <= 1.0:
        raise ValueError(f"lam must be in [0, 1], got {lam}")
    A = jnp.asarray(A)
    dt = jnp.asarray(dt, dtype=A.dtype if jnp.iscomplexobj(A) else jnp.float64)
    alpha = jnp.exp(A * dt)
    beta = (1.0 - lam) * dt * alpha
    gamma = lam * dt
    return alpha, beta, gamma


# ---------------------------------------------------------------------------
# §10.2 — forced exact solution + sequential integration + order measurement
#
# Test problem:  dx/dt = A x + sin(omega t),  x(0) = x0.   (B = 1, scalar)
# The forced (non-constant-input) case is the ONLY one that reveals the order
# difference, because the homogeneous transition exp(A dt) is exact for both ZOH
# and exp-trapezoidal.
# ---------------------------------------------------------------------------


def forced_exact(
    A: jnp.ndarray | float,
    omega: float,
    t: jnp.ndarray,
    x0: jnp.ndarray | float = 0.0,
) -> jnp.ndarray:
    r"""Exact solution of $x' = A x + \sin(\omega t)$, $x(0) = x_0$.

    By variation of constants the particular solution is
    $x_p(t) = -(A\sin\omega t + \omega\cos\omega t)/(A^2 + \omega^2)$, so

    .. math::

        x(t) = \Big(x_0 + \tfrac{\omega}{A^2+\omega^2}\Big) e^{A t}
             - \frac{A\sin\omega t + \omega\cos\omega t}{A^2 + \omega^2}.

    Used as ground truth for the forced-system order sweep.

    Parameters
    ----------
    A : scalar
        Continuous mode (must satisfy $A^2 + \omega^2 \neq 0$).
    omega : float
        Forcing angular frequency.
    t : jnp.ndarray
        Sample times.
    x0 : scalar, default 0.0
        Initial condition.

    Returns
    -------
    x : jnp.ndarray, same shape as ``t``.
    """
    A = jnp.asarray(A)
    t = jnp.asarray(t, dtype=A.dtype if jnp.iscomplexobj(A) else jnp.float64)
    denom = A * A + omega * omega
    if jnp.any(jnp.abs(denom) < 1e-300):
        raise ValueError("A^2 + omega^2 must be nonzero")
    c = x0 + omega / denom
    return c * jnp.exp(A * t) - (A * jnp.sin(omega * t) + omega * jnp.cos(omega * t)) / denom


def integrate(
    scheme: str,
    A: jnp.ndarray | float,
    dt: float,
    n_steps: int,
    omega: float | None,
    x0: jnp.ndarray | float = 0.0,
    lam: float = 0.5,
) -> jnp.ndarray:
    r"""Integrate $x' = A x + u(t)$ for ``n_steps`` of step ``dt`` via ``lax.scan``.

    The forcing is $u(t) = \sin(\omega t)$ when ``omega`` is a float, or
    $u \equiv 0$ (homogeneous) when ``omega is None`` — the latter exposes the
    order-blindness of exponential schemes on autonomous systems.

    Parameters
    ----------
    scheme : {"zoh", "bilinear", "exp_trapezoidal"}
    A : scalar
    dt : float
    n_steps : int
    omega : float or None
        Forcing frequency; ``None`` for the homogeneous system.
    x0 : scalar, default 0.0
    lam : float, default 0.5
        Interpolation parameter for exp-trapezoidal.

    Returns
    -------
    xs : jnp.ndarray, shape (n_steps + 1,)
        States $x_0, x_1, \ldots, x_{n_steps}$ at $t = 0, \Delta, \ldots$.
    """
    A = jnp.asarray(A)
    dtype = A.dtype if jnp.iscomplexobj(A) else jnp.float64
    ks = jnp.arange(n_steps + 1)
    t = ks * dt

    def u(time: jnp.ndarray) -> jnp.ndarray:
        if omega is None:
            return jnp.zeros_like(time)
        return jnp.sin(omega * time)

    u_grid = u(t)  # (n_steps + 1,)

    if scheme == "zoh":
        alpha, beta = discretize_zoh(A, dt)

        def step(x, k):  # uses u_{k-1} (left endpoint)
            x_new = alpha * x + beta * u_grid[k - 1]
            return x_new, x_new

    elif scheme == "bilinear":
        alpha, beta = discretize_bilinear(A, dt)

        def step(x, k):  # weights both endpoints (u_{k-1} + u_k)
            x_new = alpha * x + beta * (u_grid[k - 1] + u_grid[k])
            return x_new, x_new

    elif scheme == "exp_trapezoidal":
        alpha, beta, gamma = discretize_exp_trapezoidal(A, dt, lam=lam)

        def step(x, k):  # left endpoint u_{k-1}, right endpoint u_k
            x_new = alpha * x + beta * u_grid[k - 1] + gamma * u_grid[k]
            return x_new, x_new

    else:
        raise ValueError(f"unknown scheme {scheme!r}; expected zoh/bilinear/exp_trapezoidal")

    x_init = jnp.asarray(x0, dtype=dtype)
    _, xs_tail = jax.lax.scan(step, x_init, jnp.arange(1, n_steps + 1))
    return jnp.concatenate([x_init[None], xs_tail])


def global_error(
    scheme: str,
    A: jnp.ndarray | float,
    dt: float,
    T: float,
    omega: float | None,
    x0: jnp.ndarray | float = 0.0,
    lam: float = 0.5,
) -> float:
    """Max absolute error over $[0, T]$ between a scheme and the exact solution.

    For ``omega is None`` (homogeneous) the exact solution is $x_0 e^{At}$, on
    which ZOH and exp-trapezoidal are both exact (the order-blindness). For a
    float ``omega`` (forced) the exact solution is :func:`forced_exact`, on which
    ZOH is first-order and exp-trapezoidal second-order.

    Returns
    -------
    err : float
        $\\max_k |x_k^{\\mathrm{scheme}} - x(t_k)|$.
    """
    n_steps = int(round(T / dt))
    xs = integrate(scheme, A, dt, n_steps, omega, x0=x0, lam=lam)
    t = jnp.arange(n_steps + 1) * dt
    if omega is None:
        exact = jnp.asarray(x0) * jnp.exp(jnp.asarray(A) * t)
    else:
        exact = forced_exact(A, omega, t, x0=x0)
    return float(jnp.max(jnp.abs(xs - exact)))


def order_sweep(
    scheme: str,
    A: jnp.ndarray | float,
    dts: jnp.ndarray,
    T: float,
    omega: float | None,
    x0: jnp.ndarray | float = 0.0,
    lam: float = 0.5,
) -> tuple[jnp.ndarray, float]:
    r"""Global error vs step size, plus the fitted convergence slope (= order).

    Fits $\log(\text{err}) = p \log(\Delta) + c$ over the two finest step sizes
    (the asymptotic regime), returning $p$ — the empirical order of accuracy.

    Parameters
    ----------
    scheme : str
    A : scalar
    dts : jnp.ndarray
        Decreasing step sizes (finest last).
    T : float
        Integration horizon (each ``dt`` must divide ``T`` reasonably).
    omega : float or None
    x0 : scalar, default 0.0
    lam : float, default 0.5

    Returns
    -------
    errors : jnp.ndarray, shape (len(dts),)
    slope : float
        Empirical order from the two finest step sizes. Near 1 for ZOH (forced),
        near 2 for exp-trapezoidal / bilinear (forced). Meaningless (errors at
        roundoff) for the homogeneous case.
    """
    errors = jnp.asarray(
        [global_error(scheme, A, float(dt), T, omega, x0=x0, lam=lam) for dt in dts]
    )
    # Slope from the two finest (smallest) step sizes.
    d = jnp.asarray(dts)
    slope = float(
        (jnp.log(errors[-1]) - jnp.log(errors[-2])) / (jnp.log(d[-1]) - jnp.log(d[-2]))
    )
    return errors, slope


# ---------------------------------------------------------------------------
# §10.3 — amplification factor (discrete-stability diagnostic)
# ---------------------------------------------------------------------------


def amplification(scheme: str, z: jnp.ndarray, lam: float = 0.5) -> jnp.ndarray:
    r"""Amplification factor $\alpha(z)$ as a function of $z = A\Delta$.

    The homogeneous recurrence is $x_k = \alpha\, x_{k-1}$, so $|\alpha(z)| \le 1$
    is the discrete-stability condition. Schemes:

    * ``zoh`` / ``exp_trapezoidal``: $\alpha = e^{z}$ (exact). $|\alpha| \le 1$ iff
      $\operatorname{Re}(z) \le 0$ (the whole left half-plane: A-stable), and
      $\alpha \to 0$ as $z \to -\infty$ (stiff modes fully damped).
    * ``bilinear``: $\alpha = (1 + z/2)/(1 - z/2)$. Also A-stable, but
      $\alpha \to -1$ as $z \to -\infty$ (stiff modes undamped).
    * ``forward_euler``: $\alpha = 1 + z$. NOT A-stable; $|\alpha| \le 1$ only on
      the disk $|1 + z| \le 1$.

    Parameters
    ----------
    scheme : {"zoh", "exp_trapezoidal", "bilinear", "forward_euler"}
    z : jnp.ndarray
        Complex (or real) values $z = A\Delta$.
    lam : float, default 0.5
        Unused (kept for a uniform call signature); exp-trapezoidal's transition
        is $e^z$ regardless of $\lambda$.

    Returns
    -------
    alpha : jnp.ndarray, same shape as ``z``.
    """
    z = jnp.asarray(z)
    if scheme in ("zoh", "exp_trapezoidal"):
        return jnp.exp(z)
    if scheme == "bilinear":
        return (1.0 + z / 2.0) / (1.0 - z / 2.0)
    if scheme == "forward_euler":
        return 1.0 + z
    raise ValueError(f"unknown scheme {scheme!r}")


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

# Forced damped oscillator written as a scalar complex mode: A = -0.5 + 2i has a
# decaying oscillation, a faithful stand-in for one Mamba-3 complex eigenvalue.
_A_FORCED = -0.5 + 2.0j
_OMEGA = 1.3
_T = 6.0
_X0 = 1.0 + 0.0j
_DTS = jnp.asarray([0.4, 0.2, 0.1, 0.05, 0.025, 0.0125])


def make_order_figure() -> Figure:
    """Log-log global error vs step size on the FORCED system: ZOH slope 1, exp-trap slope 2."""
    import numpy as np

    from companions._shared.plot_utils import (
        SSM_COLORS,
        apply_style,
        create_tufte_figure,
        set_tufte_labels,
        set_tufte_title,
    )

    apply_style()
    dts = np.asarray(_DTS)
    err_zoh, s_zoh = (np.asarray(order_sweep("zoh", _A_FORCED, _DTS, _T, _OMEGA, x0=_X0)[0]),
                      order_sweep("zoh", _A_FORCED, _DTS, _T, _OMEGA, x0=_X0)[1])
    err_trap, s_trap = (np.asarray(order_sweep("exp_trapezoidal", _A_FORCED, _DTS, _T, _OMEGA, x0=_X0)[0]),
                        order_sweep("exp_trapezoidal", _A_FORCED, _DTS, _T, _OMEGA, x0=_X0)[1])

    fig, ax = create_tufte_figure(figsize=(6.5, 4.2))
    ax.loglog(dts, err_zoh, "o-", color=SSM_COLORS["accent"], label=f"ZOH (slope {s_zoh:.2f})")
    ax.loglog(dts, err_trap, "s-", color=SSM_COLORS["alert"],
              label=f"exp-trapezoidal (slope {s_trap:.2f})")
    # Reference slope-1 and slope-2 guides.
    ax.loglog(dts, err_zoh[-1] * (dts / dts[-1]) ** 1, ":", color=SSM_COLORS["baseline"], lw=0.8)
    ax.loglog(dts, err_trap[-1] * (dts / dts[-1]) ** 2, ":", color=SSM_COLORS["baseline"], lw=0.8)
    set_tufte_title(ax, "Forced system: order is visible")
    set_tufte_labels(ax, xlabel=r"step size $\Delta$", ylabel="max error over $[0, T]$")
    ax.legend(loc="upper left", fontsize=8, frameon=False)
    fig.tight_layout()
    return fig


def make_stability_figure() -> Figure:
    """Two panels: A-stability regions (FE disk vs LHP) and stiff-mode damping |alpha| vs -z."""
    import numpy as np

    from companions._shared.plot_utils import (
        SSM_COLORS,
        apply_style,
        create_tufte_figure,
        set_tufte_labels,
        set_tufte_title,
    )

    apply_style()
    fig, (ax1, ax2) = create_tufte_figure(ncols=2, figsize=(11.0, 4.3))

    # Panel 1: stability region boundaries in the complex z = A*dt plane.
    re = np.linspace(-4.0, 2.0, 400)
    im = np.linspace(-3.0, 3.0, 400)
    RE, IM = np.meshgrid(re, im)
    Z = RE + 1j * IM
    mag_exp = np.abs(np.exp(Z))  # ZOH / exp-trap: |e^z| <= 1 iff Re z <= 0
    mag_fe = np.abs(1.0 + Z)  # forward Euler: disk |1+z| <= 1
    ax1.contourf(RE, IM, (mag_exp <= 1.0).astype(float), levels=[0.5, 1.5],
                 colors=[SSM_COLORS["highlight"]], alpha=0.25)
    ax1.contour(RE, IM, mag_fe, levels=[1.0], colors=[SSM_COLORS["alert"]], linewidths=1.3)
    ax1.axvline(0.0, color=SSM_COLORS["accent"], lw=1.3)
    ax1.plot([], [], color=SSM_COLORS["accent"], lw=1.3, label="ZOH / exp-trap (Re z = 0)")
    ax1.plot([], [], color=SSM_COLORS["alert"], lw=1.3, label="forward Euler (disk)")
    ax1.fill_between([], [], color=SSM_COLORS["highlight"], alpha=0.25, label="A-stable region")
    set_tufte_title(ax1, "A-stability: LHP vs the forward-Euler disk")
    set_tufte_labels(ax1, xlabel=r"$\operatorname{Re}(A\Delta)$", ylabel=r"$\operatorname{Im}(A\Delta)$")
    ax1.legend(loc="upper left", fontsize=7.5, frameon=False)

    # Panel 2: stiff-mode damping along the negative real axis.
    x = np.linspace(0.0, 12.0, 300)  # |z| on the negative real axis (z = -x)
    z = -x
    ax2.plot(x, np.abs(amplification("exp_trapezoidal", z)), "-",
             color=SSM_COLORS["accent"], label=r"ZOH / exp-trap: $e^z \to 0$")
    ax2.plot(x, np.abs(amplification("bilinear", z)), "-",
             color=SSM_COLORS["alert"], label=r"bilinear: $\to 1$ (undamped)")
    ax2.axhline(1.0, color=SSM_COLORS["baseline"], lw=0.8, ls=":")
    set_tufte_title(ax2, "Stiff modes: exponential schemes damp, bilinear does not")
    set_tufte_labels(ax2, xlabel=r"$-A\Delta$ (stiffness)", ylabel=r"$|\alpha|$")
    ax2.set_ylim(-0.05, 1.3)
    ax2.legend(loc="center right", fontsize=8, frameon=False)

    fig.tight_layout()
    return fig


def main() -> None:
    import matplotlib.pyplot as plt

    from companions._shared.plot_utils import save_figure

    print("Chapter 10 — discretization.py")
    print("=" * 64)

    # The headline: order is invisible on the homogeneous system, visible on forced.
    err_h_zoh, s_h_zoh = order_sweep("zoh", _A_FORCED, _DTS, _T, None, x0=_X0)
    err_h_trap, s_h_trap = order_sweep("exp_trapezoidal", _A_FORCED, _DTS, _T, None, x0=_X0)
    print("  HOMOGENEOUS (u = 0): exponential transition is exact for both")
    print(f"    ZOH      max error over dt-sweep = {float(jnp.max(err_h_zoh)):.2e}  (roundoff)")
    print(f"    exp-trap max error over dt-sweep = {float(jnp.max(err_h_trap)):.2e}  (roundoff)")
    print("    -> order is INVISIBLE here (§10.2 homogeneous-blindness; cf. Ch 4 Ex 4.3)")

    err_zoh, s_zoh = order_sweep("zoh", _A_FORCED, _DTS, _T, _OMEGA, x0=_X0)
    err_trap, s_trap = order_sweep("exp_trapezoidal", _A_FORCED, _DTS, _T, _OMEGA, x0=_X0)
    err_bl, s_bl = order_sweep("bilinear", _A_FORCED, _DTS, _T, _OMEGA, x0=_X0)
    print(f"  FORCED (u = sin {_OMEGA} t): order is visible")
    print(f"    ZOH             slope = {s_zoh:.3f}  (first-order)")
    print(f"    exp-trapezoidal slope = {s_trap:.3f}  (second-order)")
    print(f"    bilinear        slope = {s_bl:.3f}  (second-order)")

    # Stiff-mode damping contrast (§10.3).
    z_stiff = jnp.asarray(-30.0)
    print(f"  STIFF mode z = {float(z_stiff):.0f}:")
    print(f"    |alpha| exp-trap = {float(jnp.abs(amplification('exp_trapezoidal', z_stiff))):.2e}  (-> 0)")
    print(f"    |alpha| bilinear = {float(jnp.abs(amplification('bilinear', z_stiff))):.4f}  (-> 1)")

    _OUT_DIR.mkdir(parents=True, exist_ok=True)
    for name, builder in (("lte-order", make_order_figure), ("stability-regions", make_stability_figure)):
        fig = builder()
        for p in save_figure(fig, _OUT_DIR / name, formats=("png",)):
            print(f"Wrote {p}")
        plt.close(fig)


if __name__ == "__main__":
    main()
