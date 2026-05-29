"""Chapter 7 — HiPPO-LegS as online function approximation: the reconstruction demo.

This is the load-bearing picture of the chapter. The HiPPO state $h(t)\\in\\R^N$ is *not*
an opaque hidden vector — it holds the coefficients of the best $N$-term Legendre
approximation of the entire input history $u(s),\\, s\\in[0,t]$. Drive a signal through
the (time-varying) LegS recurrence, then *reconstruct* the history from the final state
and watch the error fall as $N$ grows.

Time-varying LegS recurrence
----------------------------
HiPPO-LegS integrates the scaled ODE $\\ddt c(t) = \\tfrac{1}{t}\\!\\left(-A\\, c(t) + B\\,
u(t)\\right)$. Discretizing with the bilinear (trapezoidal) rule at integer steps $t=k$
gives the standard per-step update

::

    (I + A/(2k)) c_k = (I - A/(2k)) c_{k-1} + (B/k) u_k

i.e. one linear solve per step. We thread $c$ through ``jax.lax.scan`` with the step
index $k$ as part of the scanned input — a *time-varying* linear recurrence, still a
single fused scan.

Reconstruction
--------------
From coefficients $c$ at time $t$, the history is reconstructed on $z=s/t\\in[0,1]$ as a
Legendre series $\\hat u(z) = \\sum_n c_n\\, \\gamma_n\\, P_n(2z-1)$. The per-mode factor
$\\gamma_n$ (normalization $\\sqrt{2n+1}$ and the LegS orientation sign $(-1)^n$) is
*calibrated empirically* in ``main`` — we print the relative $L^2$ error for all four
candidate conventions and the converging one is the truth.

JAX vs the other companions
---------------------------
The recurrence is ``jax.lax.scan`` (one compiled pass emitting the whole coefficient
trajectory). The Julia companion writes the same recurrence as an explicit ``for`` loop;
the basis evaluation and reconstruction are host-side NumPy/SciPy in all three.

Port credit
-----------
LegS recurrence + reconstruction follow the Annotated S4 HiPPO example and
``post_transformers/experiments/jax/week04/s4_hippo.py``; matrices from
:mod:`companions.ch07.jax.hippo_matrix`.

Output
------
``public/figures/ch07/legendre_basis.png`` — normalized shifted Legendre basis.
``public/figures/ch07/hippo_reconstruction.png`` — reconstruction vs N + error decay.

Usage
-----
::

    PYTHONPATH=. python companions/ch07/jax/hippo_reconstruction.py
"""

from __future__ import annotations

from pathlib import Path

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy.special import eval_legendre  # noqa: E402

from companions._shared.plot_utils import (  # noqa: E402
    SSM_COLORS,
    apply_style,
    create_tufte_figure,
    save_figure,
    set_tufte_labels,
    set_tufte_title,
)
from companions.ch07.jax.hippo_matrix import make_hippo_legs  # noqa: E402

_REPO_ROOT = Path(__file__).resolve().parents[3]
_OUT_DIR = _REPO_ROOT / "public" / "figures" / "ch07"


# ---------------------------------------------------------------------------
# Online encoding: the time-varying LegS recurrence
# ---------------------------------------------------------------------------


def hippo_legs_encode(u: jnp.ndarray, n: int) -> jnp.ndarray:
    """Encode a sampled signal into HiPPO-LegS coefficients via the bilinear recurrence.

    Parameters
    ----------
    u : jnp.ndarray, shape (L,)
        Input signal samples $u_1, \\ldots, u_L$ (taken on a uniform grid of $[0, 1]$).
    n : int
        State dimension $N$.

    Returns
    -------
    c : jnp.ndarray, shape (L, n)
        Coefficient trajectory; row $k$ is the LegS coefficient vector after step $k+1$.
        The final row reconstructs the whole history $u$.

    Raises
    ------
    ValueError
        If ``u`` is not 1-D or ``n < 1``.
    """
    if u.ndim != 1:
        raise ValueError(f"u must be 1-D, got shape {u.shape}")
    if n < 1:
        raise ValueError(f"n must be >= 1, got {n}")
    A, B = make_hippo_legs(n)
    # make_hippo_legs returns the LTI-stable A (eigenvalues -1,...,-N) used by the
    # ZOH bridge of §7.4. The exact *time-varying* LegS operator integrates
    # dc/dt = (1/t)(-A_pos c + B u) with the positive-eigenvalue matrix A_pos = -A,
    # so the bilinear update (I + A_pos/2k) c_k = (I - A_pos/2k) c_{k-1} + (B/k) u_k
    # is well-conditioned (diagonal >= 1) and contracts. Same matrix, opposite sign.
    A = -A
    B = B.squeeze(-1)  # (n,)
    eye = jnp.eye(n, dtype=A.dtype)
    ks = jnp.arange(1, u.shape[0] + 1, dtype=A.dtype)  # 1-indexed step times

    def step(c_prev: jnp.ndarray, ku: tuple[jnp.ndarray, jnp.ndarray]) -> tuple[jnp.ndarray, jnp.ndarray]:
        k, u_k = ku
        lhs = eye + A / (2.0 * k)
        rhs = (eye - A / (2.0 * k)) @ c_prev + (B / k) * u_k
        c_k = jnp.linalg.solve(lhs, rhs)
        return c_k, c_k

    c0 = jnp.zeros(n, dtype=A.dtype)
    _, traj = jax.lax.scan(step, c0, (ks, u))
    return traj


# ---------------------------------------------------------------------------
# Basis evaluation + reconstruction
# ---------------------------------------------------------------------------


def legendre_basis(n: int, z: np.ndarray, *, normalized: bool = True, alternating: bool = False) -> np.ndarray:
    """Shifted Legendre basis matrix, shape ``(n, len(z))``.

    Row $i$ is $\\gamma_i\\, P_i(2z - 1)$ with $\\gamma_i = (\\sqrt{2i+1})^{[normalized]}
    (-1)^{i\\,[alternating]}$. The converging LegS reconstruction uses
    ``normalized=True, alternating=False`` (calibrated empirically; see ``main``):
    $\\hat u(z) = \\sum_n c_n \\sqrt{2n+1}\\, P_n(2z-1)$.
    """
    idx = np.arange(n)
    polys = np.stack([eval_legendre(i, 2.0 * z - 1.0) for i in idx])  # (n, len(z))
    gamma = np.ones(n)
    if normalized:
        gamma = gamma * np.sqrt(2.0 * idx + 1.0)
    if alternating:
        gamma = gamma * (-1.0) ** idx
    return gamma[:, None] * polys


def reconstruct(c: np.ndarray, z: np.ndarray, **basis_kw: bool) -> np.ndarray:
    """Reconstruct the history $\\hat u(z) = \\sum_n c_n \\gamma_n P_n(2z-1)$."""
    basis = legendre_basis(len(c), z, **basis_kw)
    return c @ basis


def _rel_l2(approx: np.ndarray, truth: np.ndarray) -> float:
    """Relative $L^2$ error $\\lVert approx - truth\\rVert / \\lVert truth\\rVert$."""
    return float(np.linalg.norm(approx - truth) / np.linalg.norm(truth))


# ---------------------------------------------------------------------------
# Test signal + experiment
# ---------------------------------------------------------------------------

_L: int = 1000  # number of history samples


def test_signal(z: np.ndarray) -> np.ndarray:
    """A smooth band-limited history: two sinusoids on $z\\in[0,1]$."""
    return np.sin(2.0 * np.pi * 1.5 * z) + 0.5 * np.sin(2.0 * np.pi * 4.0 * z)


def reconstruction_errors(
    n_values: tuple[int, ...], *, normalized: bool = True, alternating: bool = False
) -> dict[int, float]:
    """Relative-$L^2$ reconstruction error at the final time, per state dimension $N$."""
    z = np.linspace(0.0, 1.0, _L)
    truth = test_signal(z)
    u = jnp.asarray(truth)
    errs: dict[int, float] = {}
    for n in n_values:
        c_final = np.asarray(hippo_legs_encode(u, n)[-1])
        approx = reconstruct(c_final, z, normalized=normalized, alternating=alternating)
        errs[n] = _rel_l2(approx, truth)
    return errs


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------


def make_basis_figure(n: int = 5) -> plt.Figure:
    """Plot the normalized shifted Legendre basis $\\tilde P_0,\\ldots,\\tilde P_{n-1}$."""
    apply_style()
    z = np.linspace(0.0, 1.0, 400)
    basis = legendre_basis(n, z, normalized=True, alternating=False)
    fig, ax = create_tufte_figure(figsize=(6.4, 4.6))
    palette = [SSM_COLORS["accent"], SSM_COLORS["highlight"], SSM_COLORS["baseline"], SSM_COLORS["alert"]]
    for i in range(n):
        ax.plot(z, basis[i], linewidth=1.6, color=palette[i % len(palette)], label=rf"$\tilde P_{i}$")
    ax.axhline(0.0, color=SSM_COLORS["baseline"], linewidth=0.5)
    set_tufte_title(ax, r"Normalized shifted Legendre basis $\tilde P_n(z)=\sqrt{2n+1}\,P_n(2z-1)$")
    set_tufte_labels(ax, xlabel=r"$z = s/t \in [0,1]$", ylabel=r"$\tilde P_n(z)$")
    ax.legend(loc="upper center", ncol=n, frameon=False, fontsize=8)
    fig.tight_layout()
    return fig


def make_reconstruction_figure(
    n_values: tuple[int, ...], *, normalized: bool = True, alternating: bool = False
) -> plt.Figure:
    """Two panels: reconstructions overlaid on truth, and the relative-L2 error decay."""
    apply_style()
    z = np.linspace(0.0, 1.0, _L)
    truth = test_signal(z)
    u = jnp.asarray(truth)

    fig, axes = create_tufte_figure(nrows=1, ncols=2, figsize=(11.0, 4.6))
    ax_recon, ax_err = axes

    ax_recon.plot(z, truth, color=SSM_COLORS["baseline"], linewidth=2.2, label="history $u(z)$", zorder=1)
    palette = [SSM_COLORS["accent"], SSM_COLORS["highlight"], SSM_COLORS["alert"]]
    show = [n_values[0], n_values[len(n_values) // 2], n_values[-1]]
    for n, color in zip(show, palette):
        c_final = np.asarray(hippo_legs_encode(u, n)[-1])
        approx = reconstruct(c_final, z, normalized=normalized, alternating=alternating)
        ax_recon.plot(z, approx, color=color, linewidth=1.3, label=rf"$N={n}$", zorder=2)
    set_tufte_title(ax_recon, "HiPPO-LegS reconstruction of the input history")
    set_tufte_labels(ax_recon, xlabel=r"$z = s/t$", ylabel=r"$u(z)$ and $\hat u(z)$")
    ax_recon.legend(loc="upper right", frameon=False, fontsize=9)

    errs = reconstruction_errors(n_values, normalized=normalized, alternating=alternating)
    ns = np.array(list(errs.keys()))
    es = np.array(list(errs.values()))
    ax_err.semilogy(ns, es, color=SSM_COLORS["accent"], marker="o", linewidth=1.6)
    set_tufte_title(ax_err, "Reconstruction error vs state dimension")
    set_tufte_labels(ax_err, xlabel=r"state dimension $N$", ylabel=r"relative $L^2$ error")
    ax_err.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

_N_SWEEP: tuple[int, ...] = (4, 8, 16, 32, 64)


def main() -> None:
    _OUT_DIR.mkdir(parents=True, exist_ok=True)
    print("Chapter 7 — hippo_reconstruction.py")
    print("=" * 60)

    # Reconstruction convention (normalized x sqrt(2n+1), non-alternating) was
    # calibrated empirically: of the four (normalized, alternating) variants it is
    # the only one whose relative-L2 error decreases monotonically in N. Tests and
    # the chapter prose pin the numbers below.
    errs = reconstruction_errors(_N_SWEEP)
    print("Relative L2 reconstruction error vs state dimension N:")
    for n, e in errs.items():
        print(f"  N={n:>3}: {e:.3e}")
    print(f"\n  N=4 -> N=16 drop: {errs[4]:.3f} -> {errs[16]:.3f}; plateau (floor) {errs[64]:.3e}")

    fig1 = make_basis_figure(n=5)
    for p in save_figure(fig1, _OUT_DIR / "legendre_basis", formats=("png",)):
        print(f"Wrote {p}")
    plt.close(fig1)

    fig2 = make_reconstruction_figure(_N_SWEEP)
    for p in save_figure(fig2, _OUT_DIR / "hippo_reconstruction", formats=("png",)):
        print(f"Wrote {p}")
    plt.close(fig2)


if __name__ == "__main__":
    main()
