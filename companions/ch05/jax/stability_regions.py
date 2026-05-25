"""Chapter 5 — stability-region plots and Butcher-tableau visualizations.

Computes the stability region $\\mathcal{S} = \\{z \\in \\mathbb{C} : |R(z)| \\le 1\\}$
for a representative set of one-step integration methods and emits three
figures referenced by Chapter 5:

1. ``butcher_tableaux.png`` — heatmaps of the (A | b, c) augmented matrix for
   forward Euler, midpoint RK2, classical RK4, and Runge-Kutta-Fehlberg 4(5).
2. ``rk_stability_regions.png`` — stability regions of forward Euler,
   midpoint RK2, and classical RK4 in the complex plane.
3. ``atlas_stability.png`` — stability regions of all the Chapter-4
   discretizations (ZOH, bilinear, exp-trapezoidal) alongside RK1 and RK4,
   visualizing the contrast between bounded (explicit) and unbounded
   (exponential/rational) stability regions.

Output paths
------------
``public/figures/ch05/{butcher_tableaux,rk_stability_regions,atlas_stability}.png``

Usage
-----
::

    PYTHONPATH=. python companions/ch05/jax/stability_regions.py
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np

from companions._shared.plot_utils import (
    SSM_COLORS,
    apply_style,
    create_tufte_figure,
    save_figure,
    set_tufte_labels,
    set_tufte_title,
)

_REPO_ROOT = Path(__file__).resolve().parents[3]
_OUT_DIR = _REPO_ROOT / "public" / "figures" / "ch05"


# ---------------------------------------------------------------------------
# Butcher tableaux
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Tableau:
    """A Runge-Kutta Butcher tableau (A, b, c) with $s$ stages."""

    name: str
    A: np.ndarray  # shape (s, s); strictly lower triangular for explicit methods
    b: np.ndarray  # shape (s,)
    c: np.ndarray  # shape (s,)
    order: int

    @property
    def s(self) -> int:
        return self.A.shape[0]


def forward_euler_tableau() -> Tableau:
    return Tableau(name="Forward Euler", A=np.array([[0.0]]), b=np.array([1.0]), c=np.array([0.0]), order=1)


def midpoint_rk2_tableau() -> Tableau:
    return Tableau(
        name="Midpoint RK2",
        A=np.array([[0.0, 0.0], [0.5, 0.0]]),
        b=np.array([0.0, 1.0]),
        c=np.array([0.0, 0.5]),
        order=2,
    )


def classical_rk4_tableau() -> Tableau:
    return Tableau(
        name="Classical RK4",
        A=np.array(
            [
                [0.0, 0.0, 0.0, 0.0],
                [0.5, 0.0, 0.0, 0.0],
                [0.0, 0.5, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
            ]
        ),
        b=np.array([1.0 / 6.0, 1.0 / 3.0, 1.0 / 3.0, 1.0 / 6.0]),
        c=np.array([0.0, 0.5, 0.5, 1.0]),
        order=4,
    )


def rkf45_tableau() -> Tableau:
    """Runge-Kutta-Fehlberg 4(5) tableau (Fehlberg 1969, 5th-order weights)."""
    A = np.array(
        [
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [1 / 4, 0.0, 0.0, 0.0, 0.0, 0.0],
            [3 / 32, 9 / 32, 0.0, 0.0, 0.0, 0.0],
            [1932 / 2197, -7200 / 2197, 7296 / 2197, 0.0, 0.0, 0.0],
            [439 / 216, -8.0, 3680 / 513, -845 / 4104, 0.0, 0.0],
            [-8 / 27, 2.0, -3544 / 2565, 1859 / 4104, -11 / 40, 0.0],
        ]
    )
    b = np.array([16 / 135, 0.0, 6656 / 12825, 28561 / 56430, -9 / 50, 2 / 55])
    c = np.array([0.0, 1 / 4, 3 / 8, 12 / 13, 1.0, 1 / 2])
    return Tableau(name="RKF 4(5)", A=A, b=b, c=c, order=5)


# ---------------------------------------------------------------------------
# Stability functions
#
# For an explicit RK method:  R(z) = 1 + z bᵀ (I - zA)⁻¹ 𝟙
# For the Chapter 4 schemes we use closed forms.
# ---------------------------------------------------------------------------


def explicit_stab_fn(tab: Tableau) -> Callable[[np.ndarray], np.ndarray]:
    """Build $R(z) = 1 + z b^\\top (I - z A)^{-1} \\mathbf{1}$ for an explicit tableau."""
    A, b = tab.A, tab.b
    ones = np.ones(tab.s)

    def R(z: np.ndarray) -> np.ndarray:
        out = np.zeros_like(z)
        flat = z.ravel()
        for idx, z_val in enumerate(flat):
            M = np.eye(tab.s) - z_val * A
            kappa = np.linalg.solve(M, ones)
            out.flat[idx] = 1.0 + z_val * (b @ kappa)
        return out

    return R


def zoh_stab_fn(z: np.ndarray) -> np.ndarray:
    """ZOH and exp-trapezoidal share $R(z) = e^z$ on the Dahlquist test problem."""
    return np.exp(z)


def bilinear_stab_fn(z: np.ndarray) -> np.ndarray:
    """Bilinear (Tustin): $R(z) = (1 + z/2)/(1 - z/2)$."""
    return (1.0 + z / 2.0) / (1.0 - z / 2.0)


# ---------------------------------------------------------------------------
# Figure: Butcher tableau heatmaps
# ---------------------------------------------------------------------------


def _augmented_matrix(tab: Tableau) -> np.ndarray:
    """Build the (s+1) x (s+1) array [[A | b]; [c^T | 0]] for visualization."""
    s = tab.s
    M = np.zeros((s + 1, s + 1))
    M[:s, :s] = tab.A
    M[:s, s] = tab.b
    M[s, :s] = tab.c
    return M


def make_butcher_tableau_figure() -> plt.Figure:
    apply_style()
    tabs = [forward_euler_tableau(), midpoint_rk2_tableau(), classical_rk4_tableau(), rkf45_tableau()]
    n = len(tabs)
    fig, axes = create_tufte_figure(nrows=1, ncols=n, figsize=(14.0, 4.0))
    if not isinstance(axes, np.ndarray):
        axes = np.array([axes])

    for ax, tab in zip(axes, tabs):
        M = _augmented_matrix(tab)
        # Symmetric colormap centered at 0 so positive and negative
        # tableau entries are visible separately.
        vmax = float(np.max(np.abs(M)))
        im = ax.imshow(M, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="equal")
        ax.set_xticks(range(tab.s + 1))
        ax.set_yticks(range(tab.s + 1))
        ax.set_xticklabels([f"$a_{{*{j+1}}}$" for j in range(tab.s)] + ["$b$"], fontsize=8)
        ax.set_yticklabels([f"$a_{{{i+1}*}}$" for i in range(tab.s)] + ["$c$"], fontsize=8)
        set_tufte_title(ax, f"{tab.name} ($s={tab.s}$, order {tab.order})")
        ax.grid(False)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.suptitle("Butcher tableaux as augmented matrices (A | b, c)", fontsize=12, y=1.02)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Figure: explicit RK stability regions
# ---------------------------------------------------------------------------


def _make_grid(
    re_range: tuple[float, float] = (-5.0, 1.5),
    im_range: tuple[float, float] = (-4.0, 4.0),
    n: int = 401,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (RE, IM, Z) meshgrid suitable for plotting stability regions."""
    re = np.linspace(re_range[0], re_range[1], n)
    im = np.linspace(im_range[0], im_range[1], n)
    RE, IM = np.meshgrid(re, im)
    Z = RE + 1j * IM
    return RE, IM, Z


def make_rk_stability_figure() -> plt.Figure:
    apply_style()
    tabs = [forward_euler_tableau(), midpoint_rk2_tableau(), classical_rk4_tableau()]
    colors = [SSM_COLORS["baseline"], SSM_COLORS["accent"], SSM_COLORS["highlight"]]

    fig, ax = create_tufte_figure(figsize=(7.0, 6.0))
    RE, IM, Z = _make_grid()
    ax.axhline(0.0, color=SSM_COLORS["baseline"], linewidth=0.5)
    ax.axvline(0.0, color=SSM_COLORS["baseline"], linewidth=0.5)

    for tab, color in zip(tabs, colors):
        R = explicit_stab_fn(tab)
        modulus = np.abs(R(Z))
        # Filled contour: region where |R| <= 1.
        ax.contourf(RE, IM, modulus, levels=[0.0, 1.0], colors=[color], alpha=0.35)
        # Boundary curve |R| = 1.
        ax.contour(RE, IM, modulus, levels=[1.0], colors=[color], linewidths=1.4)
        ax.plot([], [], color=color, linewidth=2.0, label=f"{tab.name} (order {tab.order})")

    set_tufte_title(ax, "Stability regions of explicit Runge–Kutta methods")
    set_tufte_labels(ax, xlabel=r"$\operatorname{Re}(z)$ with $z = \lambda \Delta$", ylabel=r"$\operatorname{Im}(z)$")
    ax.set_xlim(-5.0, 1.5)
    ax.set_ylim(-4.0, 4.0)
    ax.set_aspect("equal", adjustable="box")
    ax.legend(loc="upper left", frameon=False, fontsize=10)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Figure: atlas stability (Ch 4 schemes + RK1, RK4)
# ---------------------------------------------------------------------------


def make_atlas_stability_figure() -> plt.Figure:
    apply_style()
    fig, ax = create_tufte_figure(figsize=(7.0, 6.0))
    RE, IM, Z = _make_grid(re_range=(-6.0, 2.0), im_range=(-5.0, 5.0), n=401)
    ax.axhline(0.0, color=SSM_COLORS["baseline"], linewidth=0.5)
    ax.axvline(0.0, color=SSM_COLORS["baseline"], linewidth=0.5)

    # Forward Euler (bounded disk)
    R_fe = explicit_stab_fn(forward_euler_tableau())(Z)
    ax.contourf(RE, IM, np.abs(R_fe), levels=[0.0, 1.0], colors=[SSM_COLORS["baseline"]], alpha=0.30)
    ax.contour(RE, IM, np.abs(R_fe), levels=[1.0], colors=[SSM_COLORS["baseline"]], linewidths=1.2)

    # Classical RK4 (bounded kidney)
    R_rk4 = explicit_stab_fn(classical_rk4_tableau())(Z)
    ax.contourf(RE, IM, np.abs(R_rk4), levels=[0.0, 1.0], colors=[SSM_COLORS["highlight"]], alpha=0.30)
    ax.contour(RE, IM, np.abs(R_rk4), levels=[1.0], colors=[SSM_COLORS["highlight"]], linewidths=1.2)

    # ZOH / exp-trap (entire LHP — shade the LHP).
    ax.axvspan(-6.0, 0.0, alpha=0.18, color=SSM_COLORS["alert"])

    # Bilinear: |1+z/2|/|1-z/2| <= 1 iff Re(z) <= 0 — same LHP, draw dashed boundary on imaginary axis.
    ax.axvline(0.0, color=SSM_COLORS["accent"], linewidth=1.2, linestyle="--", alpha=0.6)

    # Legend proxies.
    ax.plot([], [], color=SSM_COLORS["baseline"], linewidth=2.0, label="Forward Euler (RK1)")
    ax.plot([], [], color=SSM_COLORS["highlight"], linewidth=2.0, label="Classical RK4")
    ax.plot([], [], color=SSM_COLORS["alert"], linewidth=6.0, alpha=0.6, label="ZOH / exp-trap (entire LHP)")
    ax.plot([], [], color=SSM_COLORS["accent"], linewidth=1.5, linestyle="--", label="Bilinear (boundary = imaginary axis)")

    set_tufte_title(ax, "Discretization atlas — stability regions")
    set_tufte_labels(ax, xlabel=r"$\operatorname{Re}(z)$ with $z = \lambda \Delta$", ylabel=r"$\operatorname{Im}(z)$")
    ax.set_xlim(-6.0, 2.0)
    ax.set_ylim(-5.0, 5.0)
    ax.set_aspect("equal", adjustable="box")
    ax.legend(loc="upper left", frameon=False, fontsize=9)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    _OUT_DIR.mkdir(parents=True, exist_ok=True)
    print("Chapter 5 — stability_regions.py")
    print("=" * 60)

    fig_bt = make_butcher_tableau_figure()
    paths = save_figure(fig_bt, _OUT_DIR / "butcher_tableaux", formats=("png",))
    for p in paths:
        print(f"Wrote {p}")
    plt.close(fig_bt)

    fig_rk = make_rk_stability_figure()
    paths = save_figure(fig_rk, _OUT_DIR / "rk_stability_regions", formats=("png",))
    for p in paths:
        print(f"Wrote {p}")
    plt.close(fig_rk)

    fig_at = make_atlas_stability_figure()
    paths = save_figure(fig_at, _OUT_DIR / "atlas_stability", formats=("png",))
    for p in paths:
        print(f"Wrote {p}")
    plt.close(fig_at)


if __name__ == "__main__":
    main()
