"""Chapter 2 — stability regions of four common ODE integrators.

Plots $\\{ z \\in \\mathbb{C} : |R(z)| \\leq 1 \\}$ for forward Euler,
backward Euler, bilinear (trapezoidal), and ZOH (matrix exponential)
integrators. The shaded regions are the values of $z = \\Delta \\lambda$
for which the integrator preserves Lyapunov stability of the test system
$\\dot x = \\lambda x$.

Visual takeaways:
* Forward Euler — tiny disk; not A-stable.
* Backward Euler — exterior of a unit disk around +1; A-stable.
* Bilinear — exactly the closed LHP; A-stable.
* ZOH — exactly the closed LHP; A-stable AND L-stable.

Output
------
``public/figures/ch02/stability_regions.png`` (referenced from §2.4).
"""

from __future__ import annotations

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
_OUTPUT_PATH = _REPO_ROOT / "public" / "figures" / "ch02" / "stability_regions"

# Stability functions R(z) for each integrator.
StabilityFn = Callable[[np.ndarray], np.ndarray]
_INTEGRATORS: dict[str, StabilityFn] = {
    "Forward Euler\n$R(z) = 1 + z$": lambda z: 1.0 + z,
    "Backward Euler\n$R(z) = 1/(1-z)$": lambda z: 1.0 / (1.0 - z),
    "Bilinear (trapezoidal)\n$R(z) = (1+z/2)/(1-z/2)$": lambda z: (1.0 + z / 2.0) / (1.0 - z / 2.0),
    "ZOH (matrix exp)\n$R(z) = e^z$": lambda z: np.exp(z),
}


def stability_grid(
    x_range: tuple[float, float] = (-6.0, 4.0),
    y_range: tuple[float, float] = (-5.0, 5.0),
    resolution: int = 400,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate a complex grid for stability-region evaluation.

    Returns
    -------
    X, Y : ndarrays of shape (resolution, resolution)
        Real and imaginary parts of the grid.
    Z : ndarray of shape (resolution, resolution)
        Complex grid Z = X + 1j*Y.
    """
    x = np.linspace(x_range[0], x_range[1], resolution)
    y = np.linspace(y_range[0], y_range[1], resolution)
    X, Y = np.meshgrid(x, y)
    Z = X + 1j * Y
    return X, Y, Z


def make_figure() -> plt.Figure:
    """Build the 2x2 stability-region grid."""
    apply_style()
    fig, axes = create_tufte_figure(nrows=2, ncols=2, figsize=(11.0, 9.0))
    X, Y, Z = stability_grid()

    for ax, (label, R) in zip(axes.flat, _INTEGRATORS.items()):  # type: ignore[union-attr]
        # |R(z)| evaluated on the grid; some integrators (backward Euler) may
        # produce NaN/inf at poles — clamp to a large value for contourf.
        with np.errstate(divide="ignore", invalid="ignore"):
            magnitude = np.abs(R(Z))
        magnitude = np.where(np.isfinite(magnitude), magnitude, np.inf)

        # Shade the stable region (|R(z)| <= 1).
        ax.contourf(X, Y, magnitude, levels=[0.0, 1.0],
                    colors=[SSM_COLORS["accent"]], alpha=0.25)
        # Boundary curve.
        ax.contour(X, Y, magnitude, levels=[1.0],
                   colors=[SSM_COLORS["accent"]], linewidths=1.4)
        # Axes.
        ax.axhline(0.0, color=SSM_COLORS["baseline"], linewidth=0.6)
        ax.axvline(0.0, color=SSM_COLORS["baseline"], linewidth=0.6)
        ax.set_xlim(-6.0, 4.0)
        ax.set_ylim(-5.0, 5.0)
        ax.set_aspect("equal")
        set_tufte_title(ax, label, fontsize=10)
        set_tufte_labels(ax, xlabel=r"$\Re(z)$", ylabel=r"$\Im(z)$")

    fig.suptitle("Stability regions $\\{ z : |R(z)| \\leq 1 \\}$ for four integrators",
                 fontsize=12, y=1.0)
    fig.tight_layout()
    return fig


def main() -> None:
    fig = make_figure()
    paths = save_figure(fig, _OUTPUT_PATH, formats=("png",))
    plt.close(fig)
    for p in paths:
        print(f"Wrote {p}")


if __name__ == "__main__":
    main()
