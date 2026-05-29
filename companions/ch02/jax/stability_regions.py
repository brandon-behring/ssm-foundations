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

Idiomatic-JAX note (this companion is a NumPy→JAX teaching example)
------------------------------------------------------------------
The stability functions $R(z)$ are closed forms, so they *already* vectorize over
the whole complex meshgrid — there is no Python loop to replace with ``vmap``.
The one genuine NumPy→JAX difference is error handling at the poles:

* **``jnp.where(jnp.isfinite(...), ...)`` instead of a ``np.errstate`` context.**
  NumPy needs ``with np.errstate(divide="ignore")`` to suppress the warning when
  $R(z) = 1/(1-z)$ hits its pole; JAX follows silent IEEE semantics (it returns
  ``inf``/``nan`` without warning), so the mask is applied directly.

Output
------
``public/figures/ch02/stability_regions.png`` (referenced from §2.4).

Usage
-----
::

    PYTHONPATH=. python companions/ch02/jax/stability_regions.py
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from companions._shared.plot_utils import (  # noqa: E402
    SSM_COLORS,
    apply_style,
    create_tufte_figure,
    save_figure,
    set_tufte_labels,
    set_tufte_title,
)

_REPO_ROOT = Path(__file__).resolve().parents[3]
_OUTPUT_PATH = _REPO_ROOT / "public" / "figures" / "ch02" / "stability_regions"


# Stability functions R(z) for each integrator (vectorize over any jnp array).
def forward_euler_R(z: jnp.ndarray) -> jnp.ndarray:
    """$R(z) = 1 + z$ — explicit/forward Euler."""
    return 1.0 + z


def backward_euler_R(z: jnp.ndarray) -> jnp.ndarray:
    """$R(z) = 1/(1 - z)$ — implicit/backward Euler."""
    return 1.0 / (1.0 - z)


def bilinear_R(z: jnp.ndarray) -> jnp.ndarray:
    """$R(z) = (1 + z/2)/(1 - z/2)$ — bilinear/trapezoidal (Cayley)."""
    return (1.0 + z / 2.0) / (1.0 - z / 2.0)


def zoh_R(z: jnp.ndarray) -> jnp.ndarray:
    """$R(z) = e^z$ — zero-order hold / exact matrix exponential."""
    return jnp.exp(z)


StabilityFn = Callable[[jnp.ndarray], jnp.ndarray]
_INTEGRATORS: dict[str, StabilityFn] = {
    "Forward Euler\n$R(z) = 1 + z$": forward_euler_R,
    "Backward Euler\n$R(z) = 1/(1-z)$": backward_euler_R,
    "Bilinear (trapezoidal)\n$R(z) = (1+z/2)/(1-z/2)$": bilinear_R,
    "ZOH (matrix exp)\n$R(z) = e^z$": zoh_R,
}


def stability_grid(
    x_range: tuple[float, float] = (-6.0, 4.0),
    y_range: tuple[float, float] = (-5.0, 5.0),
    resolution: int = 400,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Generate a complex grid for stability-region evaluation.

    Returns
    -------
    X, Y : jnp.ndarrays of shape (resolution, resolution)
        Real and imaginary parts of the grid.
    Z : jnp.ndarray of shape (resolution, resolution)
        Complex grid Z = X + 1j*Y.
    """
    x = jnp.linspace(x_range[0], x_range[1], resolution)
    y = jnp.linspace(y_range[0], y_range[1], resolution)
    X, Y = jnp.meshgrid(x, y)
    Z = X + 1j * Y
    return X, Y, Z


def stability_magnitude(R: StabilityFn, Z: jnp.ndarray) -> jnp.ndarray:
    """$|R(z)|$ over the grid, with non-finite values (poles) sent to $+\\infty$.

    The ``jnp.where(isfinite, ...)`` mask is the JAX replacement for NumPy's
    ``with np.errstate(divide='ignore'): ...`` pole handling.
    """
    magnitude = jnp.abs(R(Z))
    return jnp.where(jnp.isfinite(magnitude), magnitude, jnp.inf)


def make_figure() -> plt.Figure:
    """Build the 2x2 stability-region grid."""
    apply_style()
    fig, axes = create_tufte_figure(nrows=2, ncols=2, figsize=(11.0, 9.0))
    X, Y, Z = stability_grid()
    X_np, Y_np = np.asarray(X), np.asarray(Y)

    for ax, (label, R) in zip(axes.flat, _INTEGRATORS.items()):  # type: ignore[union-attr]
        magnitude = np.asarray(stability_magnitude(R, Z))

        # Shade the stable region (|R(z)| <= 1).
        ax.contourf(X_np, Y_np, magnitude, levels=[0.0, 1.0],
                    colors=[SSM_COLORS["accent"]], alpha=0.25)
        # Boundary curve.
        ax.contour(X_np, Y_np, magnitude, levels=[1.0],
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
