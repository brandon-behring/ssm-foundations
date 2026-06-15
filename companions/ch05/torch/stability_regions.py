"""Chapter 5 (PyTorch companion) — RK stability functions $R(z)$ over the complex plane.

Mirrors ``companions/ch05/jax/stability_regions.py`` for the JAX↔PyTorch contrast.
This is the **compute-and-parity** port: it reproduces the numerical core (the stability
function $R(z) = 1 + z\\,b^\\top (I - zA)^{-1}\\mathbf{1}$ for explicit Runge-Kutta methods,
plus the closed-form ZOH and bilinear functions) but draws no figures — the figure-
generating helpers (``make_*_figure`` / ``save_figure``) live only in the JAX companion.

JAX↔PyTorch contrast
--------------------
* **Per-grid-point solve.** The JAX companion maps a single-point ``jnp.linalg.solve``
  over the flattened complex grid with ``jax.vmap``. PyTorch has no ``vmap`` requirement
  here: ``torch.linalg.solve`` is natively batched, so we stack the per-point matrices
  $(I - z A)$ along a leading axis and solve them all in one call. (The NumPy original
  ran a Python ``for z_val in z.ravel()`` loop; both frameworks vectorise it.)
* **Complex inputs.** The complex grid itself is built only in the JAX figure module;
  this compute-and-parity port coerces whatever real/complex array it is handed to
  ``complex128`` (vs JAX's ``jnp.asarray`` on the precomputed ``RE + 1j*IM`` meshgrid).
  ``jnp.abs`` on a complex array becomes ``torch.abs``.
* **Closed forms.** ``zoh_stab_fn`` ($e^z$) and ``bilinear_stab_fn`` already vectorise;
  only ``jnp`` → ``torch`` changes.
* **Precision.** JAX enables float64 globally (``jax_enable_x64``); PyTorch sets
  precision per tensor, so real grids use ``torch.float64`` and complex grids
  ``torch.complex128`` explicitly.

Usage
-----
::

    PYTHONPATH=. python companions/ch05/torch/stability_regions.py
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import torch

_DTYPE = torch.float64
_CDTYPE = torch.complex128


# ---------------------------------------------------------------------------
# Butcher tableaux
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Tableau:
    """A Runge-Kutta Butcher tableau (A, b, c) with $s$ stages."""

    name: str
    A: torch.Tensor  # shape (s, s); strictly lower triangular for explicit methods
    b: torch.Tensor  # shape (s,)
    c: torch.Tensor  # shape (s,)
    order: int

    @property
    def s(self) -> int:
        return self.A.shape[0]


def forward_euler_tableau() -> Tableau:
    return Tableau(
        name="Forward Euler",
        A=torch.tensor([[0.0]], dtype=_DTYPE),
        b=torch.tensor([1.0], dtype=_DTYPE),
        c=torch.tensor([0.0], dtype=_DTYPE),
        order=1,
    )


def midpoint_rk2_tableau() -> Tableau:
    return Tableau(
        name="Midpoint RK2",
        A=torch.tensor([[0.0, 0.0], [0.5, 0.0]], dtype=_DTYPE),
        b=torch.tensor([0.0, 1.0], dtype=_DTYPE),
        c=torch.tensor([0.0, 0.5], dtype=_DTYPE),
        order=2,
    )


def classical_rk4_tableau() -> Tableau:
    return Tableau(
        name="Classical RK4",
        A=torch.tensor(
            [
                [0.0, 0.0, 0.0, 0.0],
                [0.5, 0.0, 0.0, 0.0],
                [0.0, 0.5, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
            ],
            dtype=_DTYPE,
        ),
        b=torch.tensor([1.0 / 6.0, 1.0 / 3.0, 1.0 / 3.0, 1.0 / 6.0], dtype=_DTYPE),
        c=torch.tensor([0.0, 0.5, 0.5, 1.0], dtype=_DTYPE),
        order=4,
    )


def rkf45_tableau() -> Tableau:
    """Runge-Kutta-Fehlberg 4(5) tableau (Fehlberg 1969, 5th-order weights)."""
    A = torch.tensor(
        [
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [1 / 4, 0.0, 0.0, 0.0, 0.0, 0.0],
            [3 / 32, 9 / 32, 0.0, 0.0, 0.0, 0.0],
            [1932 / 2197, -7200 / 2197, 7296 / 2197, 0.0, 0.0, 0.0],
            [439 / 216, -8.0, 3680 / 513, -845 / 4104, 0.0, 0.0],
            [-8 / 27, 2.0, -3544 / 2565, 1859 / 4104, -11 / 40, 0.0],
        ],
        dtype=_DTYPE,
    )
    b = torch.tensor(
        [16 / 135, 0.0, 6656 / 12825, 28561 / 56430, -9 / 50, 2 / 55], dtype=_DTYPE
    )
    c = torch.tensor([0.0, 1 / 4, 3 / 8, 12 / 13, 1.0, 1 / 2], dtype=_DTYPE)
    return Tableau(name="RKF 4(5)", A=A, b=b, c=c, order=5)


# ---------------------------------------------------------------------------
# Stability functions
#
# For an explicit RK method:  R(z) = 1 + z bᵀ (I - zA)⁻¹ 𝟙
# For the Chapter 4 schemes we use closed forms.
# ---------------------------------------------------------------------------


def explicit_stab_fn(tab: Tableau) -> Callable[[torch.Tensor | np.ndarray], torch.Tensor]:
    """Build $R(z) = 1 + z b^\\top (I - z A)^{-1} \\mathbf{1}$ for an explicit tableau.

    The per-point linear solve is *batched* by ``torch.linalg.solve`` over the
    flattened complex grid (the PyTorch counterpart of the JAX companion's
    ``jax.vmap`` over ``z.ravel()`` — see the module note).

    Parameters
    ----------
    tab
        Explicit Butcher tableau; ``tab.A`` is strictly lower triangular.

    Returns
    -------
    Callable
        ``R(z)`` accepting a real or complex array/tensor of any shape and
        returning a ``complex128`` tensor of the same shape.
    """
    A = tab.A.to(_CDTYPE)
    b = tab.b.to(_CDTYPE)
    s = tab.s
    ones = torch.ones(s, dtype=_CDTYPE)
    eye = torch.eye(s, dtype=_CDTYPE)

    def R(z: torch.Tensor | np.ndarray) -> torch.Tensor:
        z = torch.as_tensor(z).to(_CDTYPE)
        flat = z.reshape(-1)  # (P,)
        # Batched (P, s, s) system  (I - z A) κ = 𝟙, solved in one call.
        mats = eye - flat[:, None, None] * A  # broadcast (P,1,1)*(s,s) -> (P,s,s)
        rhs = ones.expand(flat.shape[0], s)  # (P, s)
        kappa = torch.linalg.solve(mats, rhs)  # (P, s)
        vals = 1.0 + flat * (kappa @ b)  # (P,)
        return vals.reshape(z.shape)

    return R


def zoh_stab_fn(z: torch.Tensor | np.ndarray) -> torch.Tensor:
    """ZOH and exp-trapezoidal share $R(z) = e^z$ on the Dahlquist test problem."""
    return torch.exp(torch.as_tensor(z).to(_CDTYPE))


def bilinear_stab_fn(z: torch.Tensor | np.ndarray) -> torch.Tensor:
    """Bilinear (Tustin): $R(z) = (1 + z/2)/(1 - z/2)$."""
    z = torch.as_tensor(z).to(_CDTYPE)
    return (1.0 + z / 2.0) / (1.0 - z / 2.0)


# ---------------------------------------------------------------------------
# Entry point (numeric summary only — no figures)
# ---------------------------------------------------------------------------


def main() -> None:
    print("Chapter 5 (torch) — stability_regions.py")
    print("=" * 60)
    z = torch.tensor(
        [-1.0 + 0j, 0.3 - 0.5j, -2.0 + 1.0j, 0.1 + 0.0j, -0.5 - 2.0j], dtype=_CDTYPE
    )
    R_fe = explicit_stab_fn(forward_euler_tableau())
    R_rk4 = explicit_stab_fn(classical_rk4_tableau())
    print(f"  sample z            : {z.numpy()}")
    print(f"  forward Euler R(z)  : {R_fe(z).numpy()}")
    print(f"  classical RK4 R(z)  : {R_rk4(z).numpy()}")
    print(f"  ZOH / exp-trap e^z  : {zoh_stab_fn(z).numpy()}")
    print(f"  bilinear R(z)       : {bilinear_stab_fn(z).numpy()}")
    print("-" * 60)
    print(
        "  bilinear |R(-1)| = {:.4f}, |R(+1)| = {:.4f}, |R(3i)| = {:.4f}".format(
            float(torch.abs(bilinear_stab_fn(-1.0 + 0j))),
            float(torch.abs(bilinear_stab_fn(1.0 + 0j))),
            float(torch.abs(bilinear_stab_fn(3j))),
        )
    )


if __name__ == "__main__":
    main()
