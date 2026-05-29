"""Chapter 3 (PyTorch companion) — condition-number growth across matrix families.

Mirrors ``companions/ch03/jax/condition_number.py`` for the JAX↔PyTorch comparison
(Phase 9 notebooks). Computes $\\kappa(A)$ vs size $N$ for the HiPPO-LegS, Hilbert,
and random-Gaussian families. The canonical Figure 3.1 stays JAX-owned; this module
is a computational counterpart — it reproduces the same numbers (notably the
HiPPO-LegS $\\kappa \\sim N^2$ growth, *not* boundedness — see 0527-F14) in idiomatic
PyTorch.

JAX↔PyTorch contrast
--------------------
* **Construction.** ``torch.where`` on an index grid mirrors ``jnp.where``; both
  replace the NumPy double loop. (PyTorch could also assign in place, since its
  tensors are mutable — see the ch02 companion.)
* **Conditioning + precision.** ``torch.linalg.cond`` replaces ``jnp.linalg.cond``;
  precision is set per tensor (``dtype=torch.float64``) rather than via JAX's global
  ``jax_enable_x64``.

Usage
-----
::

    PYTHONPATH=. python companions/ch03/torch/condition_number.py
"""

from __future__ import annotations

import numpy as np
import torch

_DTYPE = torch.float64


def hippo_legs(N: int) -> torch.Tensor:
    """HiPPO-LegS matrix via two ``torch.where`` masks on the $(i, j)$ index grid.

    $A_{ij} = -\\sqrt{(2i+1)(2j+1)}$ for $i>j$; $-(i+1)$ for $i=j$; $0$ otherwise.

    Raises
    ------
    ValueError
        If ``N < 1``.
    """
    if N < 1:
        raise ValueError(f"N must be positive, got {N}")
    i = torch.arange(N, dtype=_DTYPE).unsqueeze(1)
    j = torch.arange(N, dtype=_DTYPE).unsqueeze(0)
    off_diag = -torch.sqrt((2.0 * i + 1.0) * (2.0 * j + 1.0))
    diag = -(i + 1.0)
    A = torch.where(i > j, off_diag, torch.zeros((), dtype=_DTYPE))
    return torch.where(i == j, diag, A)


def hilbert(N: int) -> torch.Tensor:
    """Hilbert matrix $H_{ij} = 1/(i + j - 1)$."""
    idx = torch.arange(1, N + 1, dtype=_DTYPE)
    return 1.0 / (idx.unsqueeze(1) + idx.unsqueeze(0) - 1.0)


def random_gaussian(N: int, seed: int = 0) -> torch.Tensor:
    """N x N standard-Gaussian matrix with a fixed seed (via ``torch.Generator``)."""
    gen = torch.Generator().manual_seed(seed)
    return torch.randn((N, N), dtype=_DTYPE, generator=gen)


def condition_numbers(sizes: list[int]) -> dict[str, np.ndarray]:
    """``torch.linalg.cond`` for each family over ``sizes`` (Hilbert capped at N<=32)."""
    gauss = np.array([float(torch.linalg.cond(random_gaussian(N))) for N in sizes])
    hippo = np.array([float(torch.linalg.cond(hippo_legs(N))) for N in sizes])
    hilb = np.array([float(torch.linalg.cond(hilbert(N))) for N in sizes if N <= 32])
    return {"gaussian": gauss, "hippo": hippo, "hilbert": hilb}


def main() -> None:
    print("Chapter 3 (torch) — condition_number.py")
    print("=" * 60)
    sizes = [8, 16, 32, 64, 128]
    kappa = condition_numbers(sizes)
    slope = float(np.polyfit(np.log(sizes), np.log(kappa["hippo"]), 1)[0])
    print(f"  HiPPO-LegS κ log-log slope ≈ {slope:.3f} (≈ 2 ⇒ quadratic growth, not bounded)")
    print(f"  κ(HiPPO, N=128)/κ(HiPPO, N=8) = {kappa['hippo'][-1] / kappa['hippo'][0]:.1f}")


if __name__ == "__main__":
    main()
