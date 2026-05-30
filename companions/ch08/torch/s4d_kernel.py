r"""Chapter 8 §8.5 (PyTorch companion) — S4D diagonal kernel as an ``nn.Module``.

Mirrors ``companions/ch08/jax/s4d_kernel.py``. The math is identical — the S4D-Lin
diagonal $A_n = -\tfrac12 + i\pi n$ and the Vandermonde kernel
$K_l = 2\,\mathrm{Re}(\sum_n C_n \frac{\bar A_n - 1}{A_n}\bar A_n^{\,l})$ — so the two
frameworks compute the same object bit-for-bit (pinned in
``tests/test_s4d_torch.py``).

JAX <-> PyTorch contrast
------------------------
* **Kernel build.** Both spell the Vandermonde sum with broadcasting; JAX uses
  ``jnp.exp``/``jnp.einsum``-style reductions, torch uses ``torch.exp`` and
  ``torch.einsum`` over ``cdouble`` tensors. No control flow — it is a closed form.
* **Parameterization as a module.** :class:`S4DKernel` registers ``log_A_real``,
  ``A_imag`` and the complex ``C`` as **buffers** (fixed init for this companion;
  the reference S4D makes them learnable ``Parameter``s with per-group learning
  rates). ``forward(L)`` returns the length-$L$ kernel.
* **Precision.** JAX enables complex128 globally (via ``jax_enable_x64``); torch
  sets dtype per tensor, so we use ``torch.complex128`` (``cdouble``) explicitly to
  match. The reference S4D used ``cfloat``; complex128 keeps the §7.5 conditioning
  story honest and the cross-framework check tight.

Port credit
-----------
Vandermonde kernel and S4D-Lin init follow
``post_transformers/experiments/refs/s4/models/s4/s4d.py`` (Gu et al., S4D,
arXiv:2206.11893), reduced to the kernel core (the ``DropoutNd`` / output-mixing
plumbing of the reference ``S4D`` block is dropped).

Usage
-----
::

    PYTHONPATH=. python companions/ch08/torch/s4d_kernel.py
"""

from __future__ import annotations

import math

import torch
from torch import nn

_RDTYPE = torch.float64
_CDTYPE = torch.complex128

__all__ = ["make_s4d_lin", "s4d_kernel", "S4DKernel"]


def make_s4d_lin(n_modes: int, *, dtype: torch.dtype = _CDTYPE) -> torch.Tensor:
    r"""The S4D-Lin diagonal $A_n = -\tfrac12 + i\pi n$ (shape ``(n_modes,)``, complex).

    Raises
    ------
    ValueError
        If ``n_modes < 1``.
    """
    if n_modes < 1:
        raise ValueError(f"n_modes must be >= 1, got {n_modes}")
    n = torch.arange(n_modes, dtype=_RDTYPE)
    return (-0.5 + 1j * math.pi * n).to(dtype)


def s4d_kernel(
    A: torch.Tensor,
    C: torch.Tensor,
    dt: float,
    L: int,
) -> torch.Tensor:
    r"""S4D Vandermonde kernel $K_l = 2\,\mathrm{Re}(\sum_n C_n \frac{\bar A_n-1}{A_n}\bar A_n^l)$.

    Parameters
    ----------
    A : torch.Tensor, shape (M,), complex
        Diagonal modes.
    C : torch.Tensor, shape (M,), complex
        Output weights.
    dt : float
        Discretization step $\Delta$.
    L : int
        Number of kernel taps.

    Returns
    -------
    K : torch.Tensor, shape (L,), real (float64)

    Raises
    ------
    ValueError
        If ``A`` and ``C`` differ in length, or ``L < 1``.
    """
    if A.shape != C.shape:
        raise ValueError(f"A and C must share shape, got {tuple(A.shape)} vs {tuple(C.shape)}")
    if L < 1:
        raise ValueError(f"L must be >= 1, got {L}")
    if torch.any(A == 0):
        raise ValueError(
            "diagonal modes A must be nonzero (a zero mode has no ZOH input (Abar-1)/A)"
        )
    dtA = A * dt
    Abar = torch.exp(dtA)
    Ctilde = C * (Abar - 1.0) / A
    taps = torch.arange(L, dtype=_RDTYPE)
    powers = torch.exp(dtA.unsqueeze(-1) * taps.unsqueeze(0))  # (M, L) = Abar^l
    return 2.0 * torch.einsum("m,ml->l", Ctilde, powers).real


class S4DKernel(nn.Module):
    """S4D-Lin kernel generator as an ``nn.Module`` (fixed-init buffers, no learnable params).

    ``forward(L)`` returns the length-``L`` real kernel. Construction uses the
    stability-by-construction parameterization $A = -e^{\\texttt{log\\_A\\_real}} +
    i\\,\\texttt{A\\_imag}$, so the modes stay in the left half-plane regardless of the
    stored values.
    """

    def __init__(self, n_modes: int, seed: int = 0) -> None:
        super().__init__()
        if n_modes < 1:
            raise ValueError(f"n_modes must be >= 1, got {n_modes}")
        self.n_modes = n_modes
        gen = torch.Generator().manual_seed(seed)
        log_A_real = torch.log(0.5 * torch.ones(n_modes, dtype=_RDTYPE))
        A_imag = math.pi * torch.arange(n_modes, dtype=_RDTYPE)
        C = torch.randn(n_modes, dtype=_RDTYPE, generator=gen) + 1j * torch.randn(
            n_modes, dtype=_RDTYPE, generator=gen
        )
        self.register_buffer("log_A_real", log_A_real)
        self.register_buffer("A_imag", A_imag)
        self.register_buffer("C", C.to(_CDTYPE))

    def diagonal(self) -> torch.Tensor:
        r"""Assemble $A = -e^{\texttt{log\_A\_real}} + i\,\texttt{A\_imag}$ (Re < 0 by construction)."""
        return -torch.exp(self.log_A_real) + 1j * self.A_imag

    def forward(self, L: int, dt: float = 0.1) -> torch.Tensor:
        """Return the length-``L`` S4D kernel at step ``dt``."""
        return s4d_kernel(self.diagonal(), self.C, dt, L)


def main() -> None:
    print("Chapter 8 (torch) — s4d_kernel.py")
    print("=" * 60)
    kernel = S4DKernel(16)
    A = kernel.diagonal()
    print(f"  S4D-Lin: max Re(A) = {float(A.real.max()):.4f}  (< 0 by construction)")
    K = kernel(64)
    print(
        f"  kernel: K[0] = {float(K[0]):.4f}, max|K| = {float(K.abs().max()):.4f}, dtype = {K.dtype}"
    )
    print(f"  S4DKernel learnable params: {sum(p.numel() for p in kernel.parameters())}")


if __name__ == "__main__":
    main()
