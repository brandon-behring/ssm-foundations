"""Chapter 7 (PyTorch companion) — HiPPO-LegS as an ``nn.Module`` projection operator.

Mirrors ``companions/ch07/jax/hippo_matrix.py`` and ``hippo_reconstruction.py`` to make
the idiomatic-JAX vs idiomatic-PyTorch contrast concrete (the Phase 9 comparison
material) and to seed the architecture-code thread: HiPPO becomes a reusable
``nn.Module`` here, which Chapter 8's S4 layer extends.

JAX <-> PyTorch contrast
------------------------
* **Recurrence.** The JAX encoder threads the coefficient vector through
  ``jax.lax.scan`` (one fused, compiled pass). PyTorch is *define-by-run* and has no
  ``scan`` primitive, so the time-varying LegS recurrence is an eager Python ``for``
  loop inside ``Module.forward`` — the loop *is* the program. (Chapter 8 will reach for
  ``torch.linalg`` FFT/associative tricks; here the loop is the honest spelling.)
* **Matrix build.** JAX used a vectorized ``jnp.where`` over a ``meshgrid``; the torch
  spelling is ``torch.where`` over ``torch.meshgrid`` — same idiom, eager evaluation.
* **State.** The fixed HiPPO matrices are registered as **buffers**, not ``Parameter``s:
  they are *initialization*, not learned. (Chapter 8 revisits which parts become
  trainable in S4.) ``HiPPOEncoder`` therefore has zero learnable parameters.
* **Precision.** JAX enables float64 globally (``jax_enable_x64``); torch sets dtype per
  tensor, so we pass ``dtype=torch.float64`` explicitly.

Port credit
-----------
HiPPO-LegS construction follows ``experiments/refs/s4/src/models/hippo/hippo.py``
(Gu et al., 2020, arXiv:2008.07669) and ``post_transformers/experiments/jax/week04/
s4_hippo.py``. The reconstruction convention (normalized, non-alternating) was
calibrated in the JAX companion; this file reproduces its numbers bit-for-bit.

Usage
-----
::

    PYTHONPATH=. python companions/ch07/torch/hippo_operator.py
"""

from __future__ import annotations

import numpy as np
import torch
from scipy.special import eval_legendre
from torch import nn

_DTYPE = torch.float64


# ---------------------------------------------------------------------------
# The HiPPO-LegS matrices
# ---------------------------------------------------------------------------


def make_hippo_legs(n: int, *, dtype: torch.dtype = _DTYPE) -> tuple[torch.Tensor, torch.Tensor]:
    """HiPPO-LegS $A\\in\\R^{N\\times N}$ (lower-triangular, eigenvalues $-1,\\ldots,-N$), $B\\in\\R^{N\\times 1}$.

    Built from the §7.3 closed form with ``torch.where`` over a ``meshgrid`` — the eager
    PyTorch counterpart of the JAX companion's ``jnp.where`` construction.

    Raises
    ------
    ValueError
        If ``n < 1``.
    """
    if n < 1:
        raise ValueError(f"state dimension n must be >= 1, got {n}")
    q = torch.arange(n, dtype=dtype)
    i_idx, j_idx = torch.meshgrid(q, q, indexing="ij")
    lower = torch.sqrt((2.0 * i_idx + 1.0) * (2.0 * j_idx + 1.0))
    diag = i_idx + 1.0
    A = -torch.where(i_idx > j_idx, lower, torch.where(i_idx == j_idx, diag, torch.zeros_like(lower)))
    B = torch.sqrt(2.0 * q + 1.0).unsqueeze(-1)
    return A, B


class HiPPOEncoder(nn.Module):
    """HiPPO-LegS online projection operator as an ``nn.Module`` (zero learnable params).

    ``forward(u)`` runs the bilinear time-varying recurrence
    ``(I + A_pos/2k) c_k = (I - A_pos/2k) c_{k-1} + (B/k) u_k`` with ``A_pos = -A`` and
    returns the full coefficient trajectory of shape ``(L, N)``. The eager ``for`` loop
    is the deliberate contrast to the JAX ``lax.scan`` encoder.
    """

    def __init__(self, n: int, *, dtype: torch.dtype = _DTYPE) -> None:
        super().__init__()
        if n < 1:
            raise ValueError(f"state dimension n must be >= 1, got {n}")
        A, B = make_hippo_legs(n, dtype=dtype)
        self.n = n
        # A_pos = -A has positive eigenvalues so the discrete update contracts; see the
        # JAX companion for the sign discussion. Buffers, not Parameters: fixed init.
        self.register_buffer("A_pos", -A)
        self.register_buffer("B", B.squeeze(-1))
        self.register_buffer("eye", torch.eye(n, dtype=dtype))

    def forward(self, u: torch.Tensor) -> torch.Tensor:
        """Encode a 1-D signal ``u`` (shape ``(L,)``) into a coefficient trajectory ``(L, N)``."""
        if u.ndim != 1:
            raise ValueError(f"u must be 1-D, got shape {tuple(u.shape)}")
        c = torch.zeros(self.n, dtype=self.eye.dtype)
        traj = []
        for k in range(1, u.shape[0] + 1):
            lhs = self.eye + self.A_pos / (2.0 * k)
            rhs = (self.eye - self.A_pos / (2.0 * k)) @ c + (self.B / k) * u[k - 1]
            c = torch.linalg.solve(lhs, rhs)
            traj.append(c)
        return torch.stack(traj)


def legs_eigenvalues(n: int) -> torch.Tensor:
    """Eigenvalues of the HiPPO-LegS $A$ (real; equal to $-1,\\ldots,-N$)."""
    return torch.linalg.eigvals(make_hippo_legs(n)[0])


# ---------------------------------------------------------------------------
# Reconstruction (host-side NumPy/SciPy — identical basis to the JAX companion)
# ---------------------------------------------------------------------------


def legendre_basis(n: int, z: np.ndarray, *, normalized: bool = True) -> np.ndarray:
    """Normalized shifted Legendre basis $\\sqrt{2i+1}\\,P_i(2z-1)$, shape ``(n, len(z))``."""
    idx = np.arange(n)
    polys = np.stack([eval_legendre(i, 2.0 * z - 1.0) for i in idx])
    gamma = np.sqrt(2.0 * idx + 1.0) if normalized else np.ones(n)
    return gamma[:, None] * polys


def reconstruct(c: np.ndarray, z: np.ndarray) -> np.ndarray:
    """Reconstruct the history $\\hat u(z)=\\sum_n c_n\\sqrt{2n+1}P_n(2z-1)$."""
    return c @ legendre_basis(len(c), z)


_L: int = 1000


def test_signal(z: np.ndarray) -> np.ndarray:
    """Smooth band-limited history: two sinusoids on $z\\in[0,1]$ (matches the JAX companion)."""
    return np.sin(2.0 * np.pi * 1.5 * z) + 0.5 * np.sin(2.0 * np.pi * 4.0 * z)


def reconstruction_error(n: int) -> float:
    """Relative $L^2$ reconstruction error at the final time for state dimension $N$."""
    z = np.linspace(0.0, 1.0, _L)
    truth = test_signal(z)
    encoder = HiPPOEncoder(n)
    with torch.no_grad():
        c_final = encoder(torch.as_tensor(truth, dtype=_DTYPE))[-1].numpy()
    approx = reconstruct(c_final, z)
    return float(np.linalg.norm(approx - truth) / np.linalg.norm(truth))


def main() -> None:
    print("Chapter 7 (torch) — hippo_operator.py")
    print("=" * 60)
    for n in (4, 8, 16):
        eigs = np.sort(legs_eigenvalues(n).real.numpy())
        print(f"  N={n:2d} eigenvalues (sorted real) = {np.round(eigs, 3)}")
    print("  Reconstruction error vs N:")
    for n in (4, 8, 16, 32, 64):
        print(f"    N={n:>3}: {reconstruction_error(n):.4e}")
    enc = HiPPOEncoder(8)
    print(f"  HiPPOEncoder(8) learnable params: {sum(p.numel() for p in enc.parameters())}")


if __name__ == "__main__":
    main()
