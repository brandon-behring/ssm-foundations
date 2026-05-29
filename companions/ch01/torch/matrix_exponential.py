"""Chapter 1 (PyTorch companion) — matrix-exponential series vs ``torch.linalg.matrix_exp``.

Mirrors ``companions/ch01/jax/matrix_exponential.py`` to make the idiomatic-JAX vs
idiomatic-PyTorch contrast concrete (this is the comparison material for the Phase 9
notebooks). Same pedagogy as §1.2: a truncated power series converges fast for small
$\\|M\\|$ and catastrophically slowly for large $\\|M\\|$.

JAX↔PyTorch contrast
--------------------
* **Recurrence.** The JAX companion threads ``(term, total)`` through ``jax.lax.scan``
  and emits every partial sum in one fused, compiled pass. PyTorch is *define-by-run*
  and has no ``scan`` primitive, so the same recurrence is a plain eager Python loop —
  the loop *is* the program. (``torch.compile`` could fuse it, but the eager loop is
  the idiomatic spelling.)
* **Per-order error sweep.** JAX used ``jax.vmap``; PyTorch's ``torch.linalg.norm`` with
  ``dim=(1, 2)`` batches the Frobenius norm over the leading axis directly.
* **Reference.** ``torch.linalg.matrix_exp`` is a *native* matrix exponential; the JAX
  companion borrowed ``scipy.linalg.expm``.
* **Precision.** JAX enables float64 globally (``jax_enable_x64``); PyTorch sets
  precision per tensor, so we pass ``dtype=torch.float64`` explicitly.

Usage
-----
::

    PYTHONPATH=. python companions/ch01/torch/matrix_exponential.py
"""

from __future__ import annotations

import numpy as np
import torch

_DTYPE = torch.float64


def series_partial_sums(M: torch.Tensor | np.ndarray, k_max: int) -> torch.Tensor:
    """All partial sums $S_0, \\ldots, S_{k_{\\max}}$ of $e^M$ via an eager loop.

    PyTorch contrast: where the JAX companion uses ``jax.lax.scan``, this runs the
    $S_k = S_{k-1} + M^k/k!$ recurrence as a define-by-run Python loop, appending each
    partial sum. Returns a stacked tensor of shape ``(k_max + 1, N, N)``.
    """
    M = torch.as_tensor(M, dtype=_DTYPE)
    n = M.shape[0]
    term = torch.eye(n, dtype=_DTYPE)
    total = torch.eye(n, dtype=_DTYPE)
    partials = [total.clone()]
    for k in range(1, k_max + 1):
        term = term @ M / k  # M^k / k!  from  M^{k-1}/(k-1)!
        total = total + term
        partials.append(total.clone())
    return torch.stack(partials)


def truncated_series(M: torch.Tensor | np.ndarray, K: int) -> torch.Tensor:
    """Single partial sum $S_K = \\sum_{k=0}^{K} M^k/k!$ (validated wrapper).

    Raises
    ------
    ValueError
        If ``K < 0`` or ``M`` is not square.
    """
    if K < 0:
        raise ValueError(f"truncation order K must be non-negative, got {K}")
    M = torch.as_tensor(M, dtype=_DTYPE)
    if M.ndim != 2 or M.shape[0] != M.shape[1]:
        raise ValueError(f"M must be a square matrix, got shape {tuple(M.shape)}")
    return series_partial_sums(M, K)[K]


def convergence_errors(M: torch.Tensor | np.ndarray, k_max: int) -> np.ndarray:
    """Relative Frobenius error of $S_0..S_{k_{\\max}}$ vs ``torch.linalg.matrix_exp``.

    The per-order error is a batched ``torch.linalg.norm`` over the stacked partial
    sums — the PyTorch counterpart of the JAX companion's ``jax.vmap``.
    """
    M = torch.as_tensor(M, dtype=_DTYPE)
    reference = torch.linalg.matrix_exp(M)
    partials = series_partial_sums(M, k_max)  # (k_max+1, N, N)
    errs = torch.linalg.norm(partials - reference, dim=(1, 2)) / torch.linalg.norm(reference)
    return errs.numpy()


def main() -> None:
    print("Chapter 1 (torch) — matrix_exponential.py")
    print("=" * 60)
    A_small = np.array([[-0.5, 1.0], [-1.0, -0.5]])
    A_large = np.array([[-5.0, 10.0], [-10.0, -5.0]])
    for name, A in (("‖M‖≈1", A_small), ("‖M‖≈14", A_large)):
        errs = convergence_errors(A, 39)
        print(f"  {name:8s}: error at K=5 = {errs[5]:.2e}, at K=39 = {errs[-1]:.2e}")


if __name__ == "__main__":
    main()
