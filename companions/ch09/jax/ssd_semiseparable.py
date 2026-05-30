r"""Chapter 9 §9.5-9.6 — SSD: the semiseparable-matrix view and attention duality.

Mamba-2 (Dao & Gu 2024, arXiv:2405.21060, "Transformers are SSMs") reads the
selective SSM of §9.1-9.3 as a single matrix multiply $y = M u + D u$, where the
$L \times L$ lower-triangular matrix

.. math::

    M_{kj} = \sum_{n} C_{k,n}\,
             \exp\!\Big(A_n \sum_{i=j+1}^{k} \Delta_i\Big)\,
             \Delta_j\, B_{j,n}, \qquad k \ge j,

collects the path-ordered taps of §9.2. Two facts make this powerful:

* **$M$ is $N$-semiseparable** (§9.5): every strictly-lower-triangular block of
  $M$ has rank $\le N$ (the state dimension). The recurrent computation of
  $y = Mu$ is the §9.3 scan ($O(LN)$); the dense computation is a masked matmul
  ($O(L^2 N)$). These are the *two modes of the same $M$* — the duality.
* **Scalar $A = a\,I$ gives masked linear attention** (§9.6): with one shared
  decay the entry factors as $M_{kj} = L_{kj}\,(C_k\!\cdot\!B_j)\,\Delta_j$, i.e.
  $M = L \circ (C B^\top)\,\mathrm{diag}(\Delta)$ where the decay mask
  $L_{kj} = \exp\!\big(a\sum_{i=j+1}^k \Delta_i\big)$ is **1-semiseparable**.
  That is attention with $C$ as queries, $B$ as keys, $u$ as values, and $L$
  replacing softmax — the bridge to Chapter 11.

The ``segsum`` trick (stable cumulative segment sum with $-\infty$ above the
diagonal, so $\exp$ zeroes the upper triangle) is ported from
``post_transformers/experiments/jax/week08/mamba2_ssd.py``; the explicit-matrix
and duality framing follow Dao & Gu (arXiv:2405.21060).

Output
------
``public/figures/ch09/semiseparable.png`` — heatmap of $M$ (left) and the
singular-value cliff of an off-diagonal block (right) that *is* the
$N$-semiseparable rank bound.

Usage
-----
::

    PYTHONPATH=. python companions/ch09/jax/ssd_semiseparable.py
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402

from companions.ch09.jax.selective_ssm import selective_apply  # noqa: E402

if TYPE_CHECKING:
    from matplotlib.figure import Figure

__all__ = [
    "segsum",
    "build_ssm_matrix",
    "ssd_apply_matmul",
    "masked_attention_form",
    "numerical_rank",
    "is_n_semiseparable",
]

_REPO_ROOT = Path(__file__).resolve().parents[3]
_OUT_DIR = _REPO_ROOT / "public" / "figures" / "ch09"


# ---------------------------------------------------------------------------
# §9.2/9.5 — the segment-sum decay and the semiseparable matrix
# ---------------------------------------------------------------------------


def segsum(x: jnp.ndarray) -> jnp.ndarray:
    r"""Stable cumulative segment sum: $y_{ij} = \sum_{k=j+1}^{i} x_k$ for $i\ge j$, else $-\infty$.

    With $-\infty$ in the strict upper triangle, ``exp(segsum(A*delta))`` is the
    lower-triangular decay matrix in one shot — upper entries exponentiate to
    zero, which is exactly causal propagation. Ported from the Week-8 source.

    Parameters
    ----------
    x : jnp.ndarray, shape (..., T)
        Per-step (already signed) log-decay values $A_n \Delta_t$.

    Returns
    -------
    y : jnp.ndarray, shape (..., T, T)
        ``y[..., i, j] = x[..., j+1] + ... + x[..., i]`` for ``i >= j`` else ``-inf``.
    """
    if x.ndim < 1:
        raise ValueError(f"x must have at least one axis, got shape {x.shape}")
    t = x.shape[-1]
    cum = jnp.cumsum(x, axis=-1)
    diff = cum[..., :, None] - cum[..., None, :]  # diff[...,i,j] = sum_{j+1..i}
    mask = jnp.tril(jnp.ones((t, t), dtype=bool), k=0)
    return jnp.where(mask, diff, -jnp.inf)


def build_ssm_matrix(
    A: jnp.ndarray,
    delta: jnp.ndarray,
    B: jnp.ndarray,
    C: jnp.ndarray,
) -> jnp.ndarray:
    r"""The $L\times L$ semiseparable matrix $M$ of the selective SSM (§9.5).

    $M_{kj} = \sum_n C_{k,n}\,\exp(A_n\sum_{i=j+1}^k\Delta_i)\,\Delta_j\,B_{j,n}$
    for $k\ge j$, and $0$ for $k<j$. Applying $M$ to the input reproduces the
    selective scan exactly: ``M @ u == selective_apply(..., D=0)`` (pinned in
    tests). This is the "quadratic / attention" mode of the same operator the
    scan computes recurrently.

    Parameters
    ----------
    A : jnp.ndarray, shape (N,)
        Diagonal modes (negative). Pass ``a * jnp.ones(N)`` for the scalar-$A$
        (Mamba-2) case.
    delta : jnp.ndarray, shape (L,)
        Positive per-step sizes.
    B, C : jnp.ndarray, shape (L, N)
        Per-step input/output matrices.

    Returns
    -------
    M : jnp.ndarray, shape (L, L)
        Lower-triangular semiseparable matrix.

    Raises
    ------
    ValueError
        If shapes are inconsistent.
    """
    if A.ndim != 1 or delta.ndim != 1:
        raise ValueError(f"A must be (N,) and delta (L,), got {A.shape} and {delta.shape}")
    n, length = A.shape[0], delta.shape[0]
    if B.shape != (length, n) or C.shape != (length, n):
        raise ValueError(f"B and C must be ({length}, {n}), got {B.shape} and {C.shape}")
    adt = A[:, None] * delta[None, :]  # (N, L)  already signed (A < 0)
    decay = jnp.exp(segsum(adt))  # (N, L, L); strict upper triangle -> 0
    # M[k,j] = sum_n C[k,n] decay[n,k,j] delta[j] B[j,n].
    return jnp.einsum("kn,nkj,j,jn->kj", C, decay, delta, B)


def ssd_apply_matmul(
    M: jnp.ndarray,
    D: jnp.ndarray | float,
    u: jnp.ndarray,
) -> jnp.ndarray:
    r"""The dense (quadratic) SSD mode: $y = M u + D u$."""
    if u.ndim != 1 or M.shape != (u.shape[0], u.shape[0]):
        raise ValueError(f"need square M (L,L) and u (L,), got {M.shape} and {u.shape}")
    return M @ u + jnp.asarray(D, dtype=M.dtype) * u


def masked_attention_form(
    a: float | jnp.ndarray,
    delta: jnp.ndarray,
    B: jnp.ndarray,
    C: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    r"""Scalar-$A$ duality (§9.6): $M = L \circ (C B^\top)\,\mathrm{diag}(\Delta)$.

    With a single shared decay $a$, the semiseparable matrix is a Hadamard
    product of the **1-semiseparable** causal decay mask
    $L_{kj} = \exp(a\sum_{i=j+1}^k \Delta_i)$ and the **attention-score** matrix
    $G_{kj} = C_k \cdot B_j$, then column-scaled by $\Delta_j$. This *is* masked
    linear attention ($C$=queries, $B$=keys, $L$=learned causal mask). It equals
    ``build_ssm_matrix(a * ones(N), delta, B, C)`` (pinned in tests).

    Parameters
    ----------
    a : float or scalar jnp.ndarray
        Shared (negative) decay rate.
    delta : jnp.ndarray, shape (L,)
    B, C : jnp.ndarray, shape (L, N)

    Returns
    -------
    M : jnp.ndarray, shape (L, L)
        The full semiseparable matrix.
    L_mask : jnp.ndarray, shape (L, L)
        The 1-semiseparable decay mask.
    G : jnp.ndarray, shape (L, L)
        The attention-score (Gram) matrix $C B^\top$.
    """
    a = jnp.asarray(a, dtype=delta.dtype)
    L_mask = jnp.exp(segsum(a * delta))  # (L, L), 1-semiseparable, upper -> 0
    G = C @ B.T  # (L, L) attention scores C_k . B_j
    M = L_mask * G * delta[None, :]  # scale source column j by delta_j
    return M, L_mask, G


# ---------------------------------------------------------------------------
# §9.5 — the semiseparable rank certificate
# ---------------------------------------------------------------------------


def numerical_rank(block: jnp.ndarray, rtol: float | None = None) -> int:
    r"""Numerical rank of a matrix block via SVD (NumPy ``matrix_rank`` convention).

    Threshold ``tol = rtol * sigma_max`` (default ``rtol`` set from machine eps
    and block size), counting singular values strictly above ``tol``. Documented
    explicitly so the §9.5 rank test is not brittle to a hidden default.
    """
    s = jnp.linalg.svd(block, compute_uv=False)
    if rtol is None:
        rtol = max(block.shape) * float(jnp.finfo(block.dtype).eps)
    tol = rtol * float(s[0]) if s.size else 0.0
    return int(jnp.sum(s > tol))


def is_n_semiseparable(
    M: jnp.ndarray,
    n_state: int,
    rtol: float | None = None,
) -> tuple[bool, list[int]]:
    r"""Check that every tested strictly-lower block of $M$ has rank $\le n\_state$.

    Samples the off-diagonal blocks $M[h:,\,:h]$ at $h \in \{L/4, L/2, 3L/4\}$
    (all entirely below the diagonal, so $k > j$ throughout). For a selective SSM
    with $N$ diagonal modes each block has rank exactly $\min(N, \text{block dim})$
    — the $N$-semiseparable property of §9.5.

    Returns
    -------
    ok : bool
        True iff all sampled block ranks are $\le n\_state$.
    ranks : list[int]
        The measured block ranks (for inspection / printing).
    """
    length = M.shape[0]
    cuts = sorted({length // 4, length // 2, (3 * length) // 4} - {0, length})
    ranks = [numerical_rank(M[h:, :h], rtol=rtol) for h in cuts]
    return all(r <= n_state for r in ranks), ranks


# ---------------------------------------------------------------------------
# Figure: M heatmap + off-diagonal singular-value cliff
# ---------------------------------------------------------------------------


def _demo_diagonal_system(n: int = 4, length: int = 48, d: int = 4, seed: int = 1):
    """A fixed diagonal-A selective system (N modes) for the semiseparable figure."""
    import numpy as np

    from companions.ch09.jax.selective_ssm import selection_from_input, stable_A

    rng = np.random.default_rng(seed)
    A = stable_A(jnp.asarray(rng.standard_normal(n)))
    x = jnp.asarray(rng.standard_normal((length, d)))
    w_delta = jnp.asarray(0.5 * rng.standard_normal(d))
    w_B = jnp.asarray(rng.standard_normal((d, n)))
    w_C = jnp.asarray(rng.standard_normal((d, n)))
    delta, B, C = selection_from_input(x, w_delta, w_B, w_C)
    return A, delta, B, C


def make_semiseparable_figure() -> Figure:
    """Left: |M| heatmap (lower-triangular). Right: off-diagonal singular values."""
    import numpy as np

    from companions._shared.plot_utils import (
        SSM_COLORS,
        apply_style,
        create_tufte_figure,
        set_tufte_labels,
        set_tufte_title,
    )

    apply_style()
    A, delta, B, C = _demo_diagonal_system()
    n = A.shape[0]
    M = np.asarray(build_ssm_matrix(A, delta, B, C))
    length = M.shape[0]

    block = jnp.asarray(M[length // 2 :, : length // 2])
    sv = np.asarray(jnp.linalg.svd(block, compute_uv=False))
    sv = sv / sv[0]  # normalize to sigma_1

    fig, (ax1, ax2) = create_tufte_figure(ncols=2, figsize=(11.0, 4.2))

    im = ax1.imshow(np.abs(M), cmap="magma", aspect="equal")
    set_tufte_title(ax1, r"$|M_{kj}|$: lower-triangular, not Toeplitz")
    set_tufte_labels(ax1, xlabel=r"source $j$", ylabel=r"target $k$")
    fig.colorbar(im, ax=ax1, fraction=0.046, pad=0.04)

    idx = np.arange(1, sv.size + 1)
    ax2.semilogy(idx, sv, "o-", color=SSM_COLORS["accent"], label="singular values")
    ax2.axvline(n + 0.5, color=SSM_COLORS["alert"], linewidth=0.8, linestyle="--",
                label=rf"$N = {n}$ (state dim)")
    set_tufte_title(ax2, "Off-diagonal block: rank $= N$")
    set_tufte_labels(ax2, xlabel="index", ylabel=r"$\sigma_i / \sigma_1$")
    ax2.legend(loc="upper right", fontsize=8, frameon=False)

    fig.tight_layout()
    return fig


def main() -> None:
    import matplotlib.pyplot as plt
    import numpy as np

    from companions._shared.plot_utils import save_figure
    from companions.ch09.jax.selective_ssm import selection_from_input, stable_A

    print("Chapter 9 — ssd_semiseparable.py")
    print("=" * 60)

    # Diagonal-A: recurrent scan == dense matmul (the two SSD modes of one M).
    rng = np.random.default_rng(2)
    n, length, d = 6, 64, 4
    A = stable_A(jnp.asarray(rng.standard_normal(n)))
    x = jnp.asarray(rng.standard_normal((length, d)))
    delta, B, C = selection_from_input(
        x,
        jnp.asarray(0.5 * rng.standard_normal(d)),
        jnp.asarray(rng.standard_normal((d, n))),
        jnp.asarray(rng.standard_normal((d, n))),
    )
    u = jnp.asarray(rng.standard_normal(length))

    y_scan = selective_apply(A, delta, B, C, 0.0, u, parallel=True)
    M = build_ssm_matrix(A, delta, B, C)
    y_mat = ssd_apply_matmul(M, 0.0, u)
    dual_resid = float(jnp.max(jnp.abs(y_scan - y_mat)))
    print(f"  recurrent scan == dense M@u: max diff = {dual_resid:.3e}  (§9.5 two modes: ~0)")

    ok, ranks = is_n_semiseparable(M, n)
    print(f"  M off-diagonal block ranks {ranks} <= N={n}: {ok}  (§9.5 N-semiseparable)")

    # Scalar-A: build_ssm_matrix == masked-attention form; mask is 1-semiseparable.
    a = -0.7
    M_scalar = build_ssm_matrix(jnp.full((n,), a), delta, B, C)
    M_attn, L_mask, _G = masked_attention_form(a, delta, B, C)
    attn_resid = float(jnp.max(jnp.abs(M_scalar - M_attn)))
    print(f"  scalar-A: build == L o (C B^T) diag(d): max diff = {attn_resid:.3e}  (§9.6 duality: ~0)")
    _ok1, mask_ranks = is_n_semiseparable(L_mask, 1)
    print(f"  decay mask L off-diagonal ranks {mask_ranks} (1-semiseparable)")

    _OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig = make_semiseparable_figure()
    for p in save_figure(fig, _OUT_DIR / "semiseparable", formats=("png",)):
        print(f"Wrote {p}")
    plt.close(fig)


if __name__ == "__main__":
    main()
