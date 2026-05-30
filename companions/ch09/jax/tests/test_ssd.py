r"""Tests for the Chapter 9 SSD / semiseparable companion (§9.5-9.6).

Pinned, load-bearing facts:

* **Two modes of one operator** (§9.5) — the recurrent selective scan and the
  dense matrix multiply $y = Mu + Du$ produce identical outputs.
* **$N$-semiseparability** (§9.5) — every strictly-lower block of $M$ has rank
  $\le N$ (the state dimension), with equality for generic modes.
* **Attention duality** (§9.6) — for scalar $A = a\,I$ the semiseparable matrix
  equals masked linear attention $L \circ (C B^\top)\,\mathrm{diag}(\Delta)$.
* **1-semiseparable decay mask** (§9.6) — the scalar-$A$ decay mask $L$ has
  rank-1 off-diagonal blocks.

All tests run in float64 and pin residuals below explicit tolerances.
"""

from __future__ import annotations

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

from companions.ch09.jax.selective_ssm import (  # noqa: E402
    selection_from_input,
    selective_apply,
    stable_A,
)
from companions.ch09.jax.ssd_semiseparable import (  # noqa: E402
    build_ssm_matrix,
    is_n_semiseparable,
    masked_attention_form,
    segsum,
    ssd_apply_matmul,
)


def _diagonal_system(n: int = 6, length: int = 48, d: int = 4, seed: int = 1):
    """A fixed diagonal-A selective system (N distinct modes)."""
    rng = np.random.default_rng(seed)
    A = stable_A(jnp.asarray(rng.standard_normal(n)))
    x = jnp.asarray(rng.standard_normal((length, d)))
    delta, B, C = selection_from_input(
        x,
        jnp.asarray(0.5 * rng.standard_normal(d)),
        jnp.asarray(rng.standard_normal((d, n))),
        jnp.asarray(rng.standard_normal((d, n))),
    )
    u = jnp.asarray(rng.standard_normal(length))
    return A, delta, B, C, u


# ---------------------------------------------------------------------------
# §9.5 — the semiseparable matrix and its two computational modes
# ---------------------------------------------------------------------------


def test_recurrent_equals_matmul() -> None:
    """The recurrent scan and the dense M@u are two modes of the same operator (§9.5)."""
    A, delta, B, C, u = _diagonal_system()
    d_feed = 0.5
    y_scan = selective_apply(A, delta, B, C, d_feed, u, parallel=True)
    matrix = build_ssm_matrix(A, delta, B, C)
    y_mat = ssd_apply_matmul(matrix, d_feed, u)
    assert jnp.max(jnp.abs(y_scan - y_mat)) < 1e-12


def test_ssm_matrix_is_n_semiseparable() -> None:
    """Every off-diagonal block of M has rank <= N, exactly N for generic modes (§9.5)."""
    n = 5
    A, delta, B, C, _ = _diagonal_system(n=n, length=48, seed=4)
    matrix = build_ssm_matrix(A, delta, B, C)
    ok, ranks = is_n_semiseparable(matrix, n)
    assert ok
    assert max(ranks) == n  # generic diagonal modes -> rank exactly N


# ---------------------------------------------------------------------------
# §9.6 — duality with masked linear attention
# ---------------------------------------------------------------------------


def test_scalar_A_equals_masked_attention() -> None:
    """Scalar A: build_ssm_matrix == masked attention L o (C B^T) diag(Delta) (§9.6)."""
    n = 6
    a = -0.7
    _, delta, B, C, _ = _diagonal_system(n=n, seed=5)
    m_build = build_ssm_matrix(jnp.full((n,), a), delta, B, C)
    m_attn, _l_mask, _g = masked_attention_form(a, delta, B, C)
    assert jnp.max(jnp.abs(m_build - m_attn)) < 1e-12


def test_decay_mask_is_one_semiseparable() -> None:
    """The scalar-A decay mask L is 1-semiseparable (rank-1 off-diagonal blocks) (§9.6)."""
    n = 6
    a = -0.7
    _, delta, B, C, _ = _diagonal_system(n=n, seed=6)
    _m, l_mask, _g = masked_attention_form(a, delta, B, C)
    ok, ranks = is_n_semiseparable(l_mask, 1)
    assert ok
    assert max(ranks) == 1


# ---------------------------------------------------------------------------
# §9.2 — the segment-sum decay helper
# ---------------------------------------------------------------------------


def test_segsum_small() -> None:
    """segsum matches a hand-computed cumulative segment sum, -inf above the diagonal."""
    x = jnp.asarray([1.0, 2.0, 3.0])
    s = segsum(x)  # s[i, j] = sum_{k=j+1}^i x_k for i >= j (diagonal = empty sum = 0)
    expected_lower = jnp.asarray([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [5.0, 3.0, 0.0]])
    lower = jnp.tril(jnp.ones((3, 3), dtype=bool))
    assert jnp.allclose(jnp.where(lower, s, 0.0), expected_lower)
    # Strict upper triangle is -inf (so exp(.) zeroes it for causal propagation).
    assert jnp.isneginf(s[0, 1]) and jnp.isneginf(s[0, 2]) and jnp.isneginf(s[1, 2])


def test_figure_builder_runs() -> None:
    """The §9.5 semiseparable figure builder executes end to end (no asserts on pixels)."""
    from companions.ch09.jax.ssd_semiseparable import make_semiseparable_figure

    assert make_semiseparable_figure() is not None
