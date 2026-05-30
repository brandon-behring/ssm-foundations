r"""Tests for the Chapter 9 selective-SSM core (§9.1-9.4).

Pinned, load-bearing facts:

* **Selective scan-equivalence** (§9.3) — the time-varying associative scan and
  the sequential recurrence produce identical states (the §8.6 primitive now fed
  input-dependent transitions).
* **Stability by construction** (§9.1) — $|\bar A_t| < 1$ for *any* parameters,
  via the $A = -e^{a_\mathrm{log}}$ / $\Delta_t = \mathrm{softplus}(\cdot)$
  parameterization (the §8.5 "sign trap defused", carried into the LTV setting).
* **Time-invariance is broken** (§9.1) — the input$\to$output map is
  lower-triangular but *not Toeplitz*, so no single convolution kernel exists.
* **Exact semantics** — ``selective_apply`` reproduces an explicit by-hand
  recurrence including the $\Delta_t B_t$ discretization and the $D$ feedthrough.

All tests run in float64 (``jax_enable_x64``) and pin residuals below explicit
tolerances so a regression in any companion fails loudly.
"""

from __future__ import annotations

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402

from companions.ch09.jax.selective_ssm import (  # noqa: E402
    discretize_selective,
    selection_from_input,
    selective_apply,
    stable_A,
)
from companions.ch09.jax.ssd_semiseparable import build_ssm_matrix  # noqa: E402


def _selective_system(n: int = 8, length: int = 48, d: int = 4, seed: int = 0):
    """A fixed reproducible input-dependent SISO selective system."""
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
# §9.3 — the selective scan is the §8.6 scan with time-varying transitions
# ---------------------------------------------------------------------------


def test_selective_scan_equals_sequential() -> None:
    """The selective associative scan equals the sequential recurrence (§9.3)."""
    A, delta, B, C, u = _selective_system()
    y_par = selective_apply(A, delta, B, C, 0.0, u, parallel=True)
    y_seq = selective_apply(A, delta, B, C, 0.0, u, parallel=False)
    assert jnp.max(jnp.abs(y_par - y_seq)) < 1e-12


def test_selective_apply_matches_hand_recurrence() -> None:
    """selective_apply reproduces an explicit by-hand recurrence (with D feedthrough)."""
    A = jnp.asarray([-1.0, -2.0])
    delta = jnp.asarray([0.5, 0.3, 0.7])
    B = jnp.asarray([[1.0, 0.0], [0.5, 1.0], [0.0, 1.0]])
    C = jnp.asarray([[1.0, 1.0], [1.0, 0.0], [0.0, 1.0]])
    u = jnp.asarray([1.0, -1.0, 2.0])
    d_feed = 0.25

    h = jnp.zeros(2)
    ys = []
    for t in range(3):
        abar_t = jnp.exp(delta[t] * A)
        h = abar_t * h + delta[t] * B[t] * u[t]
        ys.append(float(C[t] @ h + d_feed * u[t]))
    y_ref = jnp.asarray(ys)

    y_seq = selective_apply(A, delta, B, C, d_feed, u, parallel=False)
    y_par = selective_apply(A, delta, B, C, d_feed, u, parallel=True)
    assert jnp.max(jnp.abs(y_seq - y_ref)) < 1e-12
    assert jnp.max(jnp.abs(y_par - y_ref)) < 1e-12


# ---------------------------------------------------------------------------
# §9.1 — stability by construction, and the death of time-invariance
# ---------------------------------------------------------------------------


def test_selective_stable_for_any_parameters() -> None:
    r"""Stability by construction (§9.1): $A = -e^{a_\mathrm{log}} < 0$, so $|\bar A_t| \le 1$ always.

    The exact (float-safe) guarantee is $\mathrm{Re}(A) < 0$ for *every* parameter
    value — $-\exp(\cdot)$ is strictly negative for any input — which gives
    $|\bar A_t| = e^{\Delta_t A} \le 1$ for $\Delta_t > 0$: the discrete transition
    can never leave the unit disk. (At pathological scales $\Delta_t A$ underflows
    and $\bar A_t$ rounds to exactly $1$; the *strict* $|\bar A_t| < 1$ holds for
    finite modes and moderate steps, checked in the next test.)
    """
    rng = np.random.default_rng(1)
    for _ in range(100):
        A = stable_A(jnp.asarray(rng.standard_normal(16) * 10))  # extreme modes
        delta = jax.nn.softplus(jnp.asarray(rng.standard_normal(8) * 10))  # positive
        B = jnp.asarray(rng.standard_normal((8, 16)))
        abar, _ = discretize_selective(A, delta, B)
        assert jnp.all(A < 0.0)  # the exact structural guarantee
        assert jnp.all(jnp.abs(abar) <= 1.0)  # never unstable


def test_selective_strictly_contractive_for_moderate_params() -> None:
    r"""For finite modes and moderate steps, $|\bar A_t| < 1$ strictly (§9.1)."""
    rng = np.random.default_rng(11)
    for _ in range(50):
        A = stable_A(jnp.asarray(rng.standard_normal(8)))  # moderate
        delta = jax.nn.softplus(jnp.asarray(rng.standard_normal(6)))
        B = jnp.asarray(rng.standard_normal((6, 8)))
        abar, _ = discretize_selective(A, delta, B)
        assert jnp.all(jnp.abs(abar) < 1.0)


def test_selective_map_not_toeplitz() -> None:
    """Input-dependence breaks time-invariance: the I/O map is lower-triangular but not Toeplitz (§9.1).

    An LTI system (constant Delta, fixed B, C) gives a Toeplitz map -- a single
    kernel K with y = K * u. A selective system does not: the diagonals of M
    vary, so no single kernel exists. This is the formal content of the §8.3
    margin note Chapter 8 deferred to here.
    """

    def toeplitz_defect(matrix: jnp.ndarray) -> float:
        diff = jnp.abs(matrix[1:, 1:] - matrix[:-1, :-1])  # compare (k,j) vs (k-1,j-1)
        mask = jnp.tril(jnp.ones(diff.shape, dtype=bool))  # keep k >= j
        return float(jnp.max(jnp.where(mask, diff, 0.0)))

    n, length = 6, 32
    rng = np.random.default_rng(2)
    A = stable_A(jnp.asarray(rng.standard_normal(n)))

    # LTI baseline: one fixed Delta and input-independent B, C (rows all equal).
    const_delta = jnp.full((length,), 0.1)
    b_row = jnp.asarray(rng.standard_normal(n))
    c_row = jnp.asarray(rng.standard_normal(n))
    b_const = jnp.broadcast_to(b_row, (length, n))
    c_const = jnp.broadcast_to(c_row, (length, n))
    m_lti = build_ssm_matrix(A, const_delta, b_const, c_const)

    # Selective: input-dependent Delta, B, C (reuse this test's own A for a fair contrast).
    _, delta, b_sel, c_sel, _ = _selective_system(n=n, length=length, seed=3)
    m_sel = build_ssm_matrix(A, delta, b_sel, c_sel)

    assert toeplitz_defect(m_lti) < 1e-12  # LTI: constant diagonals (a single kernel)
    assert toeplitz_defect(m_sel) > 1e-2  # selective: diagonals vary (no single kernel)


# ---------------------------------------------------------------------------
# Validation + figure smoke
# ---------------------------------------------------------------------------


def test_discretize_selective_validates_shapes() -> None:
    """discretize_selective raises ValueError on inconsistent shapes (no silent broadcast)."""
    A = stable_A(jnp.zeros(4))
    with pytest.raises(ValueError):
        discretize_selective(A, jnp.ones(5), jnp.ones((6, 4)))  # delta/B length mismatch
    with pytest.raises(ValueError):
        discretize_selective(jnp.zeros((4, 4)), jnp.ones(5), jnp.ones((5, 4)))  # A not 1-D


def test_figure_builders_run() -> None:
    """The §9.2-9.4 figure builders execute end to end (no asserts on pixels)."""
    from companions.ch09.jax.selective_scan_demo import make_memory_figure, make_scan_figure
    from companions.ch09.jax.selective_vs_lti import make_selectivity_figure

    assert make_selectivity_figure() is not None
    assert make_scan_figure() is not None
    assert make_memory_figure() is not None
