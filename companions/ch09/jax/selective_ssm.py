r"""Chapter 9 §9.1-9.3 — Selective (input-dependent) SSM: the LTV core.

Mamba-1 (Gu & Dao 2023, arXiv:2312.00752) breaks the linear *time-invariance*
that Chapter 8's S4/S4D/S5 relied on: the step size $\Delta$, the input matrix
$B$, and the output matrix $C$ all become functions of the current input token,
while the diagonal state matrix $A$ stays a (stable) parameter. The discrete
transition $\bar A_t = e^{\Delta_t A}$ is therefore *different at every step* —
the system is now **linear time-varying (LTV) / non-autonomous** (§9.2).

Three consequences this module makes concrete:

* **§9.1** — there is no longer a single convolution kernel $K$ with
  $y = K * u$: the tap from step $j$ to step $k$ is
  $C_k\,\Phi(k,j)\,\bar B_j$ with the path-ordered product
  $\Phi(k,j) = \bar A_k \bar A_{k-1}\cdots \bar A_{j+1}$, which depends on the
  whole path, not on $k - j$. The input$\to$output map is lower-triangular but
  **not Toeplitz** (built and checked in ``ssd_semiseparable.py``).
* **§9.2** — $\Phi$ is the discrete state-transition operator of the LTV
  recurrence; the per-mode decay $\exp(A_n \sum_{i=j+1}^k \Delta_i)$ is what
  ``ssd_semiseparable.segsum`` materializes.
* **§9.3** — Chapter 8 §8.6's associative scan *still* computes the states,
  because the operator $(a_1,b_1)\bullet(a_2,b_2)=(a_2\odot a_1,\,a_2\odot b_1+b_2)$
  never assumed the $a$'s were equal. Feeding the *time-varying* pairs
  $(\bar A_t,\,\bar B_t u_t)$ is the **selective scan** — now the only route,
  since the convolution is gone.

Teaching choices
----------------
* **Single-channel (SISO)** state $h \in \R^N$, real diagonal $A$, no batch
  axis. The chapter is about the *dynamics* (LTV, scan, semiseparability,
  duality), not the full Mamba block (depthwise conv, gating, projections —
  those are in the Week-7 source cited below). The batched block is one extra
  ``einsum`` axis away.
* **Discretization follows Mamba's simplified form** $\bar A_t = e^{\Delta_t A}$,
  $\bar B_t = \Delta_t B_t$ (Gu & Dao §3.4), **not** Chapter 8's full ZOH
  $\bar B = (\bar A - 1)/A \cdot B$. Keeping $\Delta_t B_t$ makes §9.5's
  semiseparable-matrix entries $M_{kj} = \text{decay}\cdot(C_k\!\cdot\!B_j)\cdot\Delta_j$
  read off directly.
* **$A = -e^{a_\mathrm{log}}$** parameterization (the §8.5 S4D "sign trap,
  defused"): $\Re(A) < 0$ for *any* real parameter, and
  $\Delta_t = \mathrm{softplus}(\cdot) > 0$, so $|\bar A_t| < 1$ always.

Idiomatic-JAX note (this companion is a NumPy->JAX teaching example)
------------------------------------------------------------------
* **``lax.associative_scan`` over time-varying transitions.** A NumPy reference
  loops ``for t: h = Abar[t]*h + bu[t]`` ($O(L)$ depth). The selective scan
  stacks the per-step pairs ``(Abar, bu)`` of shape ``(L, N)`` and composes them
  with the affine-map operator in $O(\log L)$ depth — the *same* primitive as
  Chapter 8's S5 (``companions/ch08/jax/s5_scan.py``), now with a per-step
  ``Abar`` instead of one broadcast ``Abar``. That single change is the whole of
  LTI $\to$ LTV.

Port credit
-----------
Selective-scan structure ported (SISO-simplified) from
``post_transformers/experiments/jax/week07/mamba1.py`` and Gu & Dao, *Mamba:
Linear-Time Sequence Modeling with Selective State Spaces* (arXiv:2312.00752).
The associative operator is the Chapter 8 §8.6 operator
(``companions/ch08/jax/s5_scan.py``).

Usage
-----
::

    PYTHONPATH=. python companions/ch09/jax/selective_ssm.py
"""

from __future__ import annotations

import jax

# Enable float64 before any jnp array is created (matches Chapters 4, 7, 8).
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402

__all__ = [
    "stable_A",
    "selection_from_input",
    "discretize_selective",
    "selective_binary_operator",
    "selective_scan",
    "selective_sequential",
    "selective_apply",
]


# ---------------------------------------------------------------------------
# §9.1 — the selection mechanism: parameters become functions of the input
# ---------------------------------------------------------------------------


def stable_A(a_log: jnp.ndarray) -> jnp.ndarray:
    r"""Stable diagonal modes $A = -e^{a_\mathrm{log}}$ (real part $< 0$ for any input).

    The §8.5 S4D "sign trap, defused": routing $A$ through $-\exp(\cdot)$ pins
    every mode in the open left half-plane, so the per-step discrete transition
    $\bar A_t = e^{\Delta_t A}$ obeys $|\bar A_t| < 1$ for *every* parameter value
    and every positive step $\Delta_t$ — no constraint to enforce during training.

    Parameters
    ----------
    a_log : jnp.ndarray, shape (N,)
        Unconstrained real log-parameters.

    Returns
    -------
    A : jnp.ndarray, shape (N,)
        Diagonal modes, all strictly negative.
    """
    return -jnp.exp(a_log)


def selection_from_input(
    x: jnp.ndarray,
    w_delta: jnp.ndarray,
    w_B: jnp.ndarray,
    w_C: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    r"""Input-dependent $(\Delta_t, B_t, C_t)$ from a feature sequence — "selection".

    This is the one change that distinguishes Mamba from S4/S4D/S5: $\Delta$, $B$,
    and $C$ are *projections of the input*, so the dynamics depend on token content
    (Gu & Dao §3.2). $\Delta_t = \mathrm{softplus}(\cdot) > 0$ is kept positive so
    the discretization is well-posed; $B_t$ and $C_t$ are linear.

    Parameters
    ----------
    x : jnp.ndarray, shape (L, d)
        Feature sequence (the conv'd token stream in a real Mamba block).
    w_delta : jnp.ndarray, shape (d,)
        Step-size projection.
    w_B, w_C : jnp.ndarray, shape (d, N)
        Input- and output-matrix projections.

    Returns
    -------
    delta : jnp.ndarray, shape (L,)
        Positive per-step sizes.
    B, C : jnp.ndarray, shape (L, N)
        Per-step input/output matrices.

    Raises
    ------
    ValueError
        If ``x`` is not 2-D or the projection shapes are inconsistent.
    """
    if x.ndim != 2:
        raise ValueError(f"x must be 2-D (L, d), got shape {x.shape}")
    d = x.shape[1]
    if w_delta.shape != (d,):
        raise ValueError(f"w_delta must be ({d},), got {w_delta.shape}")
    if w_B.ndim != 2 or w_B.shape[0] != d:
        raise ValueError(f"w_B must be ({d}, N), got {w_B.shape}")
    if w_C.shape != w_B.shape:
        raise ValueError(f"w_C must match w_B shape {w_B.shape}, got {w_C.shape}")
    delta = jax.nn.softplus(x @ w_delta)  # (L,)
    B = x @ w_B  # (L, N)
    C = x @ w_C  # (L, N)
    return delta, B, C


# ---------------------------------------------------------------------------
# §9.2 — per-step (time-varying) discretization
# ---------------------------------------------------------------------------


def discretize_selective(
    A: jnp.ndarray,
    delta: jnp.ndarray,
    B: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    r"""Time-varying ZOH: $\bar A_t = e^{\Delta_t A}$, $\bar B_t = \Delta_t B_t$.

    Unlike Chapter 8's *single* $(\bar A, \bar B)$, every step gets its own
    transition because $\Delta_t$ (and $B_t$) vary — the defining feature of the
    LTV system. $A$ is diagonal, so $\bar A_t$ is the elementwise exponential.

    Parameters
    ----------
    A : jnp.ndarray, shape (N,)
        Diagonal modes (negative; see :func:`stable_A`).
    delta : jnp.ndarray, shape (L,)
        Positive per-step sizes.
    B : jnp.ndarray, shape (L, N)
        Per-step input matrix.

    Returns
    -------
    Abar : jnp.ndarray, shape (L, N)
        Per-step discrete transitions $\bar A_t$.
    Bbar : jnp.ndarray, shape (L, N)
        Per-step discrete input matrices $\bar B_t = \Delta_t B_t$ (multiply by
        $u_t$ to get the scan driving term).

    Raises
    ------
    ValueError
        If shapes are inconsistent.
    """
    if A.ndim != 1:
        raise ValueError(f"A must be 1-D (N,), got shape {A.shape}")
    if delta.ndim != 1:
        raise ValueError(f"delta must be 1-D (L,), got shape {delta.shape}")
    n = A.shape[0]
    if B.shape != (delta.shape[0], n):
        raise ValueError(f"B must be ({delta.shape[0]}, {n}), got {B.shape}")
    Abar = jnp.exp(delta[:, None] * A[None, :])  # (L, N)
    Bbar = delta[:, None] * B  # (L, N)
    return Abar, Bbar


# ---------------------------------------------------------------------------
# §9.3 — the selective scan: the §8.6 primitive, now the only route
# ---------------------------------------------------------------------------


def selective_binary_operator(
    left: tuple[jnp.ndarray, jnp.ndarray],
    right: tuple[jnp.ndarray, jnp.ndarray],
) -> tuple[jnp.ndarray, jnp.ndarray]:
    r"""Associative operator $(a_1,b_1)\bullet(a_2,b_2) = (a_2 a_1,\, a_2 b_1 + b_2)$.

    Identical to Chapter 8 §8.6 (``s5_binary_operator``). The point of Chapter 9
    is that nothing here assumed $a_1 = a_2$: the *same* operator composes
    *time-varying* affine maps $h \mapsto a_t h + b_t$, which is exactly what an
    LTV recurrence needs. ``left`` is the earlier segment, ``right`` the later.
    """
    a_i, b_i = left
    a_j, b_j = right
    return a_j * a_i, a_j * b_i + b_j


def selective_scan(Abar: jnp.ndarray, Bu: jnp.ndarray) -> jnp.ndarray:
    r"""States $h_k$ of the LTV recurrence via :func:`jax.lax.associative_scan`.

    Composes the *per-step* pairs $(\bar A_t, \bar B_t u_t)$ with
    :func:`selective_binary_operator` in critical-path depth
    $\lceil \log_2 L \rceil$. Because $\bar A_t$ varies with $t$ there is no
    convolution to fall back on (§9.1); this scan is the only parallel route.

    Parameters
    ----------
    Abar : jnp.ndarray, shape (L, N)
        Per-step transitions $\bar A_t$ (the time axis is axis 0).
    Bu : jnp.ndarray, shape (L, N)
        Per-step driving terms $\bar B_t u_t$.

    Returns
    -------
    hs : jnp.ndarray, shape (L, N)
        Inclusive-prefix states $h_0, \ldots, h_{L-1}$ (from $h_{-1} = 0$).
    """
    if Abar.shape != Bu.shape:
        raise ValueError(f"Abar shape {Abar.shape} must match Bu shape {Bu.shape}")
    _, hs = jax.lax.associative_scan(selective_binary_operator, (Abar, Bu))
    return hs


def selective_sequential(Abar: jnp.ndarray, Bu: jnp.ndarray) -> jnp.ndarray:
    r"""The same states $h_k$ via a sequential :func:`jax.lax.scan` (the $O(L)$ oracle)."""
    if Abar.shape != Bu.shape:
        raise ValueError(f"Abar shape {Abar.shape} must match Bu shape {Bu.shape}")

    def step(h: jnp.ndarray, ab: tuple[jnp.ndarray, jnp.ndarray]) -> tuple[jnp.ndarray, jnp.ndarray]:
        a_t, bu_t = ab
        h_new = a_t * h + bu_t
        return h_new, h_new

    h0 = jnp.zeros(Abar.shape[1], dtype=Abar.dtype)
    _, hs = jax.lax.scan(step, h0, (Abar, Bu))
    return hs


def selective_apply(
    A: jnp.ndarray,
    delta: jnp.ndarray,
    B: jnp.ndarray,
    C: jnp.ndarray,
    D: jnp.ndarray | float,
    u: jnp.ndarray,
    parallel: bool = True,
) -> jnp.ndarray:
    r"""Run the SISO selective SSM on input ``u`` (shape ``(L,)``).

    .. math::

        h_k = \bar A_k \odot h_{k-1} + \bar B_k\, u_k, \qquad
        y_k = C_k \cdot h_k + D\, u_k .

    With ``parallel=True`` the states come from the associative scan (§9.3);
    ``parallel=False`` uses the sequential oracle. The two are pinned equal in
    ``tests/test_selective.py`` (the chapter's §9.3 equivalence claim).

    Parameters
    ----------
    A : jnp.ndarray, shape (N,)
        Diagonal modes (negative).
    delta : jnp.ndarray, shape (L,)
        Positive per-step sizes.
    B, C : jnp.ndarray, shape (L, N)
        Per-step input/output matrices.
    D : jnp.ndarray or float
        Feedthrough scalar.
    u : jnp.ndarray, shape (L,)
        Input sequence.
    parallel : bool, default True
        Associative scan (True) or sequential reference (False).

    Returns
    -------
    y : jnp.ndarray, shape (L,)
        Output sequence.
    """
    if u.ndim != 1:
        raise ValueError(f"u must be 1-D (L,), got shape {u.shape}")
    if C.shape != B.shape:
        raise ValueError(f"C shape {C.shape} must match B shape {B.shape}")
    Abar, Bbar = discretize_selective(A, delta, B)
    Bu = Bbar * u[:, None]  # (L, N)
    hs = selective_scan(Abar, Bu) if parallel else selective_sequential(Abar, Bu)
    return jnp.sum(C * hs, axis=1) + jnp.asarray(D, dtype=hs.dtype) * u


# ---------------------------------------------------------------------------
# Smoke check: echo the load-bearing numbers the prose and tests rely on
# ---------------------------------------------------------------------------


def _demo_system(n: int = 8, length: int = 64, d: int = 4, seed: int = 0):
    """A fixed, reproducible input-dependent SISO selective system."""
    import numpy as np

    rng = np.random.default_rng(seed)
    A = stable_A(jnp.asarray(rng.standard_normal(n)))
    x = jnp.asarray(rng.standard_normal((length, d)))
    w_delta = jnp.asarray(rng.standard_normal(d))
    w_B = jnp.asarray(rng.standard_normal((d, n)))
    w_C = jnp.asarray(rng.standard_normal((d, n)))
    delta, B, C = selection_from_input(x, w_delta, w_B, w_C)
    u = jnp.asarray(rng.standard_normal(length))
    return A, delta, B, C, jnp.asarray(0.0), u


def main() -> None:
    print("Chapter 9 — selective_ssm.py")
    print("=" * 60)

    A, delta, B, C, D, u = _demo_system()
    L, N = u.shape[0], A.shape[0]

    y_par = selective_apply(A, delta, B, C, D, u, parallel=True)
    y_seq = selective_apply(A, delta, B, C, D, u, parallel=False)
    resid = float(jnp.max(jnp.abs(y_par - y_seq)))
    print(f"  L={L}, N={N}: max|y_scan - y_seq| = {resid:.3e}  (§9.3 selective-scan-equivalence: ~0)")

    Abar, _ = discretize_selective(A, delta, B)
    all_contractive = bool(jnp.all(jnp.abs(Abar) < 1.0))
    print(f"  all |Abar_t| < 1: {all_contractive}  (§9.1 stable by the -exp(A_log) param)")
    print(f"  delta range: [{float(delta.min()):.3f}, {float(delta.max()):.3f}]  (softplus > 0)")


if __name__ == "__main__":
    main()
