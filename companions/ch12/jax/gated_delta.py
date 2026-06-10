r"""Chapter 12 §12.6 — the gated delta rule (Gated DeltaNet's state update).

Gated DeltaNet (Yang et al., arXiv:2412.06464) composes the two forgetting
mechanisms this part of the book has built separately: the *scalar decay* of
Mamba-2's SSD (Chapter 9) and the *selective erasure* of the delta rule
(``delta_rule.py``):

.. math::

    S_t = \gamma_t\, S_{t-1}\,(I - \beta_t k_t k_t^\top)
          + \beta_t\, v_t k_t^\top,
    \qquad \gamma_t \in (0, 1],\ \beta_t \in [0, 1].

The two limits recover the two parents — pinned as exact reductions:

* $\gamma_t \equiv 1$: plain DeltaNet (``delta_rule_recurrent``), erase-only;
* $\beta_t \equiv 0$: pure exponential decay $S_t = \gamma_t S_{t-1}$, the
  scalar-decay contraction of Chapter 9's SSD (with a matrix state).

And the division of labour is exact: a stored association whose key is
*orthogonal* to everything written afterwards is untouched by the delta-rule
erasure but decays through the gate by precisely $\prod_t \gamma_t$ — gating
forgets *uniformly*, the delta rule forgets *selectively*. The demo helper
pins that product law to machine precision.

Kimi Linear's KDA (arXiv:2510.26692) is a production refinement of this same
update (per-channel diagonal gates in place of the scalar $\gamma_t$); the
chapter discusses it at architecture level only, so no KDA implementation is
claimed or shipped here.

Idiomatic-JAX note (NumPy->JAX teaching point)
----------------------------------------------
Same ``lax.scan``-vs-Python-loop oracle pairing as the sibling modules. Note
the gate multiplies the *erased* state but not the fresh write — placing the
``gamma_t *`` inside the scan body wrongly (on the whole right-hand side) is
the natural NumPy-port bug, and the $\beta = 0$ reduction test is the guard
that catches it.

Port credit
-----------
Greenfield: the predecessor week13 module is a stub, so this update is
authored from the paper math (arXiv:2412.06464 §3). The recurrent/naive
structure mirrors ``delta_rule.py``.

Usage
-----
::

    PYTHONPATH=. python companions/ch12/jax/gated_delta.py
"""

from __future__ import annotations

import jax

# Enable float64 before any jnp array is created (matches Chapters 4, 7-11).
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402

from companions.ch12.jax.delta_rule import delta_rule_recurrent  # noqa: E402

__all__ = [
    "gated_delta_step",
    "gated_delta_recurrent",
    "gated_delta_naive",
    "stale_retrieval_after_orthogonal_writes",
]


def gated_delta_step(
    state: jnp.ndarray,
    key: jnp.ndarray,
    value: jnp.ndarray,
    beta: jnp.ndarray | float,
    gamma: jnp.ndarray | float,
) -> jnp.ndarray:
    r"""One gated-delta update $S \leftarrow \gamma\,S(I - \beta kk^\top) + \beta vk^\top$.

    Rank-one form ($O(d_v d_k)$, no materialised projector): the gate scales
    the erased state, then the write lands ungated.

    Parameters
    ----------
    state : jnp.ndarray, shape (d_v, d_k)
    key : jnp.ndarray, shape (d_k,)
    value : jnp.ndarray, shape (d_v,)
    beta : scalar
        Write strength.
    gamma : scalar
        Decay gate in $(0, 1]$; $\gamma = 1$ is plain DeltaNet.

    Returns
    -------
    jnp.ndarray, shape (d_v, d_k)
    """
    if state.shape != (value.shape[0], key.shape[0]):
        raise ValueError(
            f"state must have shape (d_v, d_k) = ({value.shape[0]}, {key.shape[0]}); "
            f"got {state.shape}"
        )
    erased = state - beta * jnp.outer(state @ key, key)
    return gamma * erased + beta * jnp.outer(value, key)


def gated_delta_recurrent(
    q: jnp.ndarray,
    k: jnp.ndarray,
    v: jnp.ndarray,
    betas: jnp.ndarray,
    gammas: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    r"""Sequential gated DeltaNet via ``lax.scan``; post-update read $o_t = S_t q_t$.

    Parameters
    ----------
    q, k : jnp.ndarray, shape (L, d_k)
    v : jnp.ndarray, shape (L, d_v)
    betas, gammas : jnp.ndarray, shape (L,)

    Returns
    -------
    outputs : jnp.ndarray, shape (L, d_v)
    final_state : jnp.ndarray, shape (d_v, d_k)
    """
    if q.shape != k.shape or v.shape[0] != q.shape[0]:
        raise ValueError(f"inconsistent q/k/v shapes: {q.shape}, {k.shape}, {v.shape}")
    if betas.shape != (q.shape[0],) or gammas.shape != (q.shape[0],):
        raise ValueError(
            f"betas and gammas must have shape (L,) = ({q.shape[0]},); "
            f"got {betas.shape}, {gammas.shape}"
        )
    d_k, d_v = q.shape[1], v.shape[1]

    def step(state: jnp.ndarray, inp) -> tuple[jnp.ndarray, jnp.ndarray]:
        q_t, k_t, v_t, beta_t, gamma_t = inp
        erased = state - beta_t * jnp.outer(state @ k_t, k_t)
        new_state = gamma_t * erased + beta_t * jnp.outer(v_t, k_t)
        return new_state, new_state @ q_t

    init = jnp.zeros((d_v, d_k), dtype=v.dtype)
    final_state, outputs = jax.lax.scan(step, init, (q, k, v, betas, gammas))
    return outputs, final_state


def gated_delta_naive(
    q: jnp.ndarray,
    k: jnp.ndarray,
    v: jnp.ndarray,
    betas: jnp.ndarray,
    gammas: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    r"""Python-loop oracle: materialises $\gamma_t (I - \beta_t k_t k_t^\top)$ each step."""
    if q.shape != k.shape or v.shape[0] != q.shape[0]:
        raise ValueError(f"inconsistent q/k/v shapes: {q.shape}, {k.shape}, {v.shape}")
    length, d_k = q.shape
    d_v = v.shape[1]
    state = jnp.zeros((d_v, d_k), dtype=v.dtype)
    identity = jnp.eye(d_k, dtype=v.dtype)
    outputs = []
    for t in range(length):
        transition = gammas[t] * (identity - betas[t] * jnp.outer(k[t], k[t]))
        state = state @ transition + betas[t] * jnp.outer(v[t], k[t])
        outputs.append(state @ q[t])
    return jnp.stack(outputs), state


def stale_retrieval_after_orthogonal_writes(
    n_writes: int,
    gamma: float,
    d_k: int = 8,
    d_v: int = 4,
    seed: int = 0,
) -> tuple[float, float]:
    r"""Retrieval norm of a stale association after ``n_writes`` orthogonal-key writes.

    Stores $(k_A, v_A)$ with $\beta = 1$ on a unit key, then performs
    ``n_writes`` gated-delta writes whose keys are exactly orthogonal to
    $k_A$. The delta-rule erasure never touches the $k_A$ direction, so the
    retrieval $\|S k_A\|$ decays *only* through the gate:

    .. math::

        \|S_{T} k_A\| = \gamma^{T}\, \|v_A\|.

    Returns the measured retrieval norm and the analytic $\gamma^{T}\|v_A\|$
    — equal to machine precision (the uniform-vs-selective forgetting pin).

    Parameters
    ----------
    n_writes : int
        Number of orthogonal writes $T$ after storing the pair.
    gamma : float
        Constant gate value in $(0, 1]$.
    d_k, d_v : int
    seed : int

    Returns
    -------
    measured, analytic : float
    """
    import numpy as np

    if not 0.0 < gamma <= 1.0:
        raise ValueError(f"gamma must be in (0, 1]; got {gamma}")
    rng = np.random.default_rng(seed)
    k_a = np.zeros(d_k)
    k_a[0] = 1.0  # unit key along e_1; later keys live in its complement
    v_a = rng.standard_normal(d_v)

    state = gated_delta_step(
        jnp.zeros((d_v, d_k)), jnp.asarray(k_a), jnp.asarray(v_a), 1.0, 1.0
    )
    for _ in range(n_writes):
        raw = rng.standard_normal(d_k)
        raw[0] = 0.0  # exactly orthogonal to k_A
        key = jnp.asarray(raw / np.linalg.norm(raw))
        value = jnp.asarray(rng.standard_normal(d_v))
        state = gated_delta_step(state, key, value, 0.8, gamma)

    measured = float(jnp.linalg.norm(state @ jnp.asarray(k_a)))
    analytic = float(gamma**n_writes * np.linalg.norm(v_a))
    return measured, analytic


def main() -> None:
    import numpy as np

    print("Chapter 12 — gated_delta.py")
    print("=" * 64)

    rng = np.random.default_rng(0)
    length, d_k, d_v = 48, 8, 6
    q = jnp.asarray(rng.standard_normal((length, d_k)))
    # Unit-norm keys + beta < 1: the stable regime (beta * ||k||^2 < 2, §12.4).
    k_raw = rng.standard_normal((length, d_k))
    k = jnp.asarray(k_raw / np.linalg.norm(k_raw, axis=1, keepdims=True))
    v = jnp.asarray(rng.standard_normal((length, d_v)))
    betas = jnp.asarray(rng.uniform(0.1, 0.9, size=length))
    gammas = jnp.asarray(rng.uniform(0.7, 1.0, size=length))

    # §12.6 scan == naive oracle.
    y_scan, s_scan = gated_delta_recurrent(q, k, v, betas, gammas)
    y_naive, s_naive = gated_delta_naive(q, k, v, betas, gammas)
    print(f"  scan == naive oracle:        max diff = "
          f"{float(jnp.max(jnp.abs(y_scan - y_naive))):.2e}")

    # §12.6 reduction 1: gamma == 1 recovers plain DeltaNet.
    ones = jnp.ones(length)
    y_g1, s_g1 = gated_delta_recurrent(q, k, v, betas, ones)
    y_dn, s_dn = delta_rule_recurrent(q, k, v, betas)
    print(f"  gamma=1  -> DeltaNet:        max diff = "
          f"{float(jnp.max(jnp.abs(y_g1 - y_dn))):.2e}")

    # §12.6 reduction 2: beta == 0 is pure exponential decay (here from S_0 = 0).
    y_b0, s_b0 = gated_delta_recurrent(q, k, v, jnp.zeros(length), gammas)
    print(f"  beta=0   -> pure decay from S_0=0: max |S_L| = "
          f"{float(jnp.max(jnp.abs(s_b0))):.2e}  (state never written)")

    # §12.6 uniform-vs-selective forgetting: gate decay is exactly gamma^T.
    for gamma, n_writes in ((0.95, 20), (0.9, 40)):
        measured, analytic = stale_retrieval_after_orthogonal_writes(n_writes, gamma)
        print(f"  gamma={gamma}, T={n_writes:2d}: |S k_A| measured = {measured:.12f}, "
              f"analytic gamma^T |v_A| = {analytic:.12f}, diff = {abs(measured - analytic):.2e}")


if __name__ == "__main__":
    main()
