r"""Chapter 10 §10.5 — the trapezoidal SSD pass: order-2 integration, kept matmul-friendly.

Chapter 9 §9.5 showed the selective-SSM output is a single matrix multiply
$y = M u$ with a structured (semiseparable) lower-triangular $M$ — the SSD form
that runs as a parallel scan ($O(LN)$) or a dense matmul ($O(L^2 N)$). Mamba-3's
second-order **exponential-trapezoidal** integrator (``discretization.py``) keeps
exactly this structure: only the *stencil* changes from ZOH's single-endpoint
injection to the trapezoid's two-endpoint injection.

The selective LTV recurrence with the exp-trapezoidal stencil is

.. math::

    h_k = \alpha_k h_{k-1}
        + \underbrace{(1-\lambda)\,\Delta_k\,\alpha_k\, B_{k-1} u_{k-1}}_{\text{left endpoint}}
        + \underbrace{\lambda\,\Delta_k\, B_k u_k}_{\text{right endpoint}},
    \qquad \alpha_k = e^{A\Delta_k},

with output $y_k = C_k \cdot h_k + D u_k$. :func:`trapezoidal_sequential` runs this
verbatim ($O(L)$) and is the ground-truth oracle.

Unrolling with the state-transition operator $\Phi(k,j) = \prod_{i=j+1}^{k}\alpha_i
= \exp\!\big(A\sum_{i=j+1}^k \Delta_i\big)$ (Chapter 9 §9.2) splits the output into
**two SSD streams sharing the same decay** $\Phi$:

.. math::

    y_k = D u_k
        + \sum_{j \le k} C_k\!\cdot\!\big(\Phi(k,j)\odot B_j\big)\,
            \underbrace{\lambda\,\Delta_j}_{\text{right}}\, u_j
        + \sum_{j < k} C_k\!\cdot\!\big(\Phi(k,j)\odot B_j\big)\,
            \underbrace{(1-\lambda)\,\Delta_{j+1}}_{\text{left}}\, u_j .

The left stream is *strictly* lower-triangular and weights source $j$ by the
**next** step's $\Delta_{j+1}$ (because that left endpoint is injected when
computing step $j+1$, then propagated). Crucially, **both streams use the same
true decay** $\Phi(k,j)$. :func:`trapezoidal_matmul` builds this combined
semiseparable matrix and matmuls; it is pinned equal to the sequential oracle in
``tests/test_trapezoidal_ssd.py`` ($< 10^{-12}$).

.. warning::

    A tempting shortcut — folding each stream's coefficient into the ``dt`` passed
    to a ZOH-style ``ssd_naive`` — is **wrong**: that kernel uses ``dt`` for *both*
    the decay segment-sum *and* the source weight, so scaling ``dt`` by $\lambda$
    corrupts the decay to $\Phi^{\lambda}$. The decay must be built from the true
    $\Delta$ once and shared; only the *source weights* differ between streams.
    This module builds the decay separately for that reason.

Idiomatic-JAX note
------------------
The decay matrix comes from the Chapter-9 ``segsum`` trick (cumulative segment sum
with $-\infty$ above the diagonal, so ``exp`` zeroes the upper triangle) — one
vectorized expression instead of the $O(L^2)$ double loop a NumPy reference writes.
The oracle uses ``lax.scan`` (sequential), the matmul uses ``einsum`` (parallel);
the two routes are the SSD duality of §9.5, now carrying a second-order stencil.

Port credit
-----------
``segsum`` is reused from Chapter 9 (``companions/ch09/jax/ssd_semiseparable.py``,
itself ported from ``post_transformers/experiments/jax/week08/mamba2_ssd.py``). The
trapezoidal two-stream decomposition is derived here for §10.5; the predecessor
``post_transformers/experiments/jax/week09/mamba3.py::trapezoidal_ssd`` folds the
stream weights into ``dt`` and so is not reused (see the warning above). Mamba-3:
Lahoti et al., arXiv:2603.15569.

Usage
-----
::

    PYTHONPATH=. python companions/ch10/jax/trapezoidal_ssd.py
"""

from __future__ import annotations

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402

from companions.ch09.jax.ssd_semiseparable import segsum  # noqa: E402

__all__ = [
    "trapezoidal_sequential",
    "trapezoidal_matmul",
    "decay_operator",
]


def _validate(A: jnp.ndarray, delta: jnp.ndarray, B: jnp.ndarray, C: jnp.ndarray, u: jnp.ndarray) -> None:
    if A.ndim != 1:
        raise ValueError(f"A must be 1-D (N,), got {A.shape}")
    if delta.ndim != 1:
        raise ValueError(f"delta must be 1-D (L,), got {delta.shape}")
    n, length = A.shape[0], delta.shape[0]
    if B.shape != (length, n):
        raise ValueError(f"B must be ({length}, {n}), got {B.shape}")
    if C.shape != (length, n):
        raise ValueError(f"C must be ({length}, {n}), got {C.shape}")
    if u.shape != (length,):
        raise ValueError(f"u must be ({length},), got {u.shape}")


def trapezoidal_sequential(
    A: jnp.ndarray,
    delta: jnp.ndarray,
    B: jnp.ndarray,
    C: jnp.ndarray,
    D: jnp.ndarray | float,
    u: jnp.ndarray,
    lam: float = 0.5,
) -> jnp.ndarray:
    r"""Ground-truth $O(L)$ exp-trapezoidal selective scan (the oracle).

    Runs the two-endpoint recurrence

    .. math::

        h_k = \alpha_k h_{k-1} + (1-\lambda)\Delta_k \alpha_k B_{k-1} u_{k-1}
            + \lambda \Delta_k B_k u_k,
        \quad y_k = C_k\!\cdot\!h_k + D u_k,

    with $\alpha_k = e^{A\Delta_k}$ (elementwise over the $N$ diagonal modes). At
    $k = 0$ there is no left endpoint ($u_{-1} = 0$).

    Parameters
    ----------
    A : jnp.ndarray, shape (N,)
        Diagonal modes (real negative, or complex with negative real part).
    delta : jnp.ndarray, shape (L,)
        Positive per-step sizes.
    B, C : jnp.ndarray, shape (L, N)
        Per-step input/output matrices.
    D : scalar
        Feedthrough.
    u : jnp.ndarray, shape (L,)
        Input.
    lam : float, default 0.5
        Trapezoid interpolation parameter ($\tfrac12$ = symmetric, order 2).

    Returns
    -------
    y : jnp.ndarray, shape (L,)
        Output (complex if ``A`` is complex; take the real part downstream).
    """
    _validate(A, delta, B, C, u)
    if not 0.0 <= lam <= 1.0:
        raise ValueError(f"lam must be in [0, 1], got {lam}")

    dtype = jnp.result_type(A, B, C, u, jnp.float64)
    alpha = jnp.exp(delta[:, None] * A[None, :])  # (L, N)
    f = (B * u[:, None]).astype(dtype)  # forcing B_k u_k, (L, N)
    f_prev = jnp.concatenate([jnp.zeros((1, A.shape[0]), dtype=dtype), f[:-1]])  # B_{k-1} u_{k-1}

    left = (1.0 - lam) * delta[:, None] * alpha * f_prev  # (L, N)
    right = lam * delta[:, None] * f  # (L, N)
    inject = left + right  # total injection at step k

    def step(h, ab):
        a_k, inj_k = ab
        h_new = a_k * h + inj_k
        return h_new, h_new

    h0 = jnp.zeros(A.shape[0], dtype=dtype)
    _, hs = jax.lax.scan(step, h0, (alpha, inject))  # (L, N)
    return jnp.sum(C.astype(dtype) * hs, axis=1) + jnp.asarray(D, dtype=dtype) * u.astype(dtype)


def decay_operator(A: jnp.ndarray, delta: jnp.ndarray) -> jnp.ndarray:
    r"""The state-transition operator $\Phi(k,j) = \exp(A\sum_{i=j+1}^k \Delta_i)$.

    Shape ``(N, L, L)``: ``Phi[n, k, j]`` is mode $n$'s decay from source $j$ to
    target $k$, zero for $k < j$ (causal) and $1$ for $k = j$. Built once from the
    true $\Delta$ via Chapter 9's :func:`segsum` and shared by both trapezoid
    streams (see the module warning).

    Parameters
    ----------
    A : jnp.ndarray, shape (N,)
    delta : jnp.ndarray, shape (L,)

    Returns
    -------
    Phi : jnp.ndarray, shape (N, L, L)
    """
    Adt = A[:, None] * delta[None, :]  # (N, L)
    return jnp.exp(segsum(Adt))  # (N, L, L); -inf above diagonal -> 0


def trapezoidal_matmul(
    A: jnp.ndarray,
    delta: jnp.ndarray,
    B: jnp.ndarray,
    C: jnp.ndarray,
    D: jnp.ndarray | float,
    u: jnp.ndarray,
    lam: float = 0.5,
) -> jnp.ndarray:
    r"""The same output as :func:`trapezoidal_sequential` via one semiseparable matmul.

    Builds the combined two-stream lower-triangular matrix $M = M^{\text{right}} +
    M^{\text{left}}$ sharing the decay $\Phi$ of :func:`decay_operator`:

    * right (inclusive $j \le k$): weight $\lambda\,\Delta_j$ on source $j$;
    * left  (strict $j < k$):      weight $(1-\lambda)\,\Delta_{j+1}$ on source $j$;

    then $y = M u + D u$. The $O(L^2 N)$ dense route of the §9.5 duality, now
    carrying the order-2 stencil — pinned equal to the oracle in the tests.

    Parameters
    ----------
    A, delta, B, C, D, u, lam : as in :func:`trapezoidal_sequential`.

    Returns
    -------
    y : jnp.ndarray, shape (L,)
    """
    _validate(A, delta, B, C, u)
    if not 0.0 <= lam <= 1.0:
        raise ValueError(f"lam must be in [0, 1], got {lam}")

    length = delta.shape[0]
    dtype = jnp.result_type(A, B, C, u, jnp.float64)

    Phi = decay_operator(A, delta).astype(dtype)  # (N, L, L): [n, k, j]
    # base[k, j] = C_k . (Phi(k,j) (.) B_j) = sum_n C[k,n] Phi[n,k,j] B[j,n].
    base = jnp.einsum("kn,nkj,jn->kj", C.astype(dtype), Phi, B.astype(dtype))  # (L, L)

    # Right stream: inclusive lower triangle, weight lam * dt[j].
    w_right = lam * delta[None, :]  # (1, L) broadcast over k; decay already 0 for k<j
    # Left stream: strict lower triangle (k > j), weight (1-lam) * dt[j+1].
    dt_next = jnp.concatenate([delta[1:], jnp.zeros((1,), dtype=delta.dtype)])  # dt[j+1], 0 at j=L-1
    strict = jnp.tril(jnp.ones((length, length), dtype=bool), k=-1)  # k > j
    w_left = jnp.where(strict, (1.0 - lam) * dt_next[None, :], 0.0)  # (L, L)

    M = base * (w_right.astype(dtype) + w_left.astype(dtype))  # (L, L)
    return M @ u.astype(dtype) + jnp.asarray(D, dtype=dtype) * u.astype(dtype)


# ---------------------------------------------------------------------------
# Smoke check
# ---------------------------------------------------------------------------


def _demo_system(n: int = 6, length: int = 32, d: int = 4, seed: int = 0, complex_modes: bool = True):
    """A fixed, reproducible selective system; complex modes by default (Mamba-3)."""
    import numpy as np

    from companions.ch09.jax.selective_ssm import selection_from_input, stable_A

    rng = np.random.default_rng(seed)
    a_real = stable_A(jnp.asarray(rng.standard_normal(n)))  # negative reals
    if complex_modes:
        A = a_real + 1j * jnp.asarray(rng.standard_normal(n))  # decay + oscillation
    else:
        A = a_real
    x = jnp.asarray(rng.standard_normal((length, d)))
    w_delta = jnp.asarray(rng.standard_normal(d))
    w_B = jnp.asarray(rng.standard_normal((d, n)))
    w_C = jnp.asarray(rng.standard_normal((d, n)))
    delta, B, C = selection_from_input(x, w_delta, w_B, w_C)
    u = jnp.asarray(rng.standard_normal(length))
    return A, delta, B, C, jnp.asarray(0.0), u


def main() -> None:
    print("Chapter 10 — trapezoidal_ssd.py")
    print("=" * 64)

    for label, cm in (("complex modes (Mamba-3)", True), ("real modes (Mamba-1/2)", False)):
        A, delta, B, C, D, u = _demo_system(complex_modes=cm)
        y_seq = trapezoidal_sequential(A, delta, B, C, D, u)
        y_mat = trapezoidal_matmul(A, delta, B, C, D, u)
        resid = float(jnp.max(jnp.abs(y_seq - y_mat)))
        L, N = u.shape[0], A.shape[0]
        print(f"  {label}: L={L}, N={N}")
        print(f"    max|y_seq - y_matmul| = {resid:.3e}  (§10.5 SSD duality, order-2 stencil: ~0)")


if __name__ == "__main__":
    main()
