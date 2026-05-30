r"""Chapter 8 — S4 core: a discretized continuous LTI state-space model.

An S4 layer *is* the zero-order-hold (ZOH) discretization (Chapter 4) of the
HiPPO-LegS continuous LTI system (Chapter 7), made trainable by a learnable
timescale $\Delta$. This module is the harness-free numerical core the Chapter 8
JAX figures and tests build on:

* :func:`make_hippo_legs` — the structured state matrix $A$ (Chapter 7);
* :func:`discretize_zoh`, :func:`discretize_bilinear` — continuous -> discrete (Ch 4);
* :func:`ssm_kernel_naive` — the SSM convolution kernel $K_k = C\,\bar A^k\,\bar B$;
* :func:`ssm_recurrent`, :func:`ssm_convolutional` — the two equivalent views.

The convolution<->recurrence duality (§8.3) is the load-bearing fact: for zero
initial state the recurrent scan (O(L) sequential, inference-friendly) and the
FFT convolution (O(L log L) parallel, training-friendly) produce *identical*
outputs. :func:`ssm_recurrent` and :func:`ssm_convolutional` are pinned equal in
``tests/test_s4.py``.

Teaching choice: the kernel is computed *naively* in O(N^2 L) by iterating
$\bar A^k$, NOT via S4's Cauchy/Woodbury trick (O(N log^2 N), §8.4). For
$N \le 64$ the naive kernel is exact and far clearer; the Cauchy kernel is taught
in prose only (§8.4).

Precision: float64 throughout, matching this book's global ``jax_enable_x64``
(Chapter 4). The W4 source ran float32; float64 keeps the §8.5 S4D conditioning
story honest at larger $N$.

Port credit
-----------
S4 numerical core ported from
``post_transformers/experiments/jax/week04/s4_hippo.py`` (Gu et al., *Efficiently
Modeling Long Sequences with Structured State Spaces* — S4, arXiv:2111.00396),
which follows ``experiments/refs/s4`` and the Annotated S4. The harness-coupled
Flax ``S4Layer`` / ``S4SeqModel`` are intentionally dropped (they import an
external optimizer harness); this module is pure functions only.

Usage
-----
::

    PYTHONPATH=. python companions/ch08/jax/s4_core.py
"""

from __future__ import annotations

import jax

# Enable float64 before any jnp array is created (matches Chapter 4 / Chapter 7).
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402
import jax.scipy.linalg  # noqa: E402

__all__ = [
    "make_hippo_legs",
    "discretize_zoh",
    "discretize_bilinear",
    "ssm_kernel_naive",
    "causal_conv_fft",
    "ssm_recurrent",
    "ssm_convolutional",
]


# ---------------------------------------------------------------------------
# §8.2 — the structured state matrix (carried over from Chapter 7)
# ---------------------------------------------------------------------------


def make_hippo_legs(n: int) -> tuple[jnp.ndarray, jnp.ndarray]:
    r"""Construct the HiPPO-LegS matrices $A \in \R^{N\times N}$, $B \in \R^{N\times 1}$.

    Built directly from the Chapter 7 closed form with a vectorized ``jnp.where``
    over a ``meshgrid`` (no Python loop, no matrix inverse)::

        A[i, j] = -sqrt((2i+1)(2j+1))   for i > j
                = -(i + 1)              for i == j
                = 0                     for i < j
        B[i]    =  sqrt(2i + 1)

    Because $A$ is lower-triangular its eigenvalues are exactly its diagonal,
    $-1, -2, \ldots, -N$ — all strictly in the open left half-plane, so the
    continuous HiPPO dynamics are asymptotically stable (Chapter 2). This is the
    *LTI-stable* convention S4 discretizes; ZOH on it gives $|\lambda(\bar A)| < 1$.

    Parameters
    ----------
    n : int
        State dimension $N$ (number of Legendre basis functions); must be >= 1.

    Returns
    -------
    A : jnp.ndarray, shape (n, n)
        Continuous-time state matrix (lower-triangular, eigenvalues $-1,\ldots,-N$).
    B : jnp.ndarray, shape (n, 1)
        Continuous-time input matrix, $B_i = \sqrt{2i+1}$.

    Raises
    ------
    ValueError
        If ``n < 1``.
    """
    if n < 1:
        raise ValueError(f"state dimension n must be >= 1, got {n}")
    q = jnp.arange(n, dtype=jnp.float64)
    i_idx, j_idx = jnp.meshgrid(q, q, indexing="ij")
    lower = jnp.sqrt((2.0 * i_idx + 1.0) * (2.0 * j_idx + 1.0))
    diag = i_idx + 1.0
    A = -jnp.where(i_idx > j_idx, lower, jnp.where(i_idx == j_idx, diag, 0.0))
    B = jnp.sqrt(2.0 * q + 1.0)[:, None]
    return A, B


# ---------------------------------------------------------------------------
# §8.1 — discretization: continuous -> discrete (Chapter 4)
# ---------------------------------------------------------------------------


def discretize_zoh(
    A: jnp.ndarray,
    B: jnp.ndarray,
    dt: jnp.ndarray | float,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    r"""Discretize a continuous SSM by the zero-order hold (ZOH), $\bar A = e^{A\Delta}$.

    Uses the stacked matrix-exponential trick (Chapter 4): exponentiate the
    augmented matrix $\begin{psmallmatrix} A\Delta & B\Delta \\ 0 & 0
    \end{psmallmatrix}$ and read $\bar A$ off the top-left block, $\bar B$ off the
    top-right. This avoids forming $A^{-1}$ (the textbook
    $\bar B = A^{-1}(e^{A\Delta}-I)B$), which is ill-conditioned for some HiPPO
    configurations.

    Parameters
    ----------
    A : jnp.ndarray, shape (N, N)
        Continuous-time state matrix.
    B : jnp.ndarray, shape (N, P)
        Continuous-time input matrix (P = 1 for SISO).
    dt : jnp.ndarray or float
        Discretization step size $\Delta$ (positive scalar).

    Returns
    -------
    Ab : jnp.ndarray, shape (N, N)
        Discrete-time state matrix $\bar A = e^{A\Delta}$.
    Bb : jnp.ndarray, shape (N, P)
        Discrete-time input matrix $\bar B = A^{-1}(e^{A\Delta} - I)B$.
    """
    dt = jnp.asarray(dt, dtype=A.dtype)
    n = A.shape[0]
    p = B.shape[1]
    top = jnp.concatenate([A * dt, B * dt], axis=1)
    bottom = jnp.zeros((p, n + p), dtype=A.dtype)
    augmented = jnp.concatenate([top, bottom], axis=0)
    exp_aug = jax.scipy.linalg.expm(augmented)
    Ab = exp_aug[:n, :n]
    Bb = exp_aug[:n, n:]
    return Ab, Bb


def discretize_bilinear(
    A: jnp.ndarray,
    B: jnp.ndarray,
    dt: jnp.ndarray | float,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    r"""Discretize a continuous SSM by the bilinear (Tustin) transform.

    The bilinear transform maps the imaginary axis exactly to the unit circle
    (second-order accurate in $\Delta$)::

        Ab = (I - dt/2 A)^{-1} (I + dt/2 A)
        Bb = (I - dt/2 A)^{-1} dt B

    Parameters
    ----------
    A : jnp.ndarray, shape (N, N)
        Continuous-time state matrix.
    B : jnp.ndarray, shape (N, P)
        Continuous-time input matrix.
    dt : jnp.ndarray or float
        Discretization step size $\Delta$ (positive scalar).

    Returns
    -------
    Ab : jnp.ndarray, shape (N, N)
        Discrete-time state matrix.
    Bb : jnp.ndarray, shape (N, P)
        Discrete-time input matrix.
    """
    dt = jnp.asarray(dt, dtype=A.dtype)
    n = A.shape[0]
    eye = jnp.eye(n, dtype=A.dtype)  # `I` would trip Ruff E741
    bl = jnp.linalg.inv(eye - (dt / 2) * A)
    Ab = bl @ (eye + (dt / 2) * A)
    Bb = (bl * dt) @ B
    return Ab, Bb


# ---------------------------------------------------------------------------
# §8.3 — the two equivalent views (conv <-> recurrence duality)
# ---------------------------------------------------------------------------


def ssm_kernel_naive(
    Ab: jnp.ndarray,
    Bb: jnp.ndarray,
    C: jnp.ndarray,
    L: int,
) -> jnp.ndarray:
    r"""SSM convolution kernel $K_k = C\,\bar A^k\,\bar B$ via naive power iteration.

    Iterates the *state vector* $v_k = \bar A^k \bar B$ (one $N\times N$ mat-vec per
    tap), giving O(N^2 L) — exact and acceptable for teaching with $N \le 64$.
    (Carrying the full matrix $\bar A^k$ instead would cost O(N^3 L); the vector
    recurrence is the right "naive".) Production S4 computes the same kernel in
    O(N log^2 N) via the Cauchy/Woodbury trick (§8.4), which this companion omits.

    Parameters
    ----------
    Ab : jnp.ndarray, shape (N, N)
        Discrete-time state matrix.
    Bb : jnp.ndarray, shape (N, P)
        Discrete-time input matrix (P = 1 for SISO).
    C : jnp.ndarray, shape (Q, N)
        Output matrix (Q = 1 for SISO).
    L : int
        Sequence length (number of kernel taps).

    Returns
    -------
    K : jnp.ndarray, shape (L,)
        Convolution kernel $K_k = C\,\bar A^k\,\bar B$ for $k = 0, \ldots, L-1$.
    """

    def body(v: jnp.ndarray, _: None) -> tuple[jnp.ndarray, jnp.ndarray]:
        k_l = (C @ v).squeeze()
        return Ab @ v, k_l  # v_{k+1} = Ab @ v_k, so v_k = Ab^k @ Bb

    _, K = jax.lax.scan(body, Bb, None, length=L)
    return K


def causal_conv_fft(u: jnp.ndarray, K: jnp.ndarray) -> jnp.ndarray:
    r"""Causal convolution $y = K * u$ via FFT, zero-padded to $2L$.

    Padding to $2L$ is load-bearing: the FFT computes a *circular* convolution,
    so an $L$-point transform would wrap tap $L-1$ back onto output position 0.
    Zero-padding to $2L$ makes the circular convolution agree with the causal
    linear one on $[0, L)$ (Exercise 8.3).

    Note: argument order is ``(u, K)`` to read as $y = K * u$ — the opposite of
    ``scipy.signal.convolve(a, v)`` where the kernel comes first.

    Parameters
    ----------
    u : jnp.ndarray, shape (L,)
        Input signal.
    K : jnp.ndarray, shape (L,)
        Convolution kernel.

    Returns
    -------
    y : jnp.ndarray, shape (L,)
        Causal convolution output.
    """
    L = u.shape[0]
    ud = jnp.fft.rfft(u, n=2 * L)
    Kd = jnp.fft.rfft(K, n=2 * L)
    return jnp.fft.irfft(ud * Kd, n=2 * L)[:L]


def ssm_recurrent(
    Ab: jnp.ndarray,
    Bb: jnp.ndarray,
    C: jnp.ndarray,
    D: jnp.ndarray,
    u: jnp.ndarray,
) -> jnp.ndarray:
    r"""SSM output via the recurrent (scan) view — O(L) sequential, from $h_0 = 0$.

    .. math::

        h_{k+1} = \bar A h_k + \bar B u_k, \qquad y_k = C h_{k+1} + D u_k.

    Parameters
    ----------
    Ab : jnp.ndarray, shape (N, N)
        Discrete-time state matrix.
    Bb : jnp.ndarray, shape (N, 1)
        Discrete-time input matrix.
    C : jnp.ndarray, shape (1, N)
        Output matrix.
    D : jnp.ndarray, shape ()
        Feedthrough scalar.
    u : jnp.ndarray, shape (L,)
        Input signal.

    Returns
    -------
    y : jnp.ndarray, shape (L,)
        Output signal.
    """
    n = Ab.shape[0]

    def step(h: jnp.ndarray, u_t: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
        h_new = Ab @ h + Bb.squeeze() * u_t
        y_t = (C @ h_new).squeeze() + D * u_t
        return h_new, y_t

    h0 = jnp.zeros(n, dtype=Ab.dtype)
    _, y = jax.lax.scan(step, h0, u)
    return y


def ssm_convolutional(
    K: jnp.ndarray,
    D: jnp.ndarray,
    u: jnp.ndarray,
) -> jnp.ndarray:
    r"""SSM output via the convolutional (FFT) view — O(L log L) parallel.

    .. math::

        y = K * u + D u.

    Identical to :func:`ssm_recurrent` for zero initial state (§8.3 duality), with
    ``K = ssm_kernel_naive(Ab, Bb, C, L)``.

    Parameters
    ----------
    K : jnp.ndarray, shape (L,)
        Convolution kernel.
    D : jnp.ndarray, shape ()
        Feedthrough scalar.
    u : jnp.ndarray, shape (L,)
        Input signal.

    Returns
    -------
    y : jnp.ndarray, shape (L,)
        Output signal.
    """
    return causal_conv_fft(u, K) + D * u


# ---------------------------------------------------------------------------
# Smoke check: echo the load-bearing numbers the prose and tests rely on
# ---------------------------------------------------------------------------


def main() -> None:
    import numpy as np

    print("Chapter 8 — s4_core.py")
    print("=" * 60)

    n, dt, L = 16, 0.1, 128
    A, B = make_hippo_legs(n)
    Ab, Bb = discretize_zoh(A, B, dt)
    spectral_radius = float(np.max(np.abs(np.linalg.eigvals(np.asarray(Ab)))))
    print(f"  N={n}, dt={dt}: max|lambda(Abar)| = {spectral_radius:.6f}  (ZOH-stable: <1)")

    # A fixed, reproducible SISO readout for the duality smoke check.
    key = jax.random.PRNGKey(0)
    C = jax.random.normal(key, (1, n), dtype=jnp.float64)
    D = jnp.asarray(0.0, dtype=jnp.float64)
    z = jnp.linspace(0.0, 1.0, L)
    u = jnp.sin(2.0 * jnp.pi * 3.0 * z)

    K = ssm_kernel_naive(Ab, Bb, C, L)
    y_rec = ssm_recurrent(Ab, Bb, C, D, u)
    y_conv = ssm_convolutional(K, D, u)
    residual = float(jnp.max(jnp.abs(y_rec - y_conv)))
    print(f"  duality residual max|y_rec - y_conv| = {residual:.3e}  (§8.3: ~0)")


if __name__ == "__main__":
    main()
