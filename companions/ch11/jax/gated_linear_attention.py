r"""Chapter 11 §11.3 — gated linear attention: the LTV reading (GLA, RetNet).

Ungated linear attention (``linear_attention.py``) accumulates forever: its
transition is $A = I$ (the LTI face, Chapter 8). **Gating** puts a decay back:

.. math::

    S_t = \operatorname{diag}(\gamma_t)\,S_{t-1} + \phi(k_t)\,v_t^\top,
    \qquad y_t = S_t^\top \phi(q_t),

with a per-feature gate $\gamma_t \in (0,1)^{d_k}$. This is the Chapter 9
selectivity move read from the attention side:

* **RetNet** (Sun et al. 2023, arXiv:2307.08621) uses a *constant* head-wise
  $\gamma$ — a fixed exponential decay, the **LTI** face with forgetting. Its
  parallel decay mask $L_{tj} = \gamma^{\,t-j}$ is **Toeplitz**.
* **GLA** (Yang et al. 2024, arXiv:2312.06635) and HGRN2 (Qin et al. 2024,
  arXiv:2404.07904) use a *data-dependent* $\gamma_t = \sigma(W x_t)$ — the
  **LTV** face. Its decay mask $L_{tj} = \prod_{i=j+1}^{t}\gamma_i$ is **not**
  Toeplitz (the diagonals vary), exactly mirroring Chapter 9's selective decay.

Theorem ``ch11:gla-ltv-duality`` (pinned here): the gated recurrence equals the
masked-parallel form $Y = (L_\gamma \circ (Q_\phi K_\phi^\top))\,V$, and for a
*scalar* per-step gate $L_\gamma$ coincides with the `ch09:ssd-duality` decay
mask under $\log\gamma_i = a_i\Delta_i$. Scalar-gated linear attention and the
scalar-$A$ selective SSM are the **same operator, read from opposite sides**.

Convention: gated linear attention here is **unnormalized** (RetNet/GLA replace
the $z_t$ normalizer with an output GroupNorm; see ``linear_attention.py`` §11.2
for why the normalizer is delicate). The parallel form rescales queries by
$e^{+g_t}$ and keys by $e^{-g_t}$ with $g_t = \sum_{i\le t}\log\gamma_i$ — the
GLA "secondary" trick. Those exponentials over/underflow for long sequences
(why GLA chunks); this teaching companion stays at modest $L$ with $\gamma$ near
$1$, where the recurrent and masked forms agree to ``< 1e-12`` in float64.

Idiomatic-JAX / port credit
---------------------------
Greenfield (no predecessor JAX gated-linear-attention implementation). The
decay mask reuses ``companions/ch09/jax/ssd_semiseparable.segsum`` — the same
stable cumulative-segment-sum that builds Chapter 9's mask — so the cross-chapter
identity is checked against the *actual* Chapter 9 code, not a re-implementation.
The recurrence is a ``lax.scan``; the parallel form is one ``einsum``-style matmul.

Usage
-----
::

    PYTHONPATH=. python companions/ch11/jax/gated_linear_attention.py
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402

from companions.ch09.jax.ssd_semiseparable import segsum  # noqa: E402
from companions.ch11.jax.linear_attention import resolve_phi  # noqa: E402

if TYPE_CHECKING:
    from matplotlib.figure import Figure

__all__ = [
    "gated_recurrent",
    "gated_masked",
    "retnet_decay_mask",
    "gla_scalar_decay_mask",
    "ch09_decay_mask",
]

_REPO_ROOT = Path(__file__).resolve().parents[3]
_OUT_DIR = _REPO_ROOT / "public" / "figures" / "ch11"


def _broadcast_log_gamma(log_gamma: jnp.ndarray, length: int, d_k: int) -> jnp.ndarray:
    """Accept a scalar-per-step (L,) or per-feature (L, d_k) log-gate; return (L, d_k)."""
    log_gamma = jnp.asarray(log_gamma)
    if log_gamma.ndim == 1:
        if log_gamma.shape[0] != length:
            raise ValueError(f"log_gamma (L,) must have L={length}, got {log_gamma.shape[0]}")
        return jnp.broadcast_to(log_gamma[:, None], (length, d_k))
    if log_gamma.shape != (length, d_k):
        raise ValueError(f"log_gamma must be (L,) or (L, d_k)=({length}, {d_k}); got {log_gamma.shape}")
    return log_gamma


# ---------------------------------------------------------------------------
# §11.3 — the gated recurrence and its masked-parallel twin
# ---------------------------------------------------------------------------


def gated_recurrent(
    q: jnp.ndarray,
    k: jnp.ndarray,
    v: jnp.ndarray,
    log_gamma: jnp.ndarray,
    feature_map: str = "elu",
) -> jnp.ndarray:
    r"""Gated linear attention via ``lax.scan``: $S_t = \mathrm{diag}(\gamma_t)S_{t-1} + \phi(k_t)v_t^\top$.

    The oracle for Theorem ``ch11:gla-ltv-duality``. Unnormalized output
    $y_t = S_t^\top\phi(q_t)$.

    Parameters
    ----------
    q, k : jnp.ndarray, shape (L, d)
    v : jnp.ndarray, shape (L, d_v)
    log_gamma : jnp.ndarray, shape (L,) or (L, d_k)
        Per-step log-gate $\log\gamma_t \le 0$. A ``(L,)`` vector is a scalar gate
        (RetNet-style if constant); a ``(L, d_k)`` matrix is a per-feature gate
        (GLA-style). ``d_k`` is the feature dimension after $\phi$.
    feature_map : default "elu"

    Returns
    -------
    y : jnp.ndarray, shape (L, d_v)
    """
    if q.ndim != 2 or k.ndim != 2 or v.ndim != 2:
        raise ValueError(f"q, k, v must be 2D; got {q.shape}, {k.shape}, {v.shape}")
    if q.shape != k.shape or v.shape[0] != q.shape[0]:
        raise ValueError(f"need q.shape==k.shape and matching L; got {q.shape}, {k.shape}, {v.shape}")
    phi = resolve_phi(feature_map)
    qf, kf = phi(q), phi(k)  # (L, d_k)
    length, d_k = qf.shape
    d_v = v.shape[1]
    gamma = jnp.exp(_broadcast_log_gamma(log_gamma, length, d_k))  # (L, d_k) in (0, 1]

    def step(s, inp):
        g_t, kf_t, v_t, qf_t = inp
        s = g_t[:, None] * s + jnp.outer(kf_t, v_t)  # diag(gamma_t) @ S + phi(k) v^T
        return s, s.T @ qf_t

    s0 = jnp.zeros((d_k, d_v), dtype=v.dtype)
    _, ys = jax.lax.scan(step, s0, (gamma, kf, v, qf))
    return ys


def gated_masked(
    q: jnp.ndarray,
    k: jnp.ndarray,
    v: jnp.ndarray,
    log_gamma: jnp.ndarray,
    feature_map: str = "elu",
) -> jnp.ndarray:
    r"""Masked-parallel gated linear attention $Y = (L_\gamma \circ (Q_\phi K_\phi^\top))V$.

    Computed by the GLA rescaling: with cumulative log-gate
    $g_{t} = \sum_{i\le t}\log\gamma_i$, set $\tilde q_t = \phi(q_t)\odot e^{g_t}$,
    $\tilde k_j = \phi(k_j)\odot e^{-g_j}$; then $\tilde q_t^\top\tilde k_j =
    \phi(q_t)^\top\mathrm{diag}(\prod_{i=j+1}^t\gamma_i)\phi(k_j)$ is the gated
    score, masked causally. Equals :func:`gated_recurrent` to ``< 1e-12``
    (Theorem ``ch11:gla-ltv-duality``).
    """
    phi = resolve_phi(feature_map)
    qf, kf = phi(q), phi(k)
    length, d_k = qf.shape
    g = jnp.cumsum(_broadcast_log_gamma(log_gamma, length, d_k), axis=0)  # (L, d_k)
    q_tilde = qf * jnp.exp(g)  # (L, d_k)
    k_tilde = kf * jnp.exp(-g)  # (L, d_k)
    scores = q_tilde @ k_tilde.T  # (L, L)
    causal = jnp.tril(jnp.ones((length, length), dtype=scores.dtype))
    return (scores * causal) @ v


# ---------------------------------------------------------------------------
# §11.3 — the decay masks (RetNet Toeplitz vs GLA non-Toeplitz vs ch09 bridge)
# ---------------------------------------------------------------------------


def retnet_decay_mask(gamma: float, length: int) -> jnp.ndarray:
    r"""RetNet constant-decay mask $L_{tj} = \gamma^{\,t-j}$ for $j\le t$, else $0$.

    Toeplitz (depends only on $t-j$) — the LTI face: a single fixed decay rate.
    """
    if not 0.0 < gamma <= 1.0:
        raise ValueError(f"gamma must be in (0, 1], got {gamma}")
    idx = jnp.arange(length)
    diff = idx[:, None] - idx[None, :]  # t - j
    return jnp.where(diff >= 0, jnp.asarray(gamma) ** diff, 0.0)


def gla_scalar_decay_mask(log_gamma_vec: jnp.ndarray) -> jnp.ndarray:
    r"""GLA scalar-gate decay mask $L_{tj} = \prod_{i=j+1}^{t}\gamma_i = e^{g_t - g_j}$, $j\le t$.

    Built from the cumulative log-gate $g_t = \sum_{i\le t}\log\gamma_i$ via the
    outer difference, masked to the causal triangle. For a *varying* (input-
    dependent) gate this is **not** Toeplitz — the LTV face.
    """
    log_gamma_vec = jnp.asarray(log_gamma_vec)
    if log_gamma_vec.ndim != 1:
        raise ValueError(f"log_gamma_vec must be 1D (L,), got shape {log_gamma_vec.shape}")
    length = log_gamma_vec.shape[0]
    g = jnp.cumsum(log_gamma_vec)
    diff = g[:, None] - g[None, :]  # g_t - g_j
    causal = jnp.tril(jnp.ones((length, length), dtype=bool))
    return jnp.where(causal, jnp.exp(diff), 0.0)


def ch09_decay_mask(a: float, delta: jnp.ndarray) -> jnp.ndarray:
    r"""Chapter 9's decay mask $L_{tj} = \exp(a\sum_{i=j+1}^t \Delta_i)$ via ``ch09.segsum``.

    The `ch09:ssd-duality` mask. Equals :func:`gla_scalar_decay_mask` evaluated at
    $\log\gamma_i = a\,\Delta_i$ — the cross-chapter round-trip (Theorem
    ``ch11:gla-ltv-duality``), pinned in ``test_gla_matches_ch09_decay_mask``.
    """
    return jnp.exp(segsum(jnp.asarray(a) * jnp.asarray(delta)))


# ---------------------------------------------------------------------------
# Figure: RetNet (Toeplitz) vs GLA (non-Toeplitz) decay masks
# ---------------------------------------------------------------------------


def make_decay_mask_figure() -> Figure:
    """Two heatmaps: RetNet constant-gamma (Toeplitz) vs GLA input-dependent gamma (not Toeplitz)."""
    import numpy as np

    from companions._shared.plot_utils import (
        apply_style,
        create_tufte_figure,
        set_tufte_labels,
        set_tufte_title,
    )

    apply_style()
    length = 40

    retnet = np.asarray(retnet_decay_mask(0.9, length))

    # GLA: an input-dependent per-step gate (sigmoid of a smooth-ish random signal).
    rng = np.random.default_rng(0)
    raw = rng.standard_normal(length)
    gate = 0.80 + 0.19 * (1.0 / (1.0 + np.exp(-raw)))  # gamma_t in ~(0.80, 0.99)
    gla = np.asarray(gla_scalar_decay_mask(jnp.asarray(np.log(gate))))

    fig, (ax1, ax2) = create_tufte_figure(ncols=2, figsize=(11.0, 4.3))
    im1 = ax1.imshow(retnet, cmap="magma", aspect="equal", vmin=0.0, vmax=1.0)
    set_tufte_title(ax1, r"RetNet: $\gamma^{t-j}$ (Toeplitz, LTI)")
    set_tufte_labels(ax1, xlabel=r"source $j$", ylabel=r"target $t$")
    fig.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)

    im2 = ax2.imshow(gla, cmap="magma", aspect="equal", vmin=0.0, vmax=1.0)
    set_tufte_title(ax2, r"GLA: $\prod_{i=j+1}^t \gamma_i$ (not Toeplitz, LTV)")
    set_tufte_labels(ax2, xlabel=r"source $j$", ylabel=r"target $t$")
    fig.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)

    fig.tight_layout()
    return fig


def main() -> None:
    import matplotlib.pyplot as plt
    import numpy as np

    from companions._shared.plot_utils import save_figure

    print("Chapter 11 — gated_linear_attention.py")
    print("=" * 64)

    rng = np.random.default_rng(0)
    length, d, d_v = 32, 8, 6
    q = jnp.asarray(rng.standard_normal((length, d)))
    k = jnp.asarray(rng.standard_normal((length, d)))
    v = jnp.asarray(rng.standard_normal((length, d_v)))
    d_k = d  # elu preserves dimension

    # Per-feature (GLA) gate near 1: recurrent == masked.
    log_gamma_feat = jnp.log(0.90 + 0.09 * jax.nn.sigmoid(jnp.asarray(rng.standard_normal((length, d_k)))))
    y_rec = gated_recurrent(q, k, v, log_gamma_feat)
    y_mask = gated_masked(q, k, v, log_gamma_feat)
    print(f"  GLA per-feature gate: recurrent == masked  max diff = {float(jnp.max(jnp.abs(y_rec - y_mask))):.2e}  (Thm gla-ltv: ~0)")

    # Scalar (RetNet) constant gate: recurrent == masked, and mask is Toeplitz.
    gamma = 0.92
    log_gamma_scalar = jnp.full((length,), jnp.log(gamma))
    y_rec_s = gated_recurrent(q, k, v, log_gamma_scalar)
    y_mask_s = gated_masked(q, k, v, log_gamma_scalar)
    print(f"  RetNet constant gate: recurrent == masked  max diff = {float(jnp.max(jnp.abs(y_rec_s - y_mask_s))):.2e}")
    mask = retnet_decay_mask(gamma, length)
    # Toeplitz check: L[t,j] == L[t+1,j+1], i.e. mask[1:,1:] == mask[:-1,:-1].
    toe = float(jnp.max(jnp.abs(mask[1:, 1:] - mask[:-1, :-1])))
    print(f"  RetNet mask Toeplitz residual = {toe:.2e}  (constant gamma -> Toeplitz)")

    # Cross-chapter round-trip: GLA scalar mask == ch09 decay mask under log gamma = a*delta.
    a = -0.3
    delta = jnp.asarray(0.5 + 0.5 * rng.random(length))  # positive steps
    gla_mask = gla_scalar_decay_mask(a * delta)
    ch09_mask = ch09_decay_mask(a, delta)
    bridge = float(jnp.max(jnp.abs(gla_mask - ch09_mask)))
    print(f"  GLA scalar mask == ch09 decay mask (log g = a*delta): max diff = {bridge:.2e}  (round-trip: ~0)")

    _OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig = make_decay_mask_figure()
    for p in save_figure(fig, _OUT_DIR / "gla-decay-mask", formats=("png",)):
        print(f"Wrote {p}")
    plt.close(fig)


if __name__ == "__main__":
    main()
