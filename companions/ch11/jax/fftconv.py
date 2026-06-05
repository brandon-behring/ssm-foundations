r"""Chapter 11 §11.4 — Hyena's primitive: the causal long convolution via FFT.

Hyena (Poli et al. 2023, arXiv:2302.10866) replaces attention with an
interleaving of implicit **long convolutions** and multiplicative gating. The
convolution is the object to teach: a causal long convolution is a *linear
time-invariant* operator (Chapter 8's convolutional view, `ch08:conv-recurrence-duality`),
computable in $O(L\log L)$ by FFT instead of the $O(L^2)$ explicit Toeplitz
product. For input $u\in\mathbb{R}^{B\times L\times D}$, per-channel filter
$k\in\mathbb{R}^{D\times L}$, feedthrough $b\in\mathbb{R}^{D}$,

.. math::

    y_{b,t,d} = \sum_{s=0}^{t} k_{d,\,t-s}\,u_{b,s,d} + b_d\,u_{b,t,d}.

Theorem ``ch11:fftconv-causal`` (pinned here): zero-padding both sequences to
length $2L$, multiplying their real FFTs, inverse-transforming and truncating to
$L$ computes exactly that lower-triangular Toeplitz product. The $2L$ padding is
**necessary and sufficient** — a length-$L$ (un-padded) cyclic FFT wraps the
late taps around onto early outputs, destroying causality
(``test_2L_padding_necessary``).

S4 vs Hyena: S4's kernel $K = (C\bar B, C\bar A\bar B, \ldots)$ is *materialized*
from a structured ODE; Hyena's kernel is a free implicit filter (an MLP over
positions, §11.5). Both end in the same FFT-convolution machinery — Hyena keeps
the unrestricted kernel and pays the $\log$ factor, where S4/Mamba fix the kernel
shape (semiseparable, Chapter 9) to buy $O(L)$.

Idiomatic-JAX note
------------------
Pure ``jnp.fft.rfft`` / ``irfft`` along the sequence axis — no scan, no loop;
the naive oracle is one masked gather + ``einsum``. Re-authored in **float64**
(the predecessor below ran float32 with a ``1e-4`` tolerance); float64 tightens
the ``fftconv == naive`` identity to ``< 1e-12``.

Port credit
-----------
Ported from ``post_transformers/experiments/jax/week11/hyena_lineage.py``
(``fftconv``, ``causal_conv1d_naive``), itself mirroring the Safari reference
``fftconv_ref``. Changes here: float64 (tighter oracle pin), the un-padded
cyclic variant for the $2L$-necessity demonstration, and book figure/credit
conventions. Hyena: Poli et al., arXiv:2302.10866.

Usage
-----
::

    PYTHONPATH=. python companions/ch11/jax/fftconv.py
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402

if TYPE_CHECKING:
    from matplotlib.figure import Figure

__all__ = [
    "fftconv",
    "causal_conv1d_naive",
    "cyclic_conv_unpadded",
]

_REPO_ROOT = Path(__file__).resolve().parents[3]
_OUT_DIR = _REPO_ROOT / "public" / "figures" / "ch11"


def _check_shapes(u: jnp.ndarray, k: jnp.ndarray, bias: jnp.ndarray) -> tuple[int, int, int]:
    if u.ndim != 3:
        raise ValueError(f"u must be 3D (B, L, D), got shape {u.shape}")
    if k.ndim != 2:
        raise ValueError(f"k must be 2D (D, L), got shape {k.shape}")
    if bias.ndim != 1:
        raise ValueError(f"bias must be 1D (D,), got shape {bias.shape}")
    batch, seqlen, channels = u.shape
    if k.shape != (channels, seqlen):
        raise ValueError(f"k shape {k.shape} does not match (D, L) = ({channels}, {seqlen})")
    if bias.shape != (channels,):
        raise ValueError(f"bias shape {bias.shape} does not match (D,) = ({channels},)")
    return batch, seqlen, channels


def fftconv(u: jnp.ndarray, k: jnp.ndarray, bias: jnp.ndarray) -> jnp.ndarray:
    r"""FFT-based causal long convolution along the sequence axis, $O(B D L\log L)$.

    Pads ``u`` and ``k`` to length $2L$ so the cyclic FFT computes the *linear*
    (causal) convolution, multiplies their real FFTs pointwise, inverse-transforms
    and truncates to $L$, then adds the feedthrough $b\,u$.

    Parameters
    ----------
    u : jnp.ndarray, shape (B, L, D)
        Input sequence (batch, length, channels).
    k : jnp.ndarray, shape (D, L)
        Per-channel causal filter; ``k[d, tau]`` is the tap at lag ``tau``.
    bias : jnp.ndarray, shape (D,)
        Per-channel feedthrough (the classical SSM ``D`` term).

    Returns
    -------
    y : jnp.ndarray, shape (B, L, D)
        ``y[b, t, d] = sum_{s<=t} k[d, t-s] u[b, s, d] + bias[d] u[b, t, d]``.

    Raises
    ------
    ValueError
        If shapes are inconsistent.
    """
    _, seqlen, _ = _check_shapes(u, k, bias)
    fft_size = 2 * seqlen  # the 2L padding that makes cyclic == linear (causal)

    u_t = jnp.transpose(u, (0, 2, 1))  # (B, D, L)
    u_f = jnp.fft.rfft(u_t, n=fft_size, axis=-1)
    k_f = jnp.fft.rfft(k, n=fft_size, axis=-1)
    y_f = u_f * k_f[None, :, :]
    y = jnp.fft.irfft(y_f, n=fft_size, axis=-1)[..., :seqlen]  # (B, D, L)
    y = y + bias[None, :, None] * u_t
    return jnp.transpose(y, (0, 2, 1))  # (B, L, D)


def causal_conv1d_naive(u: jnp.ndarray, k: jnp.ndarray, bias: jnp.ndarray) -> jnp.ndarray:
    r"""$O(L^2)$ reference: the explicit lower-triangular Toeplitz product.

    Same contract as :func:`fftconv`; builds ``K[d, t, s] = k[d, t-s]`` (zero
    above the diagonal) and contracts directly. Ground-truth oracle for the
    ``< 1e-12`` identity pin.
    """
    _, seqlen, _ = _check_shapes(u, k, bias)
    idx = jnp.arange(seqlen)
    diff = idx[:, None] - idx[None, :]  # (L, L); t - s, negative above diagonal
    valid = diff >= 0
    diff_safe = jnp.where(valid, diff, 0)
    big_k = jnp.where(valid[None, :, :], k[:, diff_safe], 0.0)  # (D, L, L)
    u_t = jnp.transpose(u, (0, 2, 1))  # (B, D, L)
    y = jnp.einsum("dts,bds->bdt", big_k, u_t) + bias[None, :, None] * u_t
    return jnp.transpose(y, (0, 2, 1))


def cyclic_conv_unpadded(u: jnp.ndarray, k: jnp.ndarray, bias: jnp.ndarray) -> jnp.ndarray:
    r"""Deliberately-wrong length-$L$ (un-padded) cyclic FFT convolution.

    Identical to :func:`fftconv` but with ``n = L`` instead of ``2L``. The cyclic
    FFT then wraps late filter taps around onto early outputs, so the result is
    **not causal** and disagrees with :func:`causal_conv1d_naive`. Used only to
    demonstrate that the $2L$ padding is necessary (``test_2L_padding_necessary``,
    Theorem ``ch11:fftconv-causal``).
    """
    _, seqlen, _ = _check_shapes(u, k, bias)
    u_t = jnp.transpose(u, (0, 2, 1))
    u_f = jnp.fft.rfft(u_t, n=seqlen, axis=-1)  # n = L: no padding -> wrap-around
    k_f = jnp.fft.rfft(k, n=seqlen, axis=-1)
    y = jnp.fft.irfft(u_f * k_f[None, :, :], n=seqlen, axis=-1)[..., :seqlen]
    y = y + bias[None, :, None] * u_t
    return jnp.transpose(y, (0, 2, 1))


# ---------------------------------------------------------------------------
# Figure: fftconv == naive (machine zero) + O(L log L) vs O(L^2) cost
# ---------------------------------------------------------------------------

_LENGTHS = (16, 32, 64, 128, 256, 512)


def _demo_inputs(length: int, channels: int = 4, batch: int = 2, seed: int = 0):
    import numpy as np

    rng = np.random.default_rng(seed)
    u = jnp.asarray(rng.standard_normal((batch, length, channels)))
    # A decaying causal filter (typical Hyena implicit-filter shape).
    taps = np.arange(length)
    k = jnp.asarray(rng.standard_normal((channels, length)) * np.exp(-0.05 * taps)[None, :])
    bias = jnp.asarray(rng.standard_normal(channels))
    return u, k, bias


def make_fftconv_figure() -> Figure:
    """Left: |fftconv - naive| vs L (machine zero). Right: O(L log L) vs O(L^2) cost."""
    import numpy as np

    from companions._shared.plot_utils import (
        SSM_COLORS,
        apply_style,
        create_tufte_figure,
        set_tufte_labels,
        set_tufte_title,
    )

    apply_style()
    lengths = np.asarray(_LENGTHS)
    resid = []
    for length in lengths:
        u, k, bias = _demo_inputs(int(length))
        resid.append(float(jnp.max(jnp.abs(fftconv(u, k, bias) - causal_conv1d_naive(u, k, bias)))))
    resid = np.asarray(resid)

    fig, (ax1, ax2) = create_tufte_figure(ncols=2, figsize=(11.0, 4.2))
    ax1.semilogy(lengths, np.maximum(resid, 1e-18), "o-", color=SSM_COLORS["accent"])
    ax1.axhline(1e-12, color=SSM_COLORS["alert"], lw=0.8, ls="--", label=r"$10^{-12}$ pin")
    set_tufte_title(ax1, "FFT conv $\\equiv$ Toeplitz oracle (float64)")
    set_tufte_labels(ax1, xlabel=r"sequence length $L$", ylabel=r"$\max|{\rm fft}-{\rm naive}|$")
    ax1.legend(loc="upper left", fontsize=8, frameon=False)

    fl = lengths.astype(float)
    ax2.loglog(fl, fl * np.log2(fl), "o-", color=SSM_COLORS["accent"], label=r"FFT $O(L\log L)$")
    ax2.loglog(fl, fl ** 2, "s-", color=SSM_COLORS["highlight"], label=r"Toeplitz $O(L^2)$")
    set_tufte_title(ax2, "Why FFT: sub-quadratic long convolution")
    set_tufte_labels(ax2, xlabel=r"sequence length $L$", ylabel="relative cost")
    ax2.legend(loc="upper left", fontsize=8, frameon=False)

    fig.tight_layout()
    return fig


def main() -> None:
    import matplotlib.pyplot as plt

    from companions._shared.plot_utils import save_figure

    print("Chapter 11 — fftconv.py")
    print("=" * 64)

    for length in (64, 256, 512):
        u, k, bias = _demo_inputs(length)
        y_fft = fftconv(u, k, bias)
        y_naive = causal_conv1d_naive(u, k, bias)
        resid = float(jnp.max(jnp.abs(y_fft - y_naive)))
        print(f"  L={length:4d}: fftconv == naive  max diff = {resid:.2e}  (Thm fftconv-causal: < 1e-12)")

    # 2L padding is necessary: the un-padded cyclic conv breaks causality.
    u, k, bias = _demo_inputs(128)
    y_naive = causal_conv1d_naive(u, k, bias)
    nopad_err = float(jnp.max(jnp.abs(cyclic_conv_unpadded(u, k, bias) - y_naive)))
    print(f"  un-padded (n=L) cyclic conv vs naive: max diff = {nopad_err:.2e}  (NOT causal -> 2L needed)")

    _OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig = make_fftconv_figure()
    for p in save_figure(fig, _OUT_DIR / "fftconv", formats=("png",)):
        print(f"Wrote {p}")
    plt.close(fig)


if __name__ == "__main__":
    main()
