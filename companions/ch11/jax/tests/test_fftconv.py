r"""Tests for ch11 fftconv: the Hyena long-convolution primitive.

Pins the §11.4 claims (Theorem ``ch11:fftconv-causal``):

* the $O(L\log L)$ FFT convolution equals the $O(L^2)$ explicit-Toeplitz oracle
  to ``< 1e-12`` in float64 (the predecessor ran float32 at ``1e-4``);
* the convolution is causal (an impulse at $t_0$ leaves all outputs before $t_0$
  at zero);
* the $2L$ padding is necessary — the un-padded length-$L$ cyclic convolution
  wraps around and disagrees with the oracle by an $O(1)$ amount.
"""

from __future__ import annotations

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402
from numpy.testing import assert_allclose  # noqa: E402

from companions.ch11.jax.fftconv import (  # noqa: E402
    causal_conv1d_naive,
    cyclic_conv_unpadded,
    fftconv,
)


def _inputs(length=128, channels=4, batch=2, seed=0):
    rng = np.random.default_rng(seed)
    u = jnp.asarray(rng.standard_normal((batch, length, channels)))
    taps = np.arange(length)
    k = jnp.asarray(rng.standard_normal((channels, length)) * np.exp(-0.05 * taps)[None, :])
    bias = jnp.asarray(rng.standard_normal(channels))
    return u, k, bias


@pytest.mark.parametrize("length", [32, 128, 512])
@pytest.mark.parametrize("seed", [0, 5])
def test_fftconv_equals_naive(length, seed):
    """FFT conv == explicit Toeplitz oracle to machine precision (float64)."""
    u, k, bias = _inputs(length=length, seed=seed)
    assert_allclose(np.asarray(fftconv(u, k, bias)),
                    np.asarray(causal_conv1d_naive(u, k, bias)), rtol=0, atol=1e-12)


def test_causality_no_wraparound():
    """An impulse at t0 leaves every output strictly before t0 at zero."""
    length, channels, t0 = 64, 3, 40
    u = jnp.zeros((1, length, channels)).at[0, t0, :].set(1.0)
    rng = np.random.default_rng(1)
    k = jnp.asarray(rng.standard_normal((channels, length)))
    bias = jnp.zeros((channels,))
    y = np.asarray(fftconv(u, k, bias))
    assert_allclose(y[0, :t0, :], 0.0, rtol=0, atol=1e-12)
    # And the response at/after t0 matches the kernel taps k[d, t - t0].
    for d in range(channels):
        assert_allclose(y[0, t0:, d], np.asarray(k[d, : length - t0]), rtol=0, atol=1e-12)


def test_2L_padding_necessary():
    """The un-padded (n=L) cyclic conv breaks causality; the 2L-padded one does not."""
    u, k, bias = _inputs(length=128, seed=2)
    y_naive = causal_conv1d_naive(u, k, bias)
    # 2L padding: matches the oracle.
    assert_allclose(np.asarray(fftconv(u, k, bias)), np.asarray(y_naive), rtol=0, atol=1e-12)
    # No padding: wraps around, disagrees by an O(1) amount.
    nopad_err = float(jnp.max(jnp.abs(cyclic_conv_unpadded(u, k, bias) - y_naive)))
    assert nopad_err > 1e-1, f"un-padded cyclic conv should break causality; got {nopad_err:.2e}"


def test_shape_validation():
    u, k, bias = _inputs(length=8, channels=2)
    with pytest.raises(ValueError):
        fftconv(u, k[:, :-1], bias)  # k length != L
    with pytest.raises(ValueError):
        fftconv(u, k, bias[:-1])  # bias length != D
    with pytest.raises(ValueError):
        causal_conv1d_naive(u[0], k, bias)  # u not 3D
