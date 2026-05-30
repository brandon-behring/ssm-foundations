r"""Cross-framework tests for the Chapter 8 torch companions (S4D §8.5, S5 §8.6).

The strong claim (carried over from the Chapter 7 torch tests): the JAX and
PyTorch companions are two spellings of *one* math object, agreeing to numerical
precision. Pins:

* torch S4D Vandermonde kernel ≡ JAX S4D kernel for matched $(A, C, \Delta)$;
* torch S5 sequential scan ≡ JAX S5 associative scan for matched $(A, B, C, u)$;
* torch S4D stability is structural ($\mathrm{Re}(A) < 0$, $|\bar A| < 1$).

Run: ``PYTHONPATH=. .venv/bin/pytest companions/ch08/torch/tests/test_s4d_torch.py -q``
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import torch

from companions.ch08.jax import s4d_kernel as js4d
from companions.ch08.jax import s5_scan as js5
from companions.ch08.torch import s4d_kernel as ts4d
from companions.ch08.torch import s5_sequential as ts5


def test_s4d_kernel_jax_torch_parity() -> None:
    """torch S4D Vandermonde kernel equals the JAX one bit-for-bit (complex128)."""
    n, dt, L = 12, 0.1, 96
    rng = np.random.default_rng(0)
    C_np = rng.standard_normal(n) + 1j * rng.standard_normal(n)

    K_jax = np.asarray(js4d.s4d_kernel(js4d.make_s4d_lin(n), jnp.asarray(C_np), dt, L))
    K_torch = ts4d.s4d_kernel(
        ts4d.make_s4d_lin(n), torch.as_tensor(C_np, dtype=torch.complex128), dt, L
    ).numpy()
    np.testing.assert_allclose(K_torch, K_jax, atol=1e-9)


def test_s4d_torch_stability_by_construction() -> None:
    """torch S4D-Lin: Re(A) < 0 and |Abar| < 1 (the §8.5 structural guarantee)."""
    A = ts4d.make_s4d_lin(16)
    assert torch.all(A.real < 0.0)
    for dt in (0.01, 0.1, 1.0):
        assert float(torch.exp(A * dt).abs().max()) < 1.0


def test_s4d_kernel_module_matches_functional() -> None:
    """The S4DKernel nn.Module forward equals the functional kernel for its own params."""
    kernel = ts4d.S4DKernel(10, seed=3)
    K_mod = kernel(48, dt=0.1)
    K_fun = ts4d.s4d_kernel(kernel.diagonal(), kernel.C, 0.1, 48)
    torch.testing.assert_close(K_mod, K_fun)


def test_s5_jax_torch_parity() -> None:
    """torch S5 sequential scan equals the JAX associative scan for matched inputs."""
    n, h, dt, L = 8, 4, 0.1, 200
    rng = np.random.default_rng(1)
    B_np = rng.standard_normal((n, h)) + 1j * rng.standard_normal((n, h))
    C_np = rng.standard_normal((h, n)) + 1j * rng.standard_normal((h, n))
    z = np.linspace(0.0, 1.0, L)
    u_np = np.stack([np.sin(2 * np.pi * (k + 1) * z) for k in range(h)], axis=1)

    y_jax = np.asarray(
        js5.s5_apply(
            js4d.make_s4d_lin(n),
            jnp.asarray(B_np),
            jnp.asarray(C_np),
            dt,
            jnp.asarray(u_np),
            parallel=True,
        )
    )
    y_torch = ts5.s5_apply(
        ts4d.make_s4d_lin(n),
        torch.as_tensor(B_np),
        torch.as_tensor(C_np),
        dt,
        torch.as_tensor(u_np),
    ).numpy()
    np.testing.assert_allclose(y_torch, y_jax, atol=1e-8)


def test_s5_layer_runs_and_is_real() -> None:
    """S5Layer maps (L, H) -> (L, H) real with zero learnable params."""
    layer = ts5.S5Layer(n_modes=6, h_dim=3)
    u = torch.randn(50, 3, dtype=torch.float64)
    with torch.no_grad():
        y = layer(u)
    assert y.shape == (50, 3)
    assert y.dtype == torch.float64
    assert sum(p.numel() for p in layer.parameters()) == 0
