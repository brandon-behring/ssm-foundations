r"""Cross-framework parity: PyTorch Ch 3 condition-number vs the JAX reference (§3.x).

Backfills JAX↔torch parity for Ch 3 (audit 0527-F27). The existing torch test
(``test_torch_condition_number.py``) guards the 0527-F14 claim against a NumPy
HiPPO-LegS build; this file locks the two companion languages together — the
deterministic HiPPO-LegS and Hilbert constructions are identical, and so is the
operator-norm condition number $\kappa$ (``jnp.linalg.cond`` vs ``torch.linalg.cond``).

The random-Gaussian family is intentionally *not* parity-checked: the two companions
seed different RNGs (NumPy ``default_rng`` vs ``torch.Generator``) by design, so the
matrices — and hence $\kappa$ — differ. The conditioning *story* does not depend on
the RNG choice (see the JAX companion's note).

Run: ``PYTHONPATH=. .venv/bin/pytest companions/ch03/torch -q``
"""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

import jax  # noqa: E402

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402

from companions.ch03.jax import condition_number as jax_cn  # noqa: E402
from companions.ch03.torch import condition_number as torch_cn  # noqa: E402

_EXACT_TOL = 1e-12


def test_hippo_legs_matches_jax() -> None:
    """Deterministic HiPPO-LegS construction is identical across frameworks."""
    for N in (1, 2, 8, 32, 64):
        j = np.asarray(jax_cn.hippo_legs(N))
        t = torch_cn.hippo_legs(N).numpy()
        assert np.max(np.abs(j - t)) < _EXACT_TOL, f"divergence at N={N}"


def test_hilbert_matches_jax() -> None:
    """The Hilbert matrix $H_{ij}=1/(i+j-1)$ is identical across frameworks."""
    for N in (4, 16, 32):
        j = np.asarray(jax_cn.hilbert(N))
        t = torch_cn.hilbert(N).numpy()
        assert np.max(np.abs(j - t)) < _EXACT_TOL, f"divergence at N={N}"


def test_hippo_condition_number_matches_jax() -> None:
    """κ(HiPPO-LegS) agrees across ``jnp.linalg.cond`` and ``torch.linalg.cond``."""
    for N in (8, 32, 128):
        j = float(jnp.linalg.cond(jax_cn.hippo_legs(N)))
        t = float(torch.linalg.cond(torch_cn.hippo_legs(N)))
        np.testing.assert_allclose(j, t, rtol=1e-6)
