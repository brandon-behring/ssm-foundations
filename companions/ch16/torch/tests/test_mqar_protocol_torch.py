"""JAX <-> torch parity for the ch16 companions.

Episodes and HMM parameters are built once on the JAX side (the canonical
generator), converted through NumPy, and fed to both frameworks; integer
decodes must agree exactly, probability paths to < 1e-9 (float64 both
sides). Parity sequences are short (L = 512) — parity needs identical
arithmetic, not the figure-scale instance.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

jax = pytest.importorskip("jax")
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402

from companions.ch14.jax.two_timescale import (  # noqa: E402
    epsilon_to_lambda,
    forward_filter_predictions,
    make_hmm,
    sample_sequence,
)
from companions.ch14.torch.two_timescale import TwoTimescaleHMM as TorchHMM  # noqa: E402
from companions.ch16.jax import mqar as jmqar  # noqa: E402
from companions.ch16.jax import protocol as jprotocol  # noqa: E402
from companions.ch16.torch import mqar as tmqar  # noqa: E402
from companions.ch16.torch import protocol as tprotocol  # noqa: E402

_EPS = 0.05
_LAM_MIS = 0.3
_WINDOW = 8


@pytest.fixture(scope="module")
def episode():
    inst = jmqar.make_mqar(jax.random.PRNGKey(2), 24, 256, 32, gap=6, n_distractors=16)
    torch_inst = tmqar.MQARInstance(
        tokens=torch.tensor(np.asarray(inst.tokens)),
        query_positions=torch.tensor(np.asarray(inst.query_positions)),
        answers=torch.tensor(np.asarray(inst.answers)),
        n_keys=inst.n_keys,
        n_values=inst.n_values,
        n_distractors=inst.n_distractors,
    )
    return inst, torch_inst


@pytest.fixture(scope="module")
def hmm_pair():
    key = jax.random.PRNGKey(31)
    hmm = make_hmm(key, 4, 12, _EPS, 0.3, 0.4)
    tokens, _ = sample_sequence(jax.random.fold_in(key, 1), hmm, 512)
    torch_hmm = TorchHMM(
        transition=torch.tensor(np.asarray(hmm.transition)),
        bigrams=torch.tensor(np.asarray(hmm.bigrams)),
    )
    return hmm, tokens, torch_hmm, torch.tensor(np.asarray(tokens))


def test_induction_reader_parity(episode) -> None:
    inst, torch_inst = episode
    j = np.asarray(jmqar.induction_reader(inst))
    t = tmqar.induction_reader(torch_inst).numpy()
    np.testing.assert_array_equal(j, t)


def test_outer_product_reader_parity(episode) -> None:
    inst, torch_inst = episode
    for dim, seed in ((16, 0), (64, 3)):
        j = np.asarray(jmqar.outer_product_reader(inst, dim, seed=seed))
        t = tmqar.outer_product_reader(torch_inst, dim, seed=seed).numpy()
        np.testing.assert_array_equal(j, t)


def test_decay_reader_parity(episode) -> None:
    inst, torch_inst = episode
    for rho in (0.95, 1.0):
        j = np.asarray(jmqar.decay_reader(inst, 32, rho))
        t = tmqar.decay_reader(torch_inst, 32, rho).numpy()
        np.testing.assert_array_equal(j, t)


def test_accuracy_parity(episode) -> None:
    inst, torch_inst = episode
    j = jmqar.accuracy(jmqar.outer_product_reader(inst, 16), inst.answers)
    t = tmqar.accuracy(tmqar.outer_product_reader(torch_inst, 16), torch_inst.answers)
    assert j == t


@pytest.mark.parametrize("lam", [None, _LAM_MIS])
def test_composite_filter_parity(hmm_pair, lam) -> None:
    hmm, tokens, torch_hmm, torch_tokens = hmm_pair
    j_preds, j_priors = jprotocol.composite_filter_predictions(hmm, tokens, _WINDOW, lam)
    t_preds, t_priors = tprotocol.composite_filter_predictions(torch_hmm, torch_tokens,
                                                               _WINDOW, lam)
    assert float(np.max(np.abs(np.asarray(j_preds) - t_preds.numpy()))) < 1e-9
    assert float(np.max(np.abs(np.asarray(j_priors) - t_priors.numpy()))) < 1e-9


def test_composite_matched_lambda_is_full_filter_torch(hmm_pair) -> None:
    # The headline identity reproduced inside torch alone.
    hmm, tokens, torch_hmm, torch_tokens = hmm_pair
    lam_star = epsilon_to_lambda(_EPS, 4)
    t_preds, _ = tprotocol.composite_filter_predictions(torch_hmm, torch_tokens,
                                                        _WINDOW, lam_star)
    full, _ = forward_filter_predictions(hmm, tokens)
    assert float(np.max(np.abs(np.asarray(full) - t_preds.numpy()))) < 1e-12


def test_ridge_probe_parity(hmm_pair) -> None:
    hmm, tokens, torch_hmm, torch_tokens = hmm_pair
    j_feats = jprotocol.filter_regime_priors(hmm, tokens, "full")
    labels = np.argmax(np.asarray(j_feats), axis=1) % 4  # deterministic synthetic labels
    j_acc = jprotocol.ridge_probe_accuracy(j_feats, jnp.asarray(labels), 4)
    t_acc = tprotocol.ridge_probe_accuracy(torch.tensor(np.asarray(j_feats)),
                                           torch.tensor(labels), 4)
    assert j_acc == t_acc


def test_torch_validation_errors(hmm_pair, episode) -> None:
    _, _, torch_hmm, torch_tokens = hmm_pair
    _, torch_inst = episode
    with pytest.raises(ValueError):
        tprotocol.composite_filter_predictions(torch_hmm, torch_tokens, 0, None)
    with pytest.raises(ValueError):
        tprotocol.ridge_probe_accuracy(torch.zeros(10, 3), torch.zeros(9, dtype=torch.long), 4)
    with pytest.raises(ValueError):
        tmqar.decay_reader(torch_inst, 8, 0.0)
    with pytest.raises(ValueError):
        tmqar.outer_product_reader(torch_inst, 0)
    with pytest.raises(ValueError):
        tmqar.induction_reader(torch_inst, beta=-1.0)
