r"""Chapter 14 §§14.3–14.5 — the hybrid composition design space, measured.

A hybrid block mixes two primitives with complementary costs:

* **sliding-window attention** — exact pairwise interaction over the last
  $w$ tokens ($O(Lw)$ compute, $O(w)$ decode cache): the *fast path*;
* **gated-decay SSM** — a per-channel EMA recurrence
  $h_t = g_t \odot h_{t-1} + (1 - g_t) \odot x_t$ ($O(L)$ compute, $O(1)$
  decode state): the *slow path* (Chapter 11's GLA decay mask, simplified to
  its diagonal skeleton).

This module materialises the three composition patterns the production
lineup of §14.5 instantiates — sequential, parallel-gated, interleaved at a
layer ratio $r\!:\!1$ — plus the decode-time cost accounting that makes the
layer-ratio a *budget* decision. Exact reductions are pinned by tests:

* window $w \ge L$ ⇒ sliding-window attention **is** full causal attention;
* parallel gate $g = 1$ ⇒ the attention branch exactly; $g = 0$ ⇒ the SSM
  branch exactly (the granularity ordering of §14.4 in executable form);
* constant input + constant gate ⇒ the EMA closed form
  $h_t = (1 - g^t)\,\bar{x}$ to machine precision;
* decode-buffer sizes == the cost formulas (the §14.3 accounting, asserted
  against real array shapes, not just arithmetic).

No learned projections, residual streams only in the interleaved stack, no
training: the object of study is the *mixing pattern*, not the head. Real
models wrap each primitive in projections/normalisation; none of that
changes the composition algebra or the decode-state accounting.

Idiomatic-JAX note (NumPy->JAX teaching point)
----------------------------------------------
The windowed attention is a *vectorised band mask* over the full score
matrix (one ``jnp.tril``-style mask, one softmax) — the idiomatic JAX form.
The per-position Python loop with an explicit window slice
(``sliding_window_attention_naive``) is the readable oracle; the pair must
agree to ``< 1e-12``, which is precisely the recurrent-vs-parallel duality
of Chapter 11 in miniature.

Port credit
-----------
Greenfield: the predecessor ``week16/hybrid_mad.py`` is a TODO stub, so this
module is authored from the architecture descriptions in the cited papers
(Griffin arXiv:2402.19427, Jamba arXiv:2403.19887, Kimi Linear
arXiv:2510.26692, Lee et al. arXiv:2510.26912). Production layer-ratio /
window data points in the figure are taken from those papers, as of
May 2026.

Usage
-----
::

    PYTHONPATH=. python companions/ch14/jax/hybrid_block.py
"""

from __future__ import annotations

from pathlib import Path

import jax

# Enable float64 before any jnp array is created (matches Chapters 4, 7-12).
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402

__all__ = [
    "full_causal_attention",
    "sliding_window_attention",
    "sliding_window_attention_naive",
    "gated_decay_ssm",
    "gated_decay_ssm_naive",
    "sequential_hybrid",
    "parallel_gated_hybrid",
    "interleave_schedule",
    "interleave_hybrid",
    "decode_state_floats",
    "decode_buffers",
    "full_attention_decode_floats",
]

_REPO_ROOT = Path(__file__).resolve().parents[3]
_OUT_DIR = _REPO_ROOT / "public" / "figures" / "ch14"


# ---------------------------------------------------------------------------
# Primitives: the fast path (windowed attention) and the slow path (EMA SSM).
# ---------------------------------------------------------------------------


def _validate_sequence(x: jnp.ndarray) -> None:
    if x.ndim != 2:
        raise ValueError(f"x must have shape (L, d); got {x.shape}")


def full_causal_attention(x: jnp.ndarray) -> jnp.ndarray:
    r"""Single-head causal self-attention with $q = k = v = x$ (no projections).

    Position $t$ attends to positions $1..t$ with scores
    $x_t^\top x_s / \sqrt{d}$. The $w \ge L$ reference for the windowed form.

    Parameters
    ----------
    x : jnp.ndarray, shape (L, d)

    Returns
    -------
    jnp.ndarray, shape (L, d)
    """
    _validate_sequence(x)
    length, d = x.shape
    scores = (x @ x.T) / jnp.sqrt(jnp.asarray(d, dtype=x.dtype))
    causal = jnp.tril(jnp.ones((length, length), dtype=bool))
    scores = jnp.where(causal, scores, -jnp.inf)
    return jax.nn.softmax(scores, axis=-1) @ x


def sliding_window_attention(x: jnp.ndarray, window: int) -> jnp.ndarray:
    r"""Causal self-attention restricted to the last ``window`` positions.

    Position $t$ attends to $\max(1, t - w + 1)..t$: a *band* mask instead of
    the full lower triangle. Decode-time cache is $O(w)$ keys/values per
    layer — the boundary-layer budget of §14.3.

    Parameters
    ----------
    x : jnp.ndarray, shape (L, d)
    window : int
        Window size $w \ge 1$. ``window >= L`` recovers full causal attention.

    Returns
    -------
    jnp.ndarray, shape (L, d)
    """
    _validate_sequence(x)
    if window < 1:
        raise ValueError(f"window must be >= 1; got {window}")
    length, d = x.shape
    scores = (x @ x.T) / jnp.sqrt(jnp.asarray(d, dtype=x.dtype))
    rows = jnp.arange(length)[:, None]
    cols = jnp.arange(length)[None, :]
    band = (cols <= rows) & (cols > rows - window)
    scores = jnp.where(band, scores, -jnp.inf)
    return jax.nn.softmax(scores, axis=-1) @ x


def sliding_window_attention_naive(x: jnp.ndarray, window: int) -> jnp.ndarray:
    r"""Python-loop oracle: explicit window slice + softmax per position."""
    _validate_sequence(x)
    if window < 1:
        raise ValueError(f"window must be >= 1; got {window}")
    length, d = x.shape
    scale = 1.0 / jnp.sqrt(jnp.asarray(d, dtype=x.dtype))
    outputs = []
    for t in range(length):
        lo = max(0, t - window + 1)
        keys = x[lo : t + 1]
        scores = (keys @ x[t]) * scale
        weights = jax.nn.softmax(scores)
        outputs.append(weights @ keys)
    return jnp.stack(outputs)


def gated_decay_ssm(x: jnp.ndarray, gates: jnp.ndarray) -> jnp.ndarray:
    r"""Per-channel gated EMA recurrence $h_t = g_t \odot h_{t-1} + (1-g_t) \odot x_t$.

    The diagonal skeleton of Chapter 11's gated linear attention: per-channel
    decay $g_t \in [0, 1]$, convex (EMA) input mixing, $h_0 = 0$. Decode
    state is $d$ floats — the slow-manifold budget of §14.3.

    Parameters
    ----------
    x : jnp.ndarray, shape (L, d)
    gates : jnp.ndarray, shape (L, d) or (d,)
        Decay gates in $[0, 1]$; a ``(d,)`` vector is broadcast across time.

    Returns
    -------
    jnp.ndarray, shape (L, d)
    """
    _validate_sequence(x)
    gates = jnp.asarray(gates)
    if gates.ndim == 1:
        gates = jnp.broadcast_to(gates, x.shape)
    if gates.shape != x.shape:
        raise ValueError(f"gates must have shape {x.shape} or ({x.shape[1]},); got {gates.shape}")
    if bool(jnp.any((gates < 0.0) | (gates > 1.0))):
        raise ValueError("gates must lie in [0, 1]")

    def step(h: jnp.ndarray, inp: tuple[jnp.ndarray, jnp.ndarray]):
        x_t, g_t = inp
        h_new = g_t * h + (1.0 - g_t) * x_t
        return h_new, h_new

    init = jnp.zeros(x.shape[1], dtype=x.dtype)
    _, hs = jax.lax.scan(step, init, (x, gates))
    return hs


def gated_decay_ssm_naive(x: jnp.ndarray, gates: jnp.ndarray) -> jnp.ndarray:
    r"""Python-loop oracle for :func:`gated_decay_ssm`."""
    _validate_sequence(x)
    gates = jnp.asarray(gates)
    if gates.ndim == 1:
        gates = jnp.broadcast_to(gates, x.shape)
    if gates.shape != x.shape:
        raise ValueError(f"gates must have shape {x.shape} or ({x.shape[1]},); got {gates.shape}")
    h = jnp.zeros(x.shape[1], dtype=x.dtype)
    outputs = []
    for t in range(x.shape[0]):
        h = gates[t] * h + (1.0 - gates[t]) * x[t]
        outputs.append(h)
    return jnp.stack(outputs)


# ---------------------------------------------------------------------------
# Compositions: sequential, parallel-gated, interleaved at ratio r:1.
# ---------------------------------------------------------------------------


def sequential_hybrid(
    x: jnp.ndarray,
    gates: jnp.ndarray,
    window: int,
    order: tuple[str, str] = ("ssm", "attn"),
) -> jnp.ndarray:
    r"""Pure composition of the two primitives in the given order.

    ``("ssm", "attn")`` computes ``attn(ssm(x))`` — the slow path feeds the
    fast path (the Jamba/Bamba-style sequential pattern, stripped to one
    pair). No residual stream: the point is the composition algebra.

    Parameters
    ----------
    x : jnp.ndarray, shape (L, d)
    gates : jnp.ndarray, shape (L, d) or (d,)
    window : int
    order : tuple of {"ssm", "attn"}
        Application order, first element applied first.

    Returns
    -------
    jnp.ndarray, shape (L, d)
    """
    if sorted(order) != ["attn", "ssm"]:
        raise ValueError(f'order must be a permutation of ("ssm", "attn"); got {order}')
    y = x
    for layer in order:
        y = gated_decay_ssm(y, gates) if layer == "ssm" else sliding_window_attention(y, window)
    return y


def parallel_gated_hybrid(
    x: jnp.ndarray,
    gate: float | jnp.ndarray,
    gates_ssm: jnp.ndarray,
    window: int,
) -> jnp.ndarray:
    r"""Gated parallel mix $g \odot \mathrm{attn}(x) + (1 - g) \odot \mathrm{ssm}(x)$.

    Both branches read the *same* input (the Hymba/GMU-style parallel
    pattern). The mixing gate ``g`` is the §14.4 granularity dial:

    * scalar $g$ — one number decides the blend for every channel;
    * vector $g \in [0,1]^d$ — per-channel blending (strictly contains the
      scalar family).

    Exact reductions $g = 1 \Rightarrow$ attention branch, $g = 0
    \Rightarrow$ SSM branch are the executable content of the §14.4
    granularity-ordering proposition.

    Parameters
    ----------
    x : jnp.ndarray, shape (L, d)
    gate : float or jnp.ndarray of shape (d,)
        Mixing gate in $[0, 1]$.
    gates_ssm : jnp.ndarray, shape (L, d) or (d,)
        Decay gates for the SSM branch.
    window : int

    Returns
    -------
    jnp.ndarray, shape (L, d)
    """
    _validate_sequence(x)
    g = jnp.asarray(gate, dtype=x.dtype)
    if g.ndim not in (0, 1) or (g.ndim == 1 and g.shape != (x.shape[1],)):
        raise ValueError(f"gate must be a scalar or shape ({x.shape[1]},); got shape {g.shape}")
    if bool(jnp.any((g < 0.0) | (g > 1.0))):
        raise ValueError("gate must lie in [0, 1]")
    return g * sliding_window_attention(x, window) + (1.0 - g) * gated_decay_ssm(x, gates_ssm)


def interleave_schedule(n_blocks: int, ratio: int) -> tuple[str, ...]:
    r"""The $r\!:\!1$ layer schedule: ``ratio`` SSM blocks, then one attention block.

    ``ratio=3, n_blocks=8`` gives ``(ssm, ssm, ssm, attn, ssm, ssm, ssm,
    attn)`` — the Kimi Linear pattern (3:1 KDA:MLA, arXiv:2510.26692).

    Parameters
    ----------
    n_blocks : int
    ratio : int
        SSM blocks per attention block, $r \ge 0$. ``ratio=0`` is pure
        attention.

    Returns
    -------
    tuple of {"ssm", "attn"}, length ``n_blocks``
    """
    if n_blocks < 1:
        raise ValueError(f"n_blocks must be >= 1; got {n_blocks}")
    if ratio < 0:
        raise ValueError(f"ratio must be >= 0; got {ratio}")
    period = ratio + 1
    return tuple("attn" if (i + 1) % period == 0 else "ssm" for i in range(n_blocks))


def interleave_hybrid(
    x: jnp.ndarray,
    schedule: tuple[str, ...],
    gates_ssm: jnp.ndarray,
    window: int,
) -> jnp.ndarray:
    r"""Residual stack following ``schedule``: $y \leftarrow y + \mathrm{layer}(y)$.

    The deep-stack form of the layer-ratio design variable. Residual
    connections are kept here (unlike the single-pair compositions above)
    because a stack of pure EMA layers contracts the signal — the residual
    stream is what makes deep interleaving well-posed, in this miniature as
    in production.

    Parameters
    ----------
    x : jnp.ndarray, shape (L, d)
    schedule : tuple of {"ssm", "attn"}
        Per-block layer types, e.g. from :func:`interleave_schedule`.
    gates_ssm : jnp.ndarray, shape (L, d) or (d,)
        Decay gates, shared across SSM blocks (a deliberate simplification).
    window : int

    Returns
    -------
    jnp.ndarray, shape (L, d)
    """
    _validate_sequence(x)
    if not schedule:
        raise ValueError("schedule must be non-empty")
    bad = sorted(set(schedule) - {"ssm", "attn"})
    if bad:
        raise ValueError(f'schedule entries must be "ssm" or "attn"; got {bad}')
    y = x
    for layer in schedule:
        if layer == "ssm":
            y = y + gated_decay_ssm(y, gates_ssm)
        else:
            y = y + sliding_window_attention(y, window)
    return y


# ---------------------------------------------------------------------------
# Decode-time cost accounting (§14.3): formulas + real buffers to audit them.
# ---------------------------------------------------------------------------


def decode_state_floats(schedule: tuple[str, ...], window: int, d: int) -> dict[str, int]:
    r"""Decode-time state budget of a hybrid stack, in float counts.

    Each attention block carries a rolling $w \times d$ key buffer and a
    $w \times d$ value buffer ($2wd$ floats); each SSM block carries its
    $d$-float state. Constant in sequence length $L$ — the whole point of
    the hybrid budget (full attention pays $2Ld$ per block instead).

    Parameters
    ----------
    schedule : tuple of {"ssm", "attn"}
    window : int
    d : int

    Returns
    -------
    dict with keys ``kv_floats``, ``ssm_floats``, ``total``
    """
    if window < 1 or d < 1:
        raise ValueError(f"window and d must be >= 1; got window={window}, d={d}")
    bad = sorted(set(schedule) - {"ssm", "attn"})
    if bad:
        raise ValueError(f'schedule entries must be "ssm" or "attn"; got {bad}')
    n_attn = sum(1 for s in schedule if s == "attn")
    n_ssm = len(schedule) - n_attn
    kv = n_attn * 2 * window * d
    ssm = n_ssm * d
    return {"kv_floats": kv, "ssm_floats": ssm, "total": kv + ssm}


def decode_buffers(schedule: tuple[str, ...], window: int, d: int) -> list[jnp.ndarray]:
    r"""Materialise the decode-time buffers the formula above counts.

    Returns the actual arrays — $(w, d)$ key and value buffers per attention
    block, a $(d,)$ state per SSM block — so tests can assert
    ``sum(b.size) == decode_state_floats(...)["total"]`` against real
    shapes rather than re-deriving the same arithmetic.
    """
    if window < 1 or d < 1:
        raise ValueError(f"window and d must be >= 1; got window={window}, d={d}")
    buffers: list[jnp.ndarray] = []
    for layer in schedule:
        if layer == "attn":
            buffers.append(jnp.zeros((window, d)))  # rolling key buffer
            buffers.append(jnp.zeros((window, d)))  # rolling value buffer
        elif layer == "ssm":
            buffers.append(jnp.zeros((d,)))
        else:
            raise ValueError(f'schedule entries must be "ssm" or "attn"; got {layer!r}')
    return buffers


def full_attention_decode_floats(n_blocks: int, context_len: int, d: int) -> int:
    r"""Decode-time KV cache of a pure full-attention stack: $2 L d$ per block."""
    if n_blocks < 1 or context_len < 1 or d < 1:
        raise ValueError(
            f"all arguments must be >= 1; got n_blocks={n_blocks}, "
            f"context_len={context_len}, d={d}"
        )
    return n_blocks * 2 * context_len * d


# ---------------------------------------------------------------------------
# Figures + measured numbers (§14.3 cost frontier, §14.4 design-space map).
# ---------------------------------------------------------------------------

# Production data points, as of May 2026 (architecture papers; see module
# docstring): (label, attention blocks, SSM blocks, annotation offset in pts).
_PRODUCTION_POINTS: list[tuple[str, int, int, tuple[int, int]]] = [
    ("Kimi Linear (3:1)", 1, 3, (8, -2)),  # arXiv:2510.26692
    ("Jamba (1:7)", 1, 7, (10, -16)),  # arXiv:2403.19887
    ("Bamba (3:29)", 3, 29, (0, 12)),  # IBM HF blog, 2024
    ("Nemotron-H (10:108)", 10, 108, (-20, -28)),  # arXiv:2504.03624
    ("Samba (1:1)", 1, 1, (8, -2)),  # arXiv:2406.07522
]

# Gating design-space placement (granularity x trigger), per the §14.4 table.
# Coordinates are categorical indices: granularity 0=scalar, 1=vector,
# 2=block; trigger 0=fixed schedule, 1=input-dependent (every step),
# 2=output/uncertainty-triggered. Trigger classifies the gate VALUE's
# dependence (a GMU's value is input-computed even though its placement
# across layers is fixed); Gated DeltaNet's decay gate is the scalar
# gamma_t per head (arXiv:2412.06464) — KDA's per-channel gates are
# precisely Kimi's refinement of it (arXiv:2510.26692).
_DESIGN_POINTS: list[tuple[str, int, int]] = [
    ("Nemotron-H", 0, 0),  # scalar schedule, fixed positions
    ("Jamba", 2, 0),  # block-structural choice, fixed
    ("Hunyuan TurboS", 2, 0),  # AMF/MF macro-blocks, fixed
    ("Gated DeltaNet", 0, 1),  # scalar gamma_t per head, input-dependent
    ("SambaY (GMU)", 1, 1),  # element-wise gate, value input-computed
    ("Griffin (RG-LRU)", 1, 1),  # per-channel recurrence gate, input-dep.
    ("Kimi Linear (KDA)", 1, 1),  # per-channel diagonal gates
    ("Hymba", 2, 1),  # parallel-head fusion, every step
    ("AMOR", 0, 2),  # scalar entropy threshold, output-triggered
]


def _fig_cost_frontier(d: int, window: int, context_len: int) -> None:
    import matplotlib.pyplot as plt

    from companions._shared.plot_utils import (
        SSM_COLORS,
        apply_style,
        create_tufte_figure,
        save_figure,
        set_tufte_labels,
        set_tufte_title,
    )

    apply_style()
    fig, ax = create_tufte_figure(figsize=(6.4, 4.0))

    # Per-block decode state is linear in the attention fraction; the dense
    # curve and every production point come from the same cost function.
    n_dense = 1000
    fractions, per_block = [], []
    for n_attn in range(n_dense + 1):
        schedule = tuple(["attn"] * n_attn + ["ssm"] * (n_dense - n_attn))
        fractions.append(n_attn / n_dense)
        per_block.append(decode_state_floats(schedule, window, d)["total"] / n_dense)
    full_per_block = full_attention_decode_floats(1, context_len, d)

    ax.plot(
        fractions,
        jnp.asarray(per_block) / 1e6,
        color=SSM_COLORS["accent"],
        lw=2.0,
        label=f"hybrid, w={window}",
    )
    ax.axhline(
        full_per_block / 1e6,
        color=SSM_COLORS["alert"],
        lw=1.5,
        ls="--",
        label=f"full attention, L={context_len:,}",
    )
    print("  cost-frontier production points (per mixing block, d=%d, w=%d):" % (d, window))
    for name, n_attn, n_ssm, offset in _PRODUCTION_POINTS:
        n_total = n_attn + n_ssm
        schedule = tuple(["attn"] * n_attn + ["ssm"] * n_ssm)
        per = decode_state_floats(schedule, window, d)["total"] / n_total
        frac = n_attn / n_total
        ax.plot([frac], [per / 1e6], "o", color=SSM_COLORS["highlight"], ms=6, zorder=5)
        ax.annotate(
            name,
            (frac, per / 1e6),
            textcoords="offset points",
            xytext=offset,
            fontsize=8,
            color="#333333",
        )
        print(f"    {name:<22} f={frac:.3f}  {per / 1e6:7.3f}M floats/block "
              f"({full_per_block / per:6.1f}x below full attention)")
    ax.set_yscale("log")
    set_tufte_title(ax, "Decode-state budget vs attention fraction (cost model, measured)")
    set_tufte_labels(ax, "attention fraction of mixing blocks",
                     "decode state per block (M floats)")
    ax.legend(frameon=False, fontsize=9, loc="lower right")
    _OUT_DIR.mkdir(parents=True, exist_ok=True)
    for p in save_figure(fig, _OUT_DIR / "cost-frontier", formats=("png",)):
        print(f"  wrote {p.relative_to(_REPO_ROOT)}")
    plt.close(fig)


def _fig_design_space() -> None:
    import matplotlib.pyplot as plt

    from companions._shared.plot_utils import (
        SSM_COLORS,
        apply_style,
        create_tufte_figure,
        save_figure,
        set_tufte_labels,
        set_tufte_title,
    )

    apply_style()
    fig, ax = create_tufte_figure(figsize=(6.4, 4.2))

    offsets = {  # jitter to keep co-located labels readable
        "Nemotron-H": (0.0, 0.0),
        "Jamba": (0.0, 0.12),
        "Hunyuan TurboS": (0.0, -0.12),
        "Gated DeltaNet": (0.0, 0.0),
        "SambaY (GMU)": (0.0, 0.0),
        "Griffin (RG-LRU)": (0.0, 0.16),
        "Kimi Linear (KDA)": (0.0, -0.16),
        "Hymba": (0.0, 0.0),
        "AMOR": (0.0, 0.0),
    }
    for name, gx, ty in _DESIGN_POINTS:
        dx, dy = offsets[name]
        ax.plot([gx + dx], [ty + dy], "o", color=SSM_COLORS["accent"], ms=7)
        ax.annotate(
            name,
            (gx + dx, ty + dy),
            textcoords="offset points",
            xytext=(8, -3),
            fontsize=8.5,
            color="#333333",
        )
    ax.set_xticks([0, 1, 2], ["scalar", "vector", "block"])
    ax.set_yticks([0, 1, 2], ["fixed schedule", "input-dependent", "output-triggered"])
    ax.set_xlim(-0.4, 2.8)
    ax.set_ylim(-0.4, 2.4)
    set_tufte_title(ax, "Gating design space: granularity vs trigger (as of May 2026)")
    set_tufte_labels(ax, "gate granularity", "gate trigger")
    _OUT_DIR.mkdir(parents=True, exist_ok=True)
    for p in save_figure(fig, _OUT_DIR / "design-space", formats=("png",)):
        print(f"  wrote {p.relative_to(_REPO_ROOT)}")
    plt.close(fig)


def main() -> None:
    import numpy as np

    print("Chapter 14 — hybrid_block.py")
    print("=" * 64)

    rng = np.random.default_rng(0)
    length, d = 96, 8
    x = jnp.asarray(rng.standard_normal((length, d)))
    gates_vec = jnp.asarray(rng.uniform(0.6, 0.95, size=d))

    # §14.3 windowed band-mask == per-position loop oracle.
    w = 16
    y_band = sliding_window_attention(x, w)
    y_loop = sliding_window_attention_naive(x, w)
    print(f"  band mask == loop oracle (w={w}):    max diff = "
          f"{float(jnp.max(jnp.abs(y_band - y_loop))):.2e}")

    # §14.3 window >= L recovers full causal attention exactly.
    y_full = full_causal_attention(x)
    y_wfull = sliding_window_attention(x, length)
    print(f"  w >= L == full causal attention:     max diff = "
          f"{float(jnp.max(jnp.abs(y_wfull - y_full))):.2e}")

    # §14.3 EMA closed form under constant input + constant gate.
    g = 0.9
    xbar = jnp.asarray(rng.standard_normal(d))
    const = jnp.broadcast_to(xbar, (length, d))
    h = gated_decay_ssm(const, jnp.full(d, g))
    t = jnp.arange(1, length + 1)[:, None]
    analytic = (1.0 - g**t) * xbar
    print(f"  EMA closed form (1-g^t)x̄ (g={g}):    max diff = "
          f"{float(jnp.max(jnp.abs(h - analytic))):.2e}")

    # §14.4 parallel-gated exact reductions.
    y_g1 = parallel_gated_hybrid(x, 1.0, gates_vec, w)
    y_g0 = parallel_gated_hybrid(x, 0.0, gates_vec, w)
    d_attn = float(jnp.max(jnp.abs(y_g1 - sliding_window_attention(x, w))))
    d_ssm = float(jnp.max(jnp.abs(y_g0 - gated_decay_ssm(x, gates_vec))))
    print(f"  parallel gate g=1 -> attention:      max diff = {d_attn:.2e}")
    print(f"  parallel gate g=0 -> SSM:            max diff = {d_ssm:.2e}")

    # §14.3 interleave schedule at the Kimi ratio.
    sched = interleave_schedule(8, 3)
    print(f"  interleave_schedule(8, 3) = {sched}")
    print(f"    -> attention blocks: {sched.count('attn')} / 8")

    # §14.3 cost accounting at a production-like config, audited vs buffers.
    n_blocks, dd, ww, ctx = 24, 1024, 1024, 65536
    for r in (3, 7):
        schedule = interleave_schedule(n_blocks, r)
        costs = decode_state_floats(schedule, ww, dd)
        buffers = decode_buffers(schedule, ww, dd)
        measured = sum(b.size for b in buffers)
        full = full_attention_decode_floats(n_blocks, ctx, dd)
        print(f"  ratio {r}:1 (n={n_blocks}, d={dd}, w={ww}): total = {costs['total']:,} floats"
              f" (buffers: {measured:,}); full-attn cache @L={ctx:,}: {full:,}"
              f" -> ratio {full / costs['total']:.1f}x")

    print("  figures:")
    _fig_cost_frontier(d=1024, window=1024, context_len=65536)
    _fig_design_space()


if __name__ == "__main__":
    main()
