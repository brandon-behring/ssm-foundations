r"""Chapter 13 §13.4 — xLSTM's matrix memory and the exponential-gate stabilizer.

xLSTM's mLSTM cell carries a **matrix** memory $C_t \in \mathbb{R}^{d_v \times d_k}$
and a normalizer vector $n_t \in \mathbb{R}^{d_k}$, written with *scalar* gates that
xLSTM moves **inside an exponential**:

.. math::

    C_t = f_t\,C_{t-1} + i_t\,v_t k_t^\top, \qquad
    n_t = f_t\,n_{t-1} + i_t\,k_t, \qquad
    h_t = \frac{C_t q_t}{\max(|n_t^\top q_t|,\ 1)},

with forget gate $f_t = \sigma(\tilde f_t) \in (0, 1]$ (so $\log f_t \le 0$) and
**input gate $i_t = \exp(\tilde i_t)$ — unbounded.** That exponential is the whole
point (it lets a single token's write dominate the running memory) and the whole
problem: $\exp(\tilde i_t)$ overflows float64 once $\tilde i_t \gtrsim 709$, and the
naive recurrence then produces ``inf``/``nan``.

The fix is a **stabilizer state**, a running max in the log domain:

.. math::

    m_t = \max(\log f_t + m_{t-1},\ \log i_t), \quad m_0 = -\infty,

with which both gates are rescaled into $(0, 1]$ and the memory is carried in
**scaled** coordinates $C_t \exp(-m_t)$, $n_t \exp(-m_t)$:

.. math::

    f'_t = \exp(\log f_t + m_{t-1} - m_t) \le 1, \qquad
    i'_t = \exp(\log i_t - m_t) \le 1.

**P2 (the stabilizer is exact, not approximate).** The rescaling is a pure change
of variables — factor $\exp(m_t)$ out of both $C_t$ and $n_t$ — so the readout
ratio is invariant. Wherever the naive recurrence does *not* overflow, the
stabilized readout equals it to machine precision:

.. math::

    \frac{\bar C_t q_t}{\max(|\bar n_t^\top q_t|, 1)}
    = \frac{C_t q_t}{\max(|n_t^\top q_t|,\ \exp(-m_t))}.

The $\max(\cdot, 1)$ floor becomes $\max(\cdot, \exp(-m_t))$ under the rescaling —
that, not an approximation, is the only change. The stabilizer buys numerical
range, costs nothing in the answer. (sLSTM, xLSTM's scalar-memory cell, carries the
same exponential gates and the same $m_t$ stabilizer with a memory-mixing recurrence;
the matrix-memory mLSTM here is the chapter's subject.)

This is the book's stability thread (Chapters 2, 5-6, 12) reappearing as a
*numerical* stability question living inside the architecture's state, not in the
choice of integrator.

Idiomatic-JAX note (NumPy->JAX teaching point)
----------------------------------------------
The stabilized recurrence is the idiomatic ``lax.scan`` (carry $= (C, n, m)$ with
$m_0 = -\infty$, exactly the online-softmax running-max trick); the naive recurrence
is the deliberately-overflow-prone Python loop, kept as the oracle whose ``< 1e-12``
agreement in the safe regime certifies P2.

Port credit
-----------
Greenfield: the predecessor week14 module is a 4-line stub, so the mLSTM recurrence
and its stabilizer are authored from the xLSTM paper (arXiv:2405.04517 §4, eqs. for
the mLSTM cell and the stabilized gates).

Usage
-----
::

    PYTHONPATH=. python companions/ch13/jax/xlstm.py
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import jax

# Enable float64 before any jnp array is created (matches Chapters 4, 7-12).
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402

if TYPE_CHECKING:
    from matplotlib.figure import Figure

__all__ = [
    "log_sigmoid",
    "mlstm_naive",
    "mlstm_stabilized",
    "make_gate_stream",
    "peak_state_magnitude",
    "readout_max_abs_diff",
    "naive_finite_fraction",
    "single_pair_recovery",
]

# A fixed (RNG-free) reference pair, shared verbatim with the Julia companion so
# ``single_pair_recovery`` is a cross-language numeric anchor.
_REF_K_RAW = (1.0, 2.0, -1.0, 0.5)
_REF_V = (0.3, -0.7, 1.1)

_REPO_ROOT = Path(__file__).resolve().parents[3]
_OUT_DIR = _REPO_ROOT / "public" / "figures" / "ch13"


def log_sigmoid(x: jnp.ndarray) -> jnp.ndarray:
    r"""Numerically stable $\log \sigma(x) = -\mathrm{softplus}(-x) \le 0$ (forget log-gate)."""
    return -jnp.logaddexp(0.0, -x)


def _check_gate_stream(
    q: jnp.ndarray, k: jnp.ndarray, v: jnp.ndarray, log_f: jnp.ndarray, log_i: jnp.ndarray
) -> None:
    length, d_k = q.shape
    if q.ndim != 2 or k.shape != q.shape:
        raise ValueError(f"q and k must share shape (L, d_k); got {q.shape}, {k.shape}")
    if v.ndim != 2 or v.shape[0] != length:
        raise ValueError(f"v must have shape (L, d_v); got {v.shape}")
    if log_f.shape != (length,) or log_i.shape != (length,):
        raise ValueError(
            f"log_f and log_i must have shape (L,) = ({length},); got {log_f.shape}, {log_i.shape}"
        )


# ---------------------------------------------------------------------------
# §13.4 — the two recurrences: naive (overflows) and stabilized (lax.scan)
# ---------------------------------------------------------------------------


def mlstm_naive(
    q: jnp.ndarray, k: jnp.ndarray, v: jnp.ndarray, log_f: jnp.ndarray, log_i: jnp.ndarray
) -> jnp.ndarray:
    r"""mLSTM readouts with raw exponential gates — the overflow-prone Python-loop oracle.

    $\bar f_t = \exp(\log f_t)$, $\bar i_t = \exp(\log i_t)$; the matrix memory and
    normalizer accumulate without rescaling, so $\log i_t \gtrsim 709$ overflows
    float64 and the readout becomes ``inf``/``nan``.

    Parameters
    ----------
    q, k : jnp.ndarray, shape (L, d_k)
    v : jnp.ndarray, shape (L, d_v)
    log_f : jnp.ndarray, shape (L,)
        Log forget gate $\le 0$.
    log_i : jnp.ndarray, shape (L,)
        Log input gate (unbounded).

    Returns
    -------
    jnp.ndarray, shape (L, d_v)
        Readouts $h_t$ (may contain non-finite entries by design).
    """
    _check_gate_stream(q, k, v, log_f, log_i)
    length, d_k = q.shape
    d_v = v.shape[1]
    cell = jnp.zeros((d_v, d_k))
    norm = jnp.zeros((d_k,))
    outputs = []
    for t in range(length):
        f = jnp.exp(log_f[t])
        i = jnp.exp(log_i[t])
        cell = f * cell + i * jnp.outer(v[t], k[t])
        norm = f * norm + i * k[t]
        denom = jnp.maximum(jnp.abs(norm @ q[t]), 1.0)
        outputs.append(cell @ q[t] / denom)
    return jnp.stack(outputs)


def mlstm_stabilized(
    q: jnp.ndarray, k: jnp.ndarray, v: jnp.ndarray, log_f: jnp.ndarray, log_i: jnp.ndarray
) -> tuple[jnp.ndarray, jnp.ndarray]:
    r"""mLSTM readouts via the log-domain max-state stabilizer (idiomatic ``lax.scan``).

    Carry $(C, n, m)$ with $m_0 = -\infty$; both rescaled gates lie in $(0, 1]$, so
    nothing overflows. The readout floor $\max(\cdot, 1)$ of the naive form becomes
    $\max(\cdot, \exp(-m_t))$ — the only change (P2).

    Parameters as :func:`mlstm_naive`.

    Returns
    -------
    outputs : jnp.ndarray, shape (L, d_v)
        Readouts (always finite for finite inputs).
    m_trajectory : jnp.ndarray, shape (L,)
        The stabilizer state $m_t$ at each step.
    """
    _check_gate_stream(q, k, v, log_f, log_i)
    length, d_k = q.shape
    d_v = v.shape[1]

    def step(carry, inp):
        cell, norm, m = carry
        q_t, k_t, v_t, lf_t, li_t = inp
        m_new = jnp.maximum(lf_t + m, li_t)
        f_p = jnp.exp(lf_t + m - m_new)  # in (0, 1]
        i_p = jnp.exp(li_t - m_new)  # in (0, 1]
        cell = f_p * cell + i_p * jnp.outer(v_t, k_t)
        norm = f_p * norm + i_p * k_t
        denom = jnp.maximum(jnp.abs(norm @ q_t), jnp.exp(-m_new))
        return (cell, norm, m_new), (cell @ q_t / denom, m_new)

    init = (jnp.zeros((d_v, d_k)), jnp.zeros((d_k,)), jnp.asarray(-jnp.inf))
    _, (outputs, m_traj) = jax.lax.scan(step, init, (q, k, v, log_f, log_i))
    return outputs, m_traj


# ---------------------------------------------------------------------------
# §13.4 — overflow demonstration (the figure data + caption pins)
# ---------------------------------------------------------------------------


def make_gate_stream(
    length: int, d_k: int, d_v: int, peak_log_i: float, seed: int = 0
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    r"""A test stream whose largest log input-gate is exactly ``peak_log_i``.

    Forget gates are $\log\sigma(\tilde f)$ with mild $\tilde f$ (memory persists);
    input gates are moderate except the middle timestep, set to ``peak_log_i`` to
    drive the naive recurrence's overflow.
    """
    import numpy as np

    rng = np.random.default_rng(seed)
    q = jnp.asarray(rng.standard_normal((length, d_k)))
    k_raw = rng.standard_normal((length, d_k))
    k = jnp.asarray(k_raw / np.linalg.norm(k_raw, axis=1, keepdims=True))
    v = jnp.asarray(rng.standard_normal((length, d_v)))
    log_f = log_sigmoid(jnp.asarray(rng.uniform(0.0, 2.0, size=length)))  # in (-0.69, -0.13)
    log_i = jnp.asarray(rng.uniform(-1.0, 1.0, size=length))
    log_i = log_i.at[length // 2].set(peak_log_i)
    return q, k, v, log_f, log_i


def peak_state_magnitude(peak_log_i: float, stabilized: bool, length: int = 16, seed: int = 0) -> float:
    r"""Peak memory magnitude $\max_t \max_{ij}|C_{t,ij}|$ over a stream with the given peak log-gate.

    Uses the max-abs entry, not the Frobenius norm: the norm's sum-of-squares
    would itself overflow at entry magnitude $\sim 10^{154}$, conflating a
    *norm-computation* overflow with the state genuinely overflowing. Max-abs
    overflows exactly when an entry exceeds the float64 ceiling, i.e. when the
    exponential input gate $\exp(\tilde i) \gtrsim 1.8\times10^{308}$
    ($\tilde i \gtrsim 709$). Naive blows up like $\exp(\text{peak\_log\_i})$;
    stabilized stays $O(1)$.
    """
    q, k, v, log_f, log_i = make_gate_stream(length, 4, 3, peak_log_i, seed)
    d_v, d_k = v.shape[1], q.shape[1]
    if stabilized:
        # Recompute the scaled-cell trajectory (the readout fn discards states).
        cell = jnp.zeros((d_v, d_k))
        norm = jnp.zeros((d_k,))
        m = jnp.asarray(-jnp.inf)
        peak = 0.0
        for t in range(length):
            m_new = jnp.maximum(log_f[t] + m, log_i[t])
            f_p = jnp.exp(log_f[t] + m - m_new)
            i_p = jnp.exp(log_i[t] - m_new)
            cell = f_p * cell + i_p * jnp.outer(v[t], k[t])
            norm = f_p * norm + i_p * k[t]
            m = m_new
            peak = max(peak, float(jnp.max(jnp.abs(cell))))
        return peak
    cell = jnp.zeros((d_v, d_k))
    peak = 0.0
    for t in range(length):
        cell = jnp.exp(log_f[t]) * cell + jnp.exp(log_i[t]) * jnp.outer(v[t], k[t])
        peak = max(peak, float(jnp.max(jnp.abs(cell))))
    return peak


def readout_max_abs_diff(peak_log_i: float, length: int = 16, seed: int = 0) -> float:
    r"""$\max|h^{\text{naive}} - h^{\text{stab}}|$ over the stream (``nan`` if naive overflowed)."""
    q, k, v, log_f, log_i = make_gate_stream(length, 4, 3, peak_log_i, seed)
    h_naive = mlstm_naive(q, k, v, log_f, log_i)
    h_stab, _ = mlstm_stabilized(q, k, v, log_f, log_i)
    return float(jnp.max(jnp.abs(h_naive - h_stab)))


def naive_finite_fraction(peak_log_i: float, length: int = 16, seed: int = 0) -> float:
    r"""Fraction of finite entries in the naive readout (1.0 = no overflow, 0-ish past threshold)."""
    q, k, v, log_f, log_i = make_gate_stream(length, 4, 3, peak_log_i, seed)
    h_naive = mlstm_naive(q, k, v, log_f, log_i)
    return float(jnp.mean(jnp.isfinite(h_naive).astype(jnp.float64)))


def single_pair_recovery(log_i_value: float) -> tuple[float, jnp.ndarray]:
    r"""Store one pair on a unit key with input log-gate ``log_i_value``; read back at $q = k$.

    A single write makes $C_1 = i'_1\,v k^\top$, $n_1 = i'_1\,k$ with $i'_1 = 1$ after
    stabilization, so the read at $q = k$ (unit key) is $C_1 k / \max(|n_1^\top k|,
    e^{-m_1}) = v\,\|k\|^2 / \max(\|k\|^2, e^{-m_1}) = v$ — **exactly the stored value,
    for any $\log i$**, including $\log i = 800$ where the naive recurrence overflows.
    The normalizer state cancels the gate; that is what the exponential input gate is
    *for*. Uses the fixed reference pair shared with the Julia companion (a
    cross-language anchor).

    Returns
    -------
    max_abs_error, readout : float, jnp.ndarray
        $\max_j|h_{1,j} - v_j|$ (machine zero) and the readout $h_1$.
    """
    k = jnp.asarray(_REF_K_RAW)
    k = k / jnp.linalg.norm(k)
    v = jnp.asarray(_REF_V)
    q = k.reshape(1, -1)
    h, _ = mlstm_stabilized(q, k.reshape(1, -1), v.reshape(1, -1), jnp.zeros(1), jnp.asarray([log_i_value]))
    return float(jnp.max(jnp.abs(h[0] - v))), h[0]


# ---------------------------------------------------------------------------
# Figure: naive overflow vs the stabilized recurrence
# ---------------------------------------------------------------------------

_SCALE_GRID = (1.0, 100.0, 300.0, 500.0, 650.0, 700.0, 705.0, 710.0, 730.0, 760.0)
_SAFE_SCALE = 700.0  # naive entries still finite (~e304) — the P2 agreement point
_OVERFLOW_SCALE = 760.0  # exp(760) overflows float64: naive readout has nan entries
_FLOAT64_MAX = 1.7976931348623157e308


def make_overflow_figure() -> "Figure":
    """Left: peak memory magnitude vs input-gate scale — naive hits the float64
    ceiling and overflows; stabilized stays $O(1)$. Right: readout agreement —
    identical where the naive survives (P2), ``nan`` past the overflow cliff."""
    import numpy as np

    from companions._shared.plot_utils import (
        SSM_COLORS,
        apply_style,
        create_tufte_figure,
        set_tufte_labels,
        set_tufte_title,
    )

    apply_style()
    fig, (ax1, ax2) = create_tufte_figure(ncols=2, figsize=(11.0, 4.2))

    scales = np.asarray(_SCALE_GRID)
    peak_naive = np.array([peak_state_magnitude(float(s), stabilized=False) for s in scales])
    peak_stab = np.array([peak_state_magnitude(float(s), stabilized=True) for s in scales])
    diffs = np.array([readout_max_abs_diff(float(s)) for s in scales])

    # Panel A — log10 of the peak memory entry on a LINEAR axis (the exponent grows
    # linearly with the gate scale). Plotting the magnitude itself on a log axis makes
    # matplotlib's inverse transform base**value overflow near the 1e308 ceiling.
    finite = np.isfinite(peak_naive)
    log_naive = np.full_like(peak_naive, np.nan)
    log_naive[finite] = np.log10(peak_naive[finite])
    log_ceiling = float(np.log10(_FLOAT64_MAX))  # ~308.25
    ax1.plot(scales[finite], log_naive[finite], "o-", color=SSM_COLORS["highlight"],
             label=r"naive (raw $\exp$ gates)")
    ax1.plot(scales, np.log10(peak_stab), "s-", color=SSM_COLORS["accent"],
             label="stabilized (log-domain)")
    ax1.axhline(log_ceiling, color=SSM_COLORS["alert"], lw=0.9, ls="--")
    ax1.annotate(r"float64 ceiling ($10^{308}$)", xy=(scales[0], log_ceiling),
                 xytext=(scales[0], 235), fontsize=8, color=SSM_COLORS["alert"])
    first_overflow = scales[~finite][0] if (~finite).any() else None
    if first_overflow is not None:
        ax1.axvline(first_overflow, color=SSM_COLORS["alert"], lw=1.0, ls=":")
        ax1.annotate(f"naive overflows\nat $\\tilde i \\approx {first_overflow:.0f}$",
                     xy=(first_overflow, 150), fontsize=8, color=SSM_COLORS["alert"], ha="center")
    set_tufte_title(ax1, "Exponential input gate: the memory explodes")
    set_tufte_labels(ax1, xlabel=r"peak log input-gate $\tilde i$",
                     ylabel=r"$\log_{10}\,\max_{ij}|C_{ij}|$")
    ax1.legend(loc="center right", fontsize=8, frameon=False)

    # Panel B — readout agreement (P2) and the nan cliff.
    finite_d = np.isfinite(diffs)
    floor = np.maximum(diffs[finite_d], 1e-18)
    ax2.semilogy(scales[finite_d], floor, "o-", color=SSM_COLORS["accent"],
                 label=r"$\max|h^{\rm naive} - h^{\rm stab}|$")
    ax2.axhline(1e-12, color=SSM_COLORS["baseline"], lw=0.8, ls="--", label=r"$10^{-12}$ (P2 pin)")
    if (~finite_d).any():
        cliff = scales[~finite_d][0]
        ax2.axvspan(cliff - 5, scales[-1] + 5, color=SSM_COLORS["alert"], alpha=0.10)
        ax2.annotate("naive = nan\n(stabilized still exact)", xy=(cliff, 1e-10),
                     fontsize=8, color=SSM_COLORS["alert"], ha="center")
    ax2.set_ylim(1e-18, 1e-6)
    set_tufte_title(ax2, "Same answer where naive survives; stabilized always")
    set_tufte_labels(ax2, xlabel=r"peak log input-gate $\tilde i$",
                     ylabel=r"naive vs stabilized readout gap")
    ax2.legend(loc="lower left", fontsize=8, frameon=False)

    fig.tight_layout()
    return fig


def main() -> None:
    import matplotlib.pyplot as plt

    from companions._shared.plot_utils import save_figure

    print("Chapter 13 — xlstm.py")
    print("=" * 64)

    # §13.4 P2: stabilized == naive in the safe regime (moderate gates), to machine precision.
    q, k, v, log_f, log_i = make_gate_stream(24, 6, 5, peak_log_i=2.0, seed=0)
    h_naive = mlstm_naive(q, k, v, log_f, log_i)
    h_stab, m_traj = mlstm_stabilized(q, k, v, log_f, log_i)
    print(f"  P2 safe regime: max |h_naive - h_stab| = "
          f"{float(jnp.max(jnp.abs(h_naive - h_stab))):.2e}  (naive all finite = "
          f"{bool(jnp.all(jnp.isfinite(h_naive)))})")
    # The enabling invariant: both rescaled gates lie in (0, 1] by construction of
    # m_t as a max (m_t need NOT be monotone — it falls when forgetting dominates).
    m_prev = jnp.concatenate([jnp.array([-jnp.inf]), m_traj[:-1]])
    i_ok = bool(jnp.all(log_i <= m_traj + 1e-12))
    f_ok = bool(jnp.all(log_f + m_prev <= m_traj + 1e-12))
    print(f"  rescaled gates in (0,1]: i' <= 1 holds = {i_ok}, f' <= 1 holds = {f_ok}")

    # §13.4 the overflow cliff: naive dies, stabilized stays finite + exact.
    print(f"  safe scale  i_tilde={_SAFE_SCALE:.0f}: peak |C| naive = "
          f"{peak_state_magnitude(_SAFE_SCALE, False):.3e}, stabilized = "
          f"{peak_state_magnitude(_SAFE_SCALE, True):.3f}; readout gap = "
          f"{readout_max_abs_diff(_SAFE_SCALE):.2e}")
    print(f"  overflow scale i_tilde={_OVERFLOW_SCALE:.0f}: naive finite fraction = "
          f"{naive_finite_fraction(_OVERFLOW_SCALE):.2f}, stabilized peak |C| = "
          f"{peak_state_magnitude(_OVERFLOW_SCALE, True):.3f} (finite)")

    # §13.4 cross-language anchor: the stabilizer recovers the stored value exactly,
    # even at log_i = 800 (naive overflows). Same fixed pair as the Julia companion.
    err0, read0 = single_pair_recovery(0.0)
    err_big, _ = single_pair_recovery(800.0)
    print(f"  single-pair recovery: err(log_i=0) = {err0:.2e}, err(log_i=800) = {err_big:.2e}"
          f" (naive overflows); readout = {[round(float(x), 6) for x in read0]}")

    _OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig = make_overflow_figure()
    for p in save_figure(fig, _OUT_DIR / "stabilizer-overflow", formats=("png",)):
        print(f"Wrote {p}")
    plt.close(fig)


if __name__ == "__main__":
    main()
