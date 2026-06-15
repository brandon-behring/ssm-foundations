r"""Chapter 13 §13.2-13.3 — the generalized linear transition (RWKV-7's lineage).

Chapter 12's delta-rule family iterates $S_t = S_{t-1} A_t + \text{write}_t$ with a
*very specific* transition: $A_t = \gamma_t (I - \beta_t k_t k_t^\top)$ — a scalar
multiple of the identity, minus a rank-one term **locked to the write key** $k_t$.
RWKV-7 ("Goose", Peng et al.) keeps the diagonal-plus-rank-one *shape* but frees
both pieces:

.. math::

    A_t = \mathrm{Diag}(w_t) - c_t\, a_t a_t^\top, \qquad \|a_t\| = 1,\ c_t \ge 0,

a full **diagonal decay** $w_t$ (not a single scalar $\gamma_t$) minus a rank-one
removal in a **learned direction** $a_t$ that need not be the key. This is the
"generalized delta rule": the erase direction is decoupled from what gets written
and from what gets queried.

Two facts this module pins, both companion-measured and quoted in the chapter:

* **P1 (spectrum).** $A_t$ is *symmetric* ($\mathrm{Diag}(w_t)$ and $a_t a_t^\top$
  both are), so its eigenvalues are real — `transition_spectrum` reads them with
  ``eigvalsh``. They are the roots of the secular equation
  $1 - c\sum_i a_i^2/(w_i - \lambda) = 0$ (matrix-determinant lemma) and *interlace*
  the sorted diagonal: a rank-one downdate pushes every eigenvalue down, none past
  the next diagonal entry. In the **scalar-diagonal** case $w \equiv w_0\mathbf 1$
  exactly one eigenvalue moves: spectrum $= \{w_0\ (\times d{-}1),\ w_0 - c\}$,
  recovering Chapter 12's $k$-direction radius $|1 - \beta\|k\|^2|$ at $w_0 = 1$
  (``stability.py``).

* **P3 (reduction).** With $w_t = \gamma_t\mathbf 1$, $a_t = k_t/\|k_t\|$,
  $c_t = \gamma_t\beta_t\|k_t\|^2$, and write $u_t b_t^\top = \beta_t v_t k_t^\top$,
  the generalized recurrence reproduces ``gated_delta_recurrent`` (Chapter 12's
  Gated DeltaNet) to machine precision. RWKV-7's transition contains the whole
  Chapter 12 lineage as the scalar-diagonal, removal-locked-to-key special case.

The third demo, ``decoupled_eviction``, makes the "learned direction" concrete:
RWKV-7 can *aim* its rank-one removal at an old stored key $k_A$ while writing
fresh pairs orthogonal to it, decaying $\|S k_A\|$ as exactly $(1-c)^T\|v_A\|$.
Chapter 12's delta rule, whose removal direction *is* the current write key,
leaves $k_A$ untouched under those same orthogonal writes (the
``stale_retrieval_after_orthogonal_writes`` baseline) — eviction is impossible
without overwriting. Decoupling the erase direction is what buys it.

Idiomatic-JAX note (NumPy->JAX teaching point)
----------------------------------------------
Same ``lax.scan``-vs-Python-loop oracle pairing as the Chapter 12 siblings: the
scan carry holds the rank-one form, the naive loop materialises the full
$(d_k, d_k)$ transition $\mathrm{Diag}(w_t) - c_t a_t a_t^\top$ each step, so the
``< 1e-12`` agreement certifies the algebra, not the plumbing.

Port credit
-----------
Greenfield: the predecessor week15 module is a 4-line stub, so the generalized
transition is authored from the RWKV-7 paper's diagonal-plus-rank-one form
(arXiv:2503.14456). The recurrent/naive structure and the spectral-radius guard
extend ``companions/ch12/jax/{gated_delta,stability}.py``.

Usage
-----
::

    PYTHONPATH=. python companions/ch13/jax/generalized_transition.py
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import jax

# Enable float64 before any jnp array is created (matches Chapters 4, 7-12).
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402

from companions.ch12.jax.gated_delta import gated_delta_recurrent  # noqa: E402

if TYPE_CHECKING:
    from matplotlib.figure import Figure

__all__ = [
    "dplr_transition",
    "transition_spectrum",
    "scalar_diagonal_spectrum",
    "secular_function",
    "generalized_delta_step",
    "generalized_delta_recurrent",
    "generalized_delta_naive",
    "gated_delta_reduction",
    "decoupled_eviction",
]

_REPO_ROOT = Path(__file__).resolve().parents[3]
_OUT_DIR = _REPO_ROOT / "public" / "figures" / "ch13"


# ---------------------------------------------------------------------------
# §13.2 — the diagonal-plus-rank-one transition and its (real) spectrum (P1)
# ---------------------------------------------------------------------------


def _check_dplr(w: jnp.ndarray, a: jnp.ndarray, c: jnp.ndarray | float) -> None:
    if w.ndim != 1 or a.ndim != 1:
        raise ValueError(f"w and a must be 1D (d,); got {w.shape}, {a.shape}")
    if w.shape != a.shape:
        raise ValueError(f"w and a must share shape (d,); got {w.shape} and {a.shape}")
    if float(c) < 0.0:
        raise ValueError(f"removal coefficient c must be >= 0; got {float(c)}")


def dplr_transition(w: jnp.ndarray, a: jnp.ndarray, c: jnp.ndarray | float) -> jnp.ndarray:
    r"""The symmetric transition $A = \mathrm{Diag}(w) - c\,a a^\top$.

    Materialised on purpose (the recurrences below never form it). $a$ should be
    a unit vector for the spectrum statements to read cleanly; non-unit $a$ is
    accepted and folds $\|a\|^2$ into the rank-one magnitude.

    Parameters
    ----------
    w : jnp.ndarray, shape (d,)
        Diagonal decay vector.
    a : jnp.ndarray, shape (d,)
        Removal direction (unit norm in the chapter's statements).
    c : scalar
        Removal coefficient $\ge 0$.

    Returns
    -------
    jnp.ndarray, shape (d, d)
        Symmetric.
    """
    _check_dplr(w, a, c)
    return jnp.diag(w) - c * jnp.outer(a, a)


def transition_spectrum(w: jnp.ndarray, a: jnp.ndarray, c: jnp.ndarray | float) -> jnp.ndarray:
    r"""Real eigenvalues of $\mathrm{Diag}(w) - c\,a a^\top$, ascending, via ``eigvalsh``.

    The matrix is symmetric, so the spectrum is real; the rank-one downdate
    pushes every eigenvalue down and they interlace the sorted diagonal.

    Parameters
    ----------
    w : jnp.ndarray, shape (d,)
    a : jnp.ndarray, shape (d,)
    c : scalar

    Returns
    -------
    jnp.ndarray, shape (d,)
        Ascending eigenvalues.
    """
    return jnp.linalg.eigvalsh(dplr_transition(w, a, c))


def scalar_diagonal_spectrum(w0: float, c: float, d: int) -> jnp.ndarray:
    r"""Closed-form spectrum of $w_0 I - c\,a a^\top$ ($a$ unit), ascending.

    Exactly one eigenvalue moves: $w_0 - c$ along $a$, and $w_0$ with
    multiplicity $d-1$ on its orthogonal complement. At $w_0 = 1$ the moving
    eigenvalue is $1 - c$ — Chapter 12's $k$-direction value with
    $c = \beta\|k\|^2$ (``deltanet_spectral_radius``).

    Parameters
    ----------
    w0 : float
        The scalar diagonal value.
    c : float
        Removal coefficient.
    d : int
        Dimension ($\ge 1$).

    Returns
    -------
    jnp.ndarray, shape (d,)
        Ascending: $[w_0 - c, w_0, \dots, w_0]$ when $c \ge 0$.
    """
    if d < 1:
        raise ValueError(f"d must be >= 1; got {d}")
    eig = jnp.concatenate([jnp.array([w0 - c]), jnp.full((d - 1,), w0)])
    return jnp.sort(eig)


def secular_function(
    lmbda: jnp.ndarray | float, w: jnp.ndarray, a: jnp.ndarray, c: jnp.ndarray | float
) -> jnp.ndarray:
    r"""The secular function $f(\lambda) = 1 - c\sum_i a_i^2/(w_i - \lambda)$.

    By the matrix-determinant lemma, $\det(\mathrm{Diag}(w) - c a a^\top - \lambda I)
    = \prod_i(w_i - \lambda)\cdot f(\lambda)$, so every eigenvalue $\lambda \notin
    \{w_i\}$ is a root of $f$. Evaluated at the computed eigenvalues it is the
    derivation-drift guard: ``eigvalsh`` roots must zero it.

    Parameters
    ----------
    lmbda : scalar
        Evaluation point (must differ from every $w_i$).
    w, a : jnp.ndarray, shape (d,)
    c : scalar

    Returns
    -------
    jnp.ndarray, scalar
    """
    return 1.0 - c * jnp.sum(a**2 / (w - lmbda))


# ---------------------------------------------------------------------------
# §13.3 — the generalized delta rule over a sequence (scan vs materialised oracle)
# ---------------------------------------------------------------------------


def _check_gen_stream(
    q: jnp.ndarray,
    w: jnp.ndarray,
    a: jnp.ndarray,
    c: jnp.ndarray,
    u: jnp.ndarray,
    b: jnp.ndarray,
) -> None:
    length, d_k = q.shape
    if q.ndim != 2:
        raise ValueError(f"q must be 2D (L, d_k); got {q.shape}")
    for name, arr in (("w", w), ("a", a), ("b", b)):
        if arr.shape != (length, d_k):
            raise ValueError(f"{name} must have shape (L, d_k) = {(length, d_k)}; got {arr.shape}")
    if u.ndim != 2 or u.shape[0] != length:
        raise ValueError(f"u must have shape (L, d_v); got {u.shape}")
    if c.shape != (length,):
        raise ValueError(f"c must have shape (L,) = ({length},); got {c.shape}")


def generalized_delta_step(
    state: jnp.ndarray,
    w: jnp.ndarray,
    a: jnp.ndarray,
    c: jnp.ndarray | float,
    u: jnp.ndarray,
    b: jnp.ndarray,
) -> jnp.ndarray:
    r"""One generalized update $S \leftarrow S(\mathrm{Diag}(w) - c\,a a^\top) + u b^\top$.

    Rank-one form, $O(d_v d_k)$: the diagonal scales the state's columns, the
    rank-one term removes the projection on $a$, and the write lands ungated.

    Parameters
    ----------
    state : jnp.ndarray, shape (d_v, d_k)
    w, a, b : jnp.ndarray, shape (d_k,)
        Diagonal decay, unit removal direction, write key.
    c : scalar
        Removal coefficient.
    u : jnp.ndarray, shape (d_v,)
        Write value.

    Returns
    -------
    jnp.ndarray, shape (d_v, d_k)
    """
    if state.shape != (u.shape[0], w.shape[0]):
        raise ValueError(
            f"state must have shape (d_v, d_k) = ({u.shape[0]}, {w.shape[0]}); got {state.shape}"
        )
    # S A = S Diag(w) - c (S a) a^T: the diagonal scales the state's columns and
    # the rank-one term removes the ORIGINAL state's projection on a (both act on
    # S_{t-1}, in parallel — not the diagonal-then-rank-one composition).
    return state * w - c * jnp.outer(state @ a, a) + jnp.outer(u, b)


def generalized_delta_recurrent(
    q: jnp.ndarray,
    w: jnp.ndarray,
    a: jnp.ndarray,
    c: jnp.ndarray,
    u: jnp.ndarray,
    b: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    r"""Sequential generalized delta rule via ``lax.scan``; post-update read $o_t = S_t q_t$.

    Parameters
    ----------
    q, w, a, b : jnp.ndarray, shape (L, d_k)
    c : jnp.ndarray, shape (L,)
    u : jnp.ndarray, shape (L, d_v)

    Returns
    -------
    outputs : jnp.ndarray, shape (L, d_v)
    final_state : jnp.ndarray, shape (d_v, d_k)
    """
    _check_gen_stream(q, w, a, c, u, b)
    d_k, d_v = q.shape[1], u.shape[1]

    def step(state: jnp.ndarray, inp) -> tuple[jnp.ndarray, jnp.ndarray]:
        q_t, w_t, a_t, c_t, u_t, b_t = inp
        new_state = state * w_t - c_t * jnp.outer(state @ a_t, a_t) + jnp.outer(u_t, b_t)
        return new_state, new_state @ q_t

    init = jnp.zeros((d_v, d_k), dtype=u.dtype)
    final_state, outputs = jax.lax.scan(step, init, (q, w, a, c, u, b))
    return outputs, final_state


def generalized_delta_naive(
    q: jnp.ndarray,
    w: jnp.ndarray,
    a: jnp.ndarray,
    c: jnp.ndarray,
    u: jnp.ndarray,
    b: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    r"""Python-loop oracle: materialises $\mathrm{Diag}(w_t) - c_t a_t a_t^\top$ each step."""
    _check_gen_stream(q, w, a, c, u, b)
    length, d_k = q.shape
    d_v = u.shape[1]
    state = jnp.zeros((d_v, d_k), dtype=u.dtype)
    outputs = []
    for t in range(length):
        transition = jnp.diag(w[t]) - c[t] * jnp.outer(a[t], a[t])  # (d_k, d_k), materialised
        state = state @ transition + jnp.outer(u[t], b[t])
        outputs.append(state @ q[t])
    return jnp.stack(outputs), state


# ---------------------------------------------------------------------------
# §13.3 — P3: the generalized rule contains Chapter 12's gated DeltaNet
# ---------------------------------------------------------------------------


def gated_delta_reduction(
    q: jnp.ndarray,
    k: jnp.ndarray,
    v: jnp.ndarray,
    betas: jnp.ndarray,
    gammas: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    r"""Run the generalized rule with the parameters that reproduce gated DeltaNet.

    Maps Chapter 12's $(\gamma_t, \beta_t, k_t, v_t)$ to the generalized
    $(w_t, a_t, c_t, u_t, b_t)$:

    * $w_t = \gamma_t \mathbf 1$ (scalar diagonal),
    * $a_t = k_t/\|k_t\|$ (removal direction = unit key),
    * $c_t = \gamma_t\,\beta_t\,\|k_t\|^2$ (so $c_t a_t a_t^\top = \gamma_t\beta_t k_t k_t^\top$),
    * $u_t b_t^\top = \beta_t v_t k_t^\top$.

    Returns the generalized outputs/state; equal to ``gated_delta_recurrent`` to
    machine precision (the P3 pin lives in the test + ``main``).

    Parameters and returns mirror ``gated_delta_recurrent``.
    """
    length, d_k = q.shape
    k_norm = jnp.linalg.norm(k, axis=1, keepdims=True)  # (L, 1)
    a = k / k_norm
    w = jnp.broadcast_to(gammas[:, None], (length, d_k))
    c = gammas * betas * (k_norm[:, 0] ** 2)
    u = betas[:, None] * v
    b = k
    return generalized_delta_recurrent(q, w, a, c, u, b)


# ---------------------------------------------------------------------------
# §13.3 — the learned direction made concrete: targeted eviction
# ---------------------------------------------------------------------------


def decoupled_eviction(
    n_writes: int,
    c: float,
    d_k: int = 8,
    d_v: int = 4,
    seed: int = 0,
) -> tuple[float, float, float]:
    r"""Retrieval norm $\|S k_A\|$ after ``n_writes`` writes, two removal policies.

    Stores $(k_A, v_A)$ on a unit key, then performs ``n_writes`` writes whose
    write keys $b_t$ are exactly orthogonal to $k_A$. Two transition policies:

    * **RWKV-7 (targeted):** removal direction $a_t = k_A$, coefficient $c$. The
      removal is *aimed at the old key* even though the writes are orthogonal, so
      $\|S_T k_A\| = (1-c)^T\,\|v_A\|$ — eviction without overwriting.
    * **Chapter 12 (locked):** removal direction $a_t = b_t$ (the write key), as
      the delta rule forces. Orthogonal writes never touch $k_A$, so
      $\|S_T k_A\| = \|v_A\|$ — the ``stale_retrieval_after_orthogonal_writes``
      baseline.

    Both use scalar diagonal $w = \mathbf 1$ (decay isolated to the rank-one term).

    Parameters
    ----------
    n_writes : int
        Number of orthogonal writes $T \ge 0$.
    c : float
        Removal coefficient in $[0, 1]$ for the targeted policy.
    d_k, d_v : int
    seed : int

    Returns
    -------
    targeted_measured, targeted_analytic, locked_measured : float
        The targeted retrieval norm $\|S_T k_A\|$, its closed form
        $(1-c)^T\|v_A\|$ (equal to machine precision), and the locked-policy
        baseline ($= \|v_A\|$).
    """
    import numpy as np

    if not 0.0 <= c <= 1.0:
        raise ValueError(f"c must be in [0, 1]; got {c}")
    if n_writes < 0:
        raise ValueError(f"n_writes must be >= 0; got {n_writes}")
    rng = np.random.default_rng(seed)
    k_a = np.zeros(d_k)
    k_a[0] = 1.0  # unit key along e_1; later write keys live in its complement
    v_a = rng.standard_normal(d_v)
    ones = jnp.ones(d_k)

    s0 = jnp.outer(jnp.asarray(v_a), jnp.asarray(k_a))  # exact store: S0 k_A = v_A
    s_targeted = s0
    s_locked = s0
    for _ in range(n_writes):
        raw = rng.standard_normal(d_k)
        raw[0] = 0.0  # write key orthogonal to k_A
        b = jnp.asarray(raw / np.linalg.norm(raw))
        u = jnp.asarray(rng.standard_normal(d_v))
        # Targeted: aim the removal at k_A (decoupled from the write key b).
        s_targeted = generalized_delta_step(s_targeted, ones, jnp.asarray(k_a), c, u, b)
        # Locked (Chapter 12): removal direction is the write key b itself.
        s_locked = generalized_delta_step(s_locked, ones, b, c, u, b)

    k_a_j = jnp.asarray(k_a)
    targeted_measured = float(jnp.linalg.norm(s_targeted @ k_a_j))
    targeted_analytic = float((1.0 - c) ** n_writes * np.linalg.norm(v_a))
    locked_measured = float(jnp.linalg.norm(s_locked @ k_a_j))
    return targeted_measured, targeted_analytic, locked_measured


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

_SPEC_W0 = 1.0
_SPEC_D = 6
_EVICT_C = 0.1


def make_spectrum_figure() -> "Figure":
    """Left: scalar diagonal — one eigenvalue slides to $w_0 - c$, crossing $-1$ at
    $c = 2$ (Chapter 12's boundary). Right: general diagonal — the rank-one downdate
    pushes the whole spectrum down, interlacing the diagonal entries."""
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

    # Panel A — scalar diagonal w0 = 1: eigenvalues {1 (x d-1), 1 - c}.
    cs = np.linspace(0.0, 2.8, 281)
    moving = _SPEC_W0 - cs
    ax1.plot(cs, np.full_like(cs, _SPEC_W0), color=SSM_COLORS["baseline"], lw=1.6,
             label=r"$w_0$ ($\times\,d{-}1$, persistent)")
    ax1.plot(cs, moving, color=SSM_COLORS["accent"], lw=1.8,
             label=r"$w_0 - c$ (along $a$)")
    ax1.axhline(1.0, color=SSM_COLORS["baseline"], lw=0.6, ls="--")
    ax1.axhline(-1.0, color=SSM_COLORS["alert"], lw=0.8, ls="--")
    ax1.axvline(2.0, color=SSM_COLORS["alert"], lw=1.0, ls=":")
    ax1.fill_between(cs, -1.0, np.minimum(moving, -1.0), where=moving < -1.0,
                     color=SSM_COLORS["alert"], alpha=0.12)
    ax1.annotate(r"boundary $c = 2$" + "\n" + r"($\beta\|k\|^2 = 2$, ch.12)",
                 xy=(2.0, -1.0), xytext=(1.15, -1.9),
                 arrowprops={"arrowstyle": "->", "color": SSM_COLORS["alert"], "linewidth": 0.8},
                 fontsize=8, color=SSM_COLORS["alert"])
    ax1.set_xlim(0.0, 2.8)
    ax1.set_ylim(-2.4, 1.4)
    set_tufte_title(ax1, "Scalar diagonal: one eigenvalue moves")
    set_tufte_labels(ax1, xlabel=r"removal coefficient $c$", ylabel=r"eigenvalue $\lambda$")
    ax1.legend(loc="lower left", fontsize=8, frameon=False)

    # Panel B — general diagonal: spectrum of Diag(w) - c a a^T, w spread, a random unit.
    rng = np.random.default_rng(0)
    w = np.sort(rng.uniform(0.55, 1.0, size=_SPEC_D))[::-1].copy()  # descending diagonal
    a = rng.standard_normal(_SPEC_D)
    a = a / np.linalg.norm(a)
    c_dense = np.linspace(0.0, 2.8, 113)
    traj = np.empty((len(c_dense), _SPEC_D))
    for i, cc in enumerate(c_dense):
        traj[i] = np.asarray(transition_spectrum(jnp.asarray(w), jnp.asarray(a), float(cc)))
    for j in range(_SPEC_D):
        ax2.plot(c_dense, traj[:, j], color=SSM_COLORS["accent"], lw=1.3, alpha=0.85)
    for wj in w:
        ax2.axhline(wj, color=SSM_COLORS["baseline"], lw=0.5, ls=":", alpha=0.6)
    ax2.axhline(-1.0, color=SSM_COLORS["alert"], lw=0.8, ls="--")
    ax2.set_xlim(0.0, 2.8)
    ax2.set_ylim(-2.4, 1.4)
    ax2.annotate("eigenvalues interlace the\ndiagonal entries (dotted)",
                 xy=(0.15, 0.7), fontsize=8, color=SSM_COLORS["baseline"])
    set_tufte_title(ax2, r"General diagonal ($d = 6$): the whole spectrum slides")
    set_tufte_labels(ax2, xlabel=r"removal coefficient $c$", ylabel=r"eigenvalues $\lambda$")

    fig.tight_layout()
    return fig


def make_learned_direction_figure() -> "Figure":
    """Targeted removal (RWKV-7) evicts an old key at $(1-c)^T$ while writing
    orthogonal pairs; the Chapter 12 key-locked removal leaves it flat."""
    import numpy as np

    from companions._shared.plot_utils import (
        SSM_COLORS,
        apply_style,
        create_tufte_figure,
        set_tufte_labels,
        set_tufte_title,
    )

    apply_style()
    fig, ax = create_tufte_figure(figsize=(6.4, 4.2))

    ts = np.arange(0, 91)
    targeted = np.empty(len(ts))
    locked = np.empty(len(ts))
    for i, t in enumerate(ts):
        tm, _, lm = decoupled_eviction(int(t), _EVICT_C)
        targeted[i] = tm
        locked[i] = lm
    # Analytic (1-c)^T scaled to the t=0 norm, for the overlay.
    analytic = targeted[0] * (1.0 - _EVICT_C) ** ts

    ax.plot(ts, locked, color=SSM_COLORS["highlight"], lw=1.8,
            label=r"ch.12 locked ($a_t = $ write key): flat")
    ax.plot(ts, targeted, "o", color=SSM_COLORS["accent"], ms=3.0,
            label=r"RWKV-7 targeted ($a_t = k_A$): measured")
    ax.plot(ts, analytic, color=SSM_COLORS["accent"], lw=1.2, ls="--",
            label=r"$(1-c)^T\,\|v_A\|$, $c = 0.1$")
    ax.set_ylim(bottom=0.0)
    set_tufte_title(ax, "Decoupling the erase direction enables eviction")
    set_tufte_labels(ax, xlabel=r"orthogonal writes $T$",
                     ylabel=r"retrieval norm $\|S_T k_A\|$")
    ax.legend(loc="upper right", fontsize=8, frameon=False)

    fig.tight_layout()
    return fig


def main() -> None:
    import matplotlib.pyplot as plt
    import numpy as np

    from companions._shared.plot_utils import save_figure

    print("Chapter 13 — generalized_transition.py")
    print("=" * 64)

    # §13.2 P1: eigvalsh == scalar-diagonal closed form; secular guard zeroes.
    rng = np.random.default_rng(0)
    d = 6
    a = rng.standard_normal(d)
    a = jnp.asarray(a / np.linalg.norm(a))
    for c in (0.5, 1.0, 2.0):
        spec_scalar = transition_spectrum(jnp.ones(d), a, c)
        closed = scalar_diagonal_spectrum(1.0, c, d)
        print(f"  scalar diag, c={c}: eigvalsh vs closed form max diff = "
              f"{float(jnp.max(jnp.abs(spec_scalar - closed))):.2e}  "
              f"(moving eigenvalue {1.0 - c:+.2f})")

    # General diagonal: secular function zeroes at the eigenvalues (the guard).
    w = jnp.asarray(np.sort(rng.uniform(0.55, 1.0, size=d)))
    c_gen = 0.8
    spec = transition_spectrum(w, a, c_gen)
    sec = max(abs(float(secular_function(lam, w, a, c_gen)))
              for lam in np.asarray(spec) if min(abs(np.asarray(w) - float(lam))) > 1e-6)
    interlace = bool(jnp.all(spec <= jnp.max(w) + 1e-12)
                     and jnp.all(spec >= jnp.min(w) - c_gen - 1e-12))
    print(f"  general diag, c={c_gen}: max |secular(lambda_j)| = {sec:.2e}; "
          f"interlaces diagonal = {interlace}")

    # §13.3 P3: the generalized rule reproduces gated DeltaNet to machine precision.
    length, d_k, d_v = 40, 6, 5
    q = jnp.asarray(rng.standard_normal((length, d_k)))
    k_raw = rng.standard_normal((length, d_k))
    k = jnp.asarray(k_raw / np.linalg.norm(k_raw, axis=1, keepdims=True))
    v = jnp.asarray(rng.standard_normal((length, d_v)))
    betas = jnp.asarray(rng.uniform(0.1, 0.9, size=length))
    gammas = jnp.asarray(rng.uniform(0.7, 1.0, size=length))
    y_gen, s_gen = gated_delta_reduction(q, k, v, betas, gammas)
    y_ch12, s_ch12 = gated_delta_recurrent(q, k, v, betas, gammas)
    print(f"  P3 reduction -> gated DeltaNet: max diff = "
          f"{float(jnp.max(jnp.abs(y_gen - y_ch12))):.2e}  (outputs)")

    # §13.3 scan == materialised-transition oracle (generalized rule).
    w_seq = jnp.asarray(rng.uniform(0.7, 1.0, size=(length, d_k)))
    a_raw = rng.standard_normal((length, d_k))
    a_seq = jnp.asarray(a_raw / np.linalg.norm(a_raw, axis=1, keepdims=True))
    c_seq = jnp.asarray(rng.uniform(0.0, 0.5, size=length))
    u_seq = jnp.asarray(rng.standard_normal((length, d_v)))
    b_seq = jnp.asarray(rng.standard_normal((length, d_k)))
    y_scan, _ = generalized_delta_recurrent(q, w_seq, a_seq, c_seq, u_seq, b_seq)
    y_naive, _ = generalized_delta_naive(q, w_seq, a_seq, c_seq, u_seq, b_seq)
    print(f"  scan == naive oracle:           max diff = "
          f"{float(jnp.max(jnp.abs(y_scan - y_naive))):.2e}")

    # §13.3 decoupled eviction: (1-c)^T law, vs the locked baseline.
    for c, T in ((0.1, 30), (0.2, 20)):
        tm, ta, lm = decoupled_eviction(T, c)
        print(f"  eviction c={c}, T={T:2d}: targeted |S k_A| = {tm:.6f} "
              f"(analytic (1-c)^T |v_A| = {ta:.6f}, diff {abs(tm - ta):.2e}); "
              f"locked baseline = {lm:.6f}")

    _OUT_DIR.mkdir(parents=True, exist_ok=True)
    for fig_fn, stem in ((make_spectrum_figure, "transition-spectrum"),
                         (make_learned_direction_figure, "learned-direction")):
        fig = fig_fn()
        for p in save_figure(fig, _OUT_DIR / stem, formats=("png",)):
            print(f"Wrote {p}")
        plt.close(fig)


if __name__ == "__main__":
    main()
