r"""Chapter 12 §12.3 — Longhorn: the implicit one-step solve.

Longhorn (Liu et al., arXiv:2407.14207) frames the state update as *amortised
online learning*: instead of one explicit gradient step on the retrieval error
(DeltaNet, ``delta_rule.py``), each step solves the regularised problem in
closed form,

.. math::

    S_t = \arg\min_S \;\tfrac{1}{2}\|S k_t - v_t\|^2
          + \tfrac{\alpha_t}{2}\,\|S - S_{t-1}\|_F^2 .

Setting the gradient to zero gives the implicit recurrence
$\alpha_t (S_t - S_{t-1}) + (S_t k_t - v_t)k_t^\top = 0$ — the backward-Euler
analogue for the recall gradient flow of §12.1. Right-multiplying by $k_t$
solves for the post-update prediction and collapses the implicit equation to a
**rank-one update structurally identical to DeltaNet's**,

.. math::

    S_t = S_{t-1} + \frac{v_t - S_{t-1}k_t}{\alpha_t + \|k_t\|^2}\,k_t^\top
    \qquad\Longleftrightarrow\qquad
    \beta_t^{\mathrm{eff}} = \frac{1}{\alpha_t + \|k_t\|^2},

the implicit step's *self-limiting* learning rate: bounded by $1/\alpha_t$
no matter how large $\|k_t\|$ grows, so $\beta^{\mathrm{eff}}\|k\|^2 =
\|k\|^2/(\alpha + \|k\|^2) < 1$ never reaches the explicit boundary
$\beta\|k\|^2 = 2$ (``stability.py``). This is Chapter 6's implicit-method
robustness, replayed on the online-learning ODE.

Load-bearing facts pinned here:

* **structural identity** — ``longhorn_step(S, k, v, alpha)`` equals
  ``delta_rule_step(S, k, v, longhorn_effective_beta(k, alpha))`` exactly;
* **exact geometric error decay** — under a repeated pair $(k, v)$ the update
  is affine in $S$, so the deviation from $S^\star = vk^\top/\|k\|^2$ contracts
  by *exactly* the $k$-direction eigenvalue per step: DeltaNet by
  $|1 - \beta\|k\|^2|$ (diverges past 2), Longhorn by
  $\alpha/(\alpha + \|k\|^2) < 1$ (unconditionally stable). The trajectory
  figure quotes measured per-step ratios against the analytic values.

Idiomatic-JAX note (NumPy->JAX teaching point)
----------------------------------------------
Same ``lax.scan``-vs-Python-loop oracle pairing as ``delta_rule.py``. The
trajectory helper deliberately returns the *whole* error path from one scan
(``lax.scan`` carries the state, emits the norm) rather than re-running t
prefixes — the O(L) idiom a NumPy reference usually misses.

Port credit
-----------
Ported from ``post_transformers/experiments/jax/week12/longhorn.py``
(arXiv:2407.14207), simplified to the unbatched ``(L, d)`` companion contract;
the exact-trajectory helper and figure are new here.

Usage
-----
::

    PYTHONPATH=. python companions/ch12/jax/longhorn.py
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import jax

# Enable float64 before any jnp array is created (matches Chapters 4, 7-11).
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402

from companions.ch12.jax.delta_rule import delta_rule_fixed_point, delta_rule_step  # noqa: E402

if TYPE_CHECKING:
    from matplotlib.figure import Figure

__all__ = [
    "longhorn_effective_beta",
    "longhorn_step",
    "longhorn_step_via_solve",
    "longhorn_recurrent",
    "longhorn_naive",
    "error_trajectory",
]

_REPO_ROOT = Path(__file__).resolve().parents[3]
_OUT_DIR = _REPO_ROOT / "public" / "figures" / "ch12"


# ---------------------------------------------------------------------------
# §12.3 — the implicit step and its self-limiting effective rate
# ---------------------------------------------------------------------------


def longhorn_effective_beta(key: jnp.ndarray, alpha: jnp.ndarray | float) -> jnp.ndarray:
    r"""Longhorn's effective rate $\beta^{\mathrm{eff}} = 1/(\alpha + \|k\|^2)$.

    The denominator is what makes the step implicit: it self-adapts to the key
    magnitude, capping the rate at $1/\alpha$ and the product
    $\beta^{\mathrm{eff}}\|k\|^2$ strictly below 1.

    Parameters
    ----------
    key : jnp.ndarray, shape (d_k,)
    alpha : scalar
        Trust-region weight; must be strictly positive for the stability cap.

    Returns
    -------
    jnp.ndarray, scalar
    """
    return 1.0 / (alpha + key @ key)


def longhorn_step(
    state: jnp.ndarray,
    key: jnp.ndarray,
    value: jnp.ndarray,
    alpha: jnp.ndarray | float,
) -> jnp.ndarray:
    r"""One implicit-step Longhorn update
    $S \leftarrow S + (v - Sk)\,k^\top / (\alpha + \|k\|^2)$.

    Same shape contract as :func:`delta_rule.delta_rule_step`; the only
    difference is the rate expression — pinned as a structural identity in
    ``tests/test_longhorn.py``.

    Parameters
    ----------
    state : jnp.ndarray, shape (d_v, d_k)
    key : jnp.ndarray, shape (d_k,)
    value : jnp.ndarray, shape (d_v,)
    alpha : scalar
        Trust-region weight; must be positive.

    Returns
    -------
    jnp.ndarray, shape (d_v, d_k)
    """
    return delta_rule_step(state, key, value, longhorn_effective_beta(key, alpha))


def longhorn_step_via_solve(
    state: jnp.ndarray,
    key: jnp.ndarray,
    value: jnp.ndarray,
    alpha: jnp.ndarray | float,
) -> jnp.ndarray:
    r"""The implicit step by *dense linear solve* — the independent oracle for the closed form.

    Solves the stationarity equation directly: $\alpha(S_t - S_{t-1}) + (S_t k -
    v)k^\top = 0$ rearranges to $S_t(\alpha I + kk^\top) = \alpha S_{t-1} +
    vk^\top$, solved here with ``jnp.linalg.solve`` on the (symmetric) system —
    sharing **no code** with the rank-one closed form of :func:`longhorn_step`.
    Their agreement (pinned ``< 1e-12`` in ``tests/test_longhorn.py``) is the
    real certificate of Theorem 12.3's closed form, where the
    delegation ``longhorn_step == delta_rule_step @ beta_eff`` is true by
    construction.

    Parameters
    ----------
    state : jnp.ndarray, shape (d_v, d_k)
    key : jnp.ndarray, shape (d_k,)
    value : jnp.ndarray, shape (d_v,)
    alpha : scalar
        Must be positive (makes $\alpha I + kk^\top$ nonsingular for every key).

    Returns
    -------
    jnp.ndarray, shape (d_v, d_k)
    """
    d_k = key.shape[0]
    system = alpha * jnp.eye(d_k, dtype=state.dtype) + jnp.outer(key, key)
    rhs = alpha * state + jnp.outer(value, key)  # (d_v, d_k); solve S_t @ system = rhs
    return jnp.linalg.solve(system, rhs.T).T


def longhorn_recurrent(
    q: jnp.ndarray,
    k: jnp.ndarray,
    v: jnp.ndarray,
    alphas: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    r"""Sequential Longhorn via ``lax.scan``; post-update read $o_t = S_t q_t$.

    Parameters
    ----------
    q, k : jnp.ndarray, shape (L, d_k)
    v : jnp.ndarray, shape (L, d_v)
    alphas : jnp.ndarray, shape (L,)
        Per-step trust-region weights; must be positive.

    Returns
    -------
    outputs : jnp.ndarray, shape (L, d_v)
    final_state : jnp.ndarray, shape (d_v, d_k)
    """
    if q.ndim != 2 or k.ndim != 2 or v.ndim != 2:
        raise ValueError(f"q, k, v must each be 2D (L, d); got {q.shape}, {k.shape}, {v.shape}")
    if q.shape != k.shape or v.shape[0] != q.shape[0] or alphas.shape != (q.shape[0],):
        raise ValueError(
            f"inconsistent stream shapes: q {q.shape}, k {k.shape}, v {v.shape}, "
            f"alphas {alphas.shape}"
        )
    d_k, d_v = q.shape[1], v.shape[1]

    def step(state: jnp.ndarray, inp) -> tuple[jnp.ndarray, jnp.ndarray]:
        q_t, k_t, v_t, alpha_t = inp
        beta_eff = 1.0 / (alpha_t + k_t @ k_t)
        new_state = state + beta_eff * jnp.outer(v_t - state @ k_t, k_t)
        return new_state, new_state @ q_t

    init = jnp.zeros((d_v, d_k), dtype=v.dtype)
    final_state, outputs = jax.lax.scan(step, init, (q, k, v, alphas))
    return outputs, final_state


def longhorn_naive(
    q: jnp.ndarray,
    k: jnp.ndarray,
    v: jnp.ndarray,
    alphas: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    r"""Python-loop oracle for Longhorn (recomputes $\beta^{\mathrm{eff}}$ each step,
    materialises the projector form). Same conventions as
    :func:`delta_rule.delta_rule_naive`."""
    if q.shape != k.shape or v.shape[0] != q.shape[0] or alphas.shape != (q.shape[0],):
        raise ValueError(
            f"inconsistent stream shapes: q {q.shape}, k {k.shape}, v {v.shape}, "
            f"alphas {alphas.shape}"
        )
    length, d_k = q.shape
    d_v = v.shape[1]
    state = jnp.zeros((d_v, d_k), dtype=v.dtype)
    identity = jnp.eye(d_k, dtype=v.dtype)
    outputs = []
    for t in range(length):
        beta_eff = 1.0 / (alphas[t] + k[t] @ k[t])
        erase = identity - beta_eff * jnp.outer(k[t], k[t])
        state = state @ erase + beta_eff * jnp.outer(v[t], k[t])
        outputs.append(state @ q[t])
    return jnp.stack(outputs), state


# ---------------------------------------------------------------------------
# §12.4 setup — exact error trajectories under a repeated pair
# ---------------------------------------------------------------------------


def error_trajectory(
    key: jnp.ndarray,
    value: jnp.ndarray,
    n_steps: int,
    beta: float | None = None,
    alpha: float | None = None,
) -> jnp.ndarray:
    r"""Frobenius deviation $\|S_t - S^\star\|_F$ under the repeated pair $(k, v)$.

    Exactly one of ``beta`` (DeltaNet, explicit) or ``alpha`` (Longhorn,
    implicit) must be given. Starting from $S_0 = 0$, the deviation
    $E_t = S_t - S^\star$ satisfies $E_t = E_{t-1}(I - \beta^{\mathrm{eff}}
    k k^\top)$ *exactly* (the update is affine in $S$), and $E_0 = -S^\star$
    lies entirely in the $k$ row-direction — so $\|E_t\|_F$ is an exact
    geometric sequence with ratio $|1 - \beta^{\mathrm{eff}}\|k\|^2|$. The
    per-step ratio of the returned array therefore *equals* the analytic
    spectral radius of ``stability.py`` to machine precision.

    Parameters
    ----------
    key : jnp.ndarray, shape (d_k,)
    value : jnp.ndarray, shape (d_v,)
    n_steps : int
    beta : float, optional
        Explicit (DeltaNet) rate.
    alpha : float, optional
        Implicit (Longhorn) trust-region weight.

    Returns
    -------
    jnp.ndarray, shape (n_steps + 1,)
        $\|S_t - S^\star\|_F$ for $t = 0, \dots, n_steps$.
    """
    if (beta is None) == (alpha is None):
        raise ValueError("give exactly one of beta (DeltaNet) or alpha (Longhorn)")
    s_star = delta_rule_fixed_point(key, value)

    def step(state: jnp.ndarray, _unused) -> tuple[jnp.ndarray, jnp.ndarray]:
        if beta is not None:
            new_state = delta_rule_step(state, key, value, beta)
        else:
            new_state = longhorn_step(state, key, value, alpha)
        return new_state, jnp.linalg.norm(new_state - s_star)

    s0 = jnp.zeros_like(s_star)
    _, norms = jax.lax.scan(step, s0, None, length=n_steps)
    return jnp.concatenate([jnp.linalg.norm(s0 - s_star)[None], norms])


# ---------------------------------------------------------------------------
# Figure: explicit divergence vs implicit contraction, with analytic ratios
# ---------------------------------------------------------------------------


def make_trajectory_figure() -> "Figure":
    """Error trajectories under a repeated pair: three DeltaNet rates straddling the
    A-stability boundary, one Longhorn run; legend quotes the analytic ratio."""
    import numpy as np

    from companions._shared.plot_utils import (
        SSM_COLORS,
        apply_style,
        create_tufte_figure,
        set_tufte_labels,
        set_tufte_title,
    )

    apply_style()
    rng = np.random.default_rng(0)
    d_k, d_v, n_steps = 8, 6, 30
    key = jnp.asarray(rng.standard_normal(d_k))
    key = key / jnp.linalg.norm(key)  # unit key: beta * ||k||^2 = beta
    value = jnp.asarray(rng.standard_normal(d_v))

    fig, ax = create_tufte_figure(figsize=(7.0, 4.2))
    t = np.arange(n_steps + 1)

    runs = [
        (dict(beta=0.5), SSM_COLORS["accent"], "-", r"DeltaNet $\beta\|k\|^2=0.5$"),
        (dict(beta=1.9), SSM_COLORS["highlight"], "-", r"DeltaNet $\beta\|k\|^2=1.9$"),
        (dict(beta=2.5), SSM_COLORS["alert"], "-", r"DeltaNet $\beta\|k\|^2=2.5$"),
        (dict(alpha=1.0), SSM_COLORS["accent"], "--", r"Longhorn $\alpha=\|k\|^2$"),
    ]
    for kwargs, color, ls, label in runs:
        traj = np.asarray(error_trajectory(key, value, n_steps, **kwargs))
        if "beta" in kwargs:
            rho = abs(1.0 - kwargs["beta"])  # unit key
        else:
            rho = float(kwargs["alpha"] / (kwargs["alpha"] + 1.0))
        ax.semilogy(t, np.maximum(traj / traj[0], 1e-18), ls, color=color,
                    label=label + rf"  ($\rho = {rho:.2f}$)")

    ax.axhline(1.0, color=SSM_COLORS["baseline"], lw=0.8, ls=":")
    set_tufte_title(ax, "Explicit step diverges past the boundary; implicit step cannot")
    set_tufte_labels(ax, xlabel=r"step $t$ (repeated pair $(k, v)$)",
                     ylabel=r"$\|S_t - S^\star\|_F \,/\, \|S_0 - S^\star\|_F$")
    ax.legend(loc="upper left", fontsize=8, frameon=False)

    fig.tight_layout()
    return fig


def main() -> None:
    import matplotlib.pyplot as plt
    import numpy as np

    from companions._shared.plot_utils import save_figure

    print("Chapter 12 — longhorn.py")
    print("=" * 64)

    rng = np.random.default_rng(0)
    length, d_k, d_v = 48, 8, 6
    q = jnp.asarray(rng.standard_normal((length, d_k)))
    k = jnp.asarray(rng.standard_normal((length, d_k)))
    v = jnp.asarray(rng.standard_normal((length, d_v)))
    alphas = jnp.asarray(rng.uniform(0.5, 2.0, size=length))

    # §12.3 scan == naive oracle.
    y_scan, s_scan = longhorn_recurrent(q, k, v, alphas)
    y_naive, s_naive = longhorn_naive(q, k, v, alphas)
    print(f"  scan == naive oracle:        max diff = "
          f"{float(jnp.max(jnp.abs(y_scan - y_naive))):.2e}")

    # §12.3 structural identity: longhorn_step == delta_rule_step at beta_eff.
    state = jnp.asarray(rng.standard_normal((d_v, d_k)))
    key1, val1, alpha1 = k[0], v[0], 0.7
    lh = longhorn_step(state, key1, val1, alpha1)
    dn = delta_rule_step(state, key1, val1, longhorn_effective_beta(key1, alpha1))
    print(f"  longhorn == delta @ beta_eff: max diff = {float(jnp.max(jnp.abs(lh - dn))):.2e}")

    # §12.3 the independent certificate: closed form == dense implicit solve.
    solved = longhorn_step_via_solve(state, key1, val1, alpha1)
    print(f"  closed form == dense solve:   max diff = "
          f"{float(jnp.max(jnp.abs(lh - solved))):.2e}")
    residual = alpha1 * (lh - state) + jnp.outer(lh @ key1 - val1, key1)
    print(f"  stationarity residual at closed form:   "
          f"{float(jnp.max(jnp.abs(residual))):.2e}")

    # §12.3 the cap: beta_eff * ||k||^2 < 1 even for huge keys (print the gap to 1).
    for scale in (1.0, 1e2, 1e4):
        kk = key1 * scale
        prod = float(longhorn_effective_beta(kk, alpha1) * (kk @ kk))
        print(f"  ||k|| x {scale:8.0e}: beta_eff * ||k||^2 = 1 - {1.0 - prod:.2e}  (< 1 always)")

    # §12.4 measured per-step ratio == analytic spectral radius (unit key).
    key_u = key1 / jnp.linalg.norm(key1)
    for kwargs, rho in ((dict(beta=2.5), 1.5), (dict(alpha=1.0), 0.5)):
        traj = np.asarray(error_trajectory(key_u, val1, 10, **kwargs))
        ratios = traj[1:] / traj[:-1]
        name = "DeltaNet beta=2.5" if "beta" in kwargs else "Longhorn alpha=1 "
        print(f"  {name}: measured ratio = {ratios.mean():.12f}  (analytic rho = {rho})")

    _OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig = make_trajectory_figure()
    for p in save_figure(fig, _OUT_DIR / "explicit-implicit-trajectories", formats=("png",)):
        print(f"Wrote {p}")
    plt.close(fig)


if __name__ == "__main__":
    main()
