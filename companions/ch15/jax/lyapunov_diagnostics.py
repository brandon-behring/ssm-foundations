r"""Chapter 15 §§15.4–15.5 — Lyapunov and regime diagnostics on constructed systems.

Two propositions, both validated against *known ground truth* on systems we
construct (no training anywhere — applying these instruments to trained networks
is pilot B's program):

* **P2′ (``ch15:lyap-correctness``)** — the Benettin QR estimator
  (``companions.ch02.jax.lyapunov_qr.qr_lyapunov``, reused verbatim) recovers the
  asymptotic log-growth rate. Two regimes, exactly as Chapter 2's tests already
  document for the ring:

  - **non-degenerate** systems (distinct $|\lambda_i|$): the sorted exponents
    converge to $\log|\lambda_i|$, top exponent to $\max_i \log|\lambda_i|$. We
    show this on a Chapter 13 diagonal-plus-rank-one (DPLR) transition and a
    Chapter 9 selective/LTV Jacobian stream — genuinely new objects, not Ch 2's ring.
  - **degenerate** systems (equal $|\lambda_i|$): the individual exponents are not
    separable by QR and carry $O(10^{-2})$ splitting noise, while their **mean is
    exact**. The S4D-Lin initialization ($\Re(\lambda) = -\tfrac12$ for every mode)
    is exactly this case — a *recognizable architecture* whose uniform decay makes
    the per-mode exponents blur but the decay rate exact.
  - the **divergence identity** $\sum_i \lambda_i = \langle \log|\det J_t|\rangle$
    holds regardless of degeneracy (the regression guard).

* **P3′ (``ch15:regime-separation``)** — on a constructed block system with $r$
  marginal modes ($|\lambda| = 1$) and $d - r$ contractive modes ($|\lambda| = w$),
  two *independent* computations agree on the memory-mode count $r$: the
  **algebraic** route (count $|\lambda_i| \ge 1 - \delta$ from ``eigvalsh``) and the
  **dynamical** route (count Lyapunov exponents $\ge -\delta$ from the QR scan). The
  **effective state size** $D_{\mathrm{eff}} = (\sum_i p_i)^2 / \sum_i p_i^2$ with
  $p_i = |\lambda_i|^2$ is the continuous soft-count: $D_{\mathrm{eff}} \to r$ as
  $w \to 0$, $\to d$ as $w \to 1$. Ground-truth-known construction + two-route
  agreement is the anti-circularity guard (a diagnostic must not be "verified"
  against its own output).

Idiomatic-JAX / port credit
---------------------------
Greenfield. The Lyapunov engine is Chapter 2's ``qr_lyapunov`` (Benettin et al.
1980), reused unchanged. The constructed systems reuse Chapter 13's
``dplr_transition`` and Chapter 9's ``discretize_selective``. The closed-form
references (``eigvals``/``eigvalsh`` magnitudes; the diagonal-LTV per-coordinate
mean) are *independent* of the QR iteration — that independence is what makes the
validation non-circular.

Usage
-----
::

    PYTHONPATH=. python companions/ch15/jax/lyapunov_diagnostics.py
"""

from __future__ import annotations

from pathlib import Path

import jax

# Lyapunov exponents need float64 to separate near-degenerate modes (Ch 2 §2.3).
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
from scipy.linalg import expm  # noqa: E402

from companions.ch01.jax.coupled_oscillators import build_ring_state_matrix  # noqa: E402
from companions.ch02.jax.lyapunov_qr import qr_lyapunov  # noqa: E402
from companions.ch09.jax.selective_ssm import discretize_selective  # noqa: E402
from companions.ch13.jax.generalized_transition import dplr_transition  # noqa: E402

__all__ = [
    "lyapunov_spectrum",
    "lyapunov_top",
    "closed_form_log_growth",
    "log_det_rate",
    "dplr_jacobians",
    "s4d_lin_transition",
    "constructed_ltv_jacobians",
    "ltv_closed_form_log_growth",
    "ring_jacobian",
    "two_regime_transition",
    "effective_state_size",
    "effective_state_size_closed_form",
    "marginal_mode_count",
]

_REPO_ROOT = Path(__file__).resolve().parents[3]
_OUT_DIR = _REPO_ROOT / "public" / "figures" / "ch15"


# ---------------------------------------------------------------------------
# The diagnostics (thin, deliberately — the engine is Chapter 2's).
# ---------------------------------------------------------------------------


def lyapunov_spectrum(jacobians: np.ndarray, n_steps: int) -> np.ndarray:
    """Lyapunov spectrum (descending) of a per-step Jacobian sequence.

    Thin pass-through to :func:`companions.ch02.jax.lyapunov_qr.qr_lyapunov` — the
    Benettin QR algorithm. Kept as a named Chapter 15 entry point so the diagnostic
    reads as a Chapter 15 object while reusing the Chapter 2 engine verbatim.
    """
    return qr_lyapunov(jacobians, n_steps)


def lyapunov_top(jacobians: np.ndarray, n_steps: int) -> float:
    """Top Lyapunov exponent (the asymptotic log-growth rate of the leading direction)."""
    return float(qr_lyapunov(jacobians, n_steps)[0])


def closed_form_log_growth(J: np.ndarray) -> np.ndarray:
    r"""Closed-form Lyapunov spectrum of an *autonomous* discrete system: $\log|\lambda_i(J)|$.

    For a constant per-step Jacobian $J$, the Lyapunov exponents are
    $\log|\lambda_i(J)|$ sorted descending — the ground-truth reference the QR
    estimator is validated against. Independent of the QR iteration (it is an
    eigendecomposition), which is what makes the P2′ check non-circular.

    Parameters
    ----------
    J : ndarray, shape (N, N)
        Autonomous discrete transition.

    Returns
    -------
    ndarray, shape (N,)
        ``sort(log(abs(eigvals(J))))[::-1]``.
    """
    J = np.asarray(J)
    if J.ndim != 2 or J.shape[0] != J.shape[1]:
        raise ValueError(f"J must be square (N, N); got {J.shape}")
    mags = np.abs(np.asarray(jnp.linalg.eigvals(jnp.asarray(J))))
    return np.sort(np.log(mags))[::-1]


def log_det_rate(jacobians: np.ndarray) -> float:
    r"""The divergence identity's right side: $\langle \log|\det J_t| \rangle$.

    The sum of the Lyapunov exponents equals this *exactly* at every trace length
    (the QR triangular factor satisfies $\prod_i |R_{ii}^{(t)}| = |\det J_t|$), so it
    is the robust summary that survives even when individual exponents are
    unresolved. Independent of the QR scan — computed straight from the Jacobians.
    """
    jacobians = np.asarray(jacobians)
    if jacobians.ndim != 3 or jacobians.shape[1] != jacobians.shape[2]:
        raise ValueError(f"jacobians must be (T, N, N); got {jacobians.shape}")
    signs, logabsdets = np.linalg.slogdet(jacobians)
    return float(np.mean(logabsdets))


# ---------------------------------------------------------------------------
# Constructed systems with known spectra (P2′): DPLR, selective/LTV, S4D-Lin.
# ---------------------------------------------------------------------------


def dplr_jacobians(w: np.ndarray, a: np.ndarray, c: float, n_steps: int) -> np.ndarray:
    r"""Autonomous Jacobian stream for the Chapter 13 DPLR transition $\mathrm{Diag}(w) - c\,a a^\top$.

    Symmetric (real spectrum), generically *non-diagonal* — so the QR frame
    genuinely rotates, exercising the estimator (unlike the diagonal LTV case).
    Returns shape ``(1, N, N)`` (the autonomous tile; ``qr_lyapunov`` cycles it).
    """
    J = np.asarray(dplr_transition(jnp.asarray(w), jnp.asarray(a), c))
    return J[np.newaxis, ...]


def s4d_lin_transition(n_modes: int, dt: float) -> np.ndarray:
    r"""A real transition whose spectrum *resembles* an S4D-Lin layer at init.

    S4D-Lin initializes diagonal modes $\mu_k = -\tfrac12 + i\pi k$; the discrete
    transition has eigenvalues $e^{\,dt\,\mu_k}$, all of modulus $e^{-dt/2}$ — a
    *degenerate* Lyapunov spectrum (every mode decays at the same rate). Realized as
    $n_{\mathrm{modes}}$ real $2\times2$ rotation-scaling blocks, so the matrix is
    real and the estimator sees genuine rotation.

    Returns
    -------
    ndarray, shape (2 n_modes, 2 n_modes)
        Block-diagonal real transition; every $|\lambda| = e^{-dt/2}$.
    """
    if n_modes < 1:
        raise ValueError(f"n_modes must be >= 1; got {n_modes}")
    if dt <= 0.0:
        raise ValueError(f"dt must be > 0; got {dt}")
    rho = float(np.exp(-0.5 * dt))
    blocks = []
    for k in range(n_modes):
        theta = dt * np.pi * (k + 1)  # distinct rotation angles; equal modulus rho
        c_, s_ = np.cos(theta), np.sin(theta)
        blocks.append(rho * np.array([[c_, -s_], [s_, c_]]))
    out = np.zeros((2 * n_modes, 2 * n_modes))
    for k, blk in enumerate(blocks):
        out[2 * k : 2 * k + 2, 2 * k : 2 * k + 2] = blk
    return out


def constructed_ltv_jacobians(A: np.ndarray, delta: np.ndarray, B: np.ndarray) -> np.ndarray:
    r"""Time-varying diagonal Jacobian stream from a Chapter 9 selective discretization.

    Each step's state map is $h_t = \bar A_t \odot h_{t-1} + \cdots$ with
    $\bar A_t = e^{\Delta_t A}$ (``discretize_selective``); the Jacobian is
    $\mathrm{diag}(\bar A_t)$. Returns shape ``(L, N, N)``. Because the system is
    diagonal it decouples — the Lyapunov spectrum has the closed form
    :func:`ltv_closed_form_log_growth`, which QR reproduces to machine precision.
    """
    Abar, _ = discretize_selective(jnp.asarray(A), jnp.asarray(delta), jnp.asarray(B))
    Abar = np.asarray(Abar)  # (L, N)
    return np.stack([np.diag(row) for row in Abar], axis=0)  # (L, N, N)


def ltv_closed_form_log_growth(Abar: np.ndarray) -> np.ndarray:
    r"""Closed-form Lyapunov spectrum of a diagonal LTV system: per-coordinate mean log-stretch.

    For decoupled coordinates, coordinate $i$ grows at the average rate
    $\frac1L \sum_t \log|\bar A_{t,i}|$; the spectrum is these averages sorted
    descending. Independent of the QR iteration.
    """
    Abar = np.asarray(Abar)
    if Abar.ndim != 2:
        raise ValueError(f"Abar must be 2-D (L, N); got {Abar.shape}")
    return np.sort(np.mean(np.log(np.abs(Abar)), axis=0))[::-1]


def ring_jacobian(n: int, c: float, dt: float) -> np.ndarray:
    r"""Discrete Jacobian of Chapter 2's damped ring — the resolution-limit instance.

    The ring of oscillators (``companions.ch01``) is *non-normal* and its $2n$ modes
    share a decay rate $\Re(\lambda) = -c/2$ but oscillate at distinct frequencies —
    distinct eigenvalues, equal modulus. That is exactly the case the Benettin
    estimator cannot resolve mode-by-mode (the individual exponents scatter by
    $O(10^{-2})$, the mean stays exact), so it is the honest in-book demonstration of
    P2′'s resolution limit — *not* a re-run of the §2.3 spectrum figure, which we
    repurpose here to show the per-mode error rather than the spectrum.
    """
    if n < 1:
        raise ValueError(f"n must be >= 1; got {n}")
    if dt <= 0.0:
        raise ValueError(f"dt must be > 0; got {dt}")
    A = np.asarray(build_ring_state_matrix(n=n, c=c, kappa=1.0))
    return np.asarray(expm(A * dt))


# ---------------------------------------------------------------------------
# Regime separation and effective state size (P3′).
# ---------------------------------------------------------------------------


def two_regime_transition(r: int, d: int, w: float, seed: int = 0) -> np.ndarray:
    r"""Symmetric transition with $r$ marginal ($|\lambda|=1$) + $d-r$ contractive ($|\lambda|=w$) modes.

    Built as $Q\,\mathrm{Diag}([1]^r \Vert [w]^{d-r})\,Q^\top$ for a random
    orthogonal $Q$ — so the transition is non-diagonal (the estimator sees a
    genuine frame) yet its spectrum is known exactly by construction. This is the
    ground-truth-known object the two-route cross-check runs on.
    """
    if not 1 <= r < d:
        raise ValueError(f"need 1 <= r < d; got r={r}, d={d}")
    if not 0.0 < w < 1.0:
        raise ValueError(f"w must be in (0, 1); got {w}")
    rng = np.random.default_rng(seed)
    q, _ = np.linalg.qr(rng.standard_normal((d, d)))
    spectrum = np.concatenate([np.ones(r), np.full(d - r, w)])
    return (q * spectrum) @ q.T


def effective_state_size(magnitudes: np.ndarray) -> float:
    r"""Participation ratio of the squared spectral magnitudes: $(\sum_i p_i)^2 / \sum_i p_i^2$.

    With $p_i = |\lambda_i|^2$, a soft count of the dominant modes: $d$ when all
    magnitudes are equal, $\to$ (number of dominant modes) when the rest vanish.
    """
    p = np.asarray(magnitudes, dtype=float) ** 2
    denom = float(np.sum(p**2))
    if denom == 0.0:
        raise ValueError("all magnitudes are zero; effective state size undefined")
    return float(np.sum(p) ** 2 / denom)


def effective_state_size_closed_form(r: int, d: int, w: float) -> float:
    r"""Closed form of :func:`effective_state_size` on the two-level spectrum $[1]^r\Vert[w]^{d-r}$.

    $D_{\mathrm{eff}} = (r + (d-r)w^2)^2 / (r + (d-r)w^4)$, which $\to r$ as
    $w \to 0$ and $\to d$ as $w \to 1$.
    """
    if not 1 <= r <= d:
        raise ValueError(f"need 1 <= r <= d; got r={r}, d={d}")
    if not 0.0 <= w <= 1.0:
        raise ValueError(f"w must be in [0, 1]; got {w}")
    num = (r + (d - r) * w**2) ** 2
    den = r + (d - r) * w**4
    return float(num / den)


def marginal_mode_count(values: np.ndarray, tol: float, mode: str = "magnitude") -> int:
    r"""Count the marginal/memory modes — the two-route cross-check's shared label.

    Parameters
    ----------
    values : ndarray
        Spectral magnitudes ($|\lambda_i|$, ``mode="magnitude"``) or Lyapunov
        exponents ($\lambda_i$, ``mode="exponent"``).
    tol : float
        Slack $\delta$ around the marginal boundary.
    mode : {"magnitude", "exponent"}
        ``"magnitude"`` counts $|\lambda_i| \ge 1 - \delta$ (algebraic route);
        ``"exponent"`` counts exponents $\ge -\delta$ (dynamical route).
    """
    values = np.asarray(values, dtype=float)
    if tol < 0.0:
        raise ValueError(f"tol must be >= 0; got {tol}")
    if mode == "magnitude":
        return int(np.sum(np.abs(values) >= 1.0 - tol))
    if mode == "exponent":
        return int(np.sum(values >= -tol))
    raise ValueError(f"mode must be 'magnitude' or 'exponent'; got {mode!r}")


# ---------------------------------------------------------------------------
# Figures + measured numbers (§§15.4–15.5).
# ---------------------------------------------------------------------------

_DPLR_W = np.linspace(0.9, 0.5, 6)
_DPLR_C = 0.3
_S4D_MODES = 4
_S4D_DT = 0.4
_LTV_LEN = 256
_LTV_SEED = 0
_RECOVER_STEPS = 4000
_RING_N = 8
_RING_C = 0.2
_RING_DT = 0.05
_RING_STEPS = 2000
_REGIME_R = 3
_REGIME_D = 8
_REGIME_W = 0.4
_REGIME_STEPS = 3000
_REGIME_TOL = 0.1
_REGIME_SEED = 0


def _dplr_a() -> np.ndarray:
    rng = np.random.default_rng(1)
    a = rng.standard_normal(_DPLR_W.shape[0])
    return a / np.linalg.norm(a)


def _ltv_system() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """A selective/LTV system with distinct per-coordinate decay rates."""
    rng = np.random.default_rng(_LTV_SEED)
    A = -np.linspace(0.2, 1.5, 5)  # distinct negative modes -> distinct decays
    delta = rng.uniform(0.3, 0.7, size=_LTV_LEN)
    B = np.ones((_LTV_LEN, 5))  # B does not enter Abar; shape-only
    return A, delta, B


def _fig_lyapunov_validation() -> None:
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

    # --- Panel A: recovery on three constructed systems ---
    a = _dplr_a()
    J_dplr = dplr_jacobians(_DPLR_W, a, _DPLR_C, _RECOVER_STEPS)
    dplr_est = lyapunov_spectrum(J_dplr, _RECOVER_STEPS)
    dplr_ref = closed_form_log_growth(J_dplr[0])

    A, delta, B = _ltv_system()
    Abar = np.asarray(discretize_selective(jnp.asarray(A), jnp.asarray(delta), jnp.asarray(B))[0])
    J_ltv = constructed_ltv_jacobians(A, delta, B)
    ltv_est = lyapunov_spectrum(J_ltv, _LTV_LEN)
    ltv_ref = ltv_closed_form_log_growth(Abar)

    J_s4d = s4d_lin_transition(_S4D_MODES, _S4D_DT)[np.newaxis, ...]
    s4d_est = lyapunov_spectrum(J_s4d, _RECOVER_STEPS)
    s4d_ref = closed_form_log_growth(J_s4d[0])  # all equal to -dt/2 (degenerate, but decoupled)

    dplr_err = float(np.max(np.abs(dplr_est - dplr_ref)))
    dplr_top_err = float(abs(dplr_est[0] - dplr_ref[0]))
    ltv_err = float(np.max(np.abs(ltv_est - ltv_ref)))
    s4d_err = float(np.max(np.abs(s4d_est - s4d_ref)))
    # The divergence identity Σλ = <log|det J|> holds exactly even on the degenerate S4D.
    dplr_div_err = float(abs(dplr_est.sum() - log_det_rate(J_dplr)))
    s4d_div_err = float(abs(s4d_est.sum() - log_det_rate(J_s4d)))

    # --- Panel B: the resolution limit on a non-normal degenerate recurrence ---
    J_ring = ring_jacobian(_RING_N, _RING_C, _RING_DT)[np.newaxis, ...]
    ring_est = lyapunov_spectrum(J_ring, _RING_STEPS) / _RING_DT
    ring_ref = np.sort(np.linalg.eigvals(np.asarray(
        build_ring_state_matrix(n=_RING_N, c=_RING_C, kappa=1.0))).real)[::-1]
    ring_scatter = float(np.max(np.abs(ring_est - ring_ref)))
    ring_mean_err = float(abs(ring_est.mean() - ring_ref.mean()))

    print("  lyapunov-validation numbers:")
    print(f"    DPLR  (distinct |lambda|, normal): max|est-ref| = {dplr_err:.3e}  "
          f"top err = {dplr_top_err:.3e}  top = {dplr_est[0]:.6f}")
    print(f"    LTV   (diagonal selective):        max|est-ref| = {ltv_err:.3e}")
    print(f"    S4D-Lin (degenerate, decoupled, |lambda|=e^-dt/2={np.exp(-0.5*_S4D_DT):.6f}): "
          f"max|est-ref| = {s4d_err:.3e}")
    print(f"    divergence identity sum(lambda) vs <log|det J|>: DPLR {dplr_div_err:.3e}, "
          f"S4D {s4d_div_err:.3e}")
    print(f"    ring (non-normal, degenerate moduli): scatter max|est-ref| = {ring_scatter:.3e} "
          f"(modes unresolved), mean err = {ring_mean_err:.3e} (rate exact)")

    fig, axes = create_tufte_figure(1, 2, figsize=(11.0, 4.3))
    ax_rec, ax_res = axes  # type: ignore[misc]

    idx_d = np.arange(1, dplr_est.shape[0] + 1)
    idx_l = np.arange(1, ltv_est.shape[0] + 1)
    idx_s = np.arange(1, s4d_est.shape[0] + 1)
    ax_rec.scatter(idx_d, dplr_est, s=46, color=SSM_COLORS["accent"], edgecolors="white",
                   linewidths=0.8, zorder=3, label="DPLR (Ch 13): QR")
    ax_rec.scatter(idx_d, dplr_ref, s=18, color=SSM_COLORS["alert"], marker="x", zorder=4,
                   label=r"DPLR: $\log|\lambda_i|$")
    ax_rec.scatter(idx_l, ltv_est, s=46, color=SSM_COLORS["highlight"], edgecolors="white",
                   linewidths=0.8, zorder=3, marker="s", label="selective/LTV (Ch 9): QR")
    ax_rec.scatter(idx_l, ltv_ref, s=18, color=SSM_COLORS["baseline"], marker="+", zorder=4,
                   label="selective/LTV: closed form")
    ax_rec.plot(idx_s, s4d_est, "D-", color="0.45",
                ms=4.5, lw=0.9, zorder=2, label="S4D-Lin init: QR (uniform decay)")
    set_tufte_title(ax_rec, "The estimator recovers known spectra")
    set_tufte_labels(ax_rec, "mode index $i$", r"Lyapunov exponent $\lambda_i$")
    ax_rec.legend(frameon=False, fontsize=7.5, loc="lower left")

    idx_r = np.arange(1, ring_est.shape[0] + 1)
    err = ring_est - ring_ref
    ax_res.scatter(idx_r, err, s=42, color=SSM_COLORS["accent"], edgecolors="white",
                   linewidths=0.8, zorder=3, label=r"per-mode error $\hat\lambda_i - \lambda_i$")
    ax_res.axhline(0.0, color=SSM_COLORS["alert"], lw=1.2, ls="--", label="zero error")
    ax_res.axhline(float(err.mean()), color=SSM_COLORS["baseline"], lw=1.0, ls=":",
                   label=rf"mean error ${err.mean():.1e}$ (exact)")
    set_tufte_title(ax_res, "Resolution limit: a non-normal degenerate recurrence")
    set_tufte_labels(ax_res, "mode index $i$", r"error $\hat\lambda_i - \lambda_i$")
    ax_res.legend(frameon=False, fontsize=8, loc="upper right")

    fig.tight_layout()
    _OUT_DIR.mkdir(parents=True, exist_ok=True)
    for p in save_figure(fig, _OUT_DIR / "lyapunov-validation", formats=("png",)):
        print(f"  wrote {p.relative_to(_REPO_ROOT)}")
    plt.close(fig)


def _fig_regime_separation() -> None:
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
    A = two_regime_transition(_REGIME_R, _REGIME_D, _REGIME_W, seed=_REGIME_SEED)
    spec_lyap = lyapunov_spectrum(A[np.newaxis, ...], _REGIME_STEPS)
    mags = np.abs(np.asarray(jnp.linalg.eigvalsh(jnp.asarray(A))))
    count_alg = marginal_mode_count(mags, _REGIME_TOL, mode="magnitude")
    count_dyn = marginal_mode_count(spec_lyap, _REGIME_TOL, mode="exponent")

    ws = np.linspace(0.05, 0.95, 19)
    deff_measured = np.asarray(
        [effective_state_size(np.abs(np.asarray(jnp.linalg.eigvalsh(
            jnp.asarray(two_regime_transition(_REGIME_R, _REGIME_D, float(wv), seed=_REGIME_SEED))))))
         for wv in ws]
    )
    deff_closed = np.asarray([effective_state_size_closed_form(_REGIME_R, _REGIME_D, float(wv)) for wv in ws])
    deff_err = float(np.max(np.abs(deff_measured - deff_closed)))

    print("  regime-separation numbers:")
    print(f"    system: r={_REGIME_R} marginal + {_REGIME_D - _REGIME_R} contractive (w={_REGIME_W}), d={_REGIME_D}")
    print(f"    algebraic marginal count (|lambda|>=1-{_REGIME_TOL}) = {count_alg}")
    print(f"    dynamical marginal count (exponent>=-{_REGIME_TOL})  = {count_dyn}")
    print(f"    two routes agree on r = {_REGIME_R}: {count_alg == count_dyn == _REGIME_R}")
    print(f"    D_eff(measured vs closed form) max err = {deff_err:.3e}")
    print(f"    D_eff at w={ws[0]:.2f} = {deff_measured[0]:.4f} (-> r={_REGIME_R});  "
          f"at w={ws[-1]:.2f} = {deff_measured[-1]:.4f} (-> d={_REGIME_D})")

    fig, axes = create_tufte_figure(1, 2, figsize=(11.0, 4.3))
    ax_spec, ax_deff = axes  # type: ignore[misc]

    idx = np.arange(1, spec_lyap.shape[0] + 1)
    ax_spec.scatter(idx, spec_lyap, s=48, color=SSM_COLORS["accent"], edgecolors="white",
                    linewidths=0.8, zorder=3, label="Lyapunov spectrum")
    ax_spec.axhline(0.0, color=SSM_COLORS["alert"], lw=1.2, ls="--", label="marginal line $\\lambda=0$")
    ax_spec.axhline(-_REGIME_TOL, color=SSM_COLORS["baseline"], lw=0.7, ls=":")
    set_tufte_title(ax_spec, rf"Dynamical route: {count_dyn} exponents at zero")
    set_tufte_labels(ax_spec, "mode index $i$", r"Lyapunov exponent $\lambda_i$")
    ax_spec.legend(frameon=False, fontsize=8, loc="lower left")

    ax_deff.plot(ws, deff_measured, "o-", color=SSM_COLORS["accent"], ms=3.5,
                 label=r"$D_{\mathrm{eff}}$ (from eigvalsh)")
    ax_deff.plot(ws, deff_closed, color=SSM_COLORS["baseline"], lw=1.0, ls=":",
                 label="closed form")
    ax_deff.axhline(_REGIME_R, color=SSM_COLORS["alert"], lw=1.0, ls="--",
                    label=rf"$r={_REGIME_R}$ (memory modes)")
    ax_deff.axhline(_REGIME_D, color=SSM_COLORS["highlight"], lw=1.0, ls="--",
                    label=rf"$d={_REGIME_D}$")
    set_tufte_title(ax_deff, "Effective state size interpolates $r \\to d$")
    set_tufte_labels(ax_deff, r"contractive magnitude $w$", r"$D_{\mathrm{eff}}$")
    ax_deff.legend(frameon=False, fontsize=8, loc="center right")

    fig.tight_layout()
    _OUT_DIR.mkdir(parents=True, exist_ok=True)
    for p in save_figure(fig, _OUT_DIR / "regime-separation", formats=("png",)):
        print(f"  wrote {p.relative_to(_REPO_ROOT)}")
    plt.close(fig)


def main() -> None:
    print("Chapter 15 — lyapunov_diagnostics.py")
    print("=" * 64)
    _fig_lyapunov_validation()
    _fig_regime_separation()


if __name__ == "__main__":
    main()
