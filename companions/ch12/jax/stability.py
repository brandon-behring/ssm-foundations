r"""Chapter 12 §12.4 — stability regions for the explicit/implicit pair.

Both DeltaNet (``delta_rule.py``) and Longhorn (``longhorn.py``) iterate the
same affine map; only the step-size rule differs. Around the fixed point
$S^\star = vk^\top/\|k\|^2$ the deviation $E_t = S_t - S^\star$ contracts as

.. math::

    E_t = E_{t-1}\,(I - \beta^{\mathrm{eff}} k k^\top),

a rank-one perturbation of the identity whose only non-unit eigenvalue is the
$k$-direction value $1 - \beta^{\mathrm{eff}}\|k\|^2$ (deviations orthogonal to
$k$ are never corrected — persistent memory, not instability). The
$k$-direction spectral radius is therefore

.. math::

    \rho_k = \bigl|\,1 - \beta^{\mathrm{eff}}\|k\|^2\,\bigr|,

and the two algorithms differ *only* in $\beta^{\mathrm{eff}}$:

* **DeltaNet** (explicit): $\beta^{\mathrm{eff}} = \beta$ free, so stability
  requires $\beta\|k\|^2 \in (0, 2)$ — past $\beta\|k\|^2 = 2$ the deviation
  alternates sign and grows. This is forward Euler leaving its stability
  interval on the test equation (Chapter 5).
* **Longhorn** (implicit): $\beta^{\mathrm{eff}} = 1/(\alpha + \|k\|^2)$, so
  $\rho_k = \alpha/(\alpha + \|k\|^2) < 1$ for every $\alpha > 0$ and every
  $\|k\|$ — unconditional stability, Chapter 6's implicit-method guarantee
  (`ch06:be-a-stable`) replayed on the online-learning ODE.

The numerical guard :func:`iteration_eigenvalue_along_k` recomputes the
$k$-direction eigenvalue from the materialised iteration matrix (a Rayleigh
quotient) and is pinned against the closed forms — derivation drift between
this module and the operator definitions fails the suite.

Port credit
-----------
Ported from ``post_transformers/experiments/jax/week12/stability_analysis.py``
and ``.../figures.py`` (the analytic formulas and the two-panel comparison
figure), restyled to this book's plot conventions.

Usage
-----
::

    PYTHONPATH=. python companions/ch12/jax/stability.py
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import jax

# Enable float64 before any jnp array is created (matches Chapters 4, 7-11).
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402

if TYPE_CHECKING:
    from matplotlib.figure import Figure

__all__ = [
    "deltanet_spectral_radius",
    "longhorn_spectral_radius",
    "deltanet_a_stability_boundary",
    "longhorn_effective_beta_k_product",
    "iteration_eigenvalue_along_k",
]

_REPO_ROOT = Path(__file__).resolve().parents[3]
_OUT_DIR = _REPO_ROOT / "public" / "figures" / "ch12"


# ---------------------------------------------------------------------------
# §12.4 — closed-form spectral radii (the punchline formulas)
# ---------------------------------------------------------------------------


def deltanet_spectral_radius(
    beta: jnp.ndarray | float, k_norm_squared: jnp.ndarray | float
) -> jnp.ndarray:
    r"""DeltaNet's $k$-direction spectral radius $\rho_k = |1 - \beta\|k\|^2|$.

    Stability requires $\rho_k < 1$, i.e. $\beta\|k\|^2 \in (0, 2)$; the
    boundary $\beta\|k\|^2 = 2$ is forward Euler's stability limit on this
    problem (cf. the interval $(-2, 0)$ for $h\lambda$ in Chapter 5).

    Parameters
    ----------
    beta : array-like
        Explicit learning rate(s).
    k_norm_squared : array-like
        $\|k\|^2$.

    Returns
    -------
    jnp.ndarray
    """
    return jnp.abs(1.0 - jnp.asarray(beta) * jnp.asarray(k_norm_squared))


def longhorn_spectral_radius(
    alpha: jnp.ndarray | float, k_norm_squared: jnp.ndarray | float
) -> jnp.ndarray:
    r"""Longhorn's $k$-direction spectral radius $\rho_k = \alpha/(\alpha + \|k\|^2)$.

    Strictly below 1 for every $\alpha > 0$ and every $\|k\|^2 \ge 0$:
    unconditional stability, the implicit-step guarantee.

    Parameters
    ----------
    alpha : array-like
        Trust-region weight(s); must be positive.
    k_norm_squared : array-like

    Returns
    -------
    jnp.ndarray
    """
    alpha = jnp.asarray(alpha)
    k_norm_squared = jnp.asarray(k_norm_squared)
    return alpha / (alpha + k_norm_squared)


def deltanet_a_stability_boundary() -> float:
    r"""The explicit-step stability boundary $\beta\|k\|^2 = 2$.

    Exposed as a named constant so tests and prose read as
    ``beta * k_sq < deltanet_a_stability_boundary()``.

    Returns
    -------
    float
        The constant 2.0.
    """
    return 2.0


def longhorn_effective_beta_k_product(
    alpha: jnp.ndarray | float, k_norm_squared: jnp.ndarray | float
) -> jnp.ndarray:
    r"""Longhorn's $\beta^{\mathrm{eff}}\|k\|^2 = \|k\|^2/(\alpha + \|k\|^2)$.

    Strictly less than 1 for $\alpha > 0$ — Longhorn sits below *half* the
    explicit boundary no matter how large the key grows. The central
    quantitative fact behind the §12.4 comparison.

    Parameters
    ----------
    alpha : array-like
    k_norm_squared : array-like

    Returns
    -------
    jnp.ndarray
    """
    alpha = jnp.asarray(alpha)
    k_norm_squared = jnp.asarray(k_norm_squared)
    return k_norm_squared / (alpha + k_norm_squared)


# ---------------------------------------------------------------------------
# §12.4 — numerical guard: eigenvalue of the materialised iteration matrix
# ---------------------------------------------------------------------------


def iteration_eigenvalue_along_k(
    key: jnp.ndarray, beta_eff: jnp.ndarray | float
) -> jnp.ndarray:
    r"""$k$-direction eigenvalue of $(I - \beta^{\mathrm{eff}} k k^\top)$, computed numerically.

    Materialises the $(d_k, d_k)$ iteration matrix and returns the Rayleigh
    quotient $k^\top M k / k^\top k$ — equal to $1 - \beta^{\mathrm{eff}}\|k\|^2$
    (signed: negative past the boundary). A guard against derivation drift
    between the closed forms above and the operator definitions in
    ``delta_rule.py`` / ``longhorn.py``.

    Parameters
    ----------
    key : jnp.ndarray, shape (d_k,)
        A single key vector; must be nonzero.
    beta_eff : scalar
        Effective learning rate.

    Returns
    -------
    jnp.ndarray, scalar
    """
    if key.ndim != 1:
        raise ValueError(f"key must be 1D (d_k,); got shape {key.shape}")
    k_sq = key @ key
    if float(k_sq) == 0.0:
        raise ValueError("key must be nonzero for the Rayleigh quotient")
    iteration_matrix = jnp.eye(key.shape[0]) - beta_eff * jnp.outer(key, key)
    return key @ (iteration_matrix @ key) / k_sq


# ---------------------------------------------------------------------------
# Figure: the two stability regions side by side
# ---------------------------------------------------------------------------


def make_stability_figure() -> "Figure":
    """Panel A: DeltaNet rho vs beta*||k||^2 with the boundary at 2.
    Panel B: Longhorn rho vs ||k||^2/alpha on a log axis — approaches but never
    crosses 1."""
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

    # Panel A — DeltaNet (explicit).
    bk = np.linspace(0.0, 3.0, 401)
    rho_dn = np.asarray(deltanet_spectral_radius(bk, 1.0))
    ax1.plot(bk, rho_dn, color=SSM_COLORS["accent"], lw=1.8)
    ax1.axhline(1.0, color=SSM_COLORS["baseline"], lw=0.8, ls="--")
    ax1.axvline(2.0, color=SSM_COLORS["alert"], lw=1.0, ls=":")
    ax1.fill_between(bk, 1.0, np.maximum(rho_dn, 1.0), color=SSM_COLORS["alert"], alpha=0.12)
    ax1.set_xlim(0.0, 3.0)
    ax1.set_ylim(0.0, 2.2)
    ax1.annotate(r"boundary $\beta\|k\|^2 = 2$", xy=(2.0, 1.0), xytext=(2.05, 1.75),
                 arrowprops={"arrowstyle": "->", "color": SSM_COLORS["alert"], "linewidth": 0.8},
                 fontsize=9, color=SSM_COLORS["alert"])
    ax1.annotate(r"unstable: $\rho > 1$", xy=(2.45, 1.25), fontsize=9,
                 color=SSM_COLORS["alert"])
    set_tufte_title(ax1, "DeltaNet (explicit step)")
    set_tufte_labels(ax1, xlabel=r"$\beta\,\|k\|^2$", ylabel=r"spectral radius $\rho_k$")

    # Panel B — Longhorn (implicit).
    ratio = np.logspace(-2, 4, 601)
    rho_lh = np.asarray(longhorn_spectral_radius(1.0, ratio))  # rho = 1/(1 + ||k||^2/alpha)
    ax2.semilogx(ratio, rho_lh, color=SSM_COLORS["accent"], lw=1.8)
    ax2.axhline(1.0, color=SSM_COLORS["baseline"], lw=0.8, ls="--")
    ax2.set_xlim(1e-2, 1e4)
    ax2.set_ylim(0.0, 1.1)
    ax2.annotate("stable for every key magnitude:\n" + r"$\rho_k = \alpha/(\alpha + \|k\|^2) < 1$",
                 xy=(1e0, 0.45), fontsize=9, color=SSM_COLORS["accent"])
    set_tufte_title(ax2, "Longhorn (implicit step)")
    set_tufte_labels(ax2, xlabel=r"$\|k\|^2 / \alpha$", ylabel=r"spectral radius $\rho_k$")

    fig.tight_layout()
    return fig


def main() -> None:
    import matplotlib.pyplot as plt
    import numpy as np

    from companions._shared.plot_utils import save_figure

    print("Chapter 12 — stability.py")
    print("=" * 64)

    # §12.4 sample radii (quoted in the figure caption).
    print(f"  boundary: beta * ||k||^2 = {deltanet_a_stability_boundary()}")
    for bk in (0.5, 1.9, 2.5):
        print(f"  DeltaNet  rho at beta*||k||^2 = {bk}: "
              f"{float(deltanet_spectral_radius(bk, 1.0)):.4f}")
    for ratio in (1.0, 100.0):
        print(f"  Longhorn  rho at ||k||^2/alpha = {ratio:5.0f}: "
              f"{float(longhorn_spectral_radius(1.0, ratio)):.4f}")
    print(f"  Longhorn  beta_eff*||k||^2 at ||k||^2/alpha = 1e4: "
          f"{float(longhorn_effective_beta_k_product(1.0, 1e4)):.6f}  (< 1)")

    # §12.4 analytic == numerical eigenvalue (the derivation-drift guard).
    rng = np.random.default_rng(0)
    worst = 0.0
    for seed_beta in np.linspace(0.05, 2.95, 25):
        key = jnp.asarray(rng.standard_normal(8))
        beta_eff = seed_beta / float(key @ key)  # so beta_eff * ||k||^2 = seed_beta
        lam_num = float(iteration_eigenvalue_along_k(key, beta_eff))
        lam_ana = 1.0 - seed_beta
        worst = max(worst, abs(lam_num - lam_ana))
    print(f"  analytic vs Rayleigh eigenvalue: max |diff| = {worst:.2e} over 25-point grid")

    _OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig = make_stability_figure()
    for p in save_figure(fig, _OUT_DIR / "stability-regions", formats=("png",)):
        print(f"Wrote {p}")
    plt.close(fig)


if __name__ == "__main__":
    main()
