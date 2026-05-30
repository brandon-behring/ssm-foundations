r"""Chapter 9 §9.2 — selectivity made visible: an input-dependent step size resets memory.

The dynamical-systems reading of Mamba: an LTI SSM (Chapter 8) has *one* fixed
transition $\bar A = e^{\Delta A}$, so it filters every token through the same
geometric decay — it cannot decide what to keep. A selective SSM is **linear
time-varying**: $\bar A_t = e^{\Delta_t A}$ changes per step. With $A < 0$,

* $\Delta_t \to 0 \Rightarrow \bar A_t \to 1$  — the state is *held* (and no new
  input is absorbed, since $\bar B_t = \Delta_t B_t \to 0$): a closed gate;
* $\Delta_t$ large $\Rightarrow \bar A_t \to 0$ — the state is *overwritten* by
  the current input: an open gate.

This figure drives a single mode with one content impulse and contrasts the
state trajectory under a fixed-$\Delta$ LTI system (which forgets) against a
selective system that writes once and then holds (which remembers) — the
mechanism behind Mamba's selective-copying ability.

Output
------
``public/figures/ch09/selective-vs-lti.png`` — top: input and selective
$\Delta_t$; bottom: held (selective) vs decaying (LTI) state.

Usage
-----
::

    PYTHONPATH=. python companions/ch09/jax/selective_vs_lti.py
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402

from companions.ch09.jax.selective_ssm import selective_apply  # noqa: E402

if TYPE_CHECKING:
    from matplotlib.figure import Figure

__all__ = ["selective_vs_lti_states", "make_selectivity_figure"]

_REPO_ROOT = Path(__file__).resolve().parents[3]
_OUT_DIR = _REPO_ROOT / "public" / "figures" / "ch09"

# A single negative mode shared by both systems; the only difference is delta_t.
_A = jnp.asarray([-1.0])
_LENGTH = 40
_WRITE_STEP = 5
_DELTA_ABSORB = 1.0  # selective: open the gate at the content token
_DELTA_HOLD = 0.005  # selective: near-closed gate elsewhere (Abar ~ 1)
_DELTA_LTI = 0.30  # LTI: one fixed step size for every token


def selective_vs_lti_states() -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    r"""State trajectories for a held (selective) vs decaying (LTI) single mode.

    Both systems see the same content impulse $u_{t_0} = 1$ and the same mode
    $A = -1$, $B = C = 1$. The selective system uses $\Delta_t = \Delta_\text{hold}$
    everywhere except a $\Delta_\text{absorb}$ spike at the content token; the LTI
    system uses a single fixed $\Delta$.

    Returns
    -------
    u : jnp.ndarray, shape (L,)
        Input (impulse at the write step).
    delta_sel : jnp.ndarray, shape (L,)
        Selective step sizes.
    h_sel : jnp.ndarray, shape (L,)
        Selective state trajectory $y_t = C h_t$ (here $C = 1$, so $y_t = h_t$).
    h_lti : jnp.ndarray, shape (L,)
        LTI state trajectory.
    """
    u = jnp.zeros(_LENGTH).at[_WRITE_STEP].set(1.0)
    ones = jnp.ones((_LENGTH, 1))  # B_t = C_t = 1 for all t

    delta_sel = jnp.full((_LENGTH,), _DELTA_HOLD).at[_WRITE_STEP].set(_DELTA_ABSORB)
    delta_lti = jnp.full((_LENGTH,), _DELTA_LTI)

    h_sel = selective_apply(_A, delta_sel, ones, ones, 0.0, u, parallel=False)
    h_lti = selective_apply(_A, delta_lti, ones, ones, 0.0, u, parallel=False)
    return u, delta_sel, h_sel, h_lti


def make_selectivity_figure() -> Figure:
    """Top: input + selective delta. Bottom: held vs decaying state."""
    import numpy as np

    from companions._shared.plot_utils import (
        SSM_COLORS,
        apply_style,
        create_tufte_figure,
        set_tufte_labels,
        set_tufte_title,
    )

    apply_style()
    u, delta_sel, h_sel, h_lti = (np.asarray(a) for a in selective_vs_lti_states())
    t = np.arange(_LENGTH)

    fig, (ax1, ax2) = create_tufte_figure(ncols=2, figsize=(11.0, 4.2))

    ax1.stem(t, u, linefmt=SSM_COLORS["baseline"], markerfmt="o", basefmt=" ", label="input $u_t$")
    ax1.plot(t, delta_sel, "-", color=SSM_COLORS["accent"], label=r"selective $\Delta_t$")
    set_tufte_title(ax1, r"Selection: $\Delta_t$ spikes on content")
    set_tufte_labels(ax1, xlabel="step $t$", ylabel="magnitude")
    ax1.legend(loc="upper right", fontsize=8, frameon=False)

    ax2.plot(t, h_sel, "o-", color=SSM_COLORS["accent"], markersize=3, label="selective (holds)")
    ax2.plot(t, h_lti, "s-", color=SSM_COLORS["alert"], markersize=3, label="LTI (forgets)")
    set_tufte_title(ax2, "Same impulse, opposite memory")
    set_tufte_labels(ax2, xlabel="step $t$", ylabel=r"state $y_t = h_t$")
    ax2.legend(loc="upper right", fontsize=8, frameon=False)

    fig.tight_layout()
    return fig


def main() -> None:
    import matplotlib.pyplot as plt
    import numpy as np

    from companions._shared.plot_utils import save_figure

    print("Chapter 9 — selective_vs_lti.py")
    print("=" * 60)

    u, delta_sel, h_sel, h_lti = selective_vs_lti_states()
    written_sel = float(h_sel[_WRITE_STEP])
    written_lti = float(h_lti[_WRITE_STEP])
    ret_sel = float(h_sel[-1]) / written_sel
    ret_lti = float(h_lti[-1]) / written_lti
    print(f"  write step t={_WRITE_STEP}, read step t={_LENGTH - 1} ({_LENGTH - 1 - _WRITE_STEP} steps later)")
    print(f"  selective retains {100 * ret_sel:.1f}% of the written value  (gate held open->closed)")
    print(f"  LTI       retains {100 * ret_lti:.3f}% of the written value  (fixed decay forgets)")

    _OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig = make_selectivity_figure()
    for p in save_figure(fig, _OUT_DIR / "selective-vs-lti", formats=("png",)):
        print(f"Wrote {p}")
    plt.close(fig)


if __name__ == "__main__":
    main()
