"""Tufte-inspired matplotlib utilities for ssm-foundations companion figures.

Minimal port of ``post_transformers/guides/shared/plot_utils.py``, keeping only
the helpers Ch 1–3 figures actually use. Color palette is Paul Tol High
Contrast (CVD-safe + grayscale-distinguishable) matching the book-scaffold-astro
academic preset's callout colors.

Authoritative style file: ``post_transformers_mpl.mplstyle`` (sibling). Apply
it once per script via :func:`apply_style`.

Reference
---------
- Tufte, E. R. (1983). The Visual Display of Quantitative Information.
- Paul Tol's qualitative color schemes (https://personal.sron.nl/~pault/).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal, overload

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure

__all__ = [
    "SSM_COLORS",
    "TUFTE_PALETTE",
    "FIG_SINGLE",
    "FIG_DOUBLE",
    "FIG_TRIPLE",
    "apply_style",
    "apply_tufte_style",
    "create_tufte_figure",
    "set_tufte_title",
    "set_tufte_labels",
    "save_figure",
]

# Paul Tol High Contrast — CVD-safe, matches the academic preset's callout palette.
SSM_COLORS: dict[str, str] = {
    "accent": "#004488",  # Navy — primary data
    "highlight": "#DDAA33",  # Gold — secondary / results
    "alert": "#BB5566",  # Dusty rose — limitations / boundaries
    "baseline": "#999999",  # Gray — reference lines
}

# Structural greys for spines, grids, text.
TUFTE_PALETTE: dict[str, str] = {
    "spine": "#cccccc",
    "grid": "#e5e5e5",
    "text": "#333333",
    "text_secondary": "#666666",
    "background": "#fafafa",
}

# Standard figure sizes (width, height) in inches.
FIG_SINGLE: tuple[float, float] = (6.0, 3.5)
FIG_DOUBLE: tuple[float, float] = (10.0, 4.0)
FIG_TRIPLE: tuple[float, float] = (14.0, 4.5)

_STYLE_PATH = Path(__file__).resolve().parent / "post_transformers_mpl.mplstyle"


def apply_style() -> None:
    """Apply the shared post_transformers matplotlib style.

    Reads the sibling ``post_transformers_mpl.mplstyle`` file. Call once at
    the top of any script that emits figures for the book.

    Raises
    ------
    FileNotFoundError
        If the style file is missing from the expected sibling path.
    """
    if not _STYLE_PATH.exists():
        raise FileNotFoundError(
            f"post_transformers_mpl.mplstyle not found at {_STYLE_PATH}. "
            "Ensure it sits alongside plot_utils.py in companions/_shared/."
        )
    plt.style.use(str(_STYLE_PATH))


def apply_tufte_style(ax: Axes) -> Axes:
    """Apply Tufte spine + tick styling to a single axes.

    Removes top/right spines, lightens left/bottom spines, refines tick
    colors. Use when the chart doesn't need a grid (high data-ink ratio).

    Parameters
    ----------
    ax : Axes
        The axes to style.

    Returns
    -------
    Axes
        The same axes, for chaining.
    """
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(TUFTE_PALETTE["spine"])
    ax.spines["bottom"].set_color(TUFTE_PALETTE["spine"])
    ax.spines["left"].set_linewidth(0.8)
    ax.spines["bottom"].set_linewidth(0.8)
    ax.tick_params(
        colors=TUFTE_PALETTE["text_secondary"],
        length=4,
        width=0.8,
        direction="out",
    )
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_color(TUFTE_PALETTE["text_secondary"])
        label.set_fontsize(9)
    return ax


@overload
def create_tufte_figure(
    nrows: Literal[1] = 1,
    ncols: Literal[1] = 1,
    figsize: tuple[float, float] | None = None,
    **kwargs: Any,
) -> tuple[Figure, Axes]: ...
@overload
def create_tufte_figure(
    nrows: int = 1,
    ncols: int = 1,
    figsize: tuple[float, float] | None = None,
    **kwargs: Any,
) -> tuple[Figure, np.ndarray]: ...
def create_tufte_figure(
    nrows: int = 1,
    ncols: int = 1,
    figsize: tuple[float, float] | None = None,
    **kwargs: Any,
) -> tuple[Figure, Axes | np.ndarray]:
    """Create a figure with Tufte styling pre-applied to every axes.

    Parameters
    ----------
    nrows, ncols : int
        Subplot grid dimensions. Defaults to a single-panel 1×1.
    figsize : tuple of float, optional
        Figure size in inches. Defaults to ``(6*ncols, 3.5*nrows)``.
    **kwargs
        Forwarded to :func:`matplotlib.pyplot.subplots`.

    Returns
    -------
    fig : Figure
    axes : Axes or ndarray of Axes
        Single :class:`Axes` for a 1×1 grid; ndarray otherwise.
    """
    if figsize is None:
        figsize = (6.0 * ncols, 3.5 * nrows)
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, **kwargs)
    fig.patch.set_facecolor(TUFTE_PALETTE["background"])
    if isinstance(axes, np.ndarray):
        for ax in axes.flat:
            apply_tufte_style(ax)
            ax.set_facecolor(TUFTE_PALETTE["background"])
    else:
        apply_tufte_style(axes)
        axes.set_facecolor(TUFTE_PALETTE["background"])
    return fig, axes


def set_tufte_title(ax: Axes, title: str, **kwargs: Any) -> None:
    """Set a left-aligned, understated title (Tufte style).

    Parameters
    ----------
    ax : Axes
    title : str
    **kwargs
        Forwarded to :func:`matplotlib.axes.Axes.set_title`.
    """
    defaults = {
        "fontsize": 11,
        "fontweight": "normal",
        "color": TUFTE_PALETTE["text"],
        "loc": "left",
        "pad": 10,
    }
    defaults.update(kwargs)
    ax.set_title(title, **defaults)


def set_tufte_labels(
    ax: Axes,
    xlabel: str | None = None,
    ylabel: str | None = None,
    **kwargs: Any,
) -> None:
    """Set Tufte-styled axis labels (lighter color, slightly smaller).

    Parameters
    ----------
    ax : Axes
    xlabel, ylabel : str, optional
        Axis labels; passes through unset axes.
    **kwargs
        Forwarded to :func:`matplotlib.axes.Axes.set_xlabel`/`set_ylabel`.
    """
    defaults = {"fontsize": 10, "color": TUFTE_PALETTE["text_secondary"]}
    defaults.update(kwargs)
    if xlabel:
        ax.set_xlabel(xlabel, **defaults)
    if ylabel:
        ax.set_ylabel(ylabel, **defaults)


def save_figure(
    fig: Figure,
    output_path: Path | str,
    formats: tuple[str, ...] = ("png",),
    dpi: int = 300,
) -> list[Path]:
    """Save a figure to one or more formats.

    Parameters
    ----------
    fig : Figure
    output_path : Path or str
        Path *without* extension. Parent directory is created if missing.
    formats : tuple of str
        File extensions to emit (e.g., ``("png",)`` or ``("png", "svg")``).
        Default: PNG only (smallest acceptable for web book builds).
    dpi : int
        Resolution for raster formats. Default 300 (print quality).

    Returns
    -------
    list of Path
        Paths to the written files.

    Raises
    ------
    ValueError
        If ``formats`` is empty.
    """
    if not formats:
        raise ValueError("formats must be non-empty (e.g., ('png',))")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    written: list[Path] = []
    for fmt in formats:
        path = output_path.with_suffix(f".{fmt}")
        fig.savefig(path, bbox_inches="tight", dpi=dpi)
        written.append(path)
    return written
