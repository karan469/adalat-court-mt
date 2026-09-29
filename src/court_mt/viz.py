"""Chart theme and figures for the tokenizer study.

Colour roles follow one rule each, so nothing is picked by eye:

* **Magnitude** (one measure ranked across tokenizers) is a single sequential
  blue. One series needs no legend, the title names it.
* **Identity** (English vs Hindi, or one line per base model) uses the
  categorical slots in fixed order, never cycled, and always with a legend.
* **Ink is ink.** Values, labels and ticks wear text tokens; colour belongs to
  the mark beside them, never to the number.

The palette is the data-viz reference instance, used unchanged. Only the first
three categorical slots are ever used here, which is the subset documented as
clearing the all-pairs colour-vision-deficiency floors in both light and dark
modes (worst pair ΔE 9.2 light / 9.4 dark, against a >=8 target). Adding a fourth
series would put yellow next to orange and break that, so a fourth dimension gets
a facet or an "other" bucket instead.

Deliberate deviation from the mark spec: matplotlib has no rounded bar ends, so
bars are square. The 2px surface gap between adjacent bars is honoured via bar
width, and it is the gap, not the rounding, that does the readability work.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd


@dataclass(frozen=True)
class Theme:
    """Every colour and size a figure needs, for one rendering mode."""

    surface: str
    text_primary: str
    text_secondary: str
    muted: str
    gridline: str
    baseline: str
    categorical: tuple[str, ...]
    sequential: str
    reference: str = "#d03b3b"
    bar_gap: float = 0.42
    line_width: float = 2.0
    marker_size: float = 8.0
    font_stack: tuple[str, ...] = field(
        default=("system-ui", "-apple-system", "Segoe UI", "Helvetica Neue", "DejaVu Sans")
    )


LIGHT = Theme(
    surface="#fcfcfb",
    text_primary="#0b0b0b",
    text_secondary="#52514e",
    muted="#898781",
    gridline="#e1e0d9",
    baseline="#c3c2b7",
    categorical=("#2a78d6", "#eb6834", "#1baf7a"),
    sequential="#2a78d6",
)

DARK = Theme(
    surface="#1a1a19",
    text_primary="#ffffff",
    text_secondary="#c3c2b7",
    muted="#898781",
    gridline="#2c2c2a",
    baseline="#383835",
    categorical=("#3987e5", "#d95926", "#199e70"),
    sequential="#3987e5",
)


def use(theme: Theme = LIGHT) -> Theme:
    """Install ``theme`` as the matplotlib default and return it."""
    mpl.rcParams.update(
        {
            "figure.facecolor": theme.surface,
            "axes.facecolor": theme.surface,
            "savefig.facecolor": theme.surface,
            "savefig.bbox": "tight",
            "savefig.dpi": 160,
            "figure.dpi": 110,
            "font.family": "sans-serif",
            "font.sans-serif": list(theme.font_stack),
            "font.size": 10,
            "text.color": theme.text_primary,
            "axes.labelcolor": theme.text_secondary,
            "axes.edgecolor": theme.baseline,
            "axes.titlecolor": theme.text_primary,
            "axes.titlesize": 12,
            "axes.titleweight": "semibold",
            "axes.titlelocation": "left",
            "axes.titlepad": 14,
            "axes.grid": True,
            "axes.axisbelow": True,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "grid.color": theme.gridline,
            "grid.linewidth": 0.8,
            "xtick.color": theme.muted,
            "ytick.color": theme.muted,
            "xtick.labelcolor": theme.text_secondary,
            "ytick.labelcolor": theme.text_secondary,
            "xtick.direction": "out",
            "legend.frameon": False,
            "legend.fontsize": 9,
            "lines.linewidth": theme.line_width,
            "lines.markersize": theme.marker_size / 2,
        }
    )
    return theme


def _finish(ax: plt.Axes, theme: Theme, *, value_axis: str, subtitle: str = "") -> None:
    """Apply the shared chrome rules: grid on the value axis only, recessive axes."""
    ax.grid(axis=value_axis, color=theme.gridline, linewidth=0.8)
    ax.grid(axis="y" if value_axis == "x" else "x", visible=False)
    ax.tick_params(length=0)
    if subtitle:
        ax.text(
            0.0,
            1.02,
            subtitle,
            transform=ax.transAxes,
            color=theme.text_secondary,
            fontsize=9.5,
            va="bottom",
        )


def ranked_bar(
    data: pd.Series,
    *,
    title: str,
    subtitle: str = "",
    value_format: str = "{:.2f}",
    reference: float | None = None,
    reference_label: str = "",
    theme: Theme = LIGHT,
    ax: plt.Axes | None = None,
    highlight: str | None = None,
) -> plt.Axes:
    """Horizontal ranked bars for a single measure, the magnitude workhorse.

    Args:
        data: measure indexed by tokenizer label, plotted in the order given.
        reference: optional vertical line marking a meaningful threshold
            (token parity at 1.0, a 1024-token context cap).
        highlight: index label to emphasise; others recede to a lighter step.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(7.4, 0.42 * len(data) + 1.9))

    colors = [
        theme.sequential if (highlight is None or label == highlight) else theme.gridline
        for label in data.index
    ]
    positions = range(len(data))
    ax.barh(list(positions), data.to_numpy(), height=1 - theme.bar_gap, color=colors, zorder=3)
    ax.set_yticks(list(positions), list(data.index))
    ax.invert_yaxis()

    if reference is not None:
        ax.axvline(reference, color=theme.reference, linewidth=1.5, linestyle=(0, (4, 3)), zorder=4)
        if reference_label:
            # x in data coordinates, y in axes fraction: the label tracks the line
            # horizontally but stays inside the plot, clear of the title block.
            ax.text(
                reference,
                0.985,
                f" {reference_label}",
                transform=ax.get_xaxis_transform(),
                color=theme.reference,
                fontsize=9,
                va="top",
                ha="left",
                zorder=4,
            )

    span = float(data.max())
    for position, value in zip(positions, data.to_numpy()):
        ax.text(
            value + span * 0.012,
            position,
            value_format.format(value),
            va="center",
            ha="left",
            fontsize=9,
            color=theme.text_secondary,
        )
    ax.set_xlim(0, span * 1.16)
    ax.set_title(title)
    _finish(ax, theme, value_axis="x", subtitle=subtitle)
    return ax


def grouped_bar(
    data: pd.DataFrame,
    *,
    title: str,
    subtitle: str = "",
    value_format: str = "{:.2f}",
    theme: Theme = LIGHT,
    ax: plt.Axes | None = None,
) -> plt.Axes:
    """Horizontal grouped bars for 2-3 series sharing one scale.

    Columns are the series (identity -> categorical slots, in order); the index is
    the tokenizer. A legend is always drawn, so identity never rests on colour
    alone.
    """
    if len(data.columns) > len(theme.categorical):
        raise ValueError(
            f"{len(data.columns)} series exceeds the {len(theme.categorical)}-slot "
            "validated palette; facet instead of cycling hues"
        )
    if ax is None:
        _, ax = plt.subplots(figsize=(7.8, 0.52 * len(data) + 2.1))

    n_series = len(data.columns)
    slot_height = (1 - theme.bar_gap) / n_series
    for offset, (name, column) in enumerate(data.items()):
        positions = [i + (offset - (n_series - 1) / 2) * slot_height for i in range(len(data))]
        ax.barh(
            positions,
            column.to_numpy(),
            height=slot_height * 0.9,
            color=theme.categorical[offset],
            label=str(name),
            zorder=3,
        )
        for position, value in zip(positions, column.to_numpy()):
            ax.text(
                value + float(data.to_numpy().max()) * 0.012,
                position,
                value_format.format(value),
                va="center",
                ha="left",
                fontsize=8.5,
                color=theme.text_secondary,
            )

    ax.set_yticks(range(len(data)), list(data.index))
    ax.invert_yaxis()
    ax.set_xlim(0, float(data.to_numpy().max()) * 1.18)
    ax.set_title(title)
    ax.legend(loc="lower right", ncols=n_series)
    _finish(ax, theme, value_axis="x", subtitle=subtitle)
    return ax


def trend(
    data: pd.DataFrame,
    *,
    title: str,
    xlabel: str,
    ylabel: str,
    subtitle: str = "",
    theme: Theme = LIGHT,
    ax: plt.Axes | None = None,
    annotate_last: bool = True,
) -> plt.Axes:
    """One line per column over a shared numeric index.

    Used for the vocabulary-extension sweep, where the x axis is "pieces added".
    Only the final point of each line is labelled, a number on every point turns
    a trend into a table.
    """
    if len(data.columns) > len(theme.categorical):
        raise ValueError("too many series for the validated palette; facet instead")
    if ax is None:
        _, ax = plt.subplots(figsize=(7.4, 4.4))

    for offset, (name, column) in enumerate(data.items()):
        color = theme.categorical[offset]
        ax.plot(data.index, column.to_numpy(), color=color, marker="o", label=str(name), zorder=3)
        if annotate_last:
            ax.annotate(
                f"{column.iloc[-1]:.3f}",
                (data.index[-1], column.iloc[-1]),
                textcoords="offset points",
                xytext=(8, 0),
                color=theme.text_secondary,
                fontsize=9,
                va="center",
            )

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if len(data.columns) > 1:
        ax.legend(loc="best")
    ax.margins(x=0.10)
    _finish(ax, theme, value_axis="y", subtitle=subtitle)
    return ax


def save(fig: plt.Figure | plt.Axes, path: Path) -> Path:
    """Write a figure to ``path``, creating parent directories."""
    figure = fig.figure if isinstance(fig, plt.Axes) else fig
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path)
    return path


def histogram(
    values: "pd.Series",
    *,
    title: str,
    xlabel: str,
    subtitle: str = "",
    bins: int = 40,
    reference: float | None = None,
    reference_label: str = "",
    theme: Theme = LIGHT,
    ax: plt.Axes | None = None,
) -> plt.Axes:
    """Distribution of a single measure, the shape, not the summary.

    Used where a mean would mislead: alignment quality lives in the tails, and a
    median length ratio of 0.93 says nothing about how many pairs sit at 0.3.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(7.4, 3.8))

    ax.hist(values.to_numpy(), bins=bins, color=theme.sequential, zorder=3)

    if reference is not None:
        ax.axvline(reference, color=theme.reference, linewidth=1.5, linestyle=(0, (4, 3)), zorder=4)
        if reference_label:
            ax.text(
                reference,
                0.985,
                f" {reference_label}",
                transform=ax.get_xaxis_transform(),
                color=theme.reference,
                fontsize=9,
                va="top",
                ha="left",
                zorder=4,
            )

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("pairs")
    _finish(ax, theme, value_axis="y", subtitle=subtitle)
    return ax
