"""Shared matplotlib style and plots, used by notebooks and generated reports.

Colours follow a fixed role mapping (never cycled): control is neutral grey,
treated arms take categorical slots 1 and 2.
"""

from __future__ import annotations

from collections.abc import Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.ticker import FuncFormatter

INK = "#0b0b0b"
INK_MUTED = "#52514e"
GRID = "#e4e3df"
SURFACE = "#fcfcfb"
NEUTRAL = "#9a9893"  # control arm / reference marks
SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]  # categorical slots 1-3 (CVD-validated)
WARN = "#e34948"

ARM_COLORS = {
    "No E-Mail": NEUTRAL,
    "Mens E-Mail": SERIES[0],
    "Womens E-Mail": SERIES[1],
    "0": NEUTRAL,
    "1": SERIES[0],
}


def apply_style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "savefig.facecolor": SURFACE,
            "axes.edgecolor": GRID,
            "axes.labelcolor": INK_MUTED,
            "axes.titlecolor": INK,
            "axes.titleweight": "bold",
            "axes.titlesize": 12,
            "axes.titlelocation": "left",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "axes.axisbelow": True,
            "grid.color": GRID,
            "grid.linewidth": 0.8,
            "xtick.color": INK_MUTED,
            "ytick.color": INK_MUTED,
            "font.size": 10,
            "legend.frameon": False,
            "figure.dpi": 110,
            "savefig.dpi": 150,
            "savefig.bbox": "tight",
        }
    )


def _pct(ax: Axes, axis: str = "y", decimals: int = 0, unit: str = "%") -> None:
    sep = " " if unit == "pp" else ""
    fmt = FuncFormatter(
        lambda v, _: (
            f"{v * 100:+.{decimals}f}{sep}{unit}"
            if unit == "pp"
            else f"{v * 100:.{decimals}f}{unit}"
        )
    )
    (ax.yaxis if axis == "y" else ax.xaxis).set_major_formatter(fmt)


def plot_rates_by_arm(rates: pd.DataFrame, outcomes: Sequence[str], title: str) -> Figure:
    """One small panel per outcome, a bar per arm, values labelled on the bars."""
    fig, axes = plt.subplots(1, len(outcomes), figsize=(3.6 * len(outcomes), 3.4))
    axes = np.atleast_1d(axes)
    order = sorted(rates.index, key=lambda a: (ARM_COLORS.get(a) != NEUTRAL, a))
    for ax, outcome in zip(axes, outcomes, strict=True):
        values = rates.loc[order, outcome]
        colors = [ARM_COLORS.get(a, SERIES[2]) for a in order]
        bars = ax.bar(order, values, color=colors, width=0.6, edgecolor=SURFACE, linewidth=2)
        is_rate = values.max() <= 1
        for bar, v in zip(bars, values, strict=True):
            label = f"{v * 100:.2f}%" if is_rate else f"{v:.2f}"
            ax.annotate(
                label,
                (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                ha="center",
                va="bottom",
                fontsize=9,
                color=INK,
                xytext=(0, 3),
                textcoords="offset points",
            )
        ax.set_title(outcome)
        ax.grid(axis="x", visible=False)
        ax.tick_params(axis="x", labelsize=8)
        if is_rate:
            _pct(ax, decimals=1)
        ax.margins(y=0.15)
    fig.suptitle(title, x=0.01, ha="left", fontweight="bold", color=INK)
    fig.tight_layout()
    return fig


def plot_effect_forest(effects: pd.DataFrame, title: str, xlabel: str) -> Figure:
    """Forest plot: point estimate + 95% CI per segment level, one colour per arm."""
    arms = list(dict.fromkeys(effects["arm"]))
    rows = list(dict.fromkeys(zip(effects["feature"], effects["level"], strict=True)))
    fig, ax = plt.subplots(figsize=(7.5, 0.22 * len(rows) * len(arms) + 1.4))
    offsets = np.linspace(-0.18, 0.18, len(arms)) if len(arms) > 1 else [0.0]
    for arm, off in zip(arms, offsets, strict=True):
        sub = effects[effects["arm"] == arm].set_index(["feature", "level"]).loc[rows]
        y = np.arange(len(rows)) + off
        color = ARM_COLORS.get(arm, SERIES[0])
        ax.hlines(y, sub["ci_low"], sub["ci_high"], color=color, linewidth=2)
        ax.plot(sub["ate"], y, "o", color=color, markersize=6, label=arm,
                markeredgecolor=SURFACE, markeredgewidth=1.5)  # fmt: skip
    ax.axvline(0, color=INK_MUTED, linewidth=1)
    ax.set_yticks(np.arange(len(rows)), [f"{f} = {lvl}" for f, lvl in rows])
    ax.invert_yaxis()
    ax.grid(axis="y", visible=False)
    _pct(ax, axis="x", decimals=0, unit="pp")
    ax.set_xlabel(xlabel)
    ax.set_title(title)
    if len(arms) > 1:
        ax.legend(loc="upper right")
    fig.tight_layout()
    return fig


def plot_balance(balance: pd.DataFrame, title: str, threshold: float = 0.1) -> Figure:
    """|SMD| per feature against the 0.1 balance threshold."""
    b = balance.sort_values("abs_smd")
    fig, ax = plt.subplots(figsize=(6.5, 0.28 * len(b) + 1.2))
    colors = [NEUTRAL if ok else WARN for ok in b["balanced"]]
    ax.hlines(b["feature"], 0, b["abs_smd"], color=colors, linewidth=2)
    ax.plot(b["abs_smd"], b["feature"], "o", color=SERIES[0], markersize=6)
    ax.axvline(threshold, color=WARN, linestyle="--", linewidth=1)
    ax.annotate(
        f"imbalance threshold {threshold}",
        (threshold, 0.02),
        xycoords=("data", "axes fraction"),
        ha="right",
        va="bottom",
        fontsize=8,
        color=INK_MUTED,
        xytext=(-4, 2),
        textcoords="offset points",
    )
    ax.set_xlim(0, max(threshold * 1.3, float(b["abs_smd"].max()) * 1.1))
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("|standardised mean difference|  treated vs control")
    ax.set_title(title)
    fig.tight_layout()
    return fig
