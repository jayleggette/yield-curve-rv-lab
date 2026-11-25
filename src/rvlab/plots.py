"""Static figures for the README (matplotlib).

Same visual rules as the companion FOMC project: one y-axis per panel,
2px lines, hairline solid grid, validated categorical colours in fixed order,
a legend whenever two or more series share a panel, neutral text, and a CSV
twin of each plotted series in outputs/tables.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

SURFACE, INK, INK2, MUTED = "#fcfcfb", "#0b0b0b", "#52514e", "#898781"
GRID, AXIS, BAND = "#e1e0d9", "#c3c2b7", "#f0efec"
S1, S2, S3 = "#2a78d6", "#eb6834", "#1baf7a"


def style() -> None:
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "axes.edgecolor": AXIS, "axes.linewidth": 0.8, "axes.labelcolor": INK2,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "axes.grid.axis": "y", "grid.color": GRID, "grid.linewidth": 0.8,
        "grid.linestyle": "-", "xtick.color": MUTED, "ytick.color": MUTED,
        "xtick.labelcolor": INK2, "ytick.labelcolor": INK2,
        "font.family": "DejaVu Sans", "font.size": 9.5, "text.color": INK,
        "axes.titlesize": 11, "axes.titleweight": "semibold", "axes.titlelocation": "left",
        "legend.frameon": False, "legend.fontsize": 9,
        "lines.linewidth": 1.6, "lines.solid_capstyle": "round", "lines.solid_joinstyle": "round",
    })


def _finish(fig, path: Path, source: str) -> None:
    fig.text(0.01, -0.01, source, fontsize=7.5, color=MUTED, ha="left", va="top")
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _years(ax, step: int = 4) -> None:
    ax.xaxis.set_major_locator(mdates.YearLocator(step))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))


def curve_snapshots(snaps: list[tuple[str, pd.Series, np.ndarray, np.ndarray]], path: Path, rmse_bp: float) -> None:
    """Observed CMT points and fitted Nelson-Siegel zero curves on a few dates."""
    style()
    fig, ax = plt.subplots(figsize=(8.6, 4.4))
    ax.grid(axis="x", visible=True)
    for (label, obs, grid_t, fit), color in zip(snaps, (S1, S2, S3)):
        ax.plot(grid_t, fit, color=color, label=f"{label} (fitted zero curve)")
        ax.scatter(obs.index.values.astype(float), obs.values, s=34, color=color, edgecolor=SURFACE,
                   lw=1.6, zorder=3)
    ax.set_xlabel("Maturity, years")
    ax.set_ylabel("Yield, percent")
    ax.set_xticks([0.25, 1, 2, 3, 5, 7, 10])
    ax.set_xticklabels(["3m", "1y", "2y", "3y", "5y", "7y", "10y"])
    ax.set_title("The Treasury curve: observed yields (dots) and Nelson-Siegel fits")
    ax.legend(loc="lower right")
    _finish(fig, path, f"Constant-maturity yields from FRED. Fitted to par yields for notes, zero yields for bills. "
                       f"Median absolute fitting error since 1985: {rmse_bp:.1f} bp.")


def factors(fm: pd.DataFrame, path: Path) -> None:
    style()
    fig, axes = plt.subplots(3, 1, figsize=(9, 6.2), sharex=True, gridspec_kw={"hspace": 0.28})
    labels = {"level": "Level (long-run yield), %", "slope": "Slope (short minus long), pp",
              "curvature": "Curvature (belly), pp"}
    for ax, col, color in zip(axes, ["level", "slope", "curvature"], (S1, S2, S3)):
        ax.plot(fm.index, fm[col], color=color)
        ax.axhline(0, color=AXIS, lw=0.8)
        ax.set_title(labels[col])
    _years(axes[-1])
    _finish(fig, path, "Month-end Nelson-Siegel factors (Diebold-Li decay, 0.7308 per year), fitted to par yields.")


def pca_loadings(load: pd.DataFrame, share: np.ndarray, path: Path) -> None:
    style()
    fig, ax = plt.subplots(figsize=(8.2, 4.0))
    ax.grid(axis="x", visible=True)
    ax.axhline(0, color=AXIS, lw=0.8)
    names = {"PC1": "Level", "PC2": "Slope", "PC3": "Curvature"}
    x = load.index.values.astype(float)
    for col, color, s in zip(load.columns, (S1, S2, S3), share):
        ax.plot(x, load[col], color=color, label=f"{names[col]}: {s * 100:.1f}% of variance")
        ax.scatter(x, load[col], s=26, color=color, edgecolor=SURFACE, lw=1.4, zorder=3)
    ax.set_xticks([0.25, 1, 2, 3, 5, 7, 10])
    ax.set_xticklabels(["3m", "1y", "2y", "3y", "5y", "7y", "10y"])
    ax.set_xlabel("Maturity")
    ax.set_ylabel("Loading")
    ax.set_title("Three factors explain almost all daily curve moves")
    ax.legend(loc="lower right")
    _finish(fig, path, "Principal components of daily yield changes, 1985 to 2026.")


def term_premium(tp: pd.DataFrame, kw: pd.Series | None, path: Path) -> None:
    style()
    fig, (ax, bx) = plt.subplots(2, 1, figsize=(9, 5.8), sharex=True,
                                 gridspec_kw={"height_ratios": [1.4, 1], "hspace": 0.18})
    ax.plot(tp.index, tp["yield"], color=S1, label="10-year zero yield (fitted)")
    ax.plot(tp.index, tp["risk_neutral"], color=S2, label="Average expected short rate, next 10 years")
    ax.set_ylabel("Percent")
    ax.set_title("The 10-year yield split into expected rates and term premium")
    ax.legend(loc="upper right")
    bx.axhline(0, color=AXIS, lw=0.8)
    bx.plot(tp.index, tp["term_premium"], color=S1, label="This model")
    if kw is not None:
        k = kw.resample("MS").last().reindex(tp.index)
        bx.plot(k.index, k, color=MUTED, label="Kim-Wright (Fed Board)")
        bx.legend(loc="upper right")
    bx.set_ylabel("Term premium, pp")
    _years(bx)
    _finish(fig, path, "VAR(1) on month-end Nelson-Siegel factors; short rate = fitted 1-month zero yield.")


def fly_panel(bt: pd.DataFrame, entry: float, split: str, path: Path, name: str) -> None:
    style()
    fig, (ax, bx) = plt.subplots(2, 1, figsize=(9, 5.6), sharex=True, gridspec_kw={"hspace": 0.2})
    ax.plot(bt.index, bt["z"], color=MUTED, lw=0.8)
    for lvl in (entry, -entry):
        ax.axhline(lvl, color=S2, lw=0.9)
    ax.axhline(0, color=AXIS, lw=0.8)
    ax.annotate(f"Entry at |z| > {entry:g}", (bt.index[30], entry), xytext=(0, 4), textcoords="offset points",
                fontsize=8, color=INK2)
    ax.set_ylabel("z-score")
    ax.set_title(f"{name} butterfly: curvature residual z-score")
    cum = bt["net"].cumsum()
    gross = (bt["gross"] + bt["carry"]).cumsum()
    bx.plot(cum.index, gross, color=MUTED, lw=1.2, label="Before costs")
    bx.plot(cum.index, cum, color=S1, label="After 0.25 bp per leg per trade")
    s = pd.Timestamp(split)
    for a in (ax, bx):
        a.axvline(s, color=INK2, lw=0.8)
    bx.annotate("Tuned on this period", (s, cum.max()), xytext=(-6, -2), textcoords="offset points",
                ha="right", va="top", fontsize=8, color=INK2)
    bx.annotate("Out of sample", (s, cum.max()), xytext=(6, -2), textcoords="offset points",
                ha="left", va="top", fontsize=8, color=INK2)
    bx.set_ylabel("Cumulative P&L, bp of body DV01")
    bx.set_title("Cumulative P&L")
    bx.legend(loc="lower right")
    _years(bx)
    _finish(fig, path, "PCA weights re-estimated monthly from the prior two years; signals traded one day late; "
                       "carry and roll-down included.")


def cum_pnl_flies(bts: dict[str, pd.DataFrame], split: str, path: Path) -> None:
    """Cumulative net P&L of each butterfly, tuning period and out of sample."""
    style()
    S4 = "#eda100"
    fig, ax = plt.subplots(figsize=(9, 4.6))
    ax.axhline(0, color=AXIS, lw=0.8)
    s = pd.Timestamp(split)
    ax.axvline(s, color=INK2, lw=0.8)
    for (name, bt), color in zip(bts.items(), (S1, S2, S3, S4)):
        cum = bt["net"].cumsum()
        ax.plot(cum.index, cum, color=color, label=name)
    lo = min(bt["net"].cumsum().min() for bt in bts.values())
    hi = max(bt["net"].cumsum().max() for bt in bts.values())
    ax.set_ylim(min(lo, 0) - 0.05 * (hi - lo), hi + 0.16 * (hi - lo))
    ytxt = hi + 0.13 * (hi - lo)
    ax.annotate("Rules tuned here (2s5s10s only)", (s, ytxt), xytext=(-6, 0), textcoords="offset points",
                ha="right", va="center", fontsize=8, color=INK2)
    ax.annotate("Out of sample, rules unchanged", (s, ytxt), xytext=(6, 0), textcoords="offset points",
                ha="left", va="center", fontsize=8, color=INK2)
    ax.set_ylabel("Cumulative P&L, bp of body DV01")
    ax.set_title("PCA butterfly mean reversion: cumulative P&L after costs")
    ax.legend(loc="lower left", ncol=4)
    _years(ax)
    _finish(fig, path, "0.25 bp per leg per trade; carry and roll-down included; weights re-estimated monthly from "
                       "the prior two years; signals traded one day late.")


def sharpe_by_fly(tab: pd.DataFrame, path: Path) -> None:
    style()
    fig, ax = plt.subplots(figsize=(8.2, 3.4))
    ax.axhline(0, color=AXIS, lw=0.8)
    flies = list(tab.index)
    xpos = np.arange(len(flies))
    wdt = 0.3
    for k, (col, color, label) in enumerate([("in_sample", MUTED, "Tuning period (1987 to 2007)"),
                                             ("out_of_sample", S1, "Out of sample (2008 to 2026)")]):
        vals = tab[col].values
        ax.bar(xpos + (k - 0.5) * (wdt + 0.04), vals, width=wdt, color=color, label=label)
        for xv, v in zip(xpos + (k - 0.5) * (wdt + 0.04), vals):
            ax.annotate(f"{v:.2f}", (xv, v), xytext=(0, 3 if v >= 0 else -11), textcoords="offset points",
                        ha="center", fontsize=8.5, color=INK)
    ax.set_xticks(xpos)
    ax.set_xticklabels(flies)
    ax.set_ylabel("Sharpe ratio, after costs")
    ax.set_title("Same rules on four butterflies: in sample vs out of sample")
    ax.legend(loc="upper right", ncol=2)
    lo = min(0, tab[["in_sample", "out_of_sample"]].values.min())
    ax.set_ylim(lo * 1.3 if lo < 0 else 0, tab[["in_sample", "out_of_sample"]].values.max() * 1.35)
    _finish(fig, path, "Parameters chosen once, on the 2s5s10s fly in the tuning period, then applied unchanged.")
