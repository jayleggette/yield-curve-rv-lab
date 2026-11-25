"""PCA-weighted butterfly relative-value backtest.

The trade
---------
A butterfly is long the body and short the two wings (or the reverse). The
weights are set in DV01 terms so the position has zero exposure to the first
two principal components of curve moves (level and slope). What is left is
the curvature residual, which tends to mean-revert:

    F_t = 100 * (w_a * y_a + y_b + w_c * y_c)        in basis points

High F means the body is cheap against the wings, so the strategy goes long
the body (pos = +1) and earns -dF. Each day's P&L is in basis points of the
body's DV01: with $10,000 of body DV01, 1 bp of P&L is $10,000.

No look-ahead
-------------
* Weights are re-estimated at each month start from the trailing two years of
  daily changes that ended the day before.
* The z-score uses the weights in force that day over a trailing window.
* A signal seen at the close of day t is traded at the close of t+1, so it
  first earns on day t+2.

Costs and carry
---------------
* Transaction cost per leg per trade (default 0.25 bp of yield), charged on
  every entry, exit and monthly weight rebalance while a position is on.
* Carry (leg yield minus the 3-month bill as a funding proxy) and roll-down
  (the slope of the curve under each leg) accrue daily on the held legs.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .pca import pca

FLIES = {
    "2s5s10s": (2.0, 5.0, 10.0),
    "2s3s5s": (2.0, 3.0, 5.0),
    "3s5s7s": (3.0, 5.0, 7.0),
    "5s7s10s": (5.0, 7.0, 10.0),
}
GRID = [{"entry": e, "exit": x, "lookback": L}
        for e in (1.0, 1.5, 2.0) for x in (0.0, 0.5) for L in (126, 252, 504)]
MAX_HOLD = 63          # trading days (about a quarter)
PCA_WINDOW = 504       # trading days of daily changes (about two years)
DAYS = 252


# --------------------------------------------------------------------------
# Weights and the fly series
# --------------------------------------------------------------------------
def fly_weights(load: pd.DataFrame, fly: tuple[float, float, float]) -> np.ndarray:
    """DV01 weights (wing a, body = 1, wing c) neutral to PC1 and PC2."""
    a, b, c = fly
    L = load.loc[[a, b, c], ["PC1", "PC2"]].values
    A = np.column_stack([L[0], L[2]])
    wa, wc = np.linalg.solve(A, -L[1])
    return np.array([wa, 1.0, wc])


def rolling_weights(curve: pd.DataFrame, fly, window: int = PCA_WINDOW) -> pd.DataFrame:
    """Monthly re-estimated weights, forward-filled to every trading day."""
    chg = curve.diff().dropna() * 100.0
    rows = {}
    for d in pd.date_range(curve.index[0], curve.index[-1], freq="MS"):
        hist = chg[chg.index < d].tail(window)
        if len(hist) < window:
            continue
        load, _ = pca(hist, 3)
        rows[d] = fly_weights(load, fly)
    w = pd.DataFrame.from_dict(rows, orient="index", columns=["wa", "wb", "wc"]).sort_index()
    return w.reindex(curve.index, method="ffill").dropna()


def fly_level(curve: pd.DataFrame, fly, w) -> pd.Series:
    a, b, c = fly
    return 100.0 * (w[0] * curve[a] + w[1] * curve[b] + w[2] * curve[c])


def zscores(curve: pd.DataFrame, fly, weights: pd.DataFrame, lookback: int) -> pd.Series:
    """z-score of the fly built with each day's own weights over a trailing window."""
    out = pd.Series(np.nan, index=weights.index)
    seg_id = (weights.diff().abs().sum(axis=1) > 0).cumsum()
    pos = {d: i for i, d in enumerate(curve.index)}
    for _, seg in weights.groupby(seg_id):
        w = seg.iloc[0].values
        i0, i1 = pos[seg.index[0]], pos[seg.index[-1]]
        hist = curve.iloc[max(0, i0 - lookback + 1): i1 + 1]
        F = fly_level(hist, fly, w)
        m = F.rolling(lookback, min_periods=lookback).mean()
        s = F.rolling(lookback, min_periods=lookback).std()
        out.loc[seg.index] = ((F - m) / s).loc[seg.index].values
    return out


# --------------------------------------------------------------------------
# Carry and roll-down per leg (bp per day, per unit of DV01)
# --------------------------------------------------------------------------
def par_mod_duration(y_pct: pd.Series | float, T: float) -> pd.Series | float:
    """Modified duration of a par note with semiannual coupons; bills use T."""
    if T < 1.5:
        return T / (1.0 + np.asarray(y_pct) / 100.0 * T)
    y = np.asarray(y_pct) / 100.0
    return (1.0 - (1.0 + y / 2.0) ** (-2.0 * T)) / y


def carry_roll(curve: pd.DataFrame) -> pd.DataFrame:
    """Daily carry plus roll-down for holding one unit of DV01 at each maturity.

    carry = (y - funding) / D, converted to bp per day; funding = 3-month bill.
    roll  = local slope of the curve below the maturity times one day.
    """
    mats = list(curve.columns)
    fund = curve[mats[0]]
    out = {}
    for j, T in enumerate(mats):
        D = par_mod_duration(curve[T], T)
        carry = (curve[T] - fund) * 100.0 / (DAYS * D)
        roll = 0.0 if j == 0 else (curve[T] - curve[mats[j - 1]]) / (T - mats[j - 1]) * 100.0 / DAYS
        out[T] = carry + roll
    return pd.DataFrame(out, index=curve.index)


# --------------------------------------------------------------------------
# Backtest
# --------------------------------------------------------------------------
def positions(z: pd.Series, entry: float, exit_: float, max_hold: int = MAX_HOLD) -> pd.Series:
    """Target position decided at each close from that day's z-score."""
    tgt, pos, held = [], 0, 0
    for v in z.values:
        if np.isnan(v):
            pos, held = 0, 0
        elif pos == 0:
            if v > entry:
                pos, held = 1, 0
            elif v < -entry:
                pos, held = -1, 0
        else:
            held += 1
            if (pos == 1 and v < exit_) or (pos == -1 and v > -exit_) or held >= max_hold:
                pos, held = 0, 0
        tgt.append(pos)
    return pd.Series(tgt, index=z.index, dtype=float)


def backtest(curve: pd.DataFrame, fly, weights: pd.DataFrame, z: pd.Series, entry: float,
             exit_: float, cost_bp: float = 0.25, cr: pd.DataFrame | None = None) -> pd.DataFrame:
    a, b, c = fly
    idx = weights.index
    y = curve.loc[idx, [a, b, c]] * 100.0
    dy = y.diff()
    W = weights.loc[idx].values
    W_prev = weights.shift(1).loc[idx].values          # weights known before today's move
    W_prev2 = weights.shift(2).loc[idx].values
    held = positions(z.loc[idx], entry, exit_).shift(2).fillna(0.0)
    h, h_prev = held.values, held.shift(1).fillna(0.0).values

    gross = -h * np.nansum(W_prev * dy.values, axis=1)
    if cr is None:
        cr = carry_roll(curve)
    cr_prev = cr.loc[idx, [a, b, c]].shift(1).values
    carry = h * np.nansum(W_prev * cr_prev, axis=1)
    trade = np.abs(h - h_prev) * np.nansum(np.abs(W_prev), axis=1)
    rebal = np.where((h == h_prev) & (h != 0), np.nansum(np.abs(W_prev - W_prev2), axis=1), 0.0)
    cost = cost_bp * (trade + rebal)
    out = pd.DataFrame({"z": z.loc[idx].values, "held": h, "gross": gross, "carry": carry,
                        "cost": cost}, index=idx)
    out["net"] = out["gross"] + out["carry"] - out["cost"]
    out.iloc[:2, out.columns.get_indexer(["gross", "carry", "cost", "net"])] = 0.0
    out["wa"], out["wb"], out["wc"] = W[:, 0], W[:, 1], W[:, 2]
    return out


@dataclass
class Stats:
    ann_bp: float
    vol_bp: float
    sharpe: float
    max_dd_bp: float
    trades: int
    hit_rate: float
    avg_days: float
    in_market: float

    def as_dict(self) -> dict:
        return self.__dict__.copy()


def stats(bt: pd.DataFrame, col: str = "net") -> Stats:
    pnl = bt[col]
    ann, vol = pnl.mean() * DAYS, pnl.std() * np.sqrt(DAYS)
    cum = pnl.cumsum()
    on = bt["held"] != 0
    trade_id = (on & ((bt["held"] != bt["held"].shift(1)) | ~on.shift(1, fill_value=False))).cumsum()
    per_trade = pnl[on].groupby(trade_id[on]).sum()
    lengths = on[on].groupby(trade_id[on]).size()
    return Stats(
        ann_bp=float(ann), vol_bp=float(vol), sharpe=float(ann / vol) if vol > 0 else float("nan"),
        max_dd_bp=float((cum - cum.cummax()).min()), trades=int(len(per_trade)),
        hit_rate=float((per_trade > 0).mean()) if len(per_trade) else float("nan"),
        avg_days=float(lengths.mean()) if len(lengths) else float("nan"), in_market=float(on.mean()),
    )


def half_life(F: pd.Series) -> float:
    """Half-life (days) of an AR(1) fitted to the fly level: dF = a + b F(-1) + e."""
    d = pd.DataFrame({"dF": F.diff(), "lag": F.shift(1)}).dropna()
    X = np.column_stack([np.ones(len(d)), d["lag"].values])
    b = np.linalg.lstsq(X, d["dF"].values, rcond=None)[0][1]
    return float(-np.log(2) / np.log(1 + b)) if -1 < b < 0 else float("inf")
