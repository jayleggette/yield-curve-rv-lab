"""Nelson-Siegel curve fitted to par yields.

Diebold and Li (2006) fix the decay parameter so the curvature factor peaks
near 30 months (lambda = 0.0609 per month = 0.7308 per year), which makes the
three factors linear in the zero curve. Treasury CMT yields of two years and
longer are PAR yields, not zero yields, so treating them as zeros biases the
fit by several basis points at the long end when the curve is steep. Here the
Nelson-Siegel curve is a ZERO curve, notes are priced off it with semiannual
coupons, and the factors are chosen to match observed par yields.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import least_squares

LAMBDA = 0.7308  # per year (Diebold-Li)
PAR_FROM = 2.0   # maturities at or above this are coupon notes priced at par


def loadings(tau, lam: float = LAMBDA) -> np.ndarray:
    """Level, slope and curvature loadings at maturities tau (years)."""
    tau = np.asarray(tau, dtype=float)
    x = lam * tau
    f2 = (1 - np.exp(-x)) / x
    f3 = f2 - np.exp(-x)
    return np.column_stack([np.ones_like(tau), f2, f3])


def zero_yield(beta, tau, lam: float = LAMBDA) -> np.ndarray:
    """Zero yields (percent, bond-equivalent) at maturities tau."""
    return loadings(tau, lam) @ np.asarray(beta, dtype=float)


def discount(beta, t, lam: float = LAMBDA) -> np.ndarray:
    z = zero_yield(beta, t, lam)
    return (1.0 + z / 200.0) ** (-2.0 * np.asarray(t, dtype=float))


def par_yield(beta, T: float, lam: float = LAMBDA) -> float:
    """Par yield (percent) of a semiannual note maturing in T years."""
    times = np.arange(1, int(round(2 * T)) + 1) / 2.0
    d = discount(beta, times, lam)
    return float(200.0 * (1.0 - d[-1]) / d.sum())


def model_yields(beta, maturities, lam: float = LAMBDA) -> np.ndarray:
    """What the CMT curve should read: zero yields for bills, par yields for notes."""
    return np.array([par_yield(beta, T, lam) if T >= PAR_FROM else zero_yield(beta, [T], lam)[0]
                     for T in maturities])


def fit_day(obs: np.ndarray, maturities, start=None, lam: float = LAMBDA):
    """Least-squares Nelson-Siegel factors for one day's curve."""
    maturities = np.asarray(maturities, dtype=float)
    if start is None:
        start = np.linalg.lstsq(loadings(maturities, lam), obs, rcond=None)[0]
    res = least_squares(lambda b: model_yields(b, maturities, lam) - obs, start, method="lm")
    return res.x, model_yields(res.x, maturities, lam) - obs


def fit_curve(curve: pd.DataFrame, lam: float = LAMBDA) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fit every row. Returns factors (level, slope, curvature) and residuals (model - observed, bp)."""
    mats = curve.columns.values.astype(float)
    betas, resid, prev = [], [], None
    for _, row in curve.iterrows():
        b, r = fit_day(row.values, mats, start=prev, lam=lam)
        betas.append(b)
        resid.append(r * 100.0)
        prev = b
    factors = pd.DataFrame(betas, index=curve.index, columns=["level", "slope", "curvature"])
    residuals = pd.DataFrame(resid, index=curve.index, columns=curve.columns)
    return factors, residuals
