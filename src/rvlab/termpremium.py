"""Split the 10-year yield into expected short rates and a term premium.

Method (a statistical decomposition in the spirit of Adrian, Crump and
Moench (2013), without their cross-sectional pricing restrictions):

1. Fit Nelson-Siegel factors to the month-end curve.
2. Fit a VAR(1) to the three factors.
3. The short rate is the fitted 1-month zero yield. Its expected path follows
   from iterating the VAR forward.
4. The risk-neutral 10-year yield is the average expected short rate over the
   next 120 months. The term premium is the fitted 10-year zero yield minus
   that average.

A caveat applies. Persistent factors make the VAR's mean reversion, and hence
the split, sensitive to sample and small-sample bias. That is why the Fed
Board's Kim-Wright estimate is used as a benchmark when available.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .curve import LAMBDA, loadings, zero_yield


def var1(X: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """OLS VAR(1): X_t = c + Phi X_{t-1} + u_t. Returns c (k,) and Phi (k, k)."""
    Y = X.values[1:]
    Z = np.column_stack([np.ones(len(Y)), X.values[:-1]])
    B = np.linalg.lstsq(Z, Y, rcond=None)[0]
    return B[0], B[1:].T


def decompose(factors: pd.DataFrame, maturity_years: float = 10.0, short_tau: float = 1 / 12,
              lam: float = LAMBDA) -> pd.DataFrame:
    c, Phi = var1(factors)
    a = loadings([short_tau], lam)[0]
    H = int(round(maturity_years * 12))
    rn = []
    for x in factors.values:
        path, e = [], x.copy()
        for _ in range(H):
            path.append(a @ e)
            e = c + Phi @ e
        rn.append(np.mean(path))
    y = pd.Series([zero_yield(b, [maturity_years], lam)[0] for b in factors.values], index=factors.index)
    out = pd.DataFrame({"yield": y, "risk_neutral": rn}, index=factors.index)
    out["term_premium"] = out["yield"] - out["risk_neutral"]
    eig = np.max(np.abs(np.linalg.eigvals(Phi)))
    out.attrs["max_eigenvalue"] = float(eig)
    return out
