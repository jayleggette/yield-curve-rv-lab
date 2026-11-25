"""Principal components of daily yield changes (Litterman and Scheinkman, 1991)."""
from __future__ import annotations

import numpy as np
import pandas as pd


def pca(changes: pd.DataFrame, k: int = 3) -> tuple[pd.DataFrame, np.ndarray]:
    """Loadings (maturity x k) and the share of variance each component explains.

    Signs are fixed so the components read the usual way: PC1 (level) loads
    positively on average, PC2 (slope) is higher at the long end than the
    short end, PC3 (curvature) is higher in the belly than at the wings.
    """
    X = changes.dropna().values
    cov = np.cov(X, rowvar=False)
    vals, vecs = np.linalg.eigh(cov)
    order = np.argsort(vals)[::-1]
    vals, vecs = vals[order], vecs[:, order]
    vecs = vecs[:, :k].copy()
    mats = changes.columns.values.astype(float)
    if vecs[:, 0].mean() < 0:
        vecs[:, 0] *= -1
    if k > 1 and vecs[-1, 1] < vecs[0, 1]:
        vecs[:, 1] *= -1
    if k > 2:
        belly = np.argmin(np.abs(mats - np.median(mats)))
        if vecs[belly, 2] < 0.5 * (vecs[0, 2] + vecs[-1, 2]):
            vecs[:, 2] *= -1
    load = pd.DataFrame(vecs, index=changes.columns, columns=[f"PC{i + 1}" for i in range(k)])
    return load, vals[:k] / vals.sum()
