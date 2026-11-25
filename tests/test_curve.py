import numpy as np
import pandas as pd
import pytest

from rvlab import curve as C
from rvlab import pca as P


def test_flat_zero_curve_prices_notes_at_par():
    # With a flat 5% semiannual zero curve, every note's par yield is 5%.
    for T in (2.0, 5.0, 10.0):
        assert C.par_yield([5.0, 0.0, 0.0], T) == pytest.approx(5.0, abs=1e-10)


def test_par_differs_from_zero_on_a_steep_curve():
    beta = [5.0, -3.0, 0.0]  # steep upward curve
    z10 = C.zero_yield(beta, [10.0])[0]
    p10 = C.par_yield(beta, 10.0)
    assert p10 < z10  # coupons are discounted at lower short rates


def test_fit_recovers_a_known_curve():
    mats = np.array([0.25, 0.5, 1, 2, 3, 5, 7, 10], dtype=float)
    truth = np.array([4.8, -0.9, 1.2])
    obs = C.model_yields(truth, mats)
    b, resid = C.fit_day(obs, mats)
    np.testing.assert_allclose(b, truth, atol=1e-6)
    assert np.abs(resid).max() < 1e-6


def test_pca_sign_conventions():
    rng = np.random.default_rng(0)
    mats = [0.25, 1, 2, 5, 10]
    n = 3000
    level = rng.normal(0, 5, n)
    slope = rng.normal(0, 2, n)
    X = np.column_stack([level + slope * (m - 3) / 7 + rng.normal(0, 0.3, n) for m in mats])
    load, share = P.pca(pd.DataFrame(X, columns=mats), 2)
    assert load["PC1"].mean() > 0
    assert load["PC2"].iloc[-1] > load["PC2"].iloc[0]
    assert share[0] > share[1]
