import numpy as np
import pandas as pd
import pytest

from rvlab import data as D
from rvlab import pca as P
from rvlab import rv as RV


def test_fly_weights_are_neutral_to_level_and_slope():
    curve = D.load_curve()
    load, _ = P.pca(curve.diff().dropna().loc["2000":"2005"] * 100, 3)
    fly = RV.FLIES["2s5s10s"]
    w = RV.fly_weights(load, fly)
    exposure = load.loc[list(fly), ["PC1", "PC2"]].values.T @ w
    np.testing.assert_allclose(exposure, 0.0, atol=1e-12)
    assert w[1] == 1.0 and w[0] < 0 and w[2] < 0


def test_rolling_weights_use_only_past_data():
    curve = D.load_curve()
    fly = RV.FLIES["3s5s7s"]
    w = RV.rolling_weights(curve.loc[:"2012"], fly)
    scrambled = curve.loc[:"2012"].copy()
    scrambled.loc["2010-06-01":] *= 1.7
    w2 = RV.rolling_weights(scrambled, fly)
    pd.testing.assert_frame_equal(w.loc[:"2010-06-01"], w2.loc[:"2010-06-01"])


def test_position_rules():
    z = pd.Series([0.0, 2.5, 1.0, -0.1, 0.0, -2.2, -0.4, 0.3])
    pos = RV.positions(z, entry=2.0, exit_=0.0).tolist()
    assert pos == [0, 1, 1, 0, 0, -1, -1, 0]


def test_max_hold_forces_exit():
    z = pd.Series([3.0] + [2.5] * 70)
    pos = RV.positions(z, entry=2.0, exit_=0.0, max_hold=5)
    assert pos.iloc[0] == 1 and pos.iloc[5] == 0


def test_signal_is_traded_one_day_late():
    """A jump in the fly on the day the signal fires must not be captured."""
    idx = pd.bdate_range("2020-01-01", periods=12)
    base = pd.DataFrame({0.25: 1.0, 2.0: 2.0, 5.0: 3.0, 10.0: 4.0}, index=idx)
    base.loc[idx[6]:, 5.0] += 0.10  # body cheapens by 10bp on day 6
    w = pd.DataFrame({"wa": -0.5, "wb": 1.0, "wc": -0.5}, index=idx)
    z = pd.Series(0.0, index=idx)
    z.iloc[6:] = 3.0  # the signal fires on the jump day itself
    bt = RV.backtest(base, (2.0, 5.0, 10.0), w, z, entry=2.0, exit_=0.0, cost_bp=0.0)
    assert bt["held"].iloc[6] == 0 and bt["held"].iloc[7] == 0 and bt["held"].iloc[8] == 1
    assert bt["gross"].iloc[6] == pytest.approx(0.0)


def test_roll_down_is_positive_on_an_upward_curve():
    idx = pd.bdate_range("2020-01-01", periods=3)
    curve = pd.DataFrame({0.25: 1.0, 2.0: 2.0, 5.0: 3.0, 10.0: 4.0}, index=idx)
    cr = RV.carry_roll(curve)
    assert (cr[[2.0, 5.0, 10.0]] > 0).all().all()


def test_half_life_of_a_known_ar1():
    rng = np.random.default_rng(0)
    phi, n = 0.98, 20000
    x = np.zeros(n)
    for t in range(1, n):
        x[t] = phi * x[t - 1] + rng.normal()
    hl = RV.half_life(pd.Series(x))
    assert hl == pytest.approx(np.log(2) / -np.log(phi), rel=0.15)
