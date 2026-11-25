"""Reproduce every table, figure and number in the README.

    python scripts/run_all.py
"""
from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from rvlab import curve as C  # noqa: E402
from rvlab import data as D  # noqa: E402
from rvlab import pca as P  # noqa: E402
from rvlab import plots as PL  # noqa: E402
from rvlab import rv as RV  # noqa: E402
from rvlab import termpremium as TP  # noqa: E402

warnings.filterwarnings("ignore")
OUT = ROOT / "outputs"
TAB, FIG = OUT / "tables", OUT / "figures"
TAB.mkdir(parents=True, exist_ok=True)
FIG.mkdir(parents=True, exist_ok=True)
SPLIT = "2008-01-01"     # tuning period ends 2007; out of sample from 2008
COSTS = (0.0, 0.25, 0.5, 1.0)


def r3(x):
    return None if x is None or (isinstance(x, float) and not np.isfinite(x)) else round(float(x), 3)


def main() -> None:
    curve = D.load_curve()
    asof = curve.index[-1]

    # ---------------------------------------------------------- curve fits
    month_end = curve.groupby(curve.index.to_period("M")).tail(1)
    fm, resid = C.fit_curve(month_end)
    fm.index = fm.index.to_period("M").to_timestamp()
    resid.index = fm.index
    fm.to_csv(TAB / "ns_factors_month_end.csv")
    resid.to_csv(TAB / "ns_fit_residuals_bp.csv")
    med_abs = float(np.median(np.abs(resid.values)))
    mats = curve.columns.values.astype(float)
    grid_t = np.linspace(0.25, 10, 120)
    snaps = []
    for d in [asof, asof - pd.DateOffset(years=1), pd.Timestamp("2021-06-30")]:
        day = curve.loc[:d].iloc[-1]
        b, _ = C.fit_day(day.values, mats)
        snaps.append((f"{day.name:%d %b %Y}", day, grid_t, C.zero_yield(b, grid_t)))
    PL.curve_snapshots(snaps, FIG / "fig1_curve_fits.png", med_abs)
    PL.factors(fm, FIG / "fig2_ns_factors.png")
    latest_b, latest_r = C.fit_day(curve.iloc[-1].values, mats)

    # ----------------------------------------------------------------- PCA
    chg = curve.diff().dropna() * 100.0
    load, share = P.pca(chg, 3)
    load.to_csv(TAB / "pca_loadings.csv")
    pre_load, pre_share = P.pca(chg.loc[:"2007"], 3)
    post_load, post_share = P.pca(chg.loc["2008":], 3)
    PL.pca_loadings(load, share, FIG / "fig3_pca_loadings.png")

    # ------------------------------------------------------- term premium
    tp = TP.decompose(fm)
    tp.to_csv(TAB / "term_premium.csv")
    kw = D.load_term_premium()
    PL.term_premium(tp, kw, FIG / "fig4_term_premium.png")
    kw_corr = None
    if kw is not None:
        k = kw.resample("MS").last().reindex(tp.index)
        kw_corr = float(pd.concat([tp["term_premium"], k], axis=1).dropna().corr().iloc[0, 1])

    # ------------------------------------------------- relative value
    cr = RV.carry_roll(curve)
    W = {k: RV.rolling_weights(curve, f) for k, f in RV.FLIES.items()}
    Z = {(k, L): RV.zscores(curve, RV.FLIES[k], W[k], L)
         for k in RV.FLIES for L in sorted({g["lookback"] for g in RV.GRID})}

    def run(k, g, cost=0.25):
        return RV.backtest(curve, RV.FLIES[k], W[k], Z[(k, g["lookback"])], g["entry"], g["exit"], cost, cr)

    grid_rows = []
    for g in RV.GRID:
        bt = run("2s5s10s", g)
        grid_rows.append({**g, "is_sharpe": RV.stats(bt.loc[:"2007"]).sharpe,
                          "oos_sharpe": RV.stats(bt.loc[SPLIT:]).sharpe})
    grid = pd.DataFrame(grid_rows)
    grid.to_csv(TAB / "parameter_grid_2s5s10s.csv", index=False)
    chosen = grid.sort_values("is_sharpe", ascending=False).iloc[0]
    g = {"entry": float(chosen["entry"]), "exit": float(chosen["exit"]), "lookback": int(chosen["lookback"])}

    fly_rows, cost_rows, bts, hl_rows = [], [], {}, []
    for k, f in RV.FLIES.items():
        bt = run(k, g)
        bts[k] = bt
        bt.to_csv(TAB / f"backtest_{k}.csv")
        ins, oos = RV.stats(bt.loc[:"2007"]), RV.stats(bt.loc[SPLIT:])
        fly_rows.append({"fly": k,
                         "weights_latest": "/".join(f"{v:.2f}" for v in W[k].iloc[-1].values),
                         **{f"is_{a}": v for a, v in ins.as_dict().items()},
                         **{f"oos_{a}": v for a, v in oos.as_dict().items()},
                         "oos_gross_sharpe": RV.stats(bt.loc[SPLIT:], "gross").sharpe})
        for c in COSTS:
            btc = run(k, g, c)
            cost_rows.append({"fly": k, "cost_bp": c, "oos_sharpe": RV.stats(btc.loc[SPLIT:]).sharpe,
                              "oos_ann_bp": RV.stats(btc.loc[SPLIT:]).ann_bp})
        w_is = W[k].loc[:"2007"].mean().values
        F = RV.fly_level(curve, f, w_is)
        hl_rows.append({"fly": k, "half_life_is_days": RV.half_life(F.loc[:"2007"]),
                        "half_life_oos_days": RV.half_life(F.loc[SPLIT:])})
    flies = pd.DataFrame(fly_rows).set_index("fly")
    flies.to_csv(TAB / "results_by_fly.csv")
    costs = pd.DataFrame(cost_rows)
    costs.to_csv(TAB / "cost_sensitivity.csv", index=False)
    hl = pd.DataFrame(hl_rows).set_index("fly")
    hl.to_csv(TAB / "half_lives.csv")

    # Diagnostic: did the PCA hedge leak level/slope risk after 2008?
    # Regress each fly's daily move (with the weights in force) on the curve's
    # own level and slope moves, by period. A working hedge gives betas near 0.
    diag_rows = []
    lvl = chg.mean(axis=1)
    slope = chg[10.0] - chg[2.0]
    for k, f in RV.FLIES.items():
        a, b, c = f
        w = W[k].shift(1)
        dF = (w["wa"] * chg[a] + w["wb"] * chg[b] + w["wc"] * chg[c]).dropna()
        for lab, sl in [("1987-2007", slice("1987", "2007")), ("2008-2021", slice("2008", "2021")),
                        ("2022-2026", slice("2022", "2026"))]:
            y = dF.loc[sl]
            X = np.column_stack([np.ones(len(y)), lvl.loc[y.index], slope.loc[y.index]])
            beta, *_ = np.linalg.lstsq(X, y.values, rcond=None)
            fitted = X @ beta
            r2 = 1 - np.sum((y.values - fitted) ** 2) / np.sum((y.values - y.values.mean()) ** 2)
            diag_rows.append({"fly": k, "period": lab, "beta_level": beta[1], "beta_slope": beta[2],
                              "r2_level_slope": r2, "fly_vol_bp_day": y.std()})
    diag = pd.DataFrame(diag_rows)
    diag.to_csv(TAB / "hedge_diagnostic.csv", index=False)

    elb = curve[0.25] < 0.5
    regime_rows = []
    for k, bt in bts.items():
        o = bt.loc[SPLIT:]
        for lab, m in [("at the lower bound", elb.loc[o.index]), ("off the lower bound", ~elb.loc[o.index])]:
            p = o.loc[m, "net"]
            regime_rows.append({"fly": k, "regime": lab, "days": int(m.sum()), "ann_bp": p.mean() * 252,
                                "sharpe": p.mean() / p.std() * np.sqrt(252) if p.std() > 0 else np.nan})
    regimes = pd.DataFrame(regime_rows)
    regimes.to_csv(TAB / "oos_by_regime.csv", index=False)
    yearly = pd.DataFrame({k: bt["net"].groupby(bt.index.year).sum() for k, bt in bts.items()})
    yearly.to_csv(TAB / "net_pnl_by_year_bp.csv")

    PL.cum_pnl_flies(bts, SPLIT, FIG / "fig5_cumulative_pnl.png")
    PL.sharpe_by_fly(flies.rename(columns={"is_sharpe": "in_sample", "oos_sharpe": "out_of_sample"})
                     [["in_sample", "out_of_sample"]], FIG / "fig6_sharpe_by_fly.png")

    # ------------------------------------------------------------ results
    results = {
        "data": {"start": str(curve.index[0].date()), "end": str(asof.date()), "days": int(len(curve))},
        "curve_fit": {"median_abs_error_bp": r3(med_abs),
                      "latest_factors": dict(zip(["level", "slope", "curvature"], map(r3, latest_b))),
                      "latest_residuals_bp": {str(m): r3(v * 100) for m, v in zip(mats, latest_r)}},
        "pca": {"share_full": list(map(r3, share)), "share_pre2008": list(map(r3, pre_share)),
                "share_post2008": list(map(r3, post_share)), "cum3_full": r3(share.sum())},
        "term_premium": {"latest_month": str(tp.index[-1].date()), "latest_tp": r3(tp["term_premium"].iloc[-1]),
                         "latest_rn": r3(tp["risk_neutral"].iloc[-1]), "latest_yield": r3(tp["yield"].iloc[-1]),
                         "mean_tp": r3(tp["term_premium"].mean()),
                         "tp_2019_avg": r3(tp.loc["2019", "term_premium"].mean()),
                         "var_max_eigenvalue": r3(tp.attrs.get("max_eigenvalue")),
                         "kim_wright_corr": r3(kw_corr)},
        "rv": {"chosen_params": g, "is_sharpe_chosen": r3(chosen["is_sharpe"]),
               "grid_oos_sharpe_2s5s10s": {"min": r3(grid["oos_sharpe"].min()),
                                           "median": r3(grid["oos_sharpe"].median()),
                                           "max": r3(grid["oos_sharpe"].max()),
                                           "share_negative": r3((grid["oos_sharpe"] < 0).mean())},
               "by_fly": {k: {kk: (r3(v) if isinstance(v, (float, np.floating)) else (int(v) if isinstance(v, (int, np.integer)) else v))
                              for kk, v in row.items()} for k, row in flies.to_dict("index").items()},
               "half_lives": {k: {kk: r3(v) for kk, v in row.items()} for k, row in hl.to_dict("index").items()},
               "costs": [{k: (r3(v) if isinstance(v, float) else v) for k, v in row.items()} for row in cost_rows],
               "regimes": [{k: (r3(v) if isinstance(v, (float, np.floating)) else v) for k, v in row.items()}
                           for row in regime_rows],
               "hedge_diagnostic": [{k: (r3(v) if isinstance(v, (float, np.floating)) else v) for k, v in row.items()}
                                    for row in diag_rows]},
    }
    (OUT / "results.json").write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
