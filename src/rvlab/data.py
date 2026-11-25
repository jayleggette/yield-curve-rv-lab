"""Load the daily Treasury curve.

FRED constant-maturity (CMT) yields are par yields on a bond-equivalent
(semiannual) basis, in percent. The bill maturities (3 and 6 months, and the
1-year point, which the Treasury builds from bills) are treated as zero-coupon
yields; 2 years and longer as par yields on semiannual-coupon notes.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[2]
RAW = REPO / "data" / "raw"

SERIES = {"DGS3MO": 0.25, "DGS6MO": 0.5, "DGS1": 1.0, "DGS2": 2.0,
          "DGS3": 3.0, "DGS5": 5.0, "DGS7": 7.0, "DGS10": 10.0}
START = "1985-01-01"


def _read(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, index_col=0)
    df.index = pd.to_datetime(df.index)
    df.index.name = "date"
    return df.apply(pd.to_numeric, errors="coerce").sort_index()


def load_curve(path: Path | None = None, start: str = START) -> pd.DataFrame:
    """Daily CMT yields (percent), columns = maturity in years, days with all points."""
    d = _read(path or RAW / "fred_daily.csv")
    curve = d[list(SERIES)].rename(columns=SERIES).dropna()
    return curve.loc[start:]


def load_term_premium(path: Path | None = None) -> pd.Series | None:
    """Kim-Wright 10-year term premium (FRED THREEFYTP10) if the optional file is present."""
    p = path or RAW / "fred_daily_long.csv"
    if not p.exists():
        return None
    d = _read(p)
    return d["THREEFYTP10"].dropna() if "THREEFYTP10" in d else None
