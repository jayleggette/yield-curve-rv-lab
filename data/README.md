# Data provenance

| File | Source | Retrieved |
|---|---|---|
| `raw/fred_daily.csv` | FRED: DFF, DFEDTAR, DFEDTARU, DFEDTARL, DGS3MO, DGS6MO, DGS1, DGS2, DGS3, DGS5, DGS7, DGS10 (daily) | 24 Nov 2026 |
| `raw/fred_monthly.csv` | FRED monthly series (GS30 and macro series; not used by the main pipeline) | 24 Nov 2026 |
| `raw/fred_daily_long.csv` (optional, not included) | FRED: DGS20, DGS30, THREEFYTP10 | add to enable the Kim-Wright comparison |

To refresh: https://fred.stlouisfed.org/graph/fredgraph.csv?id=DFF,DFEDTAR,DFEDTARU,DFEDTARL,DGS3MO,DGS6MO,DGS1,DGS2,DGS3,DGS5,DGS7,DGS10 (FRED returns a zip when frequencies differ; merge the daily files on date).
Optional benchmark: https://fred.stlouisfed.org/graph/fredgraph.csv?id=DGS20,DGS30,THREEFYTP10
