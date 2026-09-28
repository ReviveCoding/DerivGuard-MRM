# Data Quality Report

The public HistoricalData.net archive contains 4,294,301 vendor
rows across 127 dates from
2022-07-01 through 2022-12-30. The normalized
SPX/SPXW subset contains
2,481,487 rows; 2,345,505 passed the model-ready filter and
135,982 have auditable exclusion codes. The processed artifact reports
127 market dates.

All historical rows lack `quote_time`; the data are standing EOD research quotes, not synchronized
NBBO. Apparent parity or static-arbitrage violations therefore remain data-quality evidence and are
not automatically attributed to model failure. Official Cboe marking data contain one prospective
date, and the DataShop integration sample was used for schema validation. Massive was skipped when
no API key was available. See `results/RESULT_AVAILABILITY.json` for the specific temporal analyses
that remain unavailable.

The empirical supplement contains 108 measured model rows: {'liquidity_filter': 54, 'near_expiry_1_to_7d': 18, 'settlement_robustness': 36}. These are cross-sectional EOD results, not synchronized P&L.
