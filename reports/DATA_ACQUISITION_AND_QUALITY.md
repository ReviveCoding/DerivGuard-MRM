# Data Acquisition and Quality — Phases 05–09

## Status

Phases 05–09 completed for the legitimately accessible public samples. Raw
artifacts are immutable and accompanied by acquisition manifests; independent
content inspection is recorded in `artifacts/data/acquisition_inspection.json`.
No paid data were purchased and no authenticated access was bypassed.

## Acquired artifacts

| Source | Actual artifact | Bytes | SHA-256 | Observed scope |
|---|---:|---:|---|---|
| HistoricalData.net options sample | `options_sample_2022H2.zip` | 238,583,748 | `8517b069f4d41e44f950dc79a3cdab694359b0fae91664190011e39dd8e48b76` | 4,294,301 rows, 127 dates, 2022-07-01 through 2022-12-30 |
| Cboe 3:00 CT marking prices | `eod_marking_prices_list.csv` | 8,279,947 | `4f381438c8c57edeb803b65934083c8acc02efa41ecb2d53bf3274d75fad5907` | 40,445 paired rows / 80,890 long rows, dated 2026-09-25 |
| Cboe DataShop quote-interval sample | `option_quote_intervals_sample.zip` | 2,628,937 | `7645a6cc94f1a984e98264a4b494f748e373c30ecde85de25ceadb83a5c56782` | Five nested CSVs, 115,617 rows total |

The HistoricalData.net archive's 127 internal SHA-256 values and row counts
were independently checked against its packaged manifest. All passed. The 34
columns were observed. `quote_time` is missing for all 4,294,301 historical
rows, so this source is classified as historical standing EOD quotes, not a
synchronized NBBO surface.

The optional Massive provider was skipped because `MASSIVE_API_KEY` was absent.
The code does not print credentials and refuses to infer endpoint entitlement
merely from the presence of a key.

## SPX/SPXW processing audit

The archive contained 845,643 SPX rows and 1,635,844 SPXW rows (2,481,487
combined). Processing used 127 bounded, restartable daily partitions through
RAW → NORMALIZED → QUALITY_FLAGGED → FILTERED → MODEL_READY. Every excluded
row remains in QUALITY_FLAGGED with deterministic source-row ID, exclusion
code/reason, and repair flag.

Measured results:

- 2,345,505 rows included; 135,982 excluded for missing critical quote fields.
- 570,764 rows carried butterfly-convexity evidence and 377,765 carried strike-
  monotonicity evidence. These are non-exclusionary diagnostics because the
  historical quotes are asynchronous.
- 9,112 rows carried an extreme-relative-spread diagnostic.
- Three rows occurred in sparse strike slices.
- SPX and SPXW settlement classes remained separate (AM and PM respectively,
  with explicit source settlement metadata taking precedence).

The Cboe marking file is classified `PROSPECTIVE_LOCKED_TEST`. Only its SPX and
SPXW roots (29,206 long rows) passed the current settlement-scope filter; 51,684
rows for other roots were retained as settlement-ambiguous exclusions. Cboe's
indicative marks are not represented as OPRA NBBO updates.

## Forward and discount estimation

F0 OLS, F1 spread-weighted least squares, F2 Huber robust regression, and F3
spread-weighted Huber regression were run by complete date/expiry/settlement
slice across four quote-quality definitions. The result artifact contains
107,521 estimator records across 6,841 distinct surfaces:

`results/raw/forward_estimator_comparison.parquet`

The narrow, liquid matched-pair subset reduced median unweighted residual RMSE
relative to all usable quotes for every method. For example, F0 median RMSE
fell from 0.224069 to 0.152719 index points. F3 median weighted RMSE fell from
0.076739 to 0.062727. These are descriptive diagnostics, not locked-test model
results. There were 313 robust fits that reached the iteration limit; these and
other implausible-discount cases are retained as `FAILED_DIAGNOSTIC`, never
coerced into successful estimates. Full measured summaries are in
`reports/tables/T02_forward_discount_estimation.csv`.

## Limitations

- Historical quote timestamps are wholly unavailable, preventing synchronized
  historical NBBO claims and limiting temporal/hedge-proxy interpretation.
- Static-arbitrage diagnostics mix market-quality/asynchrony evidence with
  possible price inconsistency and must not be labeled model failure.
- The DataShop artifact is an integration sample only; it conveys no paid
  full-history entitlement. Its documented quote-size methodology changes on
  2026-06-22.
- The current Cboe file provides one prospective observation date. It cannot be
  backfilled or used retroactively for tuning.
- The fixed 25% narrow-spread threshold used in the descriptive D03/D04 study
  is provisional and is not a locked validation threshold.

## Verification

Targeted data tests: **17 passed**. Ruff checks for the data modules and tests:
**passed**. Tests cover immutable writes, provider skip/block behavior,
settlement timing, F0–F3 recovery and robustness, source adapters, explicit
exclusion preservation, and static-arbitrage diagnostics.

