# Portfolio Materiality Slices

All real-data findings are cross-sectional on deterministic released-calibration samples. Historical quotes lack quote timestamps, so temporal OOS, hedge outcomes, and P&L remain unavailable.

Portfolio leg-level liquidity is unavailable in the inherited aggregate artifact and is not imputed.

| quote_date | split | portfolio_id | description | model_id | vix_regime | maturity_exposure | moneyness_exposure | valuation_materiality | delta_materiality | gamma_materiality | vega_materiality |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2022-12-12 | LOCKED_TEST | P01 | ATM straddle | M03_SSVI | MEDIUM | FRONT_MATURITY | ATM_DOMINANT | 19.9139 | 0.0858296 | 0.000805363 | 11.2988 |
| 2022-12-12 | LOCKED_TEST | P03 | butterfly | M03_SSVI | MEDIUM | FRONT_MATURITY | WING_STRUCTURE | 15.3849 | 0.105434 | 0.0008685 | 31.1507 |
| 2022-12-12 | LOCKED_TEST | P05 | skew portfolio | M03_SSVI | MEDIUM | FRONT_MATURITY | WING_OR_SKEW | 15.3849 | 0.105434 | 0.0008685 | 31.1507 |
| 2022-11-22 | VALIDATION | P03 | butterfly | M03_SSVI | LOW | FRONT_MATURITY | WING_STRUCTURE | 13.5714 | 0.0640797 | 0.00324037 | 63.3854 |
| 2022-11-22 | VALIDATION | P05 | skew portfolio | M03_SSVI | LOW | FRONT_MATURITY | WING_OR_SKEW | 13.5714 | 0.0640797 | 0.00324037 | 63.3854 |
| 2022-11-23 | LOCKED_TEST | P03 | butterfly | M03_SSVI | LOW | FRONT_MATURITY | WING_STRUCTURE | 13.457 | 0.0536027 | 0.0041464 | 60.8501 |
| 2022-11-23 | LOCKED_TEST | P05 | skew portfolio | M03_SSVI | LOW | FRONT_MATURITY | WING_OR_SKEW | 13.457 | 0.0536027 | 0.0041464 | 60.8501 |
| 2022-11-23 | LOCKED_TEST | P01 | ATM straddle | M03_SSVI | LOW | FRONT_MATURITY | ATM_DOMINANT | 13.4049 | 0.0702662 | 0.00412718 | 76.7499 |
| 2022-11-22 | VALIDATION | P01 | ATM straddle | M03_SSVI | LOW | FRONT_MATURITY | ATM_DOMINANT | 13.3329 | 0.0731835 | 0.00271346 | 68.3672 |
| 2022-11-04 | VALIDATION | P01 | ATM straddle | M03_SSVI | MEDIUM | FRONT_MATURITY | ATM_DOMINANT | 12.8137 | 0.0817341 | 0.000548312 | 8.31851 |
| 2022-12-12 | LOCKED_TEST | P06 | mixed standardized book | M03_SSVI | MEDIUM | MIXED_FRONT_BACK | MIXED_ATM_WING | 12.7905 | 0.0230233 | 0.000335557 | 495.361 |
| 2022-11-04 | VALIDATION | P04 | calendar spread | M03_SSVI | MEDIUM | TERM_STRUCTURE | ATM_DOMINANT | 12.4333 | 0.0362411 | 0.000287919 | 859.52 |
