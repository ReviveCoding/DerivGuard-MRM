# Calibration Diagnostics

All 135 valid released Heston calibrations retain objective, price/IV/vega-weighted and spread-normalized errors, interval hit rate, signed bias, optimizer status/counts, boundary proximity, Feller status, and market-support covariates. Each calibration additionally has CUDA-FP64 price Jacobians with 400 CPU-FP64 linearized bid/ask quote-bootstrap propagations, parameter correlations, five local nuisance-profile widths, and four actual bounded local optimization starts. Multistart dispersion retains every finite terminal solution, including unsuccessful optimizer statuses, while reporting the successful-start count separately. The linearized bootstrap and local quadratic profile are diagnostic approximations, not replacements for the frozen release fit.

## Split summary

| split | calibrations | median_objective | median_price_rmse | median_iv_rmse | median_spread_normalized_mae | median_inside_bid_ask_rate | feller_rate |
| --- | --- | --- | --- | --- | --- | --- | --- |
| DEV | 79 | 1.22277 | 0.633949 | 0.024468 | 1.24461 | 0.5 | 0 |
| LOCKED_TEST | 28 | 1.99631 | 0.936691 | 0.0205199 | 1.81509 | 0.333333 | 0 |
| VALIDATION | 28 | 1.70333 | 0.906165 | 0.0280071 | 1.59171 | 0.416667 | 0 |

## Settlement, support, and liquidity summary

| split | settlement_class | calibrations | median_moneyness_support | median_maturity_support_days | median_relative_spread | median_price_rmse |
| --- | --- | --- | --- | --- | --- | --- |
| DEV | AM | 3 | 0.679809 | 364 | 0.0153699 | 2.39858 |
| DEV | PM | 76 | 0.614539 | 325.5 | 0.0180624 | 0.619837 |
| LOCKED_TEST | AM | 3 | 0.684258 | 399 | 0.0127136 | 5.52511 |
| LOCKED_TEST | PM | 25 | 0.611654 | 295 | 0.024068 | 0.912444 |
| VALIDATION | AM | 3 | 0.693147 | 392 | 0.0233689 | 1.9107 |
| VALIDATION | PM | 25 | 0.60134 | 322 | 0.0214616 | 0.830572 |

## Date, VIX, liquidity, moneyness-support, and maturity-support slices

| split | dimension | value | calibrations | market_dates | median_objective | median_price_rmse | median_iv_rmse | median_vega_weighted_iv_error | median_spread_normalized_mae | median_inside_bid_ask_rate | mean_signed_residual_bias | feller_rate |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| DEV | liquidity_regime | MEDIUM | 77 | 75 | 1.21871 | 0.633949 | 0.0250835 | 0.00338463 | 1.23168 | 0.5 | -0.0668032 | 0 |
| DEV | liquidity_regime | NARROW | 2 | 2 | 3.63149 | 3.34455 | 0.0214642 | 0.00508217 | 2.93767 | 0.341667 | 0.700352 | 0 |
| LOCKED_TEST | liquidity_regime | MEDIUM | 23 | 22 | 2.00427 | 0.918091 | 0.0217806 | 0.00497013 | 1.81589 | 0.333333 | -0.0495492 | 0 |
| LOCKED_TEST | liquidity_regime | NARROW | 2 | 2 | 4.65887 | 5.76888 | 0.0209611 | 0.00666098 | 3.7334 | 0.183333 | 1.17474 | 0 |
| LOCKED_TEST | liquidity_regime | WIDE | 3 | 3 | 1.29497 | 0.730324 | 0.0127804 | 0.00385211 | 1.21803 | 0.583333 | -1.13114 | 0 |
| VALIDATION | liquidity_regime | MEDIUM | 25 | 23 | 1.87796 | 0.830572 | 0.0289121 | 0.00460374 | 1.71015 | 0.416667 | -0.0820158 | 0 |
| VALIDATION | liquidity_regime | NARROW | 1 | 1 | 4.62135 | 3.44518 | 0.0242509 | 0.00551483 | 3.70737 | 0.166667 | -0.0448766 | 0 |
| VALIDATION | liquidity_regime | WIDE | 2 | 2 | 1.10381 | 3.05849 | 0.0317909 | 0.00519374 | 1.14398 | 0.5 | -1.17909 | 0 |
| DEV | maturity_support_regime | BROAD | 25 | 24 | 1.04342 | 0.61035 | 0.0296097 | 0.00377752 | 1.12888 | 0.583333 | -0.00366782 | 0 |
| DEV | maturity_support_regime | MEDIUM | 27 | 26 | 1.21871 | 0.525419 | 0.015568 | 0.00232912 | 1.24461 | 0.583333 | -0.0541124 | 0 |
| DEV | maturity_support_regime | SHORT | 27 | 27 | 1.33509 | 0.710684 | 0.0252912 | 0.003921 | 1.36597 | 0.416667 | -0.0811263 | 0 |
| LOCKED_TEST | maturity_support_regime | BROAD | 10 | 9 | 2.17896 | 1.17568 | 0.0222472 | 0.00545641 | 1.91928 | 0.333333 | 0.18076 | 0 |
| LOCKED_TEST | maturity_support_regime | SHORT | 18 | 18 | 1.87862 | 0.801282 | 0.0201448 | 0.00441112 | 1.78088 | 0.375 | -0.221732 | 0 |
| VALIDATION | maturity_support_regime | BROAD | 3 | 3 | 1.91601 | 1.9107 | 0.0271022 | 0.0055369 | 1.81107 | 0.333333 | -0.00218017 | 0 |
| VALIDATION | maturity_support_regime | MEDIUM | 20 | 20 | 1.34181 | 0.830349 | 0.0277528 | 0.00455516 | 1.28931 | 0.458333 | -0.154692 | 0 |
| VALIDATION | maturity_support_regime | SHORT | 5 | 5 | 2.51332 | 0.830572 | 0.0346158 | 0.00491996 | 2.13559 | 0.416667 | -0.270617 | 0 |
| DEV | moneyness_support_regime | BROAD | 22 | 22 | 1.2603 | 0.789332 | 0.0353667 | 0.00525352 | 1.32378 | 0.416667 | 0.0184525 | 0 |
| DEV | moneyness_support_regime | MEDIUM | 28 | 28 | 1.16399 | 0.619837 | 0.0254515 | 0.00376556 | 1.21469 | 0.5 | -0.0873546 | 0 |
| DEV | moneyness_support_regime | NARROW | 29 | 29 | 1.21871 | 0.525419 | 0.0146237 | 0.00230451 | 1.20765 | 0.583333 | -0.05873 | 0 |
| LOCKED_TEST | moneyness_support_regime | BROAD | 7 | 7 | 2.02003 | 1.19906 | 0.0208199 | 0.00517908 | 1.81428 | 0.333333 | 0.303869 | 0 |
| LOCKED_TEST | moneyness_support_regime | MEDIUM | 15 | 15 | 1.98835 | 0.918091 | 0.0356611 | 0.00516798 | 1.83025 | 0.333333 | -0.0877846 | 0 |
| LOCKED_TEST | moneyness_support_regime | NARROW | 6 | 6 | 1.89206 | 0.729196 | 0.0109387 | 0.00288099 | 1.70423 | 0.458333 | -0.498981 | 0 |
| VALIDATION | moneyness_support_regime | BROAD | 4 | 4 | 1.58715 | 1.63951 | 0.0280071 | 0.00552587 | 1.58243 | 0.333333 | -0.0335954 | 0 |
| VALIDATION | moneyness_support_regime | MEDIUM | 18 | 18 | 1.40464 | 0.906165 | 0.0294416 | 0.00460274 | 1.38026 | 0.416667 | -0.149169 | 0 |
| VALIDATION | moneyness_support_regime | NARROW | 6 | 6 | 2.12127 | 0.634417 | 0.0257558 | 0.00466114 | 1.8289 | 0.541667 | -0.272338 | 0 |
| DEV | vix_regime | HIGH | 26 | 25 | 1.017 | 0.692791 | 0.0317315 | 0.00446279 | 1.12701 | 0.5 | -0.0854581 | 0 |
| DEV | vix_regime | LOW | 27 | 26 | 1.52145 | 0.525419 | 0.0155766 | 0.00256464 | 1.53132 | 0.5 | -0.0233919 | 0 |
| DEV | vix_regime | MEDIUM | 26 | 25 | 1.229 | 0.656139 | 0.0248796 | 0.00347322 | 1.26563 | 0.5 | -0.0342174 | 0 |
| LOCKED_TEST | vix_regime | LOW | 26 | 24 | 1.95033 | 0.915267 | 0.0201448 | 0.00441112 | 1.81026 | 0.375 | -0.0789965 | 0 |
| LOCKED_TEST | vix_regime | MEDIUM | 2 | 1 | 7.24402 | 4.60399 | 0.0317302 | 0.0116299 | 5.4611 | 0.166667 | -0.0648295 | 0 |

## Per-calibration bootstrap uncertainty summary

| split | parameter | calibrations | median_bootstrap_std | p90_bootstrap_std |
| --- | --- | --- | --- | --- |
| DEV | kappa | 79 | 0.0724984 | 0.101079 |
| DEV | rho | 79 | 0.0038871 | 0.00649688 |
| DEV | sigma_v | 79 | 0.0145765 | 0.0224696 |
| DEV | theta | 79 | 0.000971929 | 0.00991913 |
| DEV | v0 | 79 | 0.000145024 | 0.000230586 |
| LOCKED_TEST | kappa | 28 | 0.0863142 | 0.134341 |
| LOCKED_TEST | rho | 28 | 0.00496503 | 0.0081186 |
| LOCKED_TEST | sigma_v | 28 | 0.0137266 | 0.0212483 |
| LOCKED_TEST | theta | 28 | 0.0387669 | 0.157601 |
| LOCKED_TEST | v0 | 28 | 0.000113772 | 0.000196357 |
| VALIDATION | kappa | 28 | 0.0827393 | 0.125744 |
| VALIDATION | rho | 28 | 0.00535262 | 0.00773933 |
| VALIDATION | sigma_v | 28 | 0.0128657 | 0.0228028 |
| VALIDATION | theta | 28 | 0.00199892 | 0.112161 |
| VALIDATION | v0 | 28 | 0.00016345 | 0.000215039 |

Coverage: 135 calibration keys, 675 profile rows, and 675 multistart parameter rows.
