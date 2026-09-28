# Post-Validation Analytics

This is an additive v1.1 analysis extension. The v1 locked evidence was not rewritten, retuned, deleted, or reinterpreted.

All real-data findings are cross-sectional on deterministic released-calibration samples. Historical quotes lack quote timestamps, so temporal OOS, hedge outcomes, and P&L remain unavailable.

## Data-driven DEV segmentation

## absolute_error

```text
|--- dte <= 378.50
|   |--- settlement_code <= 0.50
|   |   |--- dte <= 18.00
|   |   |   |--- value: [0.38]
|   |   |--- dte >  18.00
|   |   |   |--- value: [0.75]
|   |--- settlement_code >  0.50
|   |   |--- relative_spread <= 0.04
|   |   |   |--- value: [2.19]
|   |   |--- relative_spread >  0.04
|   |   |   |--- value: [0.72]
|--- dte >  378.50
|   |--- value: [4.31]

```

## heston_svi_difference

```text
|--- dte <= 378.50
|   |--- settlement_code <= 0.50
|   |   |--- relative_spread <= 0.03
|   |   |   |--- value: [0.86]
|   |   |--- relative_spread >  0.03
|   |   |   |--- value: [0.40]
|   |--- settlement_code >  0.50
|   |   |--- relative_spread <= 0.03
|   |   |   |--- value: [2.47]
|   |   |--- relative_spread >  0.03
|   |   |   |--- value: [1.01]
|--- dte >  378.50
|   |--- value: [4.93]

```

## challenger_dispersion

```text
|--- dte <= 103.00
|   |--- relative_spread <= 0.06
|   |   |--- dte <= 29.50
|   |   |   |--- value: [1.63]
|   |   |--- dte >  29.50
|   |   |   |--- value: [2.98]
|   |--- relative_spread >  0.06
|   |   |--- relative_spread <= 0.10
|   |   |   |--- value: [0.74]
|   |   |--- relative_spread >  0.10
|   |   |   |--- value: [0.19]
|--- dte >  103.00
|   |--- log_moneyness <= -0.14
|   |   |--- log_moneyness <= -0.21
|   |   |   |--- value: [2.09]
|   |   |--- log_moneyness >  -0.21
|   |   |   |--- value: [4.14]
|   |--- log_moneyness >  -0.14
|   |   |--- log_moneyness <= 0.17
|   |   |   |--- value: [8.70]
|   |   |--- log_moneyness >  0.17
|   |   |   |--- value: [3.38]

```

## greek_disagreement

```text
|--- relative_spread <= 0.15
|   |--- log_moneyness <= -0.07
|   |   |--- relative_spread <= 0.09
|   |   |   |--- value: [2.20]
|   |   |--- relative_spread >  0.09
|   |   |   |--- value: [8.42]
|   |--- log_moneyness >  -0.07
|   |   |--- dte <= 203.50
|   |   |   |--- value: [0.31]
|   |   |--- dte >  203.50
|   |   |   |--- value: [2.21]
|--- relative_spread >  0.15
|   |--- log_moneyness <= -0.17
|   |   |--- value: [133.57]
|   |--- log_moneyness >  -0.17
|   |   |--- log_moneyness <= -0.13
|   |   |   |--- value: [26.14]
|   |   |--- log_moneyness >  -0.13
|   |   |   |--- value: [5.45]

```

## Leaf transfer summaries

| leaf_id | count | dates | target_mean | target_median | split | target |
| --- | --- | --- | --- | --- | --- | --- |
| 3 | 468 | 76 | 0.379187 | 0.240906 | DEV | absolute_error |
| 4 | 498 | 76 | 0.749133 | 0.603478 | DEV | absolute_error |
| 6 | 49 | 3 | 2.18892 | 1.63785 | DEV | absolute_error |
| 7 | 21 | 3 | 0.721611 | 0.445025 | DEV | absolute_error |
| 8 | 20 | 2 | 4.30699 | 1.53561 | DEV | absolute_error |
| 3 | 162 | 25 | 0.712044 | 0.304152 | VALIDATION | absolute_error |
| 4 | 192 | 25 | 1.47975 | 0.770918 | VALIDATION | absolute_error |
| 6 | 38 | 3 | 1.95306 | 0.999366 | VALIDATION | absolute_error |
| 7 | 22 | 3 | 0.725229 | 0.569067 | VALIDATION | absolute_error |
| 8 | 30 | 3 | 1.77275 | 0.975652 | VALIDATION | absolute_error |
| 3 | 162 | 25 | 0.757276 | 0.32406 | LOCKED_TEST | absolute_error |
| 4 | 192 | 25 | 2.0416 | 0.963962 | LOCKED_TEST | absolute_error |
