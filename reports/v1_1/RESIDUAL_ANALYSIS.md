# Residual Analysis

All real-data findings are cross-sectional on deterministic released-calibration samples. Historical quotes lack quote timestamps, so temporal OOS, hedge outcomes, and P&L remain unavailable.

Residual is model price minus observed midpoint; positive values indicate overpricing.

| split | model_id | slice_value | mean_signed_error | median_signed_error | mae |
| --- | --- | --- | --- | --- | --- |
| DEV | M00_BS_FLAT | atm | 7.60827 | 4.67862 | 9.85923 |
| DEV | M00_BS_FLAT | call_wing | 27.4211 | 14.7578 | 27.4214 |
| DEV | M00_BS_FLAT | deep_call_wing | 44.2501 | 44.4837 | 44.2538 |
| DEV | M00_BS_FLAT | deep_put_wing | -36.8263 | -42.3763 | 36.8645 |
| DEV | M00_BS_FLAT | near_atm_call | 12.788 | 4.79525 | 13.4255 |
| DEV | M00_BS_FLAT | near_atm_put | -0.702716 | -0.226379 | 6.51609 |
| DEV | M00_BS_FLAT | put_wing | -10.3482 | -1.50582 | 12.2918 |
| DEV | M02_SVI | atm | -0.625604 | -0.606863 | 0.978495 |
| DEV | M02_SVI | call_wing | 0.0758814 | -0.028744 | 0.381272 |
| DEV | M02_SVI | deep_call_wing | 0.0262468 | -0.04762 | 0.13042 |
| DEV | M02_SVI | deep_put_wing | -0.0768133 | -0.0447326 | 0.335104 |
| DEV | M02_SVI | near_atm_call | -0.0614068 | 0.0316651 | 0.795142 |


## Validation and locked distribution diagnostics

| model_id | count | mean | std | q05 | median | q95 | skew |
| --- | --- | --- | --- | --- | --- | --- | --- |
| M00_BS_FLAT | 888 | -0.079653 | 30.6981 | -53.5871 | 0.0046091 | 59.0282 | 0.219418 |
| M02_SVI | 888 | -0.101173 | 0.833741 | -1.45017 | -0.019809 | 1.114 | 0.96572 |
| M03_SSVI | 888 | 1.66267 | 4.88732 | -4.81804 | 0.459001 | 10.8911 | 1.18359 |
| M10_HESTON | 888 | -0.0451245 | 2.97639 | -3.59967 | -0.124591 | 3.56621 | 0.355075 |
| M21_BATES | 888 | 4.45073 | 5.91331 | -0.682248 | 2.29659 | 15.9586 | 1.53309 |
