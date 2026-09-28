# Model Winner Map

Winners use predefined spread-normalized MAE and are reported only for sufficiently populated cells, with market-date bootstrap uncertainty. The unqualified fixed Bates sensitivity is excluded from winner selection; LocalVol is unavailable without contract-level qualification. No global-best claim is made.

| split | moneyness_slice | maturity_slice | winner | second_best | winner_error | second_error | winner_margin | winner_bootstrap_probability | winner_margin_ci_low | winner_margin_ci_high |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| DEV | atm | 1-7D | M10_HESTON | M02_SVI | 1.44924 | 1.78589 | 0.33665 | 0.885 | -0.28177 | 0.896148 |
| DEV | atm | 180D+ | M10_HESTON | M02_SVI | 0.520605 | 0.67435 | 0.153745 | 0.865 | -0.162362 | 0.344333 |
| DEV | call_wing | 1-7D | M02_SVI | M10_HESTON | 0.786495 | 2.68899 | 1.9025 | 1 | 1.38499 | 2.51654 |
| DEV | call_wing | 180D+ | M02_SVI | M10_HESTON | 0.215399 | 0.588809 | 0.37341 | 1 | 0.136726 | 0.756878 |
| DEV | deep_call_wing | 180D+ | M02_SVI | M10_HESTON | 0.132273 | 0.454914 | 0.32264 | 1 | 0.164413 | 0.558814 |
| DEV | deep_put_wing | 1-7D | M02_SVI | M03_SSVI | 0.00441669 | 1.28948 | 1.28506 | 1 | 1.04677 | 1.50591 |
| DEV | deep_put_wing | 180D+ | M02_SVI | M10_HESTON | 0.311051 | 0.558452 | 0.2474 | 1 | 0.137626 | 0.359927 |
| DEV | near_atm_call | 1-7D | M02_SVI | M10_HESTON | 2.11907 | 2.49584 | 0.376762 | 0.6775 | -0.805281 | 1.65814 |
| DEV | near_atm_call | 180D+ | M02_SVI | M10_HESTON | 0.516663 | 0.796685 | 0.280022 | 0.6375 | -0.162827 | 0.99437 |
| DEV | near_atm_put | 1-7D | M10_HESTON | M02_SVI | 1.28876 | 1.50214 | 0.213381 | 0.8175 | -0.3378 | 0.669095 |
| DEV | near_atm_put | 180D+ | M10_HESTON | M02_SVI | 0.470038 | 0.699209 | 0.229171 | 0.9375 | -0.0638843 | 0.409969 |
| DEV | put_wing | 1-7D | M02_SVI | M10_HESTON | 0.378055 | 3.5451 | 3.16705 | 1 | 2.66158 | 3.58189 |

## Pairwise date wins

| split | model | opponent | win_rate | dates | win_rate_ci_low | win_rate_ci_high |
| --- | --- | --- | --- | --- | --- | --- |
| DEV | M00_BS_FLAT | M02_SVI | 0 | 76 | 0 | 0 |
| DEV | M00_BS_FLAT | M03_SSVI | 0 | 76 | 0 | 0 |
| DEV | M00_BS_FLAT | M10_HESTON | 0 | 76 | 0 | 0 |
| DEV | M02_SVI | M00_BS_FLAT | 1 | 76 | 1 | 1 |
| DEV | M02_SVI | M03_SSVI | 1 | 76 | 1 | 1 |
| DEV | M02_SVI | M10_HESTON | 0.815789 | 76 | 0.723355 | 0.894737 |
| DEV | M03_SSVI | M00_BS_FLAT | 1 | 76 | 1 | 1 |
| DEV | M03_SSVI | M02_SVI | 0 | 76 | 0 | 0 |
| DEV | M03_SSVI | M10_HESTON | 0 | 76 | 0 | 0 |
| DEV | M10_HESTON | M00_BS_FLAT | 1 | 76 | 1 | 1 |
| DEV | M10_HESTON | M02_SVI | 0.184211 | 76 | 0.104934 | 0.289474 |
| DEV | M10_HESTON | M03_SSVI | 1 | 76 | 1 | 1 |
