# Challenger Disagreement Attribution

All real-data findings are cross-sectional on deterministic released-calibration samples. Historical quotes lack quote timestamps, so temporal OOS, hedge outcomes, and P&L remain unavailable.

SVI and SSVI are qualified contract-level challengers. Bates is retained only as the inherited fixed jump sensitivity and cannot support winner claims; LocalVol attribution is unavailable without contract-level qualification.

| quote_date | split | challenger | moneyness_slice | maturity_slice | model_price_disagreement | delta_disagreement | gamma_disagreement | vega_disagreement |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2022-11-25 | LOCKED_TEST | M03_SSVI | near_atm_call | 180D+ | 29.902 | 0.0792909 | 0.000229249 | 1060.65 |
| 2022-08-12 | DEV | M03_SSVI | near_atm_call | 180D+ | 29.0863 | 0.118981 | 0.000129649 | 946.467 |
| 2022-11-25 | LOCKED_TEST | M03_SSVI | near_atm_put | 180D+ | 28.5006 | 0.110868 | 5.83684e-05 | 1033.18 |
| 2022-10-26 | VALIDATION | M03_SSVI | near_atm_call | 180D+ | 27.8692 | 0.0975258 | 0.000143752 | 358.558 |
| 2022-11-30 | LOCKED_TEST | M03_SSVI | near_atm_call | 180D+ | 24.9409 | 0.10697 | 8.58692e-05 | 1115.84 |
| 2022-08-15 | DEV | M03_SSVI | near_atm_call | 180D+ | 24.6279 | 0.114805 | 0.000142567 | 1121.74 |
| 2022-11-15 | VALIDATION | M03_SSVI | near_atm_call | 180D+ | 23.2957 | 0.102115 | 0.000104872 | 1011.07 |
| 2022-07-29 | DEV | M03_SSVI | near_atm_call | 180D+ | 22.5726 | 0.111023 | 0.000112443 | 1045.16 |
| 2022-09-20 | DEV | M03_SSVI | call_wing | 180D+ | 22.4104 | 0.11223 | 5.41186e-05 | 934.187 |
| 2022-11-10 | VALIDATION | M03_SSVI | near_atm_call | 180D+ | 22.1147 | 0.0910116 | 8.3368e-05 | 968.358 |
| 2022-08-16 | DEV | M03_SSVI | near_atm_call | 180D+ | 21.9485 | 0.114082 | 0.000149241 | 1174.18 |
| 2022-08-10 | DEV | M03_SSVI | near_atm_call | 180D+ | 21.7938 | 0.114072 | 0.00012989 | 1011.77 |
