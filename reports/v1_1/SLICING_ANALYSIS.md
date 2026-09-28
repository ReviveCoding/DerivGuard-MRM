# Fixed Economic Slicing Analysis

All real-data findings are cross-sectional on deterministic released-calibration samples. Historical quotes lack quote timestamps, so temporal OOS, hedge outcomes, and P&L remain unavailable.

0DTE is retained as a separate stress domain but is unavailable for released-surface pricing because the preserved samples excluded sub-7D contracts and the EOD files do not contain settlement-relative quote time.

| split | model_id | slice_dimension | slice_value | count | rmse | mae | spread_normalized_mae | within_bid_ask_rate |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| DEV | M00_BS_FLAT | moneyness_slice | atm | 117 | 15.324 | 9.85923 | 14.1343 | 0.00854701 |
| DEV | M00_BS_FLAT | moneyness_slice | call_wing | 151 | 39.8663 | 27.4214 | 18.4052 | 0.0397351 |
| DEV | M00_BS_FLAT | moneyness_slice | deep_call_wing | 98 | 48.4565 | 44.2538 | 44.586 | 0 |
| DEV | M00_BS_FLAT | moneyness_slice | deep_put_wing | 206 | 42.984 | 36.8645 | 28.3793 | 0 |
| DEV | M00_BS_FLAT | moneyness_slice | near_atm_call | 117 | 21.8459 | 13.4255 | 20.1896 | 0 |
| DEV | M00_BS_FLAT | moneyness_slice | near_atm_put | 143 | 10.8709 | 6.51609 | 9.50346 | 0.0559441 |
| DEV | M00_BS_FLAT | moneyness_slice | put_wing | 224 | 21.2967 | 12.2918 | 11.0793 | 0 |
| DEV | M02_SVI | moneyness_slice | atm | 117 | 1.20221 | 0.978495 | 1.60631 | 0.162393 |
| DEV | M02_SVI | moneyness_slice | call_wing | 151 | 0.677037 | 0.381272 | 0.737393 | 0.655629 |
| DEV | M02_SVI | moneyness_slice | deep_call_wing | 98 | 0.286069 | 0.13042 | 0.230265 | 0.918367 |
| DEV | M02_SVI | moneyness_slice | deep_put_wing | 206 | 0.449145 | 0.335104 | 0.358869 | 0.898058 |
| DEV | M02_SVI | moneyness_slice | near_atm_call | 117 | 1.08438 | 0.795142 | 1.73174 | 0.324786 |
