# DVE v1 Failure Analysis

This is descriptive analysis of immutable inherited evidence and was not used to tune v1.1.

Across 51 locked/OOD cases, Full DVE produced 6 false positives and 4 false negatives. Half of the false positives were in LONG|WING|LIQUID. FitOnly produced 2 false positives, 0 false negatives, recall 1.000, and FPR 0.049; Full DVE produced recall 0.600 and FPR 0.146. Full DVE added five false positives, lost four FitOnly true positives, rescued no FitOnly misses, and removed one FitOnly false positive.

## Failure slices

| value | outcome | count | dimension |
| --- | --- | --- | --- |
| LIQUID | TRUE_NEGATIVE | 35 | liquidity_regime |
| MID | TRUE_NEGATIVE | 27 | moneyness_regime |
| HESTON | TRUE_NEGATIVE | 26 | truth_family |
| SHORT | TRUE_NEGATIVE | 21 | maturity_regime |
| LONG | TRUE_NEGATIVE | 14 | maturity_regime |
| MEDIUM | TRUE_NEGATIVE | 12 | noise_proxy_regime |
| HIGH | TRUE_NEGATIVE | 12 | noise_proxy_regime |
| LOW | TRUE_NEGATIVE | 11 | noise_proxy_regime |
| WING | TRUE_NEGATIVE | 8 | moneyness_regime |
| HESTON | FALSE_POSITIVE | 6 | truth_family |
| HESTON | TRUE_POSITIVE | 6 | truth_family |
| LONG | TRUE_POSITIVE | 5 | maturity_regime |
| LIQUID | TRUE_POSITIVE | 5 | liquidity_regime |
| LONG | FALSE_POSITIVE | 5 | maturity_regime |
| HESTON | FALSE_NEGATIVE | 4 | truth_family |
| WING | TRUE_POSITIVE | 4 | moneyness_regime |
| WING | FALSE_POSITIVE | 4 | moneyness_regime |
| LIQUID | FALSE_NEGATIVE | 4 | liquidity_regime |
| LIQUID | FALSE_POSITIVE | 4 | liquidity_regime |
| HIGH | FALSE_POSITIVE | 4 | noise_proxy_regime |
| MEDIUM | TRUE_POSITIVE | 4 | noise_proxy_regime |
| LOW | FALSE_NEGATIVE | 4 | noise_proxy_regime |
| F00 | TRUE_NEGATIVE | 3 | fault_id |
| F16 | FALSE_POSITIVE | 3 | fault_id |
| BATES | TRUE_NEGATIVE | 3 | truth_family |
| F14 | TRUE_NEGATIVE | 3 | fault_id |
| F11 | TRUE_POSITIVE | 3 | fault_id |
| F03 | TRUE_NEGATIVE | 3 | fault_id |
| F05 | TRUE_NEGATIVE | 3 | fault_id |
| F01 | TRUE_NEGATIVE | 3 | fault_id |

## Conditional evidence signal

| evidence | material_error | count | mean | median | p90 |
| --- | --- | --- | --- | --- | --- |
| E_data | False | 41 | 0.654389 | 0.29576 | 2.16252 |
| E_data | True | 10 | 0.263223 | 0.242276 | 0.360912 |
| E_num | False | 41 | 0.43165 | 0.00856773 | 1.70421 |
| E_num | True | 10 | 0.18648 | 0.00727052 | 0.190038 |
| E_cal | False | 41 | 0.151209 | 0.0982793 | 0.133868 |
| E_cal | True | 10 | 1.17196 | 0.113694 | 2.16547 |
| E_ident | False | 41 | 0.292683 | 0 | 2 |
| E_ident | True | 10 | 0 | 0 | 0 |
| E_param | False | 41 | 0.61254 | 0 | 4.18569 |
| E_param | True | 10 | 0 | 0 | 0 |
| E_form | False | 41 | 1.46185 | 0.751737 | 3.86975 |
| E_form | True | 10 | 8.99902 | 5.08886 | 21.4423 |
| E_greek | False | 41 | 0.177594 | 0.0871581 | 0.373265 |
| E_greek | True | 10 | 0.287294 | 0.238005 | 0.484724 |
| E_extra | False | 41 | 0.31001 | 0 | 1.64286 |
| E_extra | True | 10 | 0.42381 | 0 | 1.37857 |
| E_outcome | False | 41 | 0 | 0 | 0 |
| E_outcome | True | 10 | 0 | 0 | 0 |

## Full DVE versus FitOnly evidence attribution

| evidence | full_vs_fitonly | count | mean | median | p90 |
| --- | --- | --- | --- | --- | --- |
| E_data | FULL_EXCESS_FALSE_POSITIVE | 5 | 1.31662 | 1.49464 | 2.32055 |
| E_data | FULL_LOST_FITONLY_TRUE_POSITIVE | 4 | 0.224429 | 0.226966 | 0.238627 |
| E_data | FULL_REMOVED_FITONLY_FALSE_POSITIVE | 1 | 0.239895 | 0.239895 | 0.239895 |
| E_data | SAME_CLASSIFICATION | 41 | 0.530279 | 0.286128 | 1.37826 |
| E_num | FULL_EXCESS_FALSE_POSITIVE | 5 | 0.536511 | 0.0148276 | 1.3425 |
| E_num | FULL_LOST_FITONLY_TRUE_POSITIVE | 4 | 0.00656814 | 0.00755175 | 0.00878809 |
| E_num | FULL_REMOVED_FITONLY_FALSE_POSITIVE | 1 | 0.00865701 | 0.00865701 | 0.00865701 |
| E_num | SAME_CLASSIFICATION | 41 | 0.410853 | 0.00737554 | 1.81081 |
| E_cal | FULL_EXCESS_FALSE_POSITIVE | 5 | 0.250878 | 0.118879 | 0.585172 |
| E_cal | FULL_LOST_FITONLY_TRUE_POSITIVE | 4 | 0.910431 | 0.915975 | 1.73875 |
| E_cal | FULL_REMOVED_FITONLY_FALSE_POSITIVE | 1 | 0.11606 | 0.11606 | 0.11606 |
| E_cal | SAME_CLASSIFICATION | 41 | 0.314805 | 0.0982793 | 0.18601 |
| E_ident | FULL_EXCESS_FALSE_POSITIVE | 5 | 0.8 | 0 | 2 |
| E_ident | FULL_LOST_FITONLY_TRUE_POSITIVE | 4 | 0 | 0 | 0 |
| E_ident | FULL_REMOVED_FITONLY_FALSE_POSITIVE | 1 | 0 | 0 | 0 |
| E_ident | SAME_CLASSIFICATION | 41 | 0.195122 | 0 | 0 |
| E_param | FULL_EXCESS_FALSE_POSITIVE | 5 | 1.67428 | 0 | 4.18569 |
| E_param | FULL_LOST_FITONLY_TRUE_POSITIVE | 4 | 0 | 0 | 0 |
| E_param | FULL_REMOVED_FITONLY_FALSE_POSITIVE | 1 | 0 | 0 | 0 |
| E_param | SAME_CLASSIFICATION | 41 | 0.40836 | 0 | 0 |
| E_form | FULL_EXCESS_FALSE_POSITIVE | 5 | 3.96101 | 4.53348 | 6.01415 |
| E_form | FULL_LOST_FITONLY_TRUE_POSITIVE | 4 | 2.1321 | 2.25837 | 2.4939 |
| E_form | FULL_REMOVED_FITONLY_FALSE_POSITIVE | 1 | 1.3101 | 1.3101 | 1.3101 |
| E_form | SAME_CLASSIFICATION | 41 | 2.93371 | 0.751737 | 5.2198 |
| E_greek | FULL_EXCESS_FALSE_POSITIVE | 5 | 0.389097 | 0.471446 | 0.654826 |
| E_greek | FULL_LOST_FITONLY_TRUE_POSITIVE | 4 | 0.165004 | 0.187102 | 0.214367 |
| E_greek | FULL_REMOVED_FITONLY_FALSE_POSITIVE | 1 | 0.0871581 | 0.0871581 | 0.0871581 |
| E_greek | SAME_CLASSIFICATION | 41 | 0.181991 | 0.0944266 | 0.373265 |
| E_extra | FULL_EXCESS_FALSE_POSITIVE | 5 | 1.03333 | 0 | 2.63333 |
| E_extra | FULL_LOST_FITONLY_TRUE_POSITIVE | 4 | 0.559524 | 0.464286 | 1.19524 |
