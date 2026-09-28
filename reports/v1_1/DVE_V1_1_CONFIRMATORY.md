# DVE v1.1 Confirmatory Study

The first preregistered design stopped before locked generation because VALIDATION seed 118000007 was CALIBRATION_UNQUALIFIED. The preserved v1.1b remediation changed only the bootstrap search budget, passed disjoint prequalification, and used wholly new 121m-124m seeds.

VALIDATION selected `top_2_evidence_mean` before locked generation. The ordinary LOCKED_TEST and stress OOD_LOCKED_TEST partitions are reported separately below. On ordinary locked data, Full v1.1 DVE and FitOnly both had recall 1.000 and Full had lower FPR (0.069 versus 0.103). On OOD, Full retained recall 1.000 but its FPR rose to 0.615 versus 0.154 for FitOnly. Combined, Full had FPR 0.238 versus 0.119 and AUPRC 0.752 versus 1.000. The preregistered primary endpoint applies to wholly fresh LOCKED_TEST cases: there Full met the near-target FPR objective and preserved FitOnly recall. OOD was a separately frozen robustness partition and failed materially, so no global superiority or robust improvement claim is supported.

## VALIDATION selection

| candidate | baseline | observations | positives | negatives | recall | false_positive_rate | material_risk_miss_rate | auroc | auprc |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| maximum_evidence_percentile | maximum_evidence_percentile | 17 | 2 | 15 | 1 | 0.133333 | 0 | 0.966667 | 0.833333 |
| top_2_evidence_mean | top_2_evidence_mean | 17 | 2 | 15 | 1 | 0.0666667 | 0 | 1 | 1 |
| top_3_evidence_mean | top_3_evidence_mean | 17 | 2 | 15 | 1 | 0 | 0 | 1 | 1 |
| logical_two_high_components | logical_two_high_components | 17 | 2 | 15 | 0 | 0 | 1 | 0.85 | 0.7 |
| materiality_gated_top_2 | materiality_gated_top_2 | 17 | 2 | 15 | 1 | 0.0666667 | 0 | 1 | 1 |

## Confirmatory metrics

| partition | baseline | observations | positives | negatives | recall | false_positive_rate | material_risk_miss_rate | auroc | auprc | recall_ci_low | recall_ci_high | fpr_ci_low | fpr_ci_high |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| LOCKED_TEST | FitOnly | 34 | 5 | 29 | 1 | 0.103448 | 0 | 1 | 1 | 1 | 1 | 0 | 0.225995 |
| LOCKED_TEST | ChallengerOnly | 34 | 5 | 29 | 1 | 0.103448 | 0 | 1 | 1 | 1 | 1 | 0 | 0.218837 |
| LOCKED_TEST | BootstrapOnly | 34 | 5 | 29 | 0.4 | 0.0344828 | 0.6 | 0.786207 | 0.597076 | 0 | 1 | 0 | 0.107143 |
| LOCKED_TEST | NumericalOnly | 34 | 5 | 29 | 0 | 0.0344828 | 1 | 0.255172 | 0.116737 | 0 | 0 | 0 | 0.111111 |
| LOCKED_TEST | NonRegimeDVE | 34 | 5 | 29 | 0.2 | 0.0689655 | 0.8 | 0.937931 | 0.678333 | 0 | 0.666667 | 0 | 0.178571 |
| LOCKED_TEST | FullV1_1DVE | 34 | 5 | 29 | 1 | 0.0689655 | 0 | 1 | 1 | 1 | 1 | 0 | 0.172414 |
| OOD_LOCKED_TEST | FitOnly | 17 | 4 | 13 | 1 | 0.153846 | 0 | 1 | 1 | 1 | 1 | 0 | 0.363636 |
| OOD_LOCKED_TEST | ChallengerOnly | 17 | 4 | 13 | 1 | 0.538462 | 0 | 0.807692 | 0.704861 | 1 | 1 | 0.272727 | 0.8 |
| OOD_LOCKED_TEST | BootstrapOnly | 17 | 4 | 13 | 0.25 | 0.0769231 | 0.75 | 0.5 | 0.466516 | 0 | 0.879167 | 0 | 0.250417 |
| OOD_LOCKED_TEST | NumericalOnly | 17 | 4 | 13 | 0 | 0.153846 | 1 | 0.25 | 0.193155 | 0 | 0 | 0 | 0.363636 |
| OOD_LOCKED_TEST | NonRegimeDVE | 17 | 4 | 13 | 0.5 | 0.384615 | 0.5 | 0.730769 | 0.590909 | 0 | 1 | 0.142857 | 0.642857 |
| OOD_LOCKED_TEST | FullV1_1DVE | 17 | 4 | 13 | 1 | 0.615385 | 0 | 0.730769 | 0.590909 | 1 | 1 | 0.333333 | 0.866667 |
| COMBINED | FitOnly | 51 | 9 | 42 | 1 | 0.119048 | 0 | 1 | 1 | 1 | 1 | 0.0243902 | 0.225 |
| COMBINED | ChallengerOnly | 51 | 9 | 42 | 1 | 0.238095 | 0 | 0.939153 | 0.775278 | 1 | 1 | 0.113573 | 0.367374 |
| COMBINED | BootstrapOnly | 51 | 9 | 42 | 0.333333 | 0.047619 | 0.666667 | 0.664021 | 0.534121 | 0 | 0.666667 | 0 | 0.119048 |
| COMBINED | NumericalOnly | 51 | 9 | 42 | 0 | 0.0714286 | 1 | 0.230159 | 0.124077 | 0 | 0 | 0 | 0.166755 |
| COMBINED | NonRegimeDVE | 51 | 9 | 42 | 0.333333 | 0.166667 | 0.666667 | 0.857143 | 0.550832 | 0 | 0.666667 | 0.0525978 | 0.295501 |
| COMBINED | FullV1_1DVE | 51 | 9 | 42 | 1 | 0.238095 | 0 | 0.936508 | 0.751918 | 1 | 1 | 0.108108 | 0.365854 |
