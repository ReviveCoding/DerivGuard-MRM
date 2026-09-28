# Findings Register

| ID | Finding | Severity | Status | Evidence |
|---|---|---|---|---|
| FND-001 | Independent ADI PDE remains outside full qualification | HIGH | OPEN | `results/aggregated/numerical_remediation_qualification.json` |
| FND-002 | Historical quote asynchrony limits temporal conclusions | HIGH | USAGE_LIMITATION_APPLIED | `results/RESULT_AVAILABILITY.json` |
| FND-003 | Exploratory Full DVE underperformed FitOnly | MEDIUM | RETAINED_NEGATIVE_RESULT | `results/exploratory/synthetic_bs_proxy_v1/synthetic_locked_test.json` |
| FND-004 | Full DVE confirmatory performance (confirmatory v6 release-aligned prequalified observable-quote study) | MEDIUM | EVALUATED | `results/confirmatory/synthetic_v5/confirmatory_metrics.json` |
| FND-005 | Monte Carlo discretization and seed sensitivity | LOW | REMEDIATED_WITH_LIMITATION | `results/aggregated/mc_qe_qem_audit.csv` |
| FND-006 | Confirmatory v3 used oracle DGP parameters for developer pricing | MEDIUM | SUPERSEDED_FOR_PRIMARY_CONFIRMATORY_INFERENCE | `results/confirmatory/synthetic_v2/confirmatory_metrics.json` |
| FND-007 | Confirmatory multistart calibration qualification | LOW | QUALIFIED_WITH_ATTEMPT_DIAGNOSTICS | `results/confirmatory/synthetic_v5/calibration_diagnostics.parquet` |
| FND-008 | Confirmatory v5 failed its frozen validation calibration qualification | HIGH | PRESERVED_PARTIAL_CALIBRATION_UNQUALIFIED | `results/confirmatory/synthetic_v4/confirmatory_metrics.json` |
