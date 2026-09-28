# Final Acceptance Report

Overall status: **PARTIAL**. Every status below is derived from
`artifacts/acceptance/gate_evidence.json`, artifact semantics, final test evidence, or a documented
availability blocker. A status-only/NaN table is never accepted as scientific evidence.

| Gate | Status | Evidence / blocker |
|---|---|---|
| ENVIRONMENT | PASS | artifacts/system/environment.json; artifacts/system/environment.json |
| DATA | PASS | artifacts/data/acquisition_inspection.json; artifacts/data/processing_summary.json; data/raw/historicaldata_spx_sample/options_sample_2022H2.zip.manifest.json; results/aggregated/data_eda.json |
| GPU | PASS | artifacts/system/environment.json; results/aggregated/numerical_benchmarks.json; results/aggregated/mc_qe_qem_audit.csv; results/aggregated/numerical_remediation_qualification.json |
| NUMERICAL | PARTIAL | artifacts/tests/final_test_summary.json; artifacts/tests/final_test_summary.json; artifacts/tests/final_test_summary.json; results/aggregated/numerical_benchmarks.json; PDE status=PARTIAL; finest case/scheme errors=[{'case': 'base', 'scheme': 'hundsdorfer_verwer', 'absolute_error_cf': 0.0004325343067339}, {'case': 'base', 'scheme': 'modified_craig_sneyd', 'absolute_error_cf': 0.0004342999242687}, {'case': 'feller_violated', 'scheme': 'hundsdorfer_verwer', 'absolute_error_cf': 0.0008833263876515}, {'case': 'feller_violated', 'scheme': 'modified_craig_sneyd', 'absolute_error_cf': 0.0009157043198495}, {'case': 'positive_rho', 'scheme': 'hundsdorfer_verwer', 'absolute_error_cf': 0.1345919623197033}, {'case': 'positive_rho', 'scheme': 'modified_craig_sneyd', 'absolute_error_cf': 0.1346169402217416}] |
| INDEPENDENCE | PARTIAL | results/aggregated/numerical_benchmarks.json; results/aggregated/mc_qe_qem_audit.csv; results/aggregated/numerical_remediation_qualification.json; results/aggregated/pde_adi_convergence.csv; PDE status=PARTIAL; finest case/scheme errors=[{'case': 'base', 'scheme': 'hundsdorfer_verwer', 'absolute_error_cf': 0.0004325343067339}, {'case': 'base', 'scheme': 'modified_craig_sneyd', 'absolute_error_cf': 0.0004342999242687}, {'case': 'feller_violated', 'scheme': 'hundsdorfer_verwer', 'absolute_error_cf': 0.0008833263876515}, {'case': 'feller_violated', 'scheme': 'modified_craig_sneyd', 'absolute_error_cf': 0.0009157043198495}, {'case': 'positive_rho', 'scheme': 'hundsdorfer_verwer', 'absolute_error_cf': 0.1345919623197033}, {'case': 'positive_rho', 'scheme': 'modified_craig_sneyd', 'absolute_error_cf': 0.1346169402217416}] |
| CALIBRATION | PASS | reports/tables/T04_CALIBRATION_OBJECTIVE_COMPARISON.csv; reports/tables/T05_OPTIMIZER_IDENTIFIABILITY.csv; results/aggregated/feller_study.csv; results/aggregated/bootstrap_calibration.parquet |
| RELEASE | PASS | artifacts/model_releases/model_release_v1.0.json; artifacts/model_releases/model_release_v1.1.json |
| SYNTHETIC | PASS | results/confirmatory/synthetic_v5/truth_qualification.json; results/confirmatory/synthetic_v5/fault_coverage.csv; results/confirmatory/synthetic_v5/confirmatory_metrics.json; results/confirmatory/synthetic_v5/calibration_diagnostics.parquet |
| DVE | PASS | results/confirmatory/synthetic_v5/dve_thresholds.freeze.json; artifacts/checkpoints/synthetic_confirmatory_v6.freeze.json; results/confirmatory/synthetic_v5/T08_DVE_BASELINES.csv |
| REAL-DATA | PARTIAL | reports/tables/T06_PRICING_MODEL_COMPARISON.csv; results/aggregated/near_expiry_study.csv; results/aggregated/liquidity_filter_study.csv; results/aggregated/am_pm_robustness.csv; All 4,294,301 public historical rows lack quote_time and the prospective Cboe marking series currently contains one date. A synchronized next-date contract-continuity study cannot be constructed defensibly.; Missing historical quote timestamps and only one prospective marking date prevent a defensible one-day option/underlying alignment. No live-P&L proxy is reported. |
| STATISTICAL | PASS | results/statistics/date_level_inference.csv; results/confirmatory/synthetic_v5/statistical_inference.json |
| ABLATION | PASS | results/confirmatory/synthetic_v5/T09_DVE_ABLATION.csv; results/aggregated/developer_method_ablation_summary.csv |
| MATERIALITY | PASS | reports/tables/T11_PORTFOLIO_MATERIALITY.csv |
| GOVERNANCE | PASS | AGENTS.md; desktop_study/DESKTOP_STUDY.md; desktop_study/literature.yaml; desktop_study/regulatory_mapping.yaml |
| REPRODUCIBILITY | PASS | artifacts/system/environment.json; artifacts/model_releases/model_release_v1.0.json; artifacts/model_releases/model_release_v1.1.json; results/confirmatory/synthetic_v5/dve_thresholds.freeze.json |
| REPORTING | PASS | reports/TECHNICAL_REPORT.md; reports/INDEPENDENT_VALIDATION_MEMORANDUM.md; artifacts/acceptance/gate_evidence.json |
