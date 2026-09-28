# Final v1.1 Acceptance Report

Date: 2026-09-27  
Scope: additive post-validation research analytics  
Protected baseline: immutable `derivguard-v1.0` evidence and locked results

## Decision

**PASS / ACCEPTED.** Implementation, execution, automated quality gates, and the independent scientific/code re-review are complete.

The evidence supports acceptance of v1.1 as a diagnostic research extension, not as a replacement release, production policy, or claim that one pricing model or DVE rule is globally superior.

## Evidence classification

### New measured real-data results — exploratory

- The reconstructed released-surface sample contains 9,720 contract-model rows across 126 dates. It is cross-sectional and bounded, not the full option universe.
- All 135 valid Heston calibrations retain objective, price RMSE, IV RMSE, vega-weighted IV error, normalized error, interval-hit rate, signed bias, optimizer counts/status, boundary proximity, Feller status, and support/liquidity covariates.
- Every calibration has CUDA-FP64 price Jacobians and 400 CPU-FP64 linearized bid/ask quote-bootstrap propagations, ten pairwise parameter correlations, five local nuisance-profile widths, and four actual bounded local optimization starts. The bootstrap/profile methods are local diagnostic approximations. Multistart dispersion retains every finite terminal solution, including unsuccessful optimizer statuses; success counts are reported separately. Multistart convergence counts were: 107 calibrations with four, 21 with three, six with two, and one with one successful start.
- Median Heston price RMSE was 0.634 on DEV, 0.906 on VALIDATION, and 0.937 on LOCKED_TEST. The observed Feller-satisfaction rate was zero in all three partitions; this adverse diagnostic is retained.
- Fixed DEV economic slices and all five required two-dimensional grids include market-date bootstrap intervals for price, IV, normalized-tail, interval-hit, and Greek-disagreement metrics. There are 475 one-dimensional and 1,285 two-dimensional metric rows; defensible rows have no missing lower confidence limits.
- Winner selection excludes the unqualified fixed Bates sensitivity. SVI won 8 of 13 defensible DEV cells, Heston won five; SVI won all 14 VALIDATION and all 12 LOCKED_TEST cells. Median bootstrap winner probability was 0.990, 0.998, and 0.980 respectively. Sixteen of 39 signed reported-winner-versus-runner-up margin intervals cross zero, so slice winners are not uniformly decisive. This heterogeneity does not establish a globally best model.
- Bates remains only an inherited fixed jump sensitivity for descriptive disagreement. LocalVol remains unavailable for contract-level winner, attribution, Greek, and stress comparisons because it lacks contract-level qualification.
- Residual distributions, challenger attribution, 170 low-valuation/high-Greek exceptions, six standardized portfolio slices, parameter stability/recalibration risk, and DEV-tree transfer are exploratory findings.
- Production-style monitoring uses DEV median/MAD, EWMA, and the declared CUSUM decision limit. Portfolio monitoring remains `UNAVAILABLE` because no DEV portfolio reference distribution exists. Statuses are transparent research signals, not JPMorgan rules.
- Controlled stress repricing produced 155,520 contract-scenario rows and 240 aggregates for spot, volatility, rates, Heston rho, Heston vol-of-vol, and spread shocks. Heston ran on the RTX in FP64. A bounded independent CPU spot check over 12 deterministic rows from a spot-shock scenario had maximum absolute error `9.095e-13`; it is not a comprehensive CPU comparison across every shock family. SVI/SSVI use a disclosed sticky-strike convention. Bates and LocalVol are excluded for qualification reasons.

### Inherited v1 result — descriptive diagnosis only

- The v1 release and its locked evidence were not rewritten, retuned, deleted, or reinterpreted. Git shows no tracked changes in the named protected v1 reports, raw/aggregate/materiality/confirmatory results, or model-release file.
- Across the preserved 51 v1 locked/OOD cases, FitOnly had two false positives, zero false negatives, recall 1.000, and FPR 0.049. Full DVE had six false positives, four false negatives, recall 0.600, and FPR 0.146.
- Relative to FitOnly, Full DVE added five false positives, lost four FitOnly true positives, rescued no FitOnly miss, and removed one FitOnly false positive. Three of its six false positives were in `LONG | WING | LIQUID`.
- Evidence-component attribution is persisted by Full-versus-FitOnly outcome. It was not used to tune v1.1.

### Fresh v1.1 confirmatory result

- The first preregistered v1.1 design stopped before locked generation because VALIDATION seed `118000007` was calibration-unqualified. Its partial results and failure record are preserved.
- The v1.1b remediation changed only the bootstrap search budget, passed disjoint prequalification, and froze hypotheses, experiment design, thresholds, aggregation, materiality, parameter ranges, and wholly new 121m-124m seeds before locked execution.
- VALIDATION selected `top_2_evidence_mean` before locked generation.
- On the primary fresh LOCKED_TEST partition (34 cases; five positives), FitOnly and Full v1.1 DVE both had recall 1.000 and AUPRC 1.000. Full had FPR 0.069 versus 0.103 for FitOnly. Thus Full met the preregistered near-target FPR/recall objective on ordinary locked data.
- On the separately frozen OOD partition (17 cases; four positives), both retained recall 1.000, but Full FPR rose to 0.615 versus 0.154 for FitOnly and Full AUPRC fell to 0.591 versus 1.000.
- Combined locked/OOD FPR was 0.238 for Full versus 0.119 for FitOnly. Therefore the ordinary locked finding is favorable, but robustness failed materially. No global superiority or production-readiness claim is supported.
- Positive counts and confidence intervals are small/wide; the exact uncertainty is reported rather than hidden.

### Unavailable analyses

- 0DTE pricing remains a separately identified but unmeasured stress domain: released real surfaces begin at seven DTE and timestamp-free EOD quotes cannot establish settlement-relative maturity.
- Clean temporal out-of-sample, hedge, and P&L outcomes remain unavailable because historical quotes are unsynchronized.
- Prospective synchronized Cboe evidence remains insufficient for those claims.
- LocalVol contract-level comparisons remain unavailable as described above.

## Scientific controls and provenance

- Real-data reconstruction, reviewed slice definitions, calibration-support definitions, monitoring thresholds, calibration uncertainty, and stress repricing have content-hashed manifests under `artifacts/v1_1/`.
- Challenger rows use validated one-to-one contract-key joins; positional matching is prohibited.
- Low-value/high-Greek ranks use DEV empirical reference distributions in every partition.
- The DVE design-module hashes still match both design freezes, the aggregation freeze references the v1.1b design freeze, and 27 DVE/failure evidence files are closure-hashed.
- New DVE seed regions have no overlap with preserved v1 seeds, and no locked label was used for selection or tuning.
- Negative and adverse findings, including the failed initial design and OOD Full-DVE failure, are retained.

## Quality gates

| Gate | Result |
| --- | --- |
| Ruff format | PASS — 97 files |
| Ruff lint | PASS |
| Mypy | PASS — 64 source files |
| Full Pytest | PASS — 279 tests |
| Numerical suite | PASS — 16 tests |
| CUDA/GPU suite | PASS — 5 tests |
| Heston stress CPU/GPU agreement | PASS — bounded 12-row spot-scenario check, max absolute error `9.095e-13` |
| DVE freeze/integrity verification | PASS |
| Independent scientific/code re-review | PASS / ACCEPTED |

The ordinary sandbox Pytest attempt before remediation reached 250 passing tests but had 28 Windows repository-`.tmp` lock-file setup errors. The identical suite, and later the expanded 279-test suite, passed with an isolated AppData basetemp; this environmental event is retained for transparency.

## Deliverables

All twelve required reports are present under `reports/v1_1/`, plus the controlled stress report and this acceptance record. Machine-readable Parquet/CSV/JSON evidence is under `results/v1_1/`; 49 publication-quality PNG figures are under `reports/v1_1/figures/`; reviewed freezes and manifests are under `artifacts/v1_1/`, with registries under `governance/` and `experiments/`.
