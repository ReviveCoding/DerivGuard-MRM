# Technical Report



## Data Risk

The public HistoricalData.net archive contains 4,294,301 vendor
rows across 127 dates from
2022-07-01 through 2022-12-30. The normalized
SPX/SPXW subset contains
2,481,487 rows; 2,345,505 passed the model-ready filter and
135,982 have auditable exclusion codes. The processed artifact reports
127 market dates.

All historical rows lack `quote_time`; the data are standing EOD research quotes, not synchronized
NBBO. Apparent parity or static-arbitrage violations therefore remain data-quality evidence and are
not automatically attributed to model failure. Official Cboe marking data contain one prospective
date, and the DataShop integration sample was used for schema validation. Massive was skipped when
no API key was available. See `results/RESULT_AVAILABILITY.json` for the specific temporal analyses
that remain unavailable.

The empirical supplement contains 108 measured model rows: {'liquidity_filter': 54, 'near_expiry_1_to_7d': 18, 'settlement_robustness': 36}. These are cross-sectional EOD results, not synchronized P&L.

## Validation Results

## 1. Executive Summary

**Simulated research conclusion: Conditionally validated for the narrow specified research use. This is not institutional approval.**

## 2. Model Purpose and Intended Use

European cash-settled SPXW vanilla research valuation, Greeks, and model-risk validation only.

## 3. Scope

Public-data audit, DEV/VALIDATION calibration studies, independent numerical engines, qualified
synthetic truth engines, fault injection, validation baselines, ablations, and portfolio
materiality. The v6 synthetic study calibrates developer Heston parameters only from observable
synthetic bid/mid/ask quotes after a prospectively frozen calibration prequalification.

## 4. Data and Lineage

Raw public inputs have manifests, source URLs, hashes, schema notes, and explicit processing stages.

## 5. Data Limitations

All 4,294,301 public historical rows lack quote_time and the prospective Cboe marking series currently contains one date. A synchronized next-date contract-continuity study cannot be constructed defensibly.; Missing historical quote timestamps and only one prospective marking date prevent a defensible one-day option/underlying alignment. No live-P&L proxy is reported.

## 6. Methodology

DVE preserves E_data, E_num, E_cal, E_ident, E_param, E_form, E_greek, E_extra, and E_outcome.
Risk evidence and economic materiality remain separate outputs.

## 7. Key Assumptions

Synthetic observation spreads, missingness, and material-error thresholds are frozen design inputs;
synthetic evidence is not production performance evidence.

## 8. Conceptual Soundness

Heston is defensible for the narrow vanilla research use but cannot represent every jump,
local-volatility, regime, or microstructure mechanism.

## 9. Implementation Verification

CF/QuantLib error is 7.194e-13; MC remediation is
`PASS`; PDE remediation remains `PARTIAL`.

## 10. Calibration Assessment

Measured C00-C04, O00-O04 where available, Feller, bootstrap, profile, multistart, and date-level
stability artifacts are reported in T04/T05 and `results/aggregated/`.

## 11. Independent Benchmarking

Independent Monte Carlo, independent ADI PDE, QuantLib, SVI/SSVI, LocalVol, and Bates sensitivity
paths are represented without treating them as interchangeable models.

## 12. Sensitivity Analysis

Objective, optimizer, Feller, numerical-grid, seed, path-count, wing/maturity availability, and
developer-method ablations were evaluated. The N10 benchmark retains FP64 as the canonical
precision after directly measuring FP32 error where its artifact is available.

## 13. Stress Testing

The v6 design covers F00-F16 when its qualified locked output exists; fault coverage is
machine-readable and proxy operators remain explicitly labeled for F04-F16.

## 14. Synthetic Ground-Truth Findings

The immutable exploratory BS/proxy study found FitOnly recall/AUPRC
1.000/0.997 versus Full DVE 0.564/0.819.
It is descriptive because registries were not frozen and must not be tuned against. The confirmatory v6 release-aligned prequalified observable-quote study measured FitOnly recall/AUPRC of 1.000/0.991 and Full DVE recall/AUPRC of 0.600/0.658. The result is retained whether favorable or unfavorable; locked observations are not used for redesign.
The confirmatory v6 release-aligned prequalified observable-quote study contains 510 selected observable-quote C04 calibrations, all 510 qualified, plus 2550 individual-start diagnostics of which 2491 reported optimizer success. Calibration inputs are bid/mid/ask observations, not latent DGP parameters. Preserved confirmatory v2 canonical-truth study: FitOnly recall/AUPRC 1.000/1.000; Full DVE 0.690/0.876. Preserved v3 oracle-parameter model-form isolation: FitOnly recall/AUPRC 0.938/0.990; Full DVE 0.500/0.510. Preserved v4 observable-quote study (calibration-unqualified): FitOnly recall/AUPRC 1.000/1.000; Full DVE 0.583/0.611. Preserved v5 release-aligned study: VALIDATION calibration qualification failed before locked generation; status PARTIAL/CALIBRATION_UNQUALIFIED. V3 remains an oracle-parameter limitation and is not
used as observable-calibration evidence for the primary conclusion.

## 15. Real-Market Outcome Analysis

Cross-sectional and real locked-date comparisons were run. Synchronized temporal outcome claims are
unavailable for the reasons in Section 5; no causality or live-P&L claim is made.
The empirical supplement contains 108 measured model rows: {'liquidity_filter': 54, 'near_expiry_1_to_7d': 18, 'settlement_robustness': 36}. These are cross-sectional EOD results, not synchronized P&L.

## 16. Ongoing Monitoring

See `ONGOING_MONITORING_PLAN.md` and `results/aggregated/monitoring_simulation.csv`.

## 17. Model Limitations

See `MODEL_LIMITATIONS_REGISTER.md`.

## 18. Findings

PDE qualification, quote asynchrony, unfavorable DVE evidence, and MC sensitivity remain visible.

## 19. Materiality Assessment

T11 contains 108 measured portfolio/model rows for six standardized portfolios, including
value, delta, gamma, vega, and separate valuation/Greek materiality.

## 20. Remediation

See `REMEDIATION_TRACKER.md`; completed remediation does not erase the initial negative evidence.

## 21. Residual Model Risk

Moderate within the narrow research use and high outside it, particularly for synchronized outcomes,
PDE-only conclusions, and prohibited products.

## 22. Validation Conclusion

Conditionally validated for the narrow specified research use; all prohibited uses remain prohibited.

## Future Extension Roadmap

- v1.1: licensed synchronized historical Cboe quotes.
- v1.2: specialized 0DTE validation.
- v2/v2.1: rates and credit derivatives.
- v3: XVA and counterparty model-risk extensions.
- Later: GPU neural surrogates as a separate model-risk case study.
