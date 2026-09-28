# Executive Summary

DerivGuard-MRM has measured environment, data, calibration, numerical-validation, synthetic,
cross-sectional, and portfolio-materiality artifacts. Developer CF agrees with QuantLib to
7.194e-13; the audited GPU MC status is
`PASS`, while the remediated ADI PDE remains `PARTIAL`.

The exploratory BS/proxy study is preserved unchanged and unfavorable to Full DVE:
FitOnly recall/AUPRC 1.000/0.997 versus Full DVE
0.564/0.819. The confirmatory v6 release-aligned prequalified observable-quote study measured FitOnly recall/AUPRC of 1.000/0.991 and Full DVE recall/AUPRC of 0.600/0.658. The result is retained whether favorable or unfavorable; locked observations are not used for redesign.
The confirmatory v6 release-aligned prequalified observable-quote study contains 510 selected observable-quote C04 calibrations, all 510 qualified, plus 2550 individual-start diagnostics of which 2491 reported optimizer success. Calibration inputs are bid/mid/ask observations, not latent DGP parameters. Preserved confirmatory v2 canonical-truth study: FitOnly recall/AUPRC 1.000/1.000; Full DVE 0.690/0.876. Preserved v3 oracle-parameter model-form isolation: FitOnly recall/AUPRC 0.938/0.990; Full DVE 0.500/0.510. Preserved v4 observable-quote study (calibration-unqualified): FitOnly recall/AUPRC 1.000/1.000; Full DVE 0.583/0.611. Preserved v5 release-aligned study: VALIDATION calibration qualification failed before locked generation; status PARTIAL/CALIBRATION_UNQUALIFIED.

The historical public sample is unsynchronized, so cross-sectional conclusions are feasible but
next-date synchronized OOS and the one-day discrete hedging proxy remain unavailable. T11 contains
real measured portfolio valuation and Greek disagreements; it is not a status placeholder.
