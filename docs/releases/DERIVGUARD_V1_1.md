# DerivGuard-MRM v1.1

DerivGuard-MRM v1.1 extends the frozen v1 model-risk research framework with
post-validation analytics, uncertainty-aware economic slicing, challenger
attribution, stress testing, monitoring, and a fresh confirmatory DVE study.

## Key results

- Independent scientific/code re-review: ACCEPTED.
- 9,720 contract-model observations across 126 market dates were used in the
  reconstructed post-validation analytics.
- 64 machine-readable result files and 49 publication figures were generated in
  the accepted internal research release.
- Ordinary LOCKED_TEST:
  - Full DVE recall: 1.000
  - FitOnly recall: 1.000
  - Full DVE FPR: 0.069
  - FitOnly FPR: 0.103
- OOD:
  - Full DVE FPR: 0.615
  - FitOnly FPR: 0.154
- Therefore the release does not claim global DVE superiority or production
  readiness.
- Sixteen of 39 signed reported-winner-versus-runner-up bootstrap margin
  intervals cross zero, so slice winners are not uniformly decisive.
- Controlled stress repricing produced 155,520 contract-scenario rows and
  240 aggregate rows across spot, volatility, rate, Heston rho, Heston
  vol-of-vol, and spread shocks.
- Heston GPU repricing used FP64. The independent CPU comparison was a bounded
  12-row deterministic spot-scenario check, with maximum absolute difference
  9.095e-13. It was not a comprehensive CPU comparison across every shock family.

## Quality gates

- Ruff formatting/lint: PASS
- Mypy: PASS
- Full local Pytest: 279 passed
- Numerical suite: 16 passed
- Qualified local CUDA/GPU suite: 5 passed
- DVE freeze/evidence integrity: VERIFIED
- Independent scientific/code re-review: ACCEPTED

## Important limitations

- DVE improvement did not transfer robustly to the OOD evaluation.
- Calibration bootstrap/profile-width methods are local diagnostic approximations.
- SVI/SSVI stress outputs use the disclosed sticky-strike convention.
- 0DTE pricing remains unavailable.
- Synchronized temporal OOS evaluation remains unavailable.
- Hedge/P&L conclusions remain unavailable.
- Vendor-derived raw/intermediate market data are not distributed in the public
  repository.
