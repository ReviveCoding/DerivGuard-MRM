# DerivGuard-MRM

[![CI](https://github.com/ReviveCoding/DerivGuard-MRM/actions/workflows/ci.yml/badge.svg)](https://github.com/ReviveCoding/DerivGuard-MRM/actions/workflows/ci.yml)

Evidence-controlled model-risk research for SPX option pricing and validation.

DerivGuard-MRM is a research framework for developing and independently validating a
Heston option-pricing model against SVI/SSVI challengers. The accepted v1.1 extension
adds calibration diagnostics and uncertainty, economic slicing and residual analysis,
model-winner and Greek-disagreement analysis, controlled stress repricing, DVE
diagnostics, and fresh ordinary/OOD confirmatory testing.

The release passed an independent scientific/code re-review. It is research software
only: it is not approved for live trading, production valuation, capital, or brokerage
execution.

## v1.1 evidence

- [Final v1.1 acceptance report](reports/v1_1/FINAL_V1_1_ACCEPTANCE_REPORT.md)
- [Independent scientific/code review](reports/v1_1/INDEPENDENT_REVIEW.md)
- [Post-validation analytics](reports/v1_1/POST_VALIDATION_ANALYTICS.md)
- [Fresh DVE confirmatory study](reports/v1_1/DVE_V1_1_CONFIRMATORY.md)
- [Publication provenance and data exclusions](docs/PUBLICATION_PROVENANCE.md)

Ordinary `LOCKED_TEST` favored Full DVE on false-positive rate (`0.069` versus
`0.103` for FitOnly), with recall `1.000` for both. That improvement did not transfer
robustly: OOD false-positive rate was `0.615` for Full DVE versus `0.154` for FitOnly.
The release therefore makes no claim of global DVE superiority or production readiness.

Sixteen of 39 signed winner-versus-runner-up bootstrap margin intervals cross zero.
0DTE pricing, synchronized temporal OOS evaluation, and hedge/P&L conclusions remain
unavailable.

## Validation and CI

The accepted local research environment recorded 279 passing Pytest tests, including
16 numerical tests and five CUDA/GPU tests on the qualified local NVIDIA environment.
GitHub-hosted CI is intentionally CPU-only: it installs the development and numerical
oracle dependencies, checks formatting, lint, typing and package dependencies, runs all
non-GPU tests plus the dedicated numerical suite, and verifies the published v1.1 DVE
freeze/evidence hashes without modifying them.

The five accepted CUDA tests are local-only and are not represented as having run on a
GitHub-hosted CPU runner. The full calibration and stress research programs are also not
reproduced in hosted CI.

## Scope and governance

The authoritative scope, intended uses, limitations, hypotheses, experiments, and
numerical tolerances are maintained in `governance/`, `experiments/`, and `configs/`.
This public history is a sanitized publication history: licensed/vendor-derived quote
data, intermediate market data, empirical checkpoints, and contract-level derived tables
are not distributed.

