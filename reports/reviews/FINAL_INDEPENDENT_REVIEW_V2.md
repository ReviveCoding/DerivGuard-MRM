# Final Independent Review V2 — Current Remediated State

Review timestamp: 2026-09-27T13:58:17Z  
Reviewed source-tree SHA-256: `9bda28e21b3ee28f7689264f7834d3285eba499e5eee8f55b8c83f64f8122e4e`  
Disposition: **PASS — no unresolved Critical or High review finding**

This is an independent scientific/code review of the current remediated repository. It is not an
institutional model approval. PASS means the implementation and evidence may proceed to final
evidence-based reporting; it does not promote the PDE numerical gate or unavailable temporal
outcome work to PASS.

No frozen confirmatory-v6 source, registry, threshold, or scientific result was changed. One
non-frozen reporting defect was corrected during review: the report generator had counted every
multistart attempt as a selected calibration. It now distinguishes 510 selected calibrations from
2,550 individual-start attempts. The complete verification suite was rerun after that edit.

## Executive conclusion

The earlier High findings concerning unqualified calibration, release-method divergence, and weak
review gating are closed:

- Developer Model Release v1.1 freezes C04/O01/unconstrained after DEV selection and VALIDATION
  confirmation. Its developer-CF, calibration, empirical, tolerance, selection, execution, and
  prior-release hashes are current. Real-date execution used five independent starts without
  cross-date warm starts; 135 of 136 jobs converged, and the failure is retained and excluded.
- Confirmatory v6 is linked to v1.1 and uses observable bid/mid/ask inputs with true O01 multistart
  C04 calibration. All 102 primary and 408 bootstrap selected fits qualified. Of 2,550 individual
  starts, 2,491 converged; failed starts were candidates, not accepted fits.
- V6 was frozen before execution, DEV/VALIDATION thresholds were frozen before locked generation,
  and the 107m/108m/109m/110m seed regions are disjoint. V5 stopped in VALIDATION without evaluating
  a locked seed. Current semantic checks pass the release, freeze, provenance, calibration, fault,
  baseline, and ablation links.
- Phase and acceptance status is evidence driven. Final test/review evidence must match the exact
  current source-tree hash, and unresolved Critical/High review counts must be zero.

The confirmatory result is adverse to Full DVE and remains unchanged. Across 51 locked/OOD cases
and 10 material positives, FitOnly achieved recall/AUPRC `1.000/0.9909`, while Full DVE achieved
`0.600/0.6575`. Full DVE's realized false-positive rate was `0.1463`, despite a 5% DEV threshold
target. Regime conditioning did not change the aggregate V05/V06 result. The evidence does not
support a claim that Full DVE dominates FitOnly.

## Scope reviewed

- developer/validator independence;
- developer Heston characteristic function and QuantLib comparison;
- independent full-truncation, QE, and QE-M Monte Carlo logic;
- nonuniform-grid MCS/HV ADI PDE, mixed derivative, variance-zero row, and convergence;
- DEV selection, VALIDATION confirmation, and locked-test leakage controls;
- Developer Model Release v1.1 and its hashes;
- v5/v6 freeze chronology, seed separation, calibration qualification, and provenance;
- DVE arithmetic, V00–V06 baselines, A01–A11 ablations, and materiality labels;
- canonical synthetic Heston/Bates/SSVI-marginal/piecewise-regime constructions and proxy faults;
- case/date-level statistics and confidence-interval limitations;
- evidence-based phase/gate logic and report generation;
- CUDA FP64 correctness, workload selection, and benchmark claims;
- real-data studies, portfolio materiality, and unavailable temporal work.

## Findings

### Accepted High project limitations — not unresolved review defects

**FR2-AH01 — Independent PDE remains unqualified.** The requested nonuniform grid, explicit mixed
term, variance-zero treatment, Rannacher damping, and MCS/HV schemes are implemented. Recorded
finest absolute CF errors are about `4.33e-4` in the base case, `8.83e-4` to `9.16e-4` under Feller
violation, and `1.346e-1` in the positive-rho case. Successive-grid and nonnegativity requirements
also fail. The PDE must remain a bounded diagnostic and the numerical gate must remain PARTIAL.

**FR2-AH02 — Synchronized temporal outcomes remain unavailable.** All 4,294,301 public historical
rows lack quote timestamps and the prospective Cboe marking series has one date. Next-date OOS and
the one-day discrete hedge proxy therefore remain UNAVAILABLE. Cross-sectional and held-out work
remains valid within its documented scope.

### Unresolved Medium limitations

**FR2-M01 — Model-form material-positive coverage is weak.** V6 runs genuine Bates,
SSVI-marginal, and piecewise-regime truths, but none of the locked F01–F03 cases exceeded the frozen
one-spread-unit median material-error threshold. The 10 material positives arose in F04, F07, F11,
F12, and F13. V6 therefore provides little direct detection-power evidence for economically
material F01/F02/F03 cases.

**FR2-M02 — F04–F16 are controlled perturbation proxies.** Seventy-eight of 102 v6 cases are
explicitly flagged proxy faults. F09/F10 alter evidence fields rather than constructing complete
corrupted/stale quote histories, and F05 perturbs disagreement rather than executing a coarse PDE.
The registry/provenance is honest, but results cannot be generalized to all degraded engines.

**FR2-M03 — Confirmatory precision is small-sample and descriptive.** Locked plus OOD contains 51
cases, 10 material positives, and three observations per fault. The 250-replicate paired case
bootstrap is descriptive, does not preserve fault strata, and is not a production performance
bound.

**FR2-M04 — The v6 freeze does not hash every transitive implementation module.** It binds v6,
its v5 calibration dependency, registries, release, configuration, schedule, and prequalification,
but not every imported truth/fault/DVE module individually. Those modules predate the freeze and
this review binds the complete current source tree. A future freeze should include a transitive
source/content manifest.

**FR2-M05 — Public real-data inference remains cross-sectional and sampled.** The 127 PM surfaces,
AM robustness, near-expiry, liquidity, wing/maturity availability, objective/optimizer,
identifiability, stability, and materiality work are measured, but heavy calibration studies use a
predeclared representative-date/quote design. Treasury yields are sensitivity references, not an
exact option discount curve.

### Unresolved Low limitation

**FR2-L01 — Some individual calibration starts do not converge.** V6 records 2,491 successes among
2,550 starts. Every selected primary/bootstrap fit converged, so this does not invalidate the locked
study, but start-level failures should remain visible.

### Resolved during this review

**FR2-R01 — Report generator conflated starts with selected fits.**
`src/derivguard/reporting/final_reports.py` now counts `start_index == -1` as selected fits and
reports candidate-start success separately. Frozen inputs/results were untouched. Reports predating
v6 must be regenerated; the reviewed generator selects v6 and preserves earlier history.

## Areas reviewed with no material defect found

- **Developer Heston CF:** stable decaying-root/Little-Heston-Trap-style formulation, FP64
  CPU/CUDA agreement, and recorded QuantLib absolute error of about `7.19e-13`.
- **Independence:** validator MC/PDE kernels do not import the developer CF. The remediation runner
  imports it only as a comparator, not inside validator kernels.
- **Monte Carlo:** Andersen moments, QE-M conditional-MGF correction, antithetic pairing, control
  variate, and CI sampling units are coherent. The audit uses multiple seeds, time grids, and path
  counts and keeps bias separate from standard error.
- **PDE:** MCS/HV splitting, scheme-specific anchors, nonuniform derivatives, mixed derivative,
  variance-zero equation, and boundaries are substantively present. Adverse convergence is retained.
- **DVE/locking:** latent truth is used for labels only; DEV alone fits scales/thresholds; locked
  evaluation does not refit; all V00–V06 and A01–A11 outputs exist.
- **Statistics:** real inference is date/surface-level, not option-row IID; synthetic inference is
  case-level and explicitly descriptive.
- **GPU:** research-scale truth, population calibration, bootstrap valuation, and MC use CUDA FP64
  with accuracy checks. Small GPU jobs may be slower, and CPU PDE is retained when CuPy is slower.
- **Portfolio materiality:** T11 has 108 rows for P01–P06 with finite value, delta, gamma, vega,
  and separate valuation/Greek materiality.

## Exact verification evidence

- Full verification after the reporting fix: `ruff format --check`, `ruff check`, and `mypy src`
  all exit 0; `pytest` reports **275 passed, 0 failed, 0 errors, 0 skipped** in 17.99 s.
- Reviewer-focused native subset: **39 passed** in 14.70 s, covering gates, empirical logic,
  portfolio materiality, v6 freeze, DVE, CF/QuantLib, MC/QE-M, PDE, and GPU checks.
- Structured full-suite evidence: `artifacts/tests/final_test_summary.json`.
- Source-tree SHA-256 in both test evidence and this review:
  `9bda28e21b3ee28f7689264f7834d3285eba499e5eee8f55b8c83f64f8122e4e`.

## Final review decision

There are zero unresolved Critical and zero unresolved High review findings. The repository may
proceed to report regeneration and final acceptance evaluation. PDE and synchronized temporal
outcome gates must remain PARTIAL/UNAVAILABLE. The final narrative must state that confirmatory
Full DVE underperformed FitOnly and that real model-form cases produced no material positives at
the frozen threshold.
