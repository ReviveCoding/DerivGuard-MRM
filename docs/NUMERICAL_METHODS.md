# Independent Heston numerical methods

This note documents the validator implementations. It does not revise the
central tolerances in `configs/numerical_tolerances.yaml`.

## Monte Carlo: full truncation, QE, and QE-M

`src/derivguard/validation/heston_mc.py` is independent of the developer
characteristic-function kernel. It implements full-truncation Euler and the
two-branch quadratic-exponential (QE) conditional variance approximation from
Andersen (2008). The `qe` stock update uses the standard symmetric
`gamma1 = gamma2 = 1/2` coefficients. The `qe_m` update applies the conditional
martingale correction: after integrating the independent stock normal, it
evaluates the conditional MGF of the QE variance approximation at
`K2 + K4/2`, then selects the drift term that makes the one-step discounted
stock expectation equal to the current discounted stock.

The quadratic branch uses the shifted non-central-chi-square MGF; the
atom-plus-exponential branch uses the mixture MGF. The implementation rejects
an invalid MGF domain instead of clipping it. QE-M corrects the discrete stock
martingale but does not make the variance approximation or finite-time-step
option value exact. Qualification therefore reports time-step bias, path-count
convergence, independent-seed dispersion, standard errors, and CF/QuantLib
coverage separately. Antithetic pairs are averaged before standard-error
estimation, so each pair is one independent sampling unit.

Primary source: Leif Andersen, “Simple and Efficient Simulation of the Heston
Stochastic Volatility Model,” *Journal of Computational Finance* 11(3), 2008,
[DOI 10.21314/JCF.2008.189](https://doi.org/10.21314/JCF.2008.189).

## PDE: nonuniform MCS and HV ADI

`src/derivguard/validation/heston_pde.py` retains the original full-operator
implicit-Euler and Crank–Nicolson reference solvers and adds two independent
ADI schemes:

- Modified Craig–Sneyd (MCS), with `theta = 1/3`;
- Hundsdorfer–Verwer (HV), with `theta = 1/2 + sqrt(3)/6`.

The Heston spatial operator is split into `A0 + A1 + A2`: `A0` contains the
mixed spot/variance derivative and is explicit, while `A1` and `A2` are the
implicitly solved spot and variance directions. Three-point nonuniform
finite-difference weights are generated from local mesh widths. Sinh grids
contain the valuation spot and initial variance exactly and concentrate nodes
near them. At variance zero, the degenerate first-order Heston PDE is applied
with a second-order forward variance derivative; no artificial reflecting
boundary is imposed there. The truncated upper-variance boundary uses a
homogeneous Neumann condition. Dirichlet asymptotic values are imposed at the
spot edges. Two initial Rannacher steps (four implicit half steps) damp the
payoff kink before ADI stepping.

Qualification varies spot nodes, variance nodes, time steps, and domain
boundaries, and includes Feller-violating and high-correlation regimes. It
compares independently with the developer CF, QuantLib, and Monte Carlo
intervals where available. Failure of the unchanged registry tolerance
remains a reported numerical finding.

Primary sources:

- K. J. in ’t Hout and S. Foulon, “ADI Finite Difference Schemes for Option
  Pricing in the Heston Model with Correlation,” *International Journal of
  Numerical Analysis and Modeling* 7(2), 2010. The project’s source record is
  `desktop_study/literature.yaml` entry `PDE-INTHOUT-FOULON-2010`.
- K. J. in ’t Hout and B. D. Welfert, “Stability of ADI Schemes Applied to
  Convection-Diffusion Equations with Mixed Derivative Terms,” *Applied
  Numerical Mathematics* 57, 2007. See the matching desktop-study record.

## Executable evidence

`run_numerical_remediation(profile="research", device="cuda")` in
`src/derivguard/validation/numerical_remediation.py` produces:

- `results/aggregated/mc_qe_qem_audit.csv`;
- `results/aggregated/pde_adi_convergence.csv`;
- `results/aggregated/numerical_remediation_qualification.json`.

The JSON status is derived from measured rows and the unchanged tolerance
registry. A PARTIAL result is retained if convergence or non-negativity does
not meet the registered criteria.
