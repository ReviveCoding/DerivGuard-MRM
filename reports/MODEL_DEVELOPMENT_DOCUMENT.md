# Model Development Document

## Intended use and release discipline

The developer Heston model is restricted to European SPXW vanilla-option research valuation and
Greeks. The selected methodology record is `SELECTED_ON_DEV_CONFIRMED_ON_VALIDATION` and states objective
`C04`, optimizer `O01`, and
Feller treatment `unconstrained`. Selection uses DEV and confirmation
uses VALIDATION; locked observations are excluded from selection.

## Calibration evidence

T04 contains 18 measured objective-study rows. On the common C04 evaluation, the lowest
measured DEV row was C04 at 3.0993.
T05 contains 5 optimizer rows plus measured identifiability diagnostics; its
lowest native optimizer objective was O01 at
3.6919. These bounded comparisons are empirical measurements,
not assertions that convergence succeeded in every run.

## Numerical verification

Developer CF and QuantLib differ by 7.194e-13 at the reference
case. The independent MC remediation status is `PASS` after Euler, QE, and
QE-M
multi-seed/path/time-step testing. Bias and sampling standard error remain separate. The independent
nonuniform MCS/HV PDE status is `PARTIAL`: remediation was seriously attempted,
but difficult regimes and convergence/nonnegativity criteria prevent a full pass.

## Cross-sectional model evidence

- M01_MARKET_IV_REFERENCE: median price RMSE 0.4343 over 135 measured rows
- M02_SVI: median price RMSE 0.7288 over 135 measured rows
- M10_HESTON: median price RMSE 0.8332 over 171 measured rows
- M03_SSVI: median price RMSE 4.8253 over 135 measured rows
- M20_LOCALVOL_MC: median price RMSE 5.0832 over 18 measured rows
- M21_BATES_SENSITIVITY: median price RMSE 7.8262 over 18 measured rows
- M00_BS_FLAT: median price RMSE 31.1123 over 135 measured rows
