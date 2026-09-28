# Model Limitations Register

1. Historical public quotes lack quote timestamps and are not synchronized NBBO.
2. The prospective Cboe marking series currently contains one date.
3. Independent nonuniform MCS/HV PDE qualification is `PARTIAL`.
4. MC passed its remediation criteria, but bias remains scheme/time-step/seed/path dependent.
5. Some real wing and maturity holdouts are unavailable when a date lacks sufficient contracts.
6. Real-market LocalVol and Bates results are challenger sensitivities, not truth. Synthetic
   `LOCALVOL_SSVI_MARGINAL` and piecewise-regime constructions are controlled European-marginal
   DGPs; they do not independently validate Dupire path dynamics or a fitted regime-switch process.
7. Synthetic proxy operators F04-F16 are controlled fault mechanisms, not historical observations.
8. Exploratory BS/proxy locked results are descriptive and cannot be used to tune the confirmatory
   design.
9. Confirmatory v3 is an oracle-parameter model-form isolation because developer values used latent
   DGP parameters; v4 supplies an observable-quote calibration study but remains unqualified.
10. Confirmatory v4 produced finite parameters for every fit, but only 20 of 510 fits satisfied the
    optimizer success criterion before the prospectively frozen iteration cap. Its locked result is
    preserved and interpreted with this calibration-convergence limitation.
11. Confirmatory v5 failed its frozen VALIDATION calibration qualification. It is preserved as
    PARTIAL/CALIBRATION_UNQUALIFIED and is not primary confirmatory evidence.
12. Historical hedge calculations cannot be called actual trading P&L.
13. The model is prohibited for American options, exotics, live trading, capital, and production
    valuation.
