# DerivGuard-MRM Desktop Study

**Study date:** 2026-09-27  
**Source retrieval date:** 2026-09-27 unless stated otherwise  
**Scope:** governance, SPX/SPXW conventions, public data, pricing and validation methods, independent numerical engines, and GPU-compute implementation implications.

## Executive conclusions

1. The current U.S. interagency model-risk baseline is **SR 26-2 (April 17, 2026)**, not SR 11-7. SR 26-2 expressly supersedes SR 11-7 and emphasizes a risk-based approach tied to inherent risk, exposure, purpose, use, and materiality. Its core validation components are conceptual soundness, outcomes analysis, and ongoing monitoring, supported by effective challenge, clear roles, a model inventory, documentation, and controls. DerivGuard is a research project rather than an institutional approval process, but its governance design should map to these principles. [Federal Reserve SR 26-2](https://www.federalreserve.gov/supervisionreg/srletters/SR2602.htm); [revised interagency guidance](https://www.federalreserve.gov/frrs/guidance/supervisory-guidance-on-model-risk-management.htm).
2. Cboe currently documents traditional SPX as AM-settled and SPXW weekly/end-of-month contracts as PM-settled; both are European-style and cash-settled. They must not be pooled without explicit settlement controls. The empirical core should therefore be PM-settled SPXW, with traditional AM-settled SPX treated as a separate robustness study. [Cboe SPX Weeklys specifications](https://www.cboe.com/tradable_products/sp_500/spx_weekly_options/specifications); [Cboe SPX product specifications](https://www.cboe.com/tradable-products/sp-500/spx-options/spx-specifications).
3. The HistoricalData.net public option sample is documented as 127 end-of-day files from 2022-07-01 through 2022-12-30, including SPX/SPXW and 34 fields. Its historical rows are not a synchronized closing NBBO: the provider states each contract records its own last standing quote, and the 2022 sample's `quote_time` is generally unavailable. The downloaded archive must be independently hashed and inspected before any coverage claim becomes an observed project result. [sample page](https://historicaldata.net/samples.html); [provider FAQ](https://historicaldata.net/faq.html); [options documentation](https://historicaldata.net/options.html).
4. Cboe publishes free daily proprietary-index marking-price CSVs for SPX/SPXW. Cboe explicitly says these indicative marks are not actual OPRA NBBO updates and may differ from the electronic-book BBO. They are suitable for a prospective locked collector, with the mark, contemporaneous BBO, retrieval time, and file hash preserved. [Cboe proprietary index marking prices](https://www.cboe.com/markets/us/options/market-statistics/product-data/proprietary-index-marking-prices/).
5. A stable Heston characteristic-function implementation must avoid complex-log branch discontinuities and be tested beyond well-behaved parameter regions. Independent Monte Carlo and PDE validators should not reuse the developer kernel. Full-truncation Euler and Andersen's QE method are defensible MC comparisons; Modified Craig-Sneyd and Hundsdorfer-Verwer are defensible ADI candidates for the mixed-derivative Heston PDE.
6. SSVI provides a smooth, static-arbitrage-aware total-variance surface suitable for Dupire differentiation. It is a challenger and preprocessing surface, not ground truth. Local volatility fits a contemporaneous surface by construction but imposes deterministic spot/time volatility dynamics. Bates adds jumps and is therefore a useful model-form challenger and synthetic truth model.
7. QuantLib is appropriate as an external oracle, not as a substitute for project implementations. Cross-checks must lock conventions, parameter order, curves, day counts, evaluation date, integration method, and tolerances.
8. PyTorch's current Windows installation guidance supports Python 3.9–3.12, so Python 3.12 is the defensible canonical environment choice rather than the host Python 3.13. GPU timing must synchronize CUDA work. Research pricing should default to FP64 and explicitly control reduced-precision/TF32 behavior where relevant. [PyTorch Windows installation](https://docs.pytorch.org/get-started/locally/); [PyTorch CUDA semantics](https://docs.pytorch.org/docs/main/notes/cuda.html).

## 1. Regulatory and governance study

### Current guidance

On 2026-04-17 the Federal Reserve, OCC, and FDIC issued revised guidance through SR 26-2. The letter states that the revision supersedes SR 11-7 and SR 21-8. The attached guidance is principles-based and says it does not set enforceable prescriptive standards; its stated applicability is chiefly banking organizations above $30 billion in assets, while allowing relevance to smaller organizations with significant model risk.

The revised framework distinguishes:

- **inherent risk:** assumptions, complexity, input quality, and data constraints;
- **model exposure and purpose:** the quantitative and qualitative drivers of materiality;
- **model use:** misuse or use beyond intended purpose can create risk even for a technically sound model;
- **effective challenge:** objective critical analysis by suitably expert and independent personnel;
- **validation:** reliability and limitations, including conceptual soundness and outcomes analysis;
- **monitoring:** performance under changing products, exposure, data relevance, and market conditions;
- **governance:** roles, accountability, inventory, documentation, remediation, and oversight of vendor products.

### DerivGuard interpretation

DerivGuard should not describe itself as complying with or receiving approval under SR 26-2. It should state that a research governance framework is **informed by** the guidance. The project-specific implications are:

- maintain a versioned model inventory and explicit intended/prohibited uses;
- separate model evidence from economic materiality because SR 26-2 explicitly connects oversight intensity to exposure and purpose;
- preserve physical and methodological independence between developer and validator implementations;
- use model-form challengers, independent numerical methods, outcomes analysis, and ongoing monitoring rather than fit metrics alone;
- document public-data and vendor limitations, including unavailable source code or methodology;
- freeze the developer release and validation thresholds before locked testing;
- maintain findings, remediation, limitations, and residual-risk records.

The detailed mapping is in [`regulatory_mapping.yaml`](regulatory_mapping.yaml).

## 2. Product and settlement conventions

Cboe's current product comparison states:

| Feature | Traditional SPX | SPXW weekly / end-of-month |
|---|---|---|
| Root ticker | SPX | SPXW |
| Settlement | AM | PM |
| Exercise | European | European |
| Delivery | Cash | Cash |
| Multiplier | 100 | 100 |
| Typical expiration trading cutoff | preceding business day for traditional SPX | expiration day for SPXW |

These conventions affect the appropriate underlying observation, time to expiry, parity regression, near-expiry behavior, and one-day outcome matching. A root label alone is not sufficient because some data vendors display SPX and SPXW in a combined chain. The data model must preserve root, expiration, exercise style, AM/PM settlement, quote timestamp if available, and source-specific contract identifier.

The Cboe page also notes holiday-specific expiration-date movement. Therefore expiration timestamps should come from contract metadata or an explicit exchange calendar, not from a fixed `date + close-time` assumption.

## 3. Data-source study

### Historical public option sample

HistoricalData.net publicly documents an options sample with:

- TSLA, KO, SPY, and SPX including SPXW;
- 127 trading dates from July through December 2022;
- one end-of-day CSV per date;
- 34 fields including bid/ask, size, volume, open interest, IV, Greeks, underlying close, settlement label, and `quote_time`;
- a ZIP, manifest, README, and verification script.

Important limitations:

- these are standing end-of-day quotes, not automatically a synchronized surface;
- the provider states each option row represents its own last standing quote;
- timestamp availability changed for ongoing capture on 2026-08-06, so the 2022 sample should not be assumed to have useful quote times;
- provider-published verification claims are not project observations until the project downloads and validates the archive;
- data license terms permit derived research publication but restrict redistribution of reconstructable raw data; the project should not republish vendor rows.

This source supports cross-sectional calibration, settlement studies, quote-quality/asynchrony sensitivity, and qualified next-date proxy work. It cannot support a claim of synchronized close execution or actual desk P&L.

### Cboe marking prices

Cboe publishes no-charge 3:00 p.m. CT indicative marking-price CSVs for proprietary index options, including SPX/SPXW, and separately lists 3:15 p.m. and end-of-month files. Cboe states:

- the files normally appear about 15 minutes after the market close;
- the indicative price and actual BBO at that time are included;
- the marks are not actual OPRA NBBO updates;
- marks may differ from the disseminated market/electronic book.

Only files collected after the project start may be classified as `PROSPECTIVE_LOCKED_TEST`; no historical reconstruction should be claimed if earlier files are unavailable.

### Cboe DataShop

The official Option Quote Intervals documentation describes one-minute or custom N-minute snapshots containing NBBO and size, OHLC, trade volume, optional open interest, and optional calculations. Historical availability is stated as January 2012 onward. A documented methodology change took effect on 2026-06-22: quote sizes thereafter are captured at the most recent price change in the interval, whereas earlier data used the latest price and size regardless of whether the price changed. This break must be stored as data lineage.

The documentation and a sample link are public. Full historical data are paid; the mandate prohibits purchasing them. Index underlying fields can also depend on a Cboe Global Indices Feed license. An accessible free sample can validate schema only and should not be represented as full historical coverage.

### Rates and covariates

- The New York Fed describes SOFR as a volume-weighted median of qualifying overnight Treasury-repo transactions and publishes SOFR averages and an index. SOFR is an overnight secured reference rate, not by itself a complete SPX option discount curve. [New York Fed reference rates](https://www.newyorkfed.org/markets/reference-rates); [methodology](https://www.newyorkfed.org/markets/reference-rates/additional-information-about-reference-rates).
- FRED provides a REST API for series metadata and observations, subject to API-key rules for Version 1. FRED is an aggregator: source/release metadata and revisions must be retained. [FRED API](https://fred.stlouisfed.org/docs/api/fred/).
- Treasury's daily curve is a **par yield curve**, derived from indicative bid-side prices and, since 2021-12-06, a monotone-convex methodology. It is not an exact derivatives discount curve. [Treasury methodology](https://home.treasury.gov/policy-issues/financing-the-government/interest-rate-statistics/treasury-yield-curve-methodology); [XML feed documentation](https://home.treasury.gov/treasury-daily-interest-rate-xml-feed).
- Cboe provides a daily VIX history from 1990 onward. VIX can define market regimes, but any threshold must be estimated on DEV and frozen before locked testing. [Cboe VIX history](https://www.cboe.com/tradable_products/vix/vix_historical_data).
- Massive offers authenticated options and index endpoints. Use is optional and conditional on an existing valid key and plan. Aggregate/trade data do not establish synchronized NBBO mids. [Massive REST documentation](https://massive.com/docs/rest/quickstart).

The complete access and licensing assessment is in [`data_source_matrix.yaml`](data_source_matrix.yaml).

## 4. Methodology study

### Forward and discount inference

For European options, put-call parity implies

\[
C(K,T)-P(K,T)=D(T)\,[F(T)-K].
\]

A cross-strike regression estimates intercept `D F` and slope `-D`. Ordinary least squares is a transparent baseline, but heteroskedastic and asynchronous quotes motivate spread weighting and robust regression. The final estimator must be selected on DEV using parity residuals, strike-subset stability, cross-date stability, and consistency checks against external rates. External Treasury or SOFR data are sensitivity inputs, not automatic truth.

### Black-Scholes and implied volatility

Black-Scholes provides the analytic baseline, parity identities, Greeks, bounds, and an independent limit case. Implied volatility inversion should use no-arbitrage bounds plus a bracketed method such as Brent's method, with explicit failure codes. Round trips must cover short/long maturities, extreme moneyness, and low/high volatility.

### SVI, SSVI, and Dupire local volatility

Raw SVI is parsimonious and flexible for smile fitting but unconstrained calibration can create butterfly or calendar arbitrage. Gatheral and Jacquier's SSVI construction gives explicit sufficient conditions for a broad class of static-arbitrage-free surfaces. DerivGuard should fit total variance, run discrete and analytic arbitrage diagnostics, and report residual violations.

Dupire local volatility infers a state/time-dependent diffusion from a smooth option-price or total-variance surface. Differentiating raw quotes is numerically unstable, so the path must be cleaned prices → forward/discount → IV → smooth SSVI/equivalent surface → derivatives → local volatility. The result is sensitive to interpolation/extrapolation and derivative regularization; it is an independent model-form challenger, not a direct observation.

### Heston characteristic-function engine

Heston supplies a semi-analytic European option price through Fourier inversion. Numerically, algebraically equivalent characteristic-function formulas can behave differently because complex logarithms are multi-valued. Kahl and Jäckel analyze rotation/branch handling; Albrecher et al. describe the “Little Heston Trap”; Lord and Kahl provide a broader treatment of complex-log discontinuities. Implementation should:

- use a stable formulation rather than a naive complex logarithm;
- validate branch behavior across maturity, correlation, vol-of-vol, and Feller-violating regions;
- separate quadrature error from calibration error;
- compare with QuantLib and independent CPU reference cases;
- use FP64 on both CPU and GPU for final prices unless a documented precision study supports another choice.

### Independent Heston Monte Carlo

Euler discretization can create negative variance. Full truncation is a defensible baseline; the QE/QE-M family offers higher practical efficiency. Tests should distinguish time-step bias from sampling error and include antithetic variates, control variates where justified, common random numbers for paired comparisons, confidence intervals, and seed robustness. The GPU engine must vectorize paths while chunking over paths/time steps to respect VRAM.

### Independent Heston PDE

The two-dimensional Heston PDE has a mixed derivative from spot/variance correlation. In 't Hout and Foulon's study compares Douglas, Craig-Sneyd, Modified Craig-Sneyd, and Hundsdorfer-Verwer ADI schemes on nonuniform grids. DerivGuard should independently implement at least MCS and HV where practical, test boundaries and convergence, and benchmark CPU versus GPU. A CuPy sparse backend is only warranted if accuracy and end-to-end speed improve; repeated host-device transfers can erase acceleration.

### Bates challenger

Bates extends stochastic volatility with jumps. It can represent short-maturity tail/skew mechanisms unavailable to diffusion-only Heston, but introduces parameter-identifiability and calibration-cost risk. It is most valuable here as a challenger and synthetic data-generating process. A failed or weakly identified Bates calibration must be recorded, not hidden.

### QuantLib oracle

QuantLib exposes analytic Heston, finite-difference Heston, Monte Carlo Heston, and Bates engines. The source documents multiple complex-log/control-variate choices. It is a useful external implementation reference, subject to explicit version and convention capture. Agreement with QuantLib alone does not prove correctness because common formulas and conventions can still be wrong.

### Calibration, uncertainty, identifiability, and validation

Price RMSE overweights expensive options; IV RMSE can overemphasize low-vega points; vega weighting, spread normalization, and robust losses each encode different assumptions. Objectives and optimizer families must therefore be compared on DEV, including local, multistart, global, and global-plus-local refinement. The Feller condition is a diagnostic and experiment dimension, not an automatic validity rule.

Bootstrap distributions, multistart dispersion, parameter correlations, and profile objectives address distinct uncertainty/identifiability questions. A good fit with flat profiles or unstable parameters is evidence that fit quality and model reliability are not equivalent.

DVE should retain separate evidence dimensions and keep risk evidence distinct from exposure-based materiality. Thresholds and regimes must be fixed before locked testing; synthetic truth supports controlled detection metrics, while real-market inference uses market dates/surfaces rather than individual option rows as IID observations.

The detailed comparison is in [`methodology_matrix.yaml`](methodology_matrix.yaml).

## 5. GPU and reproducibility implications

PyTorch's Windows guidance currently lists Python 3.9–3.12, which supports choosing CPython 3.12 for the project-local environment. CUDA work is asynchronous; benchmarks must use CUDA events or explicit synchronization. Memory accounting should distinguish allocated and reserved memory. The project should favor a single long-lived GPU executor, large vectorized batches, adaptive chunking, and logged OOM reductions.

CuPy provides NumPy/SciPy-like dense and sparse APIs backed by CUDA libraries, including FP64 and complex128 sparse data support, but not every SciPy operation is implemented. Any PDE port must verify the exact solver operations and index limitations against the installed CuPy version. [CuPy installation](https://docs.cupy.dev/en/stable/install.html); [CuPy sparse arrays](https://docs.cupy.dev/en/latest/reference/scipy_sparse.html).

Reproducibility requires more than seeding: capture package and CUDA runtime versions, device, precision, tolerance registry, config/content hashes, exact source hashes, and synchronized timing procedure. GPU and CPU agreement must be tested before research-scale execution.

## 6. Access limitations and unresolved source questions

- This phase researched and documented sources; it did **not** download or inspect any market-data archive. Provider claims about sample contents remain to be verified in the acquisition phase.
- The original 1994 Dupire article is publicly viewable as a Risk-hosted PDF, but it has no verified DOI in the sources reviewed here.
- “The Little Heston Trap” was verified through an institutional bibliographic record and an author-hosted/preprint copy; no DOI was found and none is asserted.
- The 2004 SVI item is a conference presentation rather than a conventional journal article. A later author-slide URL returned HTTP 502 during this study; the bibliographic record was cross-checked through Guo et al. (2016). The peer-reviewed SSVI reference is Gatheral and Jacquier (2014).
- Cboe DataShop's documentation and sample link are public, but full historical interval data are paid and were not purchased.
- Massive access depends on an existing API key and plan; no availability is assumed.
- Cboe marking files are rotating/current publications. Durable prospective collection is required because a historical archive is not promised by the page.

## 7. Source-integrity rules for subsequent phases

1. Store retrieval timestamp, original URL, bytes, SHA-256, schema version, row count, date range, license note, and redistribution restriction for every raw artifact.
2. Cite the actual downloaded artifact and its hash in result manifests; do not turn web-page claims into measured dataset facts.
3. Record unavailable or paid sources as unavailable; never reconstruct absent observations.
4. Preserve AM/PM settlement and quote-timing metadata through every data stage.
5. Keep raw inputs immutable and retain every exclusion code at the source-row level.
6. Treat Cboe indicative marks, HistoricalData.net standing quotes, DataShop NBBO snapshots, and trade aggregates as distinct observation types.
7. Recheck living software and data documentation at execution time and pin the versions actually used.

## Primary-source index

- [Federal Reserve SR 26-2 letter](https://www.federalreserve.gov/supervisionreg/srletters/SR2602.htm)
- [Federal Reserve revised model-risk guidance](https://www.federalreserve.gov/frrs/guidance/supervisory-guidance-on-model-risk-management.htm)
- [Cboe SPX/SPXW product comparison](https://www.cboe.com/tradable_products/sp_500/spx_weekly_options/specifications)
- [Cboe SPX contract specifications](https://www.cboe.com/tradable-products/sp-500/spx-options/spx-specifications)
- [Cboe proprietary index marking prices](https://www.cboe.com/markets/us/options/market-statistics/product-data/proprietary-index-marking-prices/)
- [Cboe DataShop Option Quote Intervals](https://datashop.cboe.com/option-quote-intervals)
- [HistoricalData.net public samples](https://historicaldata.net/samples.html)
- [New York Fed reference-rate methodology](https://www.newyorkfed.org/markets/reference-rates/additional-information-about-reference-rates)
- [FRED API documentation](https://fred.stlouisfed.org/docs/api/fred/)
- [U.S. Treasury yield-curve methodology](https://home.treasury.gov/policy-issues/financing-the-government/interest-rate-statistics/treasury-yield-curve-methodology)
- [Heston (1993), DOI](https://doi.org/10.1093/rfs/6.2.327)
- [Gatheral and Jacquier (2014), DOI](https://doi.org/10.1080/14697688.2013.819986)
- [Dupire (1994), Risk-hosted article](https://www.risk.net/sites/default/files/import_unmanaged/risk.net/data/Pay_per_view/risk/technical/1994/risk_0194_volatility.pdf)
- [Bates (1996), DOI](https://doi.org/10.1093/rfs/9.1.69)
- [Andersen (2008), DOI](https://doi.org/10.21314/JCF.2008.189)
- [Lord, Koekkoek, and van Dijk (2010), DOI](https://doi.org/10.1080/14697680802392496)
- [Kahl and Jäckel, stable Heston logarithms](https://www2.math.uni-wuppertal.de/~kahl/publications/NotSoComplexLogarithmsInTheHestonModel.pdf)
- [Albrecher et al., The Little Heston Trap](https://tugraz.elsevierpure.com/en/publications/the-little-heston-trap/)
- [In 't Hout and Foulon (2010)](https://www.global-sci.com/ijnam/article/view/9955)
- [QuantLib analytic Heston engine source](https://github.com/lballabio/QuantLib/blob/master/ql/pricingengines/vanilla/analytichestonengine.hpp)
- [PyTorch Windows installation](https://docs.pytorch.org/get-started/locally/)
- [PyTorch CUDA semantics](https://docs.pytorch.org/docs/main/notes/cuda.html)
- [CuPy documentation](https://docs.cupy.dev/en/stable/)
