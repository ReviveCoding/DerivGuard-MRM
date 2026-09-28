# Independent Scientific and Code Re-review

Date: 2026-09-27  
Verdict: **ACCEPT**

The independent re-review found the two prior scientific blockers resolved:

- Winner-margin intervals hold the reported winner and runner-up fixed and calculate the signed runner-up-minus-winner margin. Independent reproduction matched all 39 intervals exactly; 16 intervals cross zero. Winner bootstrap probability remains a separate statistic.
- Median signed-error date-block-bootstrap intervals are present in the one- and two-dimensional slice tables, with no missing intervals in defensible cells.

The reviewer also verified current stress source hashes, calibration-method disclosures, Bates exclusion, LocalVol unavailability, separation of ordinary LOCKED_TEST from OOD findings, freeze integrity, and the absence of a new v1 immutability or leakage blocker.

Acceptance remains qualified as follows:

- Slice winners are not uniformly decisive because 16 of 39 signed margin intervals cross zero.
- Calibration bootstrap and profile widths are local diagnostic approximations.
- Multistart dispersion includes finite unsuccessful terminal solutions; success counts are reported separately.
- SVI/SSVI stress results use sticky strike, and the Heston CPU comparison is limited to 12 deterministic spot-scenario rows.
- The Full v1.1 DVE result is favorable only on the small ordinary locked sample; OOD robustness failed materially.
- 0DTE pricing, synchronized temporal OOS analysis, and hedge/P&L conclusions remain unavailable.
