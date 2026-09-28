"""Final report assembly for the additive v1.1 analytics extension."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

import matplotlib
import numpy as np
import pandas as pd
import seaborn as sns  # type: ignore[import-untyped]
from matplotlib import pyplot as plt

matplotlib.use("Agg")

EVIDENCE_NAMES = (
    "E_data",
    "E_num",
    "E_cal",
    "E_ident",
    "E_param",
    "E_form",
    "E_greek",
    "E_extra",
    "E_outcome",
)


def _markdown(frame: pd.DataFrame, rows: int = 30) -> str:
    if frame.empty:
        return "No rows available."
    selected = frame.head(rows)
    columns = list(selected.columns)
    header = "| " + " | ".join(columns) + " |"
    separator = "| " + " | ".join("---" for _ in columns) + " |"
    body = [
        "| "
        + " | ".join(
            (f"{value:.6g}" if isinstance(value, float) else str(value)).replace("|", "\\|")
            for value in row
        )
        + " |"
        for row in selected.itertuples(index=False, name=None)
    ]
    return "\n".join([header, separator, *body])


def _stress_matrix(rows: pd.DataFrame) -> pd.DataFrame:
    """Apply coherent local valuation shocks to the measured contract rows.

    Spot uses delta-gamma expansion and volatility uses measured model vega.
    Spread shocks change normalized materiality only. Rate and stochastic-
    volatility-parameter shocks are recorded as unavailable rather than mixed
    with model-specific dynamics in a common local approximation.
    """

    records: list[dict[str, object]] = []
    scenarios = [
        *(f"SPOT_{value:+.0%}" for value in (-0.10, -0.05, 0.05, 0.10)),
        *(f"VOL_{value:+.0%}" for value in (-0.50, -0.20, 0.20, 0.50)),
        "SPREAD_X2",
        "SPREAD_X4",
    ]
    for (split, model), group in rows.groupby(["split", "model_id"]):
        for scenario in scenarios:
            if scenario.startswith("SPOT_"):
                shock = float(scenario.removeprefix("SPOT_").removesuffix("%")) / 100.0
                move = group["spot"] * shock
                change = group["delta"] * move + 0.5 * group["gamma"] * np.square(move)
                normalized = np.abs(change) / np.maximum(group["ask"] - group["bid"], 0.01)
                method = "delta-gamma local expansion"
            elif scenario.startswith("VOL_"):
                shock = float(scenario.removeprefix("VOL_").removesuffix("%")) / 100.0
                change = group["vega"] * group["market_iv"] * shock
                normalized = np.abs(change) / np.maximum(group["ask"] - group["bid"], 0.01)
                method = "vega local expansion"
            else:
                multiplier = float(scenario.removeprefix("SPREAD_X"))
                change = pd.Series(0.0, index=group.index)
                normalized = np.abs(group["residual"]) / (
                    np.maximum(group["ask"] - group["bid"], 0.01) * multiplier
                )
                method = "spread-only materiality shock"
            records.append(
                {
                    "split": split,
                    "model_id": model,
                    "scenario": scenario,
                    "method": method,
                    "count": len(group),
                    "mean_value_change": float(change.mean()),
                    "mean_absolute_value_change": float(np.abs(change).mean()),
                    "p95_absolute_value_change": float(np.abs(change).quantile(0.95)),
                    "mean_spread_normalized_effect": float(normalized.mean()),
                    "base_mean_abs_delta": float(np.abs(group["delta"]).mean()),
                    "base_mean_abs_gamma": float(np.abs(group["gamma"]).mean()),
                    "base_mean_abs_vega": float(np.abs(group["vega"]).mean()),
                    "status": "MEASURED_LOCAL_SENSITIVITY",
                }
            )
    for scenario in (
        "RATE_-200BP",
        "RATE_-100BP",
        "RATE_+100BP",
        "RATE_+200BP",
        "HESTON_RHO_SHOCKS",
        "VOL_OF_VOL_SHOCKS",
    ):
        records.append(
            {
                "split": "ALL",
                "model_id": "CROSS_MODEL",
                "scenario": scenario,
                "method": "not mixed with local common-Greek approximation",
                "status": "UNAVAILABLE_MODEL_SPECIFIC_REPRICING_NOT_EXECUTED",
            }
        )
    return pd.DataFrame(records)


def _heatmap(table: pd.DataFrame, title: str, path: Path, fmt: str = ".2f") -> None:
    if table.empty:
        return
    fig, ax = plt.subplots(figsize=(9, 5.5))
    sns.heatmap(table, cmap="viridis", annot=True, fmt=fmt, ax=ax)
    ax.set_title(title)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _augment_v1_failure_diagnosis(root: Path, result_dir: Path) -> dict[str, Any]:
    source = pd.read_parquet(
        root / "results/confirmatory/synthetic_v5/evidence_score_records.parquet"
    )
    locked = source[source["partition"].isin(["LOCKED_TEST", "OOD_LOCKED_TEST"])].copy()
    thresholds = cast(
        dict[str, Any],
        json.loads(
            (root / "results/confirmatory/synthetic_v5/dve_thresholds.freeze.json").read_text(
                encoding="utf-8"
            )
        ),
    )
    fit_threshold = float(thresholds["thresholds"]["global_thresholds"]["V00"])
    locked["fitonly_prediction"] = locked["score_V00"] > fit_threshold
    locked["full_prediction"] = locked["V06_prediction"].fillna(False).astype(bool)
    locked["full_vs_fitonly"] = np.select(
        [
            locked["full_prediction"] & ~locked["fitonly_prediction"] & ~locked["material_error"],
            ~locked["full_prediction"] & locked["fitonly_prediction"] & locked["material_error"],
            locked["full_prediction"] & ~locked["fitonly_prediction"] & locked["material_error"],
            ~locked["full_prediction"] & locked["fitonly_prediction"] & ~locked["material_error"],
        ],
        [
            "FULL_EXCESS_FALSE_POSITIVE",
            "FULL_LOST_FITONLY_TRUE_POSITIVE",
            "FULL_RESCUED_FITONLY_MISS",
            "FULL_REMOVED_FITONLY_FALSE_POSITIVE",
        ],
        default="SAME_CLASSIFICATION",
    )
    records = []
    for evidence in EVIDENCE_NAMES:
        for outcome, group in locked.groupby("full_vs_fitonly"):
            records.append(
                {
                    "evidence": evidence,
                    "full_vs_fitonly": outcome,
                    "count": len(group),
                    "mean": float(group[evidence].mean()),
                    "median": float(group[evidence].median()),
                    "p90": float(group[evidence].quantile(0.90)),
                }
            )
    attribution = pd.DataFrame(records)
    attribution.to_csv(
        result_dir / "dve_failure/full_vs_fitonly_evidence_attribution.csv", index=False
    )
    locked.to_parquet(result_dir / "dve_failure/v1_full_vs_fitonly_cases.parquet", index=False)
    positive = locked["material_error"].astype(bool)
    negative = ~positive
    summary = {
        "cases": len(locked),
        "positives": int(positive.sum()),
        "negatives": int(negative.sum()),
        "fitonly_false_positives": int((locked["fitonly_prediction"] & negative).sum()),
        "fitonly_false_negatives": int((~locked["fitonly_prediction"] & positive).sum()),
        "fitonly_recall": float((locked["fitonly_prediction"] & positive).sum() / positive.sum()),
        "fitonly_fpr": float((locked["fitonly_prediction"] & negative).sum() / negative.sum()),
        "full_false_positives": int((locked["full_prediction"] & negative).sum()),
        "full_false_negatives": int((~locked["full_prediction"] & positive).sum()),
        "full_recall": float((locked["full_prediction"] & positive).sum() / positive.sum()),
        "full_fpr": float((locked["full_prediction"] & negative).sum() / negative.sum()),
        "comparison_counts": locked["full_vs_fitonly"].value_counts().to_dict(),
    }
    (result_dir / "dve_failure/full_vs_fitonly_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return summary


def generate(root: Path = Path(".")) -> dict[str, Any]:
    report_dir = root / "reports/v1_1"
    result_dir = root / "results/v1_1"
    report_dir.mkdir(parents=True, exist_ok=True)
    failure = cast(
        dict[str, Any],
        json.loads((result_dir / "dve_failure/diagnosis_summary.json").read_text(encoding="utf-8")),
    )
    failure_slices = pd.read_csv(result_dir / "dve_failure/failure_slices.csv")
    evidence = pd.read_csv(result_dir / "dve_failure/conditional_evidence_signal.csv")
    fit_comparison = _augment_v1_failure_diagnosis(root, result_dir)
    attribution = pd.read_csv(result_dir / "dve_failure/full_vs_fitonly_evidence_attribution.csv")
    (report_dir / "DVE_FAILURE_ANALYSIS.md").write_text(
        "# DVE v1 Failure Analysis\n\n"
        "This is descriptive analysis of immutable inherited evidence and was not used "
        "to tune v1.1.\n\n"
        f"Across {failure['locked_cases']} locked/OOD cases, Full DVE produced "
        f"{failure['false_positives']} false positives and {failure['false_negatives']} "
        "false negatives. Half of the false positives were in LONG|WING|LIQUID. "
        f"FitOnly produced {fit_comparison['fitonly_false_positives']} false positives, "
        f"{fit_comparison['fitonly_false_negatives']} false negatives, recall "
        f"{fit_comparison['fitonly_recall']:.3f}, and FPR {fit_comparison['fitonly_fpr']:.3f}; "
        f"Full DVE produced recall {fit_comparison['full_recall']:.3f} and FPR "
        f"{fit_comparison['full_fpr']:.3f}. Full DVE added five false positives, lost four "
        "FitOnly true positives, rescued no FitOnly misses, and removed one FitOnly false "
        "positive.\n\n"
        "## Failure slices\n\n"
        + _markdown(failure_slices.sort_values("count", ascending=False))
        + "\n\n## Conditional evidence signal\n\n"
        + _markdown(evidence)
        + "\n\n## Full DVE versus FitOnly evidence attribution\n\n"
        + _markdown(attribution)
        + "\n",
        encoding="utf-8",
    )
    status = cast(
        dict[str, Any],
        json.loads(
            (result_dir / "dve_v1_1b/confirmatory_metrics.json").read_text(encoding="utf-8")
        ),
    )
    metrics = pd.read_csv(result_dir / "dve_v1_1b/locked_baseline_metrics.csv")
    validation = pd.read_csv(result_dir / "dve_v1_1b/validation_candidate_selection.csv")
    initial_failure = cast(
        dict[str, Any],
        json.loads(
            (result_dir / "dve_v1_1/validation_records.failure.json").read_text(encoding="utf-8")
        ),
    )
    (report_dir / "DVE_V1_1_CONFIRMATORY.md").write_text(
        "# DVE v1.1 Confirmatory Study\n\n"
        "The first preregistered design stopped before locked generation because "
        f"{initial_failure['partition']} seed {initial_failure['seed']} was "
        "CALIBRATION_UNQUALIFIED. The preserved v1.1b remediation changed only the "
        "bootstrap search budget, passed disjoint prequalification, and used wholly "
        "new 121m-124m seeds.\n\n"
        f"VALIDATION selected `{status['selected_transparent_aggregation']}` before "
        "locked generation. The ordinary LOCKED_TEST and stress OOD_LOCKED_TEST partitions "
        "are reported separately below. On ordinary locked data, Full v1.1 DVE and FitOnly "
        "both had recall 1.000 and Full had lower FPR (0.069 versus 0.103). On OOD, Full "
        "retained recall 1.000 but its FPR rose to 0.615 versus 0.154 for FitOnly. Combined, "
        "Full had FPR 0.238 versus 0.119 and AUPRC 0.752 versus 1.000. The preregistered "
        "primary endpoint applies to wholly fresh LOCKED_TEST cases: there Full met the "
        "near-target FPR objective and preserved FitOnly recall. OOD was a separately frozen "
        "robustness partition and failed materially, so no global superiority or robust "
        "improvement claim is supported.\n\n"
        "## VALIDATION selection\n\n"
        + _markdown(validation)
        + "\n\n## Confirmatory metrics\n\n"
        + _markdown(metrics)
        + "\n",
        encoding="utf-8",
    )
    contract = pd.read_parquet(result_dir / "contract_model_results_sliced.parquet")
    stress_manifest = root / "artifacts/v1_1/stress_repricing_manifest.json"
    if stress_manifest.exists():
        stress = pd.read_parquet(result_dir / "stress_sensitivity_matrix.parquet")
        measured = stress[stress["status"] == "MEASURED_CONTROLLED_REPRICING"]
        stress_description = (
            "Spot, volatility, rates, Heston rho, and Heston vol-of-vol use controlled "
            "repricing. Heston ran in CUDA FP64 and passed independent CPU spot checks; "
            "SVI/SSVI use an explicit sticky-strike convention. Rho and vol-of-vol are not "
            "applicable to the Black-Scholes surface models, which remain unchanged references. "
            "Bates was excluded because it is an unqualified fixed sensitivity, and LocalVol "
            "lacks contract-level qualification. Spread shocks affect normalization only."
        )
    else:
        stress = _stress_matrix(contract)
        stress.to_parquet(result_dir / "stress_sensitivity_matrix.parquet", index=False)
        measured = stress[stress["status"] == "MEASURED_LOCAL_SENSITIVITY"]
        stress_description = (
            "Preliminary local Greek approximation only; controlled repricing has not yet run."
        )
    (report_dir / "STRESS_SENSITIVITY_MATRIX.md").write_text(
        "# Stress and Sensitivity Matrix\n\n"
        + stress_description
        + " No impossible scenario combinations were applied.\n\n"
        + _markdown(measured.sort_values("mean_spread_normalized_effect", ascending=False))
        + "\n",
        encoding="utf-8",
    )
    # Complete inherited diagnostic tables requested by the calibration deep dive.
    profiles = pd.read_parquet(root / "results/aggregated/identifiability_profiles.parquet")
    profile_rows = []
    for parameter, group in profiles[profiles["study"] == "profile_objective"].groupby("parameter"):
        finite = group[np.isfinite(group["objective_value"])].sort_values("fixed_value")
        if finite.empty:
            continue
        cutoff = float(finite["objective_value"].min()) * 1.10
        accepted = finite[finite["objective_value"] <= cutoff]
        profile_rows.append(
            {
                "parameter": parameter,
                "relative_objective_cutoff": 0.10,
                "profile_width": float(
                    accepted["fixed_value"].max() - accepted["fixed_value"].min()
                ),
                "grid_points": len(finite),
                "accepted_points": len(accepted),
            }
        )
    pd.DataFrame(profile_rows).to_csv(result_dir / "profile_objective_widths.csv", index=False)
    profiles[profiles["study"] == "identifiability"][
        ["parameter", "dispersion", "successful_solutions", "covariance_condition_number"]
    ].to_csv(result_dir / "multiple_start_parameter_dispersion.csv", index=False)
    uncertainty_path = result_dir / "calibration_uncertainty_by_calibration.parquet"
    if uncertainty_path.exists():
        uncertainty = pd.read_parquet(uncertainty_path)
        calibration_slices = pd.read_parquet(result_dir / "calibration_diagnostic_slices.parquet")
        per_calibration_profiles = pd.read_parquet(
            result_dir / "calibration_profile_widths.parquet"
        )
        per_calibration_multistart = pd.read_parquet(
            result_dir / "calibration_multistart_dispersion.parquet"
        )
        calibration = pd.read_parquet(result_dir / "calibration_deep_dive.parquet")
        grouped = calibration.groupby("split", as_index=False).agg(
            calibrations=("objective_value", "size"),
            median_objective=("objective_value", "median"),
            median_price_rmse=("price_rmse", "median"),
            median_iv_rmse=("iv_rmse", "median"),
            median_spread_normalized_mae=("spread_normalized_mae", "median"),
            median_inside_bid_ask_rate=("inside_bid_ask_rate", "median"),
            feller_rate=("feller_satisfied", "mean"),
        )
        support = calibration.groupby(["split", "settlement_class"], as_index=False).agg(
            calibrations=("objective_value", "size"),
            median_moneyness_support=("moneyness_support", "median"),
            median_maturity_support_days=("maturity_support_days", "median"),
            median_relative_spread=("median_relative_spread", "median"),
            median_price_rmse=("price_rmse", "median"),
        )
        uncertainty_summary = uncertainty.groupby(["split", "parameter"], as_index=False).agg(
            calibrations=("quote_date", "size"),
            median_bootstrap_std=("bootstrap_std", "median"),
            p90_bootstrap_std=("bootstrap_std", lambda value: float(value.quantile(0.90))),
        )
        (report_dir / "CALIBRATION_DIAGNOSTICS.md").write_text(
            "# Calibration Diagnostics\n\n"
            "All 135 valid released Heston calibrations retain objective, price/IV/vega-"
            "weighted and spread-normalized errors, interval hit rate, signed bias, optimizer "
            "status/counts, boundary proximity, Feller status, and market-support covariates. "
            "Each calibration additionally has CUDA-FP64 price Jacobians with 400 CPU-FP64 "
            "linearized bid/ask quote-bootstrap propagations, parameter correlations, five "
            "local nuisance-profile widths, and four actual bounded local optimization starts. "
            "Multistart dispersion retains every finite terminal solution, including unsuccessful "
            "optimizer statuses, while reporting the successful-start count separately. The "
            "linearized bootstrap and local quadratic profile are diagnostic approximations, not "
            "replacements for the frozen release fit.\n\n"
            "## Split summary\n\n"
            + _markdown(grouped)
            + "\n\n## Settlement, support, and liquidity summary\n\n"
            + _markdown(support)
            + "\n\n## Date, VIX, liquidity, moneyness-support, and maturity-support slices\n\n"
            + _markdown(
                calibration_slices[calibration_slices["dimension"] != "quote_date"].sort_values(
                    ["dimension", "split", "value"]
                )
            )
            + "\n\n## Per-calibration bootstrap uncertainty summary\n\n"
            + _markdown(uncertainty_summary)
            + "\n\nCoverage: "
            + f"{uncertainty[['quote_date', 'settlement_class']].drop_duplicates().shape[0]} "
            + "calibration keys, "
            + f"{len(per_calibration_profiles)} profile rows, and "
            + f"{len(per_calibration_multistart)} multistart parameter rows.\n",
            encoding="utf-8",
        )
    figure_dir = report_dir / "figures"
    locked_contract = contract[contract["split"].isin(["VALIDATION", "LOCKED_TEST"])]
    for model, group in locked_contract.groupby("model_id"):
        residual_heatmap = group.pivot_table(
            index="moneyness_slice",
            columns="maturity_slice",
            values="residual",
            aggfunc="mean",
        )
        _heatmap(
            residual_heatmap,
            f"{model} mean signed residual",
            figure_dir / f"{str(model).lower()}_residual_heatmap.png",
        )
    residual_summary = locked_contract.groupby("model_id", as_index=False).agg(
        count=("residual", "size"),
        mean=("residual", "mean"),
        std=("residual", "std"),
        q05=("residual", lambda value: float(value.quantile(0.05))),
        median=("residual", "median"),
        q95=("residual", lambda value: float(value.quantile(0.95))),
        skew=("residual", "skew"),
    )
    residual_summary.to_csv(result_dir / "residual_distribution_summary.csv", index=False)
    residual_report = report_dir / "RESIDUAL_ANALYSIS.md"
    residual_base = residual_report.read_text(encoding="utf-8").split(
        "\n\n## Validation and locked distribution diagnostics", maxsplit=1
    )[0]
    residual_report.write_text(
        residual_base
        + "\n\n## Validation and locked distribution diagnostics\n\n"
        + _markdown(residual_summary)
        + "\n",
        encoding="utf-8",
    )
    fig, ax = plt.subplots(figsize=(10, 6))
    clipped = locked_contract.copy()
    limit = float(clipped["residual"].abs().quantile(0.99))
    clipped["residual"] = clipped["residual"].clip(-limit, limit)
    sns.violinplot(data=clipped, x="model_id", y="residual", inner="quartile", cut=0, ax=ax)
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.set_title("Validation and locked residual distributions (1st-99th percentile clipped)")
    ax.tick_params(axis="x", rotation=25)
    fig.tight_layout()
    fig.savefig(figure_dir / "residual_distribution_violin.png", dpi=180)
    plt.close(fig)
    price_wide = locked_contract.pivot_table(
        index=["quote_date", "source_row_id", "moneyness_slice", "maturity_slice"],
        columns="model_id",
        values="model_price",
    )
    challenger_columns = [
        name for name in ("M02_SVI", "M03_SSVI", "M21_BATES") if name in price_wide
    ]
    price_wide["challenger_dispersion"] = price_wide[challenger_columns].std(axis=1)
    dispersion = price_wide.reset_index().pivot_table(
        index="moneyness_slice",
        columns="maturity_slice",
        values="challenger_dispersion",
        aggfunc="mean",
    )
    _heatmap(
        dispersion,
        "Challenger price dispersion",
        figure_dir / "challenger_dispersion_heatmap.png",
    )
    disagreement = pd.read_parquet(result_dir / "challenger_disagreement.parquet")
    for column in ("delta_disagreement", "gamma_disagreement", "vega_disagreement"):
        scale = max(float(disagreement[column].median()), 1.0e-12)
        disagreement[f"scaled_{column}"] = disagreement[column] / scale
    disagreement["greek_disagreement_score"] = disagreement[
        [
            "scaled_delta_disagreement",
            "scaled_gamma_disagreement",
            "scaled_vega_disagreement",
        ]
    ].max(axis=1)
    greek = disagreement.pivot_table(
        index="moneyness_slice",
        columns="maturity_slice",
        values="greek_disagreement_score",
        aggfunc="mean",
    )
    _heatmap(greek, "Scaled Greek disagreement", figure_dir / "greek_disagreement_heatmap.png")
    two_dim = pd.read_parquet(result_dir / "two_dimensional_slice_metrics.parquet")
    evaluation = two_dim[
        two_dim["split"].isin(["VALIDATION", "LOCKED_TEST"]) & two_dim["defensible"]
    ]
    pairs = evaluation[["dimension_1", "dimension_2"]].drop_duplicates()
    for pair in pairs.itertuples(index=False):
        subset = evaluation[
            (evaluation["dimension_1"] == pair.dimension_1)
            & (evaluation["dimension_2"] == pair.dimension_2)
        ]
        label = f"{pair.dimension_1}_x_{pair.dimension_2}".replace("_slice", "")
        pivoted = subset.pivot_table(
            index=["value_1", "value_2"], columns="model_id", values="rmse", aggfunc="mean"
        )
        for model in ("M10_HESTON", "M02_SVI", "M03_SSVI"):
            if model in pivoted:
                _heatmap(
                    pivoted[model].unstack(),
                    f"{model} RMSE: {label}",
                    figure_dir / f"{label}_{model.lower()}_rmse.png",
                )
        if "M10_HESTON" in pivoted:
            for challenger in ("M02_SVI", "M03_SSVI"):
                if challenger in pivoted:
                    _heatmap(
                        (pivoted["M10_HESTON"] - pivoted[challenger]).unstack(),
                        f"Heston minus {challenger} RMSE: {label}",
                        figure_dir / f"{label}_heston_minus_{challenger.lower()}.png",
                    )
        challengers = [name for name in ("M02_SVI", "M03_SSVI") if name in pivoted]
        if len(challengers) == 2:
            _heatmap(
                pivoted[challengers].std(axis=1).unstack(),
                f"Challenger RMSE dispersion: {label}",
                figure_dir / f"{label}_challenger_dispersion.png",
            )
        greek_subset = subset[subset["model_id"].isin(("M02_SVI", "M03_SSVI"))]
        if not greek_subset.empty:
            greek_table = greek_subset.pivot_table(
                index="value_1",
                columns="value_2",
                values="vega_disagreement",
                aggfunc="mean",
            )
            _heatmap(
                greek_table,
                f"Mean vega disagreement: {label}",
                figure_dir / f"{label}_vega_disagreement.png",
            )
    portfolio = pd.read_parquet(result_dir / "portfolio_materiality_slices.parquet")
    materiality = portfolio.pivot_table(
        index="portfolio_id",
        columns="model_id",
        values="valuation_materiality",
        aggfunc="mean",
    )
    _heatmap(
        materiality,
        "Mean portfolio valuation materiality",
        figure_dir / "portfolio_materiality_heatmap.png",
    )
    return {
        "status": "V1_1_REPORTS_GENERATED",
        "selected_aggregation": status["selected_transparent_aggregation"],
        "confirmatory_conclusion": "LOCKED_MET_PRIMARY_OOD_FAILED_NO_GLOBAL_SUPERIORITY",
        "stress_measured_rows": len(measured),
    }
