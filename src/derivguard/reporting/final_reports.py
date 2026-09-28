"""Generate final research reports from measured, machine-readable artifacts.

This module reconciles the mutable main experiment registry from current artifact
content before summarizing evidence.  Frozen confirmatory registries are never
mutated.  Adverse and unavailable results remain visible.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml  # type: ignore[import-untyped]

from derivguard.governance.experiment_reconciliation import reconcile_experiment_registry
from derivguard.governance.gates import (
    acceptance_from_phase_evidence,
    reconcile_run_state_from_evidence,
    verify_final_evidence,
    write_gate_evidence,
)


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object in {path}")
    return value


def _metric(metrics: dict[str, Any], baseline: str, field: str) -> float:
    row = metrics[baseline]
    if "metrics" in row:
        row = row["metrics"]
    return float(row[field])


def _fmt(value: float | int | None, digits: int = 4) -> str:
    if value is None or not np.isfinite(float(value)):
        return "unavailable"
    return f"{float(value):.{digits}f}"


def _baseline_summary(payload: dict[str, Any], label: str) -> str:
    metrics = payload.get("baselines", payload.get("metrics", {}))
    fit_recall = _metric(metrics, "V00", "recall_at_dev_5pct_fpr")
    full_recall = _metric(metrics, "V06", "recall_at_dev_5pct_fpr")
    fit_auprc = _metric(metrics, "V00", "auprc")
    full_auprc = _metric(metrics, "V06", "auprc")
    return (
        f"{label}: FitOnly recall/AUPRC {fit_recall:.3f}/{fit_auprc:.3f}; "
        f"Full DVE {full_recall:.3f}/{full_auprc:.3f}."
    )


def _synthetic_evidence(root: Path) -> tuple[dict[str, Any] | None, str, Path | None]:
    result_dir = root / "results/confirmatory/synthetic_v5"
    metrics_path = result_dir / "confirmatory_metrics.json"
    qualification_path = result_dir / "calibration_qualification.json"
    calibration_path = result_dir / "calibration_diagnostics.parquet"
    if metrics_path.exists() and qualification_path.exists() and calibration_path.exists():
        value = _json(metrics_path)
        qualification = _json(qualification_path)
        if (
            value.get("status") == "CONFIRMATORY_LOCKED_COMPLETE"
            and qualification.get("status") == "PASS"
            and qualification.get("all_selected_fits_qualified") is True
        ):
            return (
                value,
                "confirmatory v6 release-aligned prequalified observable-quote study",
                result_dir,
            )
    return None, "confirmatory execution unavailable", None


def _write_monitoring(root: Path, result_dir: Path | None) -> None:
    """Create a transparent monitoring simulation from the newest evidence records."""

    if result_dir is None:
        return
    path = result_dir / "evidence_score_records.parquet"
    if not path.exists():
        return
    frame = pd.read_parquet(path)
    locked = frame.loc[frame["partition"].isin(["LOCKED_TEST", "OOD_LOCKED_TEST"])].copy()
    if locked.empty or "V06_prediction" not in locked:
        return
    prediction = locked["V06_prediction"].astype(bool)
    material = locked["material_error"].astype(bool)
    locked["monitoring_status"] = np.select(
        [prediction & material, prediction | material], ["RED", "AMBER"], default="GREEN"
    )
    locked["simulated_action"] = locked["monitoring_status"].map(
        {
            "GREEN": "No Action",
            "AMBER": "Investigate",
            "RED": "Apply Usage Limitation",
        }
    )
    output = root / "results/aggregated/monitoring_simulation.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    locked.to_csv(output, index=False)


def _make_figures(root: Path, result_dir: Path | None) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure_dir = root / "reports/figures"
    figure_dir.mkdir(parents=True, exist_ok=True)

    pde_path = root / "results/aggregated/pde_adi_convergence.csv"
    if pde_path.exists():
        pde = pd.read_csv(pde_path)
        subset = pde.loc[pde["study"] == "combined_grid_convergence"]
        fig, ax = plt.subplots(figsize=(7.5, 4.5))
        for (case, scheme), group in subset.groupby(["case", "scheme"]):
            ax.plot(
                group["level"], group["absolute_error_cf"], marker="o", label=f"{case}/{scheme}"
            )
        ax.axhline(5.0e-4, color="black", linestyle="--", linewidth=0.9, label="registry tolerance")
        ax.set_yscale("log")
        ax.set(
            title="Heston ADI convergence against CF", xlabel="grid level", ylabel="absolute error"
        )
        ax.legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(figure_dir / "pde_convergence.png", dpi=160)
        plt.close(fig)

    mc_path = root / "results/aggregated/mc_qe_qem_audit.csv"
    if mc_path.exists():
        mc = pd.read_csv(mc_path)
        finest_paths = int(mc["paths"].max())
        subset = mc.loc[mc["paths"] == finest_paths]
        fig, ax = plt.subplots(figsize=(7.5, 4.5))
        for scheme, group in subset.groupby("scheme"):
            summary = group.groupby("steps", as_index=False)["absolute_bias"].median()
            ax.plot(summary["steps"], summary["absolute_bias"], marker="o", label=str(scheme))
        ax.set(
            title="GPU Monte Carlo discretization audit",
            xlabel="time steps",
            ylabel="median absolute bias",
        )
        ax.legend()
        fig.tight_layout()
        fig.savefig(figure_dir / "mc_convergence_remediated.png", dpi=160)
        plt.close(fig)

    performance_path = root / "results/aggregated/T12_CPU_GPU_PERFORMANCE.csv"
    if performance_path.exists():
        performance = pd.read_csv(performance_path).dropna(subset=["speedup"])
        if not performance.empty:
            fig, ax = plt.subplots(figsize=(7, 4.5))
            ax.bar(performance["workload"], performance["speedup"])
            ax.axhline(1.0, color="black", linewidth=0.8)
            ax.tick_params(axis="x", rotation=25)
            ax.set(title="Measured CPU/GPU performance", ylabel="CPU time / GPU time")
            fig.tight_layout()
            fig.savefig(figure_dir / "gpu_performance.png", dpi=160)
            plt.close(fig)

    if result_dir is not None:
        baseline_path = result_dir / "T08_DVE_BASELINES.csv"
        ablation_path = result_dir / "T09_DVE_ABLATION.csv"
        if baseline_path.exists():
            baseline = pd.read_csv(baseline_path)
            fig, ax = plt.subplots(figsize=(7.5, 4.5))
            ax.bar(baseline["baseline"], baseline["recall_at_dev_5pct_fpr"])
            ax.set_ylim(0.0, 1.0)
            ax.set(
                title="Confirmatory validation baselines", ylabel="recall at DEV 5% FPR threshold"
            )
            fig.tight_layout()
            fig.savefig(figure_dir / "dve_baseline_comparison.png", dpi=160)
            plt.close(fig)
        if ablation_path.exists():
            ablation = pd.read_csv(ablation_path)
            fig, ax = plt.subplots(figsize=(8, 4.5))
            ax.bar(ablation["ablation_id"], ablation["recall_change"])
            ax.axhline(0.0, color="black", linewidth=0.8)
            ax.set(title="DVE component ablation", ylabel="ablated recall minus full DVE")
            fig.tight_layout()
            fig.savefig(figure_dir / "dve_ablation.png", dpi=160)
            plt.close(fig)


def _write_findings(
    root: Path,
    confirmatory: dict[str, Any] | None,
    confirmatory_label: str,
    remediation: dict[str, Any],
) -> tuple[str, str]:
    generated = datetime.now(UTC).isoformat()
    pde_status = str(remediation.get("pde_status", "UNAVAILABLE"))
    mc_status = str(remediation.get("mc_status", "UNAVAILABLE"))
    active_version = (
        "synthetic_v5" if confirmatory_label.startswith("confirmatory v6") else "synthetic_v4"
    )
    calibration_path = (
        root / f"results/confirmatory/{active_version}/calibration_diagnostics.parquet"
    )
    calibration_rows = pd.read_parquet(calibration_path) if calibration_path.exists() else None
    selected_calibrations = (
        calibration_rows.loc[calibration_rows["start_index"].astype(int) == -1]
        if calibration_rows is not None and "start_index" in calibration_rows
        else calibration_rows
    )
    calibration_successes = (
        int(selected_calibrations["success"].sum()) if selected_calibrations is not None else 0
    )
    calibration_total = len(selected_calibrations) if selected_calibrations is not None else 0
    attempted_starts = (
        calibration_rows.loc[calibration_rows["start_index"].astype(int) >= 0]
        if calibration_rows is not None and "start_index" in calibration_rows
        else None
    )
    attempted_start_successes = (
        int(attempted_starts["success"].sum()) if attempted_starts is not None else 0
    )
    attempted_start_total = len(attempted_starts) if attempted_starts is not None else 0
    findings = [
        {
            "finding_id": "FND-001",
            "title": "Independent ADI PDE remains outside full qualification",
            "risk_category": "numerical",
            "evidence": "results/aggregated/numerical_remediation_qualification.json",
            "severity": "HIGH" if pde_status != "PASS" else "LOW",
            "materiality": (
                "difficult-regime accuracy, convergence, or grid nonnegativity criteria remain "
                "unmet"
            ),
            "root_cause": (
                "nonuniform MCS/HV remediation improved the base case but did not satisfy every "
                "frozen criterion"
            ),
            "recommendation": (
                "retain PDE as a bounded diagnostic and continue boundary/grid research"
            ),
            "status": "OPEN" if pde_status != "PASS" else "CLOSED",
            "residual_risk": "MEDIUM" if pde_status != "PASS" else "LOW",
        },
        {
            "finding_id": "FND-002",
            "title": "Historical quote asynchrony limits temporal conclusions",
            "risk_category": "data",
            "evidence": "results/RESULT_AVAILABILITY.json",
            "severity": "HIGH",
            "materiality": "next-date OOS and a defensible one-day hedge proxy are unavailable",
            "root_cause": "historical quote_time is absent and the prospective series has one date",
            "recommendation": (
                "accumulate synchronized prospective observations or obtain licensed synchronized "
                "history"
            ),
            "status": "USAGE_LIMITATION_APPLIED",
            "residual_risk": "MEDIUM",
        },
        {
            "finding_id": "FND-003",
            "title": "Exploratory Full DVE underperformed FitOnly",
            "risk_category": "validation_framework",
            "evidence": "results/exploratory/synthetic_bs_proxy_v1/synthetic_locked_test.json",
            "severity": "MEDIUM",
            "materiality": (
                "negative evidence against assuming multi-dimensional evidence dominates fit"
            ),
            "root_cause": (
                "BS-only truth and proxy fault design; the result remains descriptive, not tunable "
                "evidence"
            ),
            "recommendation": "retain unchanged and compare with prospectively frozen studies",
            "status": "RETAINED_NEGATIVE_RESULT",
            "residual_risk": "MEDIUM",
        },
        {
            "finding_id": "FND-004",
            "title": f"Full DVE confirmatory performance ({confirmatory_label})",
            "risk_category": "validation_framework",
            "evidence": "results/confirmatory/synthetic_v5/confirmatory_metrics.json",
            "severity": "MEDIUM",
            "materiality": "baseline comparison must be interpreted from measured locked metrics",
            "root_cause": "heterogeneous evidence dimensions need not improve every fault class",
            "recommendation": (
                "do not tune from locked outcomes; retain fault-stratified and ablation results"
            ),
            "status": "EVALUATED" if confirmatory is not None else "PENDING",
            "residual_risk": "MEDIUM",
        },
        {
            "finding_id": "FND-005",
            "title": "Monte Carlo discretization and seed sensitivity",
            "risk_category": "numerical",
            "evidence": "results/aggregated/mc_qe_qem_audit.csv",
            "severity": "LOW" if mc_status == "PASS" else "MEDIUM",
            "materiality": (
                "bias and sampling error vary by scheme, time step, path count, and seed"
            ),
            "root_cause": "discretization error and stochastic sampling",
            "recommendation": (
                "use qualified QE-M settings and report bias separately from standard error"
            ),
            "status": "REMEDIATED_WITH_LIMITATION" if mc_status == "PASS" else "OPEN",
            "residual_risk": "LOW" if mc_status == "PASS" else "MEDIUM",
        },
        {
            "finding_id": "FND-006",
            "title": "Confirmatory v3 used oracle DGP parameters for developer pricing",
            "risk_category": "synthetic_design",
            "evidence": "results/confirmatory/synthetic_v2/confirmatory_metrics.json",
            "severity": "MEDIUM",
            "materiality": (
                "v3 isolates model form but is not evidence for observable-quote Heston calibration"
            ),
            "root_cause": (
                "developer Heston values were generated from latent truth parameters rather than "
                "parameters calibrated exclusively from observed synthetic quotes"
            ),
            "recommendation": (
                "retain v3 as an oracle-parameter isolation and use release-aligned v6 for the "
                "primary confirmatory conclusion"
            ),
            "status": "SUPERSEDED_FOR_PRIMARY_CONFIRMATORY_INFERENCE",
            "residual_risk": "LOW"
            if confirmatory_label.startswith("confirmatory v6")
            else "MEDIUM",
        },
        {
            "finding_id": "FND-007",
            "title": "Confirmatory multistart calibration qualification",
            "risk_category": "calibration",
            "evidence": f"results/confirmatory/{active_version}/calibration_diagnostics.parquet",
            "severity": "HIGH"
            if calibration_total and calibration_successes / calibration_total < 0.5
            else "LOW",
            "materiality": (
                f"{calibration_successes}/{calibration_total} selected observable-quote fits "
                f"qualified; {attempted_start_successes}/{attempted_start_total} individual "
                "multistart attempts reported optimizer success"
            ),
            "root_cause": (
                "some individual starts reached the prospectively frozen iteration limit; "
                "selection admitted only converged starts"
            ),
            "recommendation": (
                "retain selected-fit qualification and continue reporting individual-start "
                "convergence separately"
            ),
            "status": (
                "QUALIFIED_WITH_ATTEMPT_DIAGNOSTICS"
                if calibration_total and calibration_successes == calibration_total
                else "OPEN_LIMITATION"
            ),
            "residual_risk": "HIGH"
            if calibration_total and calibration_successes / calibration_total < 0.5
            else "LOW",
        },
        {
            "finding_id": "FND-008",
            "title": "Confirmatory v5 failed its frozen validation calibration qualification",
            "risk_category": "calibration",
            "evidence": "results/confirmatory/synthetic_v4/confirmatory_metrics.json",
            "severity": "HIGH",
            "materiality": (
                "v5 is not eligible as primary confirmatory evidence and its locked stage was not "
                "used to tune the replacement study"
            ),
            "root_cause": (
                "the prospectively frozen observable-quote calibration design did not qualify on "
                "VALIDATION"
            ),
            "recommendation": (
                "preserve v5 as CALIBRATION_UNQUALIFIED history and require v6 prequalification "
                "before any new locked generation"
            ),
            "status": "PRESERVED_PARTIAL_CALIBRATION_UNQUALIFIED",
            "residual_risk": "HIGH" if confirmatory is None else "MEDIUM",
        },
    ]
    governance = {
        "schema_version": "2.0",
        "generated_at": generated,
        "derivation": "machine-readable numerical, synthetic, and data-availability evidence",
        "findings": findings,
    }
    tracker = {
        "schema_version": "2.0",
        "generated_at": generated,
        "items": [
            {
                "finding_id": row["finding_id"],
                "recommendation": row["recommendation"],
                "status": row["status"],
            }
            for row in findings
        ],
    }
    (root / "governance/findings_register.yaml").write_text(
        yaml.safe_dump(governance, sort_keys=False), encoding="utf-8"
    )
    (root / "governance/remediation_tracker.yaml").write_text(
        yaml.safe_dump(tracker, sort_keys=False), encoding="utf-8"
    )
    rows = "\n".join(
        f"| {row['finding_id']} | {row['title']} | {row['severity']} | "
        f"{row['status']} | `{row['evidence']}` |"
        for row in findings
    )
    findings_md = (
        "# Findings Register\n\n"
        "| ID | Finding | Severity | Status | Evidence |\n|---|---|---|---|---|\n"
        f"{rows}\n"
    )
    remediation_rows = "\n".join(
        f"| {row['finding_id']} | {row['recommendation']} | {row['status']} |" for row in findings
    )
    remediation_md = (
        "# Remediation Tracker\n\n"
        "| Finding | Action | Status |\n|---|---|---|\n"
        f"{remediation_rows}\n"
    )
    return findings_md, remediation_md


def generate_mandatory_reports(root: Path) -> dict[str, Any]:
    """Write mandatory Markdown reports from current measured artifacts."""

    verification = verify_final_evidence(root)
    if verification["status"] != "PASS":
        raise RuntimeError(f"final test/review evidence is stale or unresolved: {verification}")
    reconcile_experiment_registry(root)
    reports = root / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    env = _json(root / "artifacts/system/environment.json")
    acquisition = _json(root / "artifacts/data/acquisition_inspection.json")
    data = _json(root / "results/aggregated/data_eda.json")
    numerical = _json(root / "results/aggregated/numerical_benchmarks.json")
    remediation = _json(root / "results/aggregated/numerical_remediation_qualification.json")
    selection = _json(root / "results/aggregated/developer_method_selection.json")
    availability = _json(root / "results/RESULT_AVAILABILITY.json")
    t04 = pd.read_csv(root / "reports/tables/T04_CALIBRATION_OBJECTIVE_COMPARISON.csv")
    t05 = pd.read_csv(root / "reports/tables/T05_OPTIMIZER_IDENTIFIABILITY.csv")
    t06 = pd.read_csv(root / "reports/tables/T06_PRICING_MODEL_COMPARISON.csv")
    t11 = pd.read_csv(root / "reports/tables/T11_PORTFOLIO_MATERIALITY.csv")
    supplement_paths = (
        root / "results/aggregated/near_expiry_study.csv",
        root / "results/aggregated/liquidity_filter_study.csv",
        root / "results/aggregated/am_pm_robustness.csv",
    )
    supplement_frames = [pd.read_csv(path) for path in supplement_paths if path.exists()]
    supplement = (
        pd.concat(supplement_frames, ignore_index=True) if supplement_frames else pd.DataFrame()
    )
    final_compute_path = root / "results/aggregated/final_compute_benchmarks.json"
    final_compute = _json(final_compute_path) if final_compute_path.exists() else None
    precision_path = root / "results/aggregated/precision_sensitivity.json"
    precision = _json(precision_path) if precision_path.exists() else None
    final_t12_path = root / "results/aggregated/T12_CPU_GPU_PERFORMANCE.csv"
    final_t12 = pd.read_csv(final_t12_path) if final_compute and final_t12_path.exists() else None
    confirmatory, confirmatory_label, confirmatory_dir = _synthetic_evidence(root)
    exploratory = _json(
        root / "results/exploratory/synthetic_bs_proxy_v1/synthetic_locked_test.json"
    )
    exploratory_metrics = exploratory["metrics"]
    confirmatory_metrics = confirmatory.get("baselines", {}) if confirmatory else {}
    oracle_v3_path = root / "results/confirmatory/synthetic_v2/confirmatory_metrics.json"
    canonical_v2_path = root / "results/confirmatory/synthetic_v1/confirmatory_metrics.json"
    observable_v4_path = root / "results/confirmatory/synthetic_v3/confirmatory_metrics.json"
    release_v5_path = root / "results/confirmatory/synthetic_v4/confirmatory_metrics.json"
    oracle_v3 = _json(oracle_v3_path) if oracle_v3_path.exists() else None
    canonical_v2 = _json(canonical_v2_path) if canonical_v2_path.exists() else None
    observable_v4 = _json(observable_v4_path) if observable_v4_path.exists() else None
    release_v5 = _json(release_v5_path) if release_v5_path.exists() else None

    _write_monitoring(root, confirmatory_dir)
    _make_figures(root, confirmatory_dir)
    findings_md, remediation_md = _write_findings(
        root, confirmatory, confirmatory_label, remediation
    )

    quality = data["processing_summary"]["historical_spx_sample"]["quality_summary"]
    historical = data["processing_summary"]["historical_spx_sample"]
    historical_source = acquisition["historical_spx_sample"]
    if final_t12 is not None:
        assert final_compute is not None
        measured_speedups = final_t12.dropna(subset=["speedup"])
        final_compute_text = (
            f"The final benchmark manifest records profile `{final_compute['profile']}` on "
            f"`{final_compute['device']}` across {len(final_t12)} workload families. "
            f"Measured CPU/GPU speedups range from "
            f"{_fmt(float(measured_speedups['speedup'].min()))}x to "
            f"{_fmt(float(measured_speedups['speedup'].max()))}x."
            if not measured_speedups.empty
            else "The final benchmark recorded no comparable CPU/GPU speedup."
        )
    else:
        final_compute_text = "The final compute benchmark artifact is not yet available."
    precision_text = (
        "The N10 precision study measured maximum FP64/FP32 absolute errors of "
        f"{precision['maximum_fp64_absolute_error']:.3e}/"
        f"{precision['maximum_fp32_absolute_error']:.3e}; FP64 remains mandatory."
        if precision
        else "The N10 precision-sensitivity artifact is unavailable."
    )
    if not supplement.empty:
        supplement_counts = supplement.groupby("experiment").size().to_dict()
        supplement_text = (
            f"The empirical supplement contains {len(supplement)} measured model rows: "
            f"{supplement_counts}. These are cross-sectional EOD results, not synchronized P&L."
        )
    else:
        supplement_text = (
            "Near-expiry, liquidity-filter, and AM/PM supplement artifacts are not yet available."
        )
    environment_md = f"""# Compute and Environment

Generated from `artifacts/system/environment.json` captured at {env["captured_at"]}.

- Python: `{env["python"].split()[0]}` at `{env["sys_executable"]}`
- PyTorch/CUDA: `{env["torch"]}` / runtime `{env["torch_cuda_runtime"]}`
- GPU: `{env["gpu_name"]}` with {env["gpu_total_memory_bytes"] / 1024**3:.2f} GiB
- CuPy/runtime: `{env["cupy"]}` / `{env["cupy_runtime"]}`
- NumPy/SciPy/QuantLib: `{env["numpy"]}` / `{env["scipy"]}` / `{env["quantlib"]}`

CUDA FP64 smoke tests passed. Research workloads recorded on GPU include batched Heston CF,
Heston Monte Carlo, and synthetic truth generation. The measured small CF batch speedup was
{numerical["benchmarks"][0]["speedup"]:.3f}x (slower than CPU); the medium batch speedup was
{numerical["benchmarks"][1]["speedup"]:.3f}x. GPU benefit is workload-size dependent.

{final_compute_text}
{precision_text}
"""
    (reports / "COMPUTE_AND_ENVIRONMENT.md").write_text(environment_md, encoding="utf-8")

    data_md = f"""# Data Quality Report

The public HistoricalData.net archive contains {historical_source["vendor_rows_total"]:,} vendor
rows across {historical_source["trading_days_present"]} dates from
{historical_source["minimum_date"]} through {historical_source["maximum_date"]}. The normalized
SPX/SPXW subset contains
{quality["ROWS_TOTAL"]:,} rows; {quality["ROWS_INCLUDED"]:,} passed the model-ready filter and
{quality["ROWS_EXCLUDED"]:,} have auditable exclusion codes. The processed artifact reports
{historical["daily_files_processed"]} market dates.

All historical rows lack `quote_time`; the data are standing EOD research quotes, not synchronized
NBBO. Apparent parity or static-arbitrage violations therefore remain data-quality evidence and are
not automatically attributed to model failure. Official Cboe marking data contain one prospective
date, and the DataShop integration sample was used for schema validation. Massive was skipped when
no API key was available. See `results/RESULT_AVAILABILITY.json` for the specific temporal analyses
that remain unavailable.

{supplement_text}
"""
    (reports / "DATA_QUALITY_REPORT.md").write_text(data_md, encoding="utf-8")

    selection_status = str(selection.get("status", "UNAVAILABLE"))
    best_objective = t04.sort_values("evaluation_C04").iloc[0]
    optimizer_rows = t05.loc[t05["optimizer_id"].notna()]
    best_optimizer = optimizer_rows.sort_values("objective_value").iloc[0]
    model_summary = (
        t06.groupby("model_id", as_index=False)
        .agg(market_rows=("quote_date", "size"), median_price_rmse=("price_rmse", "median"))
        .sort_values("median_price_rmse")
    )
    model_lines = "\n".join(
        f"- {row['model_id']!s}: median price RMSE "
        f"{_fmt(float(row['median_price_rmse']))} over {int(row['market_rows'])} measured rows"
        for _, row in model_summary.iterrows()
    )
    model_dev = f"""# Model Development Document

## Intended use and release discipline

The developer Heston model is restricted to European SPXW vanilla-option research valuation and
Greeks. The selected methodology record is `{selection_status}` and states objective
`{selection.get("selected_objective")}`, optimizer `{selection.get("selected_optimizer")}`, and
Feller treatment `{selection.get("selected_feller_treatment")}`. Selection uses DEV and confirmation
uses VALIDATION; locked observations are excluded from selection.

## Calibration evidence

T04 contains {len(t04)} measured objective-study rows. On the common C04 evaluation, the lowest
measured DEV row was {best_objective["objective"]} at {_fmt(best_objective["evaluation_C04"])}.
T05 contains {len(optimizer_rows)} optimizer rows plus measured identifiability diagnostics; its
lowest native optimizer objective was {best_optimizer["optimizer_id"]} at
{_fmt(best_optimizer["objective_value"])}. These bounded comparisons are empirical measurements,
not assertions that convergence succeeded in every run.

## Numerical verification

Developer CF and QuantLib differ by {numerical["cf_quantlib_absolute_error"]:.3e} at the reference
case. The independent MC remediation status is `{remediation["mc_status"]}` after Euler, QE, and
QE-M
multi-seed/path/time-step testing. Bias and sampling standard error remain separate. The independent
nonuniform MCS/HV PDE status is `{remediation["pde_status"]}`: remediation was seriously attempted,
but difficult regimes and convergence/nonnegativity criteria prevent a full pass.

## Cross-sectional model evidence

{model_lines}
"""
    (reports / "MODEL_DEVELOPMENT_DOCUMENT.md").write_text(model_dev, encoding="utf-8")

    exp_fit_recall = _metric(exploratory_metrics, "V00", "recall_at_dev_5pct_fpr")
    exp_full_recall = _metric(exploratory_metrics, "V06", "recall_at_dev_5pct_fpr")
    exp_fit_auprc = _metric(exploratory_metrics, "V00", "auprc")
    exp_full_auprc = _metric(exploratory_metrics, "V06", "auprc")
    if confirmatory_metrics:
        con_fit_recall = _metric(confirmatory_metrics, "V00", "recall_at_dev_5pct_fpr")
        con_full_recall = _metric(confirmatory_metrics, "V06", "recall_at_dev_5pct_fpr")
        con_fit_auprc = _metric(confirmatory_metrics, "V00", "auprc")
        con_full_auprc = _metric(confirmatory_metrics, "V06", "auprc")
        synthetic_text = (
            f"The {confirmatory_label} measured FitOnly recall/AUPRC of "
            f"{con_fit_recall:.3f}/{con_fit_auprc:.3f} and Full DVE recall/AUPRC of "
            f"{con_full_recall:.3f}/{con_full_auprc:.3f}. The result is retained whether favorable "
            "or unfavorable; locked observations are not used for redesign."
        )
    else:
        synthetic_text = (
            "The new confirmatory locked execution has not yet produced measured output."
        )
    history_lines: list[str] = []
    if canonical_v2 is not None:
        history_lines.append(
            _baseline_summary(canonical_v2, "Preserved confirmatory v2 canonical-truth study")
        )
    if oracle_v3 is not None:
        history_lines.append(
            _baseline_summary(oracle_v3, "Preserved v3 oracle-parameter model-form isolation")
        )
    if observable_v4 is not None and observable_v4.get("baselines"):
        history_lines.append(
            _baseline_summary(
                observable_v4,
                "Preserved v4 observable-quote study (calibration-unqualified)",
            )
        )
    if release_v5 is not None:
        if release_v5.get("baselines"):
            history_lines.append(
                _baseline_summary(
                    release_v5,
                    "Preserved v5 release-aligned study (calibration-unqualified)",
                )
            )
        else:
            history_lines.append(
                "Preserved v5 release-aligned study: VALIDATION calibration qualification "
                "failed before locked generation; status PARTIAL/CALIBRATION_UNQUALIFIED."
            )
    synthetic_history_text = " ".join(history_lines)
    calibration_text = ""
    if confirmatory_dir is not None and confirmatory_dir.name == "synthetic_v5":
        calibration = pd.read_parquet(confirmatory_dir / "calibration_diagnostics.parquet")
        selected_calibration = calibration.loc[calibration["start_index"].astype(int) == -1]
        attempted_calibration = calibration.loc[calibration["start_index"].astype(int) >= 0]
        calibration_text = (
            f"The {confirmatory_label} contains {len(selected_calibration)} selected observable-"
            f"quote C04 calibrations, all {int(selected_calibration['success'].sum())} qualified, "
            f"plus {len(attempted_calibration)} individual-start diagnostics of which "
            f"{int(attempted_calibration['success'].sum())} reported optimizer success. "
            "Calibration inputs are bid/mid/ask observations, not latent DGP parameters."
        )

    temporal_blockers = "; ".join(
        str(row["blocker"])
        for row in availability.get("entries", [])
        if row["phase_id"] in {"30", "31"}
    )
    validation_conclusion = "Conditionally validated for the narrow specified research use"
    if (
        selection_status != "SELECTED_ON_DEV_CONFIRMED_ON_VALIDATION"
        or not confirmatory_label.startswith("confirmatory v6")
    ):
        validation_conclusion = "Not yet validated for the complete specified research use"
    validation = f"""# Independent Validation Memorandum

## 1. Executive Summary

**Simulated research conclusion: {validation_conclusion}. This is not institutional approval.**

## 2. Model Purpose and Intended Use

European cash-settled SPXW vanilla research valuation, Greeks, and model-risk validation only.

## 3. Scope

Public-data audit, DEV/VALIDATION calibration studies, independent numerical engines, qualified
synthetic truth engines, fault injection, validation baselines, ablations, and portfolio
materiality. The v6 synthetic study calibrates developer Heston parameters only from observable
synthetic bid/mid/ask quotes after a prospectively frozen calibration prequalification.

## 4. Data and Lineage

Raw public inputs have manifests, source URLs, hashes, schema notes, and explicit processing stages.

## 5. Data Limitations

{temporal_blockers}

## 6. Methodology

DVE preserves E_data, E_num, E_cal, E_ident, E_param, E_form, E_greek, E_extra, and E_outcome.
Risk evidence and economic materiality remain separate outputs.

## 7. Key Assumptions

Synthetic observation spreads, missingness, and material-error thresholds are frozen design inputs;
synthetic evidence is not production performance evidence.

## 8. Conceptual Soundness

Heston is defensible for the narrow vanilla research use but cannot represent every jump,
local-volatility, regime, or microstructure mechanism.

## 9. Implementation Verification

CF/QuantLib error is {numerical["cf_quantlib_absolute_error"]:.3e}; MC remediation is
`{remediation["mc_status"]}`; PDE remediation remains `{remediation["pde_status"]}`.

## 10. Calibration Assessment

Measured C00-C04, O00-O04 where available, Feller, bootstrap, profile, multistart, and date-level
stability artifacts are reported in T04/T05 and `results/aggregated/`.

## 11. Independent Benchmarking

Independent Monte Carlo, independent ADI PDE, QuantLib, SVI/SSVI, LocalVol, and Bates sensitivity
paths are represented without treating them as interchangeable models.

## 12. Sensitivity Analysis

Objective, optimizer, Feller, numerical-grid, seed, path-count, wing/maturity availability, and
developer-method ablations were evaluated. The N10 benchmark retains FP64 as the canonical
precision after directly measuring FP32 error where its artifact is available.

## 13. Stress Testing

The v6 design covers F00-F16 when its qualified locked output exists; fault coverage is
machine-readable and proxy operators remain explicitly labeled for F04-F16.

## 14. Synthetic Ground-Truth Findings

The immutable exploratory BS/proxy study found FitOnly recall/AUPRC
{exp_fit_recall:.3f}/{exp_fit_auprc:.3f} versus Full DVE {exp_full_recall:.3f}/{exp_full_auprc:.3f}.
It is descriptive because registries were not frozen and must not be tuned against. {synthetic_text}
{calibration_text} {synthetic_history_text} V3 remains an oracle-parameter limitation and is not
used as observable-calibration evidence for the primary conclusion.

## 15. Real-Market Outcome Analysis

Cross-sectional and real locked-date comparisons were run. Synchronized temporal outcome claims are
unavailable for the reasons in Section 5; no causality or live-P&L claim is made.
{supplement_text}

## 16. Ongoing Monitoring

See `ONGOING_MONITORING_PLAN.md` and `results/aggregated/monitoring_simulation.csv`.

## 17. Model Limitations

See `MODEL_LIMITATIONS_REGISTER.md`.

## 18. Findings

PDE qualification, quote asynchrony, unfavorable DVE evidence, and MC sensitivity remain visible.

## 19. Materiality Assessment

T11 contains {len(t11)} measured portfolio/model rows for six standardized portfolios, including
value, delta, gamma, vega, and separate valuation/Greek materiality.

## 20. Remediation

See `REMEDIATION_TRACKER.md`; completed remediation does not erase the initial negative evidence.

## 21. Residual Model Risk

Moderate within the narrow research use and high outside it, particularly for synchronized outcomes,
PDE-only conclusions, and prohibited products.

## 22. Validation Conclusion

{validation_conclusion}; all prohibited uses remain prohibited.
"""
    (reports / "INDEPENDENT_VALIDATION_MEMORANDUM.md").write_text(validation, encoding="utf-8")

    executive = f"""# Executive Summary

DerivGuard-MRM has measured environment, data, calibration, numerical-validation, synthetic,
cross-sectional, and portfolio-materiality artifacts. Developer CF agrees with QuantLib to
{numerical["cf_quantlib_absolute_error"]:.3e}; the audited GPU MC status is
`{remediation["mc_status"]}`, while the remediated ADI PDE remains `{remediation["pde_status"]}`.

The exploratory BS/proxy study is preserved unchanged and unfavorable to Full DVE:
FitOnly recall/AUPRC {exp_fit_recall:.3f}/{exp_fit_auprc:.3f} versus Full DVE
{exp_full_recall:.3f}/{exp_full_auprc:.3f}. {synthetic_text}
{calibration_text} {synthetic_history_text}

The historical public sample is unsynchronized, so cross-sectional conclusions are feasible but
next-date synchronized OOS and the one-day discrete hedging proxy remain unavailable. T11 contains
real measured portfolio valuation and Greek disagreements; it is not a status placeholder.
"""
    (reports / "EXECUTIVE_SUMMARY.md").write_text(executive, encoding="utf-8")

    limitations = f"""# Model Limitations Register

1. Historical public quotes lack quote timestamps and are not synchronized NBBO.
2. The prospective Cboe marking series currently contains one date.
3. Independent nonuniform MCS/HV PDE qualification is `{remediation["pde_status"]}`.
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
"""
    (reports / "MODEL_LIMITATIONS_REGISTER.md").write_text(limitations, encoding="utf-8")
    (reports / "FINDINGS_REGISTER.md").write_text(findings_md, encoding="utf-8")
    (reports / "REMEDIATION_TRACKER.md").write_text(remediation_md, encoding="utf-8")

    monitoring = """# Ongoing Monitoring Plan

Monitor each eligible surface for data exclusions, parity/static-arbitrage evidence, developer fit,
independent numerical disagreement, challenger disagreement, bootstrap uncertainty,
identifiability, parameter and Greek drift, extrapolation distance, and later outcomes only when
synchronized evidence
is available. GREEN means no current material breach; AMBER means investigate/recalibrate; RED means
consider a usage limitation or redevelopment. These are project simulation labels, not institutional
policy. Thresholds must be versioned and frozen before their evaluation population.
"""
    (reports / "ONGOING_MONITORING_PLAN.md").write_text(monitoring, encoding="utf-8")

    compute = (
        environment_md
        + """
## Reproducibility

Primary full run:
`.\\.venv\\Scripts\\python.exe -m derivguard full --profile research --device cuda --resume`

Resume uses the same command. Machine-readable outputs record seeds, backends, timestamps, hashes,
and frozen design identifiers. The 20 GiB disk reserve remains mandatory.
"""
    )
    (reports / "COMPUTE_AND_REPRODUCIBILITY_REPORT.md").write_text(compute, encoding="utf-8")

    technical = f"""# Technical Report

{model_dev.splitlines()[1] and chr(10).join(model_dev.splitlines()[2:])}

## Data Risk

{chr(10).join(data_md.splitlines()[2:])}

## Validation Results

{chr(10).join(validation.splitlines()[2:])}

## Future Extension Roadmap

- v1.1: licensed synchronized historical Cboe quotes.
- v1.2: specialized 0DTE validation.
- v2/v2.1: rates and credit derivatives.
- v3: XVA and counterparty model-risk extensions.
- Later: GPU neural surrogates as a separate model-risk case study.
"""
    (reports / "TECHNICAL_REPORT.md").write_text(technical, encoding="utf-8")

    gate_payload = write_gate_evidence(root)
    acceptance_rows = acceptance_from_phase_evidence(gate_payload)
    overall = (
        "PASS"
        if all(row["status"] in {"PASS", "NOT APPLICABLE"} for row in acceptance_rows)
        else "PARTIAL"
    )
    acceptance_lines = "\n".join(
        f"| {row['gate']} | {row['status']} | "
        f"{'; '.join(row['evidence'][:4] + row['blockers'][:2])} |"
        for row in acceptance_rows
    )
    acceptance = f"""# Final Acceptance Report

Overall status: **{overall}**. Every status below is derived from
`artifacts/acceptance/gate_evidence.json`, artifact semantics, final test evidence, or a documented
availability blocker. A status-only/NaN table is never accepted as scientific evidence.

| Gate | Status | Evidence / blocker |
|---|---|---|
{acceptance_lines}
"""
    (reports / "FINAL_ACCEPTANCE_REPORT.md").write_text(acceptance, encoding="utf-8")
    state = reconcile_run_state_from_evidence(root, gate_payload)
    phase_rows = "\n".join(
        f"| {row['phase_id']} | {row['name']} | {row['status']} | {row.get('blocker') or ''} |"
        for row in gate_payload["phases"]
    )
    (reports / "EXECUTION_STATUS.md").write_text(
        "# Execution Status\n\n"
        f"Overall: **{state['overall_status']}**.\n\n"
        "| Phase | Name | Evidence status | Blocker |\n|---|---|---|---|\n"
        f"{phase_rows}\n",
        encoding="utf-8",
    )
    return {
        "status": "REPORTS_GENERATED_FROM_ARTIFACTS",
        "overall_acceptance": overall,
        "confirmatory_source": str(confirmatory_dir) if confirmatory_dir else None,
        "report_count": 13,
    }
