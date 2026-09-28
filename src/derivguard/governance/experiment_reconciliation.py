"""Reconcile the experiment registry against measured project artifacts.

The registry is a catalogue, not an execution log.  This module changes an
entry to ``COMPLETED`` only when a semantic predicate over a non-empty result
artifact succeeds.  Missing evidence remains ``PARTIAL`` (or ``UNAVAILABLE``
when an attempted external-data blocker is recorded).  No experiment ID is a
completion allow-list.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from hashlib import sha256
from pathlib import Path
from typing import Any

import pandas as pd
import yaml  # type: ignore[import-untyped]


@dataclass(frozen=True)
class ArtifactSnapshot:
    path: str
    sha256: str
    rows: int | None


@dataclass(frozen=True)
class ExperimentEvaluation:
    status: str
    feasibility: str
    reason: str
    predicates: dict[str, bool]
    evidence: tuple[ArtifactSnapshot, ...]


def _sha256_file(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sha256_json(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, default=str).encode()
    return sha256(encoded).hexdigest()


class _Evidence:
    def __init__(self, root: Path) -> None:
        self.root = root
        self._frames: dict[str, pd.DataFrame | None] = {}
        self._objects: dict[str, dict[str, Any] | None] = {}

    def frame(self, relative: str) -> pd.DataFrame | None:
        if relative not in self._frames:
            path = self.root / relative
            if not path.is_file():
                self._frames[relative] = None
            elif path.suffix == ".parquet":
                self._frames[relative] = pd.read_parquet(path)
            else:
                self._frames[relative] = pd.read_csv(path)
        return self._frames[relative]

    def object(self, relative: str) -> dict[str, Any] | None:
        if relative not in self._objects:
            path = self.root / relative
            if not path.is_file():
                self._objects[relative] = None
            else:
                value = json.loads(path.read_text(encoding="utf-8"))
                self._objects[relative] = value if isinstance(value, dict) else None
        return self._objects[relative]

    def snapshot(self, relative: str) -> ArtifactSnapshot | None:
        path = self.root / relative
        if not path.is_file():
            return None
        rows: int | None = None
        if path.suffix in {".csv", ".parquet"}:
            frame = self.frame(relative)
            rows = None if frame is None else len(frame)
        return ArtifactSnapshot(relative, _sha256_file(path), rows)

    def snapshots(self, *paths: str) -> tuple[ArtifactSnapshot, ...]:
        values = (self.snapshot(path) for path in paths)
        return tuple(value for value in values if value is not None)


def _evaluation(
    predicates: dict[str, bool],
    reason: str,
    evidence: tuple[ArtifactSnapshot, ...],
    *,
    unavailable: bool = False,
) -> ExperimentEvaluation:
    if unavailable:
        return ExperimentEvaluation(
            "UNAVAILABLE", "EXTERNAL_DATA_BLOCKER", reason, predicates, evidence
        )
    passed = bool(predicates) and all(predicates.values())
    status = "COMPLETED" if passed else "PARTIAL"
    feasibility = "EXECUTED_MEASURED" if passed else "ATTEMPTED_INCOMPLETE"
    if not evidence:
        reason = f"No measured artifact found. {reason}"
        feasibility = "NO_EXECUTION_EVIDENCE"
    return ExperimentEvaluation(status, feasibility, reason, predicates, evidence)


def _column_values(frame: pd.DataFrame | None, column: str) -> set[str]:
    if frame is None or column not in frame:
        return set()
    return set(frame[column].dropna().astype(str))


def _data_evaluation(entry: dict[str, Any], evidence: _Evidence) -> ExperimentEvaluation:
    experiment_id = str(entry["id"])
    forward_path = "results/raw/forward_estimator_comparison.parquet"
    processing_path = "artifacts/data/processing_summary.json"
    acquisition_path = "artifacts/data/acquisition_inspection.json"
    forward = evidence.frame(forward_path)
    processing = evidence.object(processing_path) or {}
    acquisition = evidence.object(acquisition_path) or {}
    supplement_by_id = {
        "D03": "results/aggregated/liquidity_filter_study.csv",
        "D06": "results/aggregated/am_pm_robustness.csv",
    }
    if experiment_id == "D01":
        historical = processing.get("historical_spx_sample", {})
        stages = historical.get("rows_by_stage", {}) if isinstance(historical, dict) else {}
        subsets = _column_values(forward, "subset_id")
        predicates = {
            "raw_and_filtered_counts_measured": int(stages.get("normalized", 0))
            > int(stages.get("filtered", 0))
            > 0,
            "quote_quality_subsets_measured": len(subsets) >= 4,
        }
        return _evaluation(
            predicates,
            f"normalized/filtered rows and {len(subsets)} quote-quality subsets were inspected",
            evidence.snapshots(processing_path, forward_path),
        )
    if experiment_id == "D02":
        availability_path = "results/RESULT_AVAILABILITY.json"
        availability = evidence.object(availability_path) or {}
        entries = availability.get("entries", [])
        blocker = next(
            (
                str(item.get("blocker"))
                for item in entries
                if isinstance(item, dict) and item.get("phase_id") == "30"
            ),
            "Synchronized multi-date public observations are unavailable.",
        )
        predicates = {
            "historical_asynchrony_documented": bool(acquisition),
            "multi_date_synchronized_comparator_available": False,
        }
        return _evaluation(
            predicates,
            blocker,
            evidence.snapshots(acquisition_path, availability_path),
        )
    if experiment_id in supplement_by_id:
        path = supplement_by_id[experiment_id]
        frame = evidence.frame(path)
        predicates = {
            "measured_rows_present": frame is not None and not frame.empty,
            "model_metrics_present": frame is not None
            and {
                "price_rmse",
                "spread_normalized_rmse",
                "inside_bid_ask_rate",
            }.issubset(frame.columns),
        }
        if experiment_id == "D03":
            predicates["multiple_quote_subsets"] = len(_column_values(frame, "quote_subset")) >= 2
        else:
            predicates["am_and_pm_measured"] = {"AM", "PM"}.issubset(
                _column_values(frame, "settlement_class")
            )
        return _evaluation(
            predicates, "cross-sectional supplement rows inspected", evidence.snapshots(path)
        )
    if experiment_id == "D04":
        successful = (
            forward.loc[forward["status"].astype(str) == "SUCCESS"]
            if forward is not None and "status" in forward
            else pd.DataFrame()
        )
        methods = _column_values(successful, "method")
        predicates = {
            "all_estimators_measured": methods == {"F0", "F1", "F2", "F3"},
            "residual_metrics_present": not successful.empty
            and {"residual_rmse", "weighted_rmse"}.issubset(successful.columns),
        }
        return _evaluation(
            predicates, f"successful estimators={sorted(methods)}", evidence.snapshots(forward_path)
        )
    if experiment_id == "D05":
        path = "reports/tables/T02_EXTERNAL_CURVE_SENSITIVITY.csv"
        frame = evidence.frame(path)
        predicates = {
            "measured_rows_present": frame is not None and not frame.empty,
            "price_shift_metrics_present": frame is not None
            and {"mean_absolute_price_shift", "rmse_price_shift"}.issubset(frame.columns),
            "dev_validation_locked_covered": {"DEV", "VALIDATION", "LOCKED_TEST"}.issubset(
                _column_values(frame, "split")
            ),
        }
        return _evaluation(
            predicates,
            "external Treasury curve sensitivity rows inspected",
            evidence.snapshots(path),
        )
    if experiment_id == "D07":
        sources = {
            key
            for key, value in acquisition.items()
            if isinstance(value, dict) and value.get("artifact_sha256")
        }
        predicates = {
            "multiple_sources_hashed": len(sources) >= 3,
            "datashop_schema_inspected": bool(
                isinstance(acquisition.get("cboe_datashop_sample"), dict)
                and acquisition["cboe_datashop_sample"].get("members")
            ),
        }
        return _evaluation(
            predicates, f"hashed sources={sorted(sources)}", evidence.snapshots(acquisition_path)
        )
    return _evaluation({}, f"No evidence contract for {experiment_id}", ())


def _numerical_evaluation(entry: dict[str, Any], evidence: _Evidence) -> ExperimentEvaluation:
    experiment_id = str(entry["id"])
    numerical_path = "results/aggregated/numerical_benchmarks.json"
    tests_path = "artifacts/tests/final_test_summary.json"
    remediation_path = "results/aggregated/numerical_remediation_qualification.json"
    numerical = evidence.object(numerical_path) or {}
    tests = evidence.object(tests_path) or {}
    remediation = evidence.object(remediation_path) or {}
    if experiment_id == "N01":
        source = "src/derivguard/market/black_scholes.py"
        predicates = {
            "implementation_present": (evidence.root / source).is_file(),
            "full_test_suite_passed": tests.get("status") == "PASS",
        }
        return _evaluation(
            predicates,
            "analytic foundation and passing full tests inspected",
            evidence.snapshots(source, tests_path),
        )
    if experiment_id in {"N02", "N03"}:
        experiment_ids = set(map(str, numerical.get("experiment_ids", [])))
        predicates = {
            "research_numerical_run": numerical.get("profile") == "research",
            "experiment_recorded": experiment_id in experiment_ids,
            "finite_reference_error": float(
                numerical.get("cf_quantlib_absolute_error", float("inf"))
            )
            < 5.0e-7,
            "full_test_suite_passed": tests.get("status") == "PASS",
        }
        return _evaluation(
            predicates,
            "CF/reference and stress-test evidence inspected",
            evidence.snapshots(numerical_path, tests_path),
        )
    mc_path = "results/aggregated/mc_qe_qem_audit.csv"
    mc = evidence.frame(mc_path)
    if experiment_id == "N06":
        path = "results/aggregated/mc_variance_reduction.csv"
        manifest_path = "results/aggregated/mc_variance_reduction.json"
        frame = evidence.frame(path)
        manifest = evidence.object(manifest_path) or {}
        variants: set[tuple[bool, bool]] = set()
        if frame is not None and {"antithetic", "control_variate"}.issubset(frame.columns):
            variants = set(
                zip(
                    frame["antithetic"].astype(bool),
                    frame["control_variate"].astype(bool),
                    strict=True,
                )
            )
        named_variants = _column_values(frame, "variant") or _column_values(frame, "method")
        seeds = _column_values(frame, "seed")
        predicates = {
            "none_antithetic_control_variants_compared": len(variants) >= 3
            or len(named_variants) >= 3,
            "multiple_independent_seeds": len(seeds) >= 3,
            "sampling_error_measured": frame is not None
            and bool({"standard_error", "variance"} & set(frame.columns)),
            "manifest_identifies_n06": manifest.get("experiment_id") == "N06"
            and manifest.get("status") in {"MEASURED", "PASS", "COMPLETED"},
        }
        return _evaluation(
            predicates,
            "variance-reduction variants="
            f"{len(variants) or len(named_variants)}; seeds={len(seeds)}",
            evidence.snapshots(path, manifest_path),
        )
    if experiment_id in {"N04", "N05"}:
        schemes = _column_values(mc, "scheme")
        steps = set(mc["steps"].dropna().astype(int)) if mc is not None and "steps" in mc else set()
        mc_seeds = (
            set(mc["seed"].dropna().astype(int)) if mc is not None and "seed" in mc else set()
        )
        predicates = {
            "multiple_time_steps": len(steps) >= 3,
            "multiple_seeds": len(mc_seeds) >= 3,
            "bias_and_standard_error_separate": mc is not None
            and {"bias", "standard_error", "reference_in_95pct_ci"}.issubset(mc.columns),
        }
        if experiment_id == "N05":
            predicates["euler_qe_and_qem_measured"] = {
                "full_truncation",
                "qe",
                "qe_m",
            }.issubset(schemes)
        return _evaluation(
            predicates,
            f"schemes={sorted(schemes)}, time levels={len(steps)}, seeds={len(mc_seeds)}",
            evidence.snapshots(mc_path, remediation_path),
        )
    if experiment_id in {"N07", "N08"}:
        pde_path = "results/aggregated/pde_adi_convergence.csv"
        pde = evidence.frame(pde_path)
        schemes = _column_values(pde, "scheme")
        studies = _column_values(pde, "study")
        predicates = {
            "mcs_and_hv_attempted": {
                "modified_craig_sneyd",
                "hundsdorfer_verwer",
            }.issubset(schemes),
            "required_convergence_dimensions_attempted": {
                "spot_grid",
                "variance_grid",
                "time_grid",
                "domain_boundary",
            }.issubset(studies),
            "frozen_pde_qualification_passed": remediation.get("pde_status") == "PASS",
        }
        return _evaluation(
            predicates,
            "PDE was attempted with schemes="
            f"{sorted(schemes)}; frozen status={remediation.get('pde_status')}",
            evidence.snapshots(pde_path, remediation_path),
        )
    if experiment_id == "N09":
        path = "results/aggregated/T12_CPU_GPU_PERFORMANCE.csv"
        frame = evidence.frame(path)
        selected = _column_values(frame, "selected_backend")
        predicates = {
            "measured_runtime_rows_present": frame is not None and not frame.empty,
            "cpu_and_gpu_backends_represented": any("cuda" in value.lower() for value in selected)
            and any("cpu" in value.lower() for value in selected),
            "accuracy_fields_present": frame is not None
            and {"absolute_error", "relative_error", "accuracy_pass", "speedup"}.issubset(
                frame.columns
            ),
        }
        return _evaluation(
            predicates, f"selected backends={sorted(selected)}", evidence.snapshots(path)
        )
    if experiment_id == "N10":
        path = "results/aggregated/precision_sensitivity.csv"
        manifest_path = "results/aggregated/precision_sensitivity.json"
        frame = evidence.frame(path)
        manifest = evidence.object(manifest_path) or {}
        predicates = {
            "fp32_and_fp64_measured": {"float32", "float64"}.issubset(
                _column_values(frame, "precision")
            ),
            "multiple_stress_cases": frame is not None
            and frame.get("case", pd.Series(dtype=str)).nunique() >= 3,
            "error_metrics_present": frame is not None
            and {"maximum_absolute_error", "maximum_relative_error"}.issubset(frame.columns),
            "manifest_identifies_n10": manifest.get("experiment_id") == "N10"
            and manifest.get("status") == "MEASURED",
        }
        return _evaluation(
            predicates,
            "FP32/FP64 sensitivity artifact inspected",
            evidence.snapshots(path, manifest_path),
        )
    return _evaluation({}, f"No evidence contract for {experiment_id}", ())


def _calibration_evaluation(entry: dict[str, Any], evidence: _Evidence) -> ExperimentEvaluation:
    experiment_id = str(entry["id"])
    t04_path = "reports/tables/T04_CALIBRATION_OBJECTIVE_COMPARISON.csv"
    t05_path = "reports/tables/T05_OPTIMIZER_IDENTIFIABILITY.csv"
    t06_path = "reports/tables/T06_PRICING_MODEL_COMPARISON.csv"
    t04 = evidence.frame(t04_path)
    t05 = evidence.frame(t05_path)
    t06 = evidence.frame(t06_path)
    path = t05_path
    predicates: dict[str, bool]
    reason: str
    if experiment_id == "C01":
        objectives = _column_values(t04, "objective")
        predicates = {
            "all_objectives_measured": objectives == {"C00", "C01", "C02", "C03", "C04"},
            "dev_selection_and_validation_confirmation": {"DEV", "VALIDATION"}.issubset(
                _column_values(t04, "split")
            ),
        }
        path, reason = t04_path, f"objectives={sorted(objectives)}"
    elif experiment_id == "C02":
        optimizers = _column_values(t05, "optimizer_id")
        predicates = {"all_optimizers_measured": optimizers == {"O00", "O01", "O02", "O03", "O04"}}
        reason = f"optimizers={sorted(optimizers)}"
    elif experiment_id == "C03":
        path = "results/aggregated/identifiability_profiles.parquet"
        frame = evidence.frame(path)
        params = set(
            str(value).removeprefix("MULTISTART_")
            for value in _column_values(frame, "diagnostic_id")
            if str(value).startswith("MULTISTART_")
        )
        predicates = {
            "all_parameters_multistarted": params == {"v0", "kappa", "theta", "sigma_v", "rho"},
            "solution_dispersion_measured": frame is not None
            and bool(frame.get("dispersion", pd.Series(dtype=float)).notna().any()),
        }
        reason = f"multistart parameters={sorted(params)}"
    elif experiment_id == "C04":
        path = "results/aggregated/feller_study.csv"
        frame = evidence.frame(path)
        variants = _column_values(frame, "feller_treatment")
        predicates = {
            "unconstrained_soft_hard_measured": variants == {"unconstrained", "soft", "hard"}
        }
        reason = f"Feller variants={sorted(variants)}"
    elif experiment_id == "C05":
        path = "results/aggregated/bootstrap_calibration.parquet"
        frame = evidence.frame(path)
        predicates = {
            "multiple_bootstrap_draws": frame is not None and len(frame) >= 5,
            "parameter_distribution_measured": frame is not None
            and {"v0", "kappa", "theta", "sigma_v", "rho", "draw"}.issubset(frame.columns),
        }
        reason = f"bootstrap rows={0 if frame is None else len(frame)}"
    elif experiment_id == "C06":
        path = "results/aggregated/identifiability_profiles.parquet"
        frame = evidence.frame(path)
        studies = _column_values(frame, "study")
        predicates = {
            "multistart_and_profile_measured": {"identifiability", "profile_objective"}.issubset(
                studies
            ),
            "profile_grid_values_present": frame is not None
            and bool(frame.get("fixed_value", pd.Series(dtype=float)).notna().any()),
        }
        reason = f"identifiability studies={sorted(studies)}"
    elif experiment_id == "C07":
        path = "results/aggregated/parameter_stability.parquet"
        frame = evidence.frame(path)
        predicates = {
            "rolling_dates_measured": frame is not None
            and frame.get("quote_date", pd.Series(dtype=str)).nunique() >= 3,
            "all_temporal_splits_present": {"DEV", "VALIDATION", "LOCKED_TEST"}.issubset(
                _column_values(frame, "split")
            ),
            "jump_score_present": frame is not None
            and bool(frame.get("jump_score", pd.Series(dtype=float)).notna().any()),
        }
        reason = f"stability rows={0 if frame is None else len(frame)}"
    elif experiment_id == "C08":
        experiments = _column_values(t06, "experiment")
        predicates = {
            "wing_and_maturity_holdouts_measured": {"wing_holdout", "maturity_holdout"}.issubset(
                experiments
            )
        }
        path, reason = t06_path, f"model experiments={sorted(experiments)}"
    elif experiment_id == "C09":
        objectives = _column_values(t04, "objective")
        predicates = {
            "raw_spread_and_robust_weightings_measured": {"C00", "C03", "C04"}.issubset(objectives)
        }
        path, reason = t04_path, f"weighting-related objectives={sorted(objectives)}"
    else:
        return _evaluation({}, f"No evidence contract for {experiment_id}", ())
    return _evaluation(predicates, reason, evidence.snapshots(path))


def _model_evaluation(entry: dict[str, Any], evidence: _Evidence) -> ExperimentEvaluation:
    experiment_id = str(entry["id"])
    comparison_path = "reports/tables/T06_PRICING_MODEL_COMPARISON.csv"
    result_dir, _, _ = _confirmatory_paths(evidence)
    truth_path = f"{result_dir}/truth_qualification.json"
    remediation_path = "results/aggregated/numerical_remediation_qualification.json"
    comparison = evidence.frame(comparison_path)
    model_ids = _column_values(comparison, "model_id")
    truth_rows = (
        json.loads((evidence.root / truth_path).read_text(encoding="utf-8"))
        if (evidence.root / truth_path).is_file()
        else []
    )
    qualified = {
        str(row.get("engine"))
        for row in truth_rows
        if isinstance(row, dict) and row.get("status") == "PASS"
    }
    required_by_id = {
        "M01": {"M00_BS_FLAT", "M01_MARKET_IV_REFERENCE"},
        "M02": {"M02_SVI", "M03_SSVI"},
        "M03": {"M10_HESTON"},
        "M04": {"M20_LOCALVOL_MC"},
        "M05": {"M21_BATES_SENSITIVITY"},
    }
    required = required_by_id.get(experiment_id, set())
    predicates = {
        "real_cross_sectional_models_measured": bool(required) and required.issubset(model_ids),
        "dev_validation_locked_splits_measured": {"DEV", "VALIDATION", "LOCKED_TEST"}.issubset(
            _column_values(comparison, "split")
        ),
    }
    paths = [comparison_path]
    if experiment_id == "M03":
        numerical = evidence.object(remediation_path) or {}
        predicates["mc_independent_validator_qualified"] = numerical.get("mc_status") == "PASS"
        predicates["pde_independent_validator_qualified"] = numerical.get("pde_status") == "PASS"
        paths.append(remediation_path)
    elif experiment_id == "M04":
        predicates["localvol_truth_engine_qualified"] = "LOCALVOL_SSVI_MARGINAL" in qualified
        paths.append(truth_path)
    elif experiment_id == "M05":
        predicates["bates_truth_engine_qualified"] = "BATES_CF" in qualified
        paths.append(truth_path)
    return _evaluation(
        predicates,
        f"required model rows={sorted(required)}; measured={sorted(required & model_ids)}",
        evidence.snapshots(*paths),
    )


def _confirmatory_paths(evidence: _Evidence) -> tuple[str, str, str]:
    """Locate the newest confirmatory result without mutating frozen inputs."""

    if (evidence.root / "results/confirmatory/synthetic_v5/confirmatory_metrics.json").is_file():
        return (
            "results/confirmatory/synthetic_v5",
            "artifacts/checkpoints/synthetic_confirmatory_v6.freeze.json",
            "src/derivguard/synthetic/confirmatory_v6.py",
        )
    if (evidence.root / "results/confirmatory/synthetic_v4/confirmatory_metrics.json").is_file():
        return (
            "results/confirmatory/synthetic_v4",
            "artifacts/checkpoints/synthetic_confirmatory_v5.freeze.json",
            "src/derivguard/synthetic/confirmatory_v5.py",
        )
    return (
        "results/confirmatory/synthetic_v3",
        "artifacts/checkpoints/synthetic_confirmatory_v4.freeze.json",
        "src/derivguard/synthetic/confirmatory_v4.py",
    )


def _confirmatory_coherent(
    evidence: _Evidence,
) -> tuple[dict[str, bool], tuple[ArtifactSnapshot, ...]]:
    result_dir, freeze_path, source_path = _confirmatory_paths(evidence)
    metrics_path = f"{result_dir}/confirmatory_metrics.json"
    threshold_path = f"{result_dir}/dve_thresholds.freeze.json"
    calibration_path = f"{result_dir}/calibration_diagnostics.parquet"
    qualification_path = f"{result_dir}/calibration_qualification.json"
    provenance_path = f"{result_dir}/case_provenance.parquet"
    freeze = evidence.object(freeze_path) or {}
    metrics = evidence.object(metrics_path) or {}
    threshold = evidence.object(threshold_path) or {}
    qualification = evidence.object(qualification_path) or {}
    calibration = evidence.frame(calibration_path)
    provenance = evidence.frame(provenance_path)
    freeze_without_hash = dict(freeze)
    freeze_without_hash.pop("freeze_hash", None)
    selected = (
        calibration.loc[calibration["start_index"].astype(int) == -1]
        if calibration is not None and "start_index" in calibration
        else calibration
        if calibration is not None
        else pd.DataFrame()
    )
    primary = (
        selected.loc[selected["draw"].astype(int) == -1] if "draw" in selected else pd.DataFrame()
    )
    bootstrap = (
        selected.loc[selected["draw"].astype(int) >= 0] if "draw" in selected else pd.DataFrame()
    )
    primary_case_ids = set(primary["case_id"].astype(str)) if "case_id" in primary else set()
    provenance_case_ids = (
        set(provenance["case_id"].astype(str))
        if provenance is not None and "case_id" in provenance
        else set()
    )
    bootstrap_draws = int((freeze.get("calibration_design") or {}).get("bootstrap_draws", 0))
    optimizer_column = (
        "optimizer_id" if calibration is not None and "optimizer_id" in calibration else "optimizer"
    )
    optimizer_values = _column_values(calibration, optimizer_column)
    release_id = str(
        freeze.get(
            "release_id",
            freeze.get("developer_release_id", freeze.get("developer_model_release_id", "")),
        )
    )
    metrics_release = str(
        metrics.get(
            "release_id",
            metrics.get("developer_release_id", metrics.get("developer_model_release_id", "")),
        )
    )
    release_path = next(
        (
            path
            for path in (evidence.root / "artifacts/model_releases").glob("model_release_*.json")
            if (evidence.object(str(path.relative_to(evidence.root))) or {}).get("release_id")
            == release_id
        ),
        None,
    )
    release_relative = "" if release_path is None else str(release_path.relative_to(evidence.root))
    release_hash = None if release_path is None else _sha256_file(release_path)
    source = evidence.root / source_path
    hypothesis = evidence.root / str(freeze.get("hypothesis_registry_path", ""))
    experiment = evidence.root / str(freeze.get("experiment_registry_path", ""))
    source_and_registry_hashes = bool(
        source.is_file()
        and hypothesis.is_file()
        and experiment.is_file()
        and freeze.get("module_hash") == _sha256_file(source)
        and freeze.get("hypothesis_registry_hash") == _sha256_file(hypothesis)
        and freeze.get("experiment_registry_hash") == _sha256_file(experiment)
    )
    release_hash_linked = bool(
        release_hash
        and freeze.get(
            "release_hash", freeze.get("developer_release_hash", freeze.get("release_sha256"))
        )
        == release_hash
        and metrics.get(
            "release_hash", metrics.get("developer_release_hash", metrics.get("release_sha256"))
        )
        == release_hash
    )
    predicates = {
        "primary_confirmatory_version_is_v6": result_dir.endswith("synthetic_v5"),
        "locked_status_complete": metrics.get("status") == "CONFIRMATORY_LOCKED_COMPLETE",
        "freeze_hash_matches": bool(freeze.get("freeze_hash"))
        and metrics.get("freeze_hash") == freeze.get("freeze_hash")
        and freeze.get("freeze_hash") == _sha256_json(freeze_without_hash),
        "observable_c04_calibrations_present": calibration is not None
        and not calibration.empty
        and _column_values(calibration, "objective") == {"C04"},
        "observable_provenance_present": provenance is not None
        and not provenance.empty
        and _column_values(provenance, "calibration_inputs") == {"observable_quotes_only"},
        "all_primary_calibrations_converged": not primary.empty
        and bool(primary["success"].eq(True).all())
        and bool(primary["qualified"].eq(True).all())
        and primary_case_ids == provenance_case_ids,
        "all_bootstrap_calibrations_converged": not bootstrap.empty
        and bool(bootstrap["success"].eq(True).all())
        and bool(bootstrap["qualified"].eq(True).all())
        and len(bootstrap) == len(provenance_case_ids) * bootstrap_draws,
        "calibration_qualification_passed": qualification.get("status") == "PASS"
        and qualification.get("all_selected_fits_qualified") is True,
        "developer_objective_is_c04": _column_values(calibration, "objective") == {"C04"},
        "developer_optimizer_explicitly_o01": optimizer_values == {"O01"},
        "release_id_linked": bool(release_id) and release_id == metrics_release,
        "release_hash_linked": release_hash_linked,
        "source_and_frozen_registries_current": source_and_registry_hashes,
        "threshold_freeze_linked_and_precedes_locked": bool(freeze.get("freeze_hash"))
        and threshold.get("design_freeze_hash") == freeze.get("freeze_hash")
        and (evidence.root / threshold_path).is_file()
        and (evidence.root / metrics_path).is_file()
        and (evidence.root / threshold_path).stat().st_mtime_ns
        < (evidence.root / metrics_path).stat().st_mtime_ns,
    }
    return predicates, evidence.snapshots(
        freeze_path,
        metrics_path,
        threshold_path,
        calibration_path,
        qualification_path,
        provenance_path,
        source_path,
        release_relative,
    )


def _synthetic_evaluation(entry: dict[str, Any], evidence: _Evidence) -> ExperimentEvaluation:
    result_dir, _, _ = _confirmatory_paths(evidence)
    coverage_path = f"{result_dir}/fault_coverage.csv"
    truth_path = f"{result_dir}/truth_qualification.json"
    coverage = evidence.frame(coverage_path)
    faults = set(map(str, entry.get("faults", [])))
    observed = _column_values(coverage, "fault_id")
    coherent, snapshots = _confirmatory_coherent(evidence)
    predicates = dict(coherent)
    predicates.update(
        {
            "declared_faults_executed": bool(faults) and faults.issubset(observed),
            "faults_have_real_implementation_labels": coverage is not None
            and not coverage.empty
            and bool(
                coverage.loc[coverage["fault_id"].astype(str).isin(faults), "implementation_label"]
                .notna()
                .all()
            ),
        }
    )
    return _evaluation(
        predicates,
        f"declared faults={sorted(faults)}; observed={sorted(faults & observed)}",
        snapshots + evidence.snapshots(coverage_path, truth_path),
    )


def _real_outcome_evaluation(entry: dict[str, Any], evidence: _Evidence) -> ExperimentEvaluation:
    availability_path = "results/RESULT_AVAILABILITY.json"
    availability = evidence.object(availability_path) or {}
    entries = availability.get("entries", [])
    blockers = [
        str(item.get("blocker"))
        for item in entries
        if isinstance(item, dict)
        and item.get("status") == "UNAVAILABLE"
        and item.get("phase_id") in {"30", "31"}
    ]
    t10_exists = (evidence.root / "reports/tables/T10_REAL_OOS_OUTCOMES.csv").is_file()
    predicates = {
        "attempt_recorded": bool(blockers),
        "defensible_temporal_result_present": t10_exists,
    }
    reason = " ".join(blockers) or "No temporal-outcome execution or blocker artifact found."
    unavailable = bool(blockers) and not t10_exists
    return _evaluation(
        predicates,
        f"{entry['id']}: {reason}",
        evidence.snapshots(availability_path),
        unavailable=unavailable,
    )


def _materiality_evaluation(entry: dict[str, Any], evidence: _Evidence) -> ExperimentEvaluation:
    path = "reports/tables/T11_PORTFOLIO_MATERIALITY.csv"
    frame = evidence.frame(path)
    portfolio_id = str(entry["id"])
    rows = (
        frame.loc[frame["portfolio_id"].astype(str) == portfolio_id]
        if frame is not None and "portfolio_id" in frame
        else pd.DataFrame()
    )
    metric_columns = {"value", "delta", "gamma", "vega"}
    materiality_columns = {
        "valuation_materiality",
        "delta_materiality",
        "gamma_materiality",
        "vega_materiality",
    }
    predicates = {
        "portfolio_rows_measured": not rows.empty,
        "multiple_models_measured": len(_column_values(rows, "model_id")) >= 3,
        "value_and_greeks_measured": metric_columns.issubset(rows.columns)
        and bool(rows[list(metric_columns)].notna().all().all()),
        "separate_materiality_measures": materiality_columns.issubset(rows.columns)
        and bool(rows[list(materiality_columns)].notna().all().all()),
    }
    return _evaluation(
        predicates, f"{portfolio_id} measured rows={len(rows)}", evidence.snapshots(path)
    )


def _ablation_evaluation(entry: dict[str, Any], evidence: _Evidence) -> ExperimentEvaluation:
    result_dir, _, _ = _confirmatory_paths(evidence)
    path = f"{result_dir}/T09_DVE_ABLATION.csv"
    frame = evidence.frame(path)
    ablation_id = str(entry["id"])
    rows = (
        frame.loc[frame["ablation_id"].astype(str) == ablation_id]
        if frame is not None and "ablation_id" in frame
        else pd.DataFrame()
    )
    predicates = {
        "confirmatory_ablation_row_present": len(rows) == 1,
        "required_changes_measured": {
            "recall_change",
            "auprc_change",
            "false_alarm_change",
            "miss_rate_change",
        }.issubset(rows.columns)
        and not rows.empty
        and bool(
            rows[["recall_change", "auprc_change", "false_alarm_change", "miss_rate_change"]]
            .notna()
            .all()
            .all()
        ),
    }
    coherent, snapshots = _confirmatory_coherent(evidence)
    predicates.update(coherent)
    return _evaluation(
        predicates, f"confirmatory ablation={ablation_id}", snapshots + evidence.snapshots(path)
    )


def _developer_ablation_evaluation(
    entry: dict[str, Any], evidence: _Evidence
) -> ExperimentEvaluation:
    experiment_id = str(entry["id"])
    summary_path = "results/aggregated/developer_method_ablation_summary.csv"
    summary = evidence.frame(summary_path)
    summary_rows = (
        summary.loc[summary["ablation_id"].astype(str) == experiment_id]
        if summary is not None and "ablation_id" in summary
        else pd.DataFrame()
    )
    if not summary_rows.empty:
        numeric = summary_rows.select_dtypes(include="number")
        predicates = {
            "exactly_one_measured_ablation_row": len(summary_rows) == 1,
            "quantitative_result_present": not numeric.empty and bool(numeric.notna().any().any()),
            "not_status_only": len(summary_rows.columns) > 3,
        }
        if "status" in summary_rows:
            statuses = _column_values(summary_rows, "status")
            predicates["execution_status_measured"] = bool(statuses) and statuses.issubset(
                {"MEASURED", "COMPLETED", "PASS"}
            )
        return _evaluation(
            predicates,
            f"measured developer-method ablation {experiment_id}",
            evidence.snapshots(summary_path),
        )
    ablation_path = "results/aggregated/developer_method_ablations.csv"
    ablations = evidence.frame(ablation_path)
    objective_variants = (
        _column_values(ablations.loc[ablations["ablation_family"] == "objective"], "variant")
        if ablations is not None and "ablation_family" in ablations
        else set()
    )
    optimizer_variants = (
        _column_values(ablations.loc[ablations["ablation_family"] == "optimizer"], "variant")
        if ablations is not None and "ablation_family" in ablations
        else set()
    )
    if experiment_id == "DA01":
        predicates = {
            "local_and_global_optimizer_variants_measured": {
                "O00",
                "O01",
                "O02",
                "O03",
                "O04",
            }.issubset(optimizer_variants)
        }
        path = ablation_path
    elif experiment_id == "DA02":
        predicates = {
            "robust_and_nonrobust_objectives_measured": {"C00", "C04"}.issubset(objective_variants)
        }
        path = ablation_path
    elif experiment_id == "DA03":
        predicates = {
            "spread_weighted_and_unweighted_objectives_measured": {"C00", "C03", "C04"}.issubset(
                objective_variants
            )
        }
        path = ablation_path
    elif experiment_id == "DA04":
        predicates = {"with_and_without_bootstrap_outcomes_measured": False}
        path = "results/aggregated/bootstrap_calibration.parquet"
    elif experiment_id == "DA05":
        path = "results/aggregated/feller_study.csv"
        frame = evidence.frame(path)
        predicates = {
            "all_feller_variants_measured": _column_values(frame, "feller_treatment")
            == {"unconstrained", "soft", "hard"}
        }
    elif experiment_id == "DA06":
        path = "results/raw/forward_estimator_comparison.parquet"
        frame = evidence.frame(path)
        predicates = {
            "all_forward_variants_measured": {"F0", "F1", "F2", "F3"}.issubset(
                _column_values(frame, "method")
            )
        }
    else:
        return _evaluation({}, f"No evidence contract for {experiment_id}", ())
    return _evaluation(
        predicates, f"developer-method ablation {experiment_id} inspected", evidence.snapshots(path)
    )


def _evaluate_family(
    family: str, entry: dict[str, Any], evidence: _Evidence
) -> ExperimentEvaluation:
    evaluators = {
        "DATA": _data_evaluation,
        "NUMERICAL": _numerical_evaluation,
        "CALIBRATION": _calibration_evaluation,
        "MODEL_COMPARISON": _model_evaluation,
        "SYNTHETIC_VALIDATION": _synthetic_evaluation,
        "REAL_OUTCOMES": _real_outcome_evaluation,
        "MATERIALITY": _materiality_evaluation,
        "DVE_ABLATION": _ablation_evaluation,
        "DEVELOPER_METHOD_ABLATION": _developer_ablation_evaluation,
    }
    evaluator = evaluators.get(family)
    if evaluator is None:
        return _evaluation({}, f"No family evidence contract for {family}", ())
    return evaluator(entry, evidence)


def _attach(entry: dict[str, Any], evaluation: ExperimentEvaluation) -> None:
    entry["status"] = evaluation.status
    entry["feasibility"] = evaluation.feasibility
    entry["status_reason"] = evaluation.reason
    entry["evidence"] = [asdict(item) for item in evaluation.evidence]
    entry["evidence_predicates"] = evaluation.predicates


def _reconcile_methods(registry: dict[str, Any], evidence: _Evidence) -> list[dict[str, Any]]:
    changes: list[dict[str, Any]] = []
    methods = registry.get("calibration_methods", {})
    contracts = (
        (
            "objectives",
            "reports/tables/T04_CALIBRATION_OBJECTIVE_COMPARISON.csv",
            "objective",
            "formula_status",
            "IMPLEMENTED_AND_MEASURED",
        ),
        (
            "optimizers",
            "reports/tables/T05_OPTIMIZER_IDENTIFIABILITY.csv",
            "optimizer_id",
            "status",
            "COMPLETED",
        ),
        (
            "forward_estimators",
            "results/raw/forward_estimator_comparison.parquet",
            "method",
            "status",
            "COMPLETED",
        ),
    )
    for section, path, column, status_field, passed_status in contracts:
        frame = evidence.frame(path)
        values = _column_values(frame, column)
        snapshot = evidence.snapshot(path)
        for entry in methods.get(section, []):
            if not isinstance(entry, dict):
                continue
            old = str(entry.get(status_field, ""))
            measured = str(entry.get("id")) in values
            entry[status_field] = passed_status if measured else "PARTIAL"
            entry["evidence"] = [] if snapshot is None else [asdict(snapshot)]
            entry["evidence_predicates"] = {"measured_rows_present": measured}
            changes.append(
                {
                    "family": f"CALIBRATION_METHODS/{section}",
                    "experiment_id": str(entry.get("id")),
                    "previous_status": old,
                    "status": entry[status_field],
                }
            )
    return changes


def _reconcile_baselines(registry: dict[str, Any], evidence: _Evidence) -> list[dict[str, Any]]:
    result_dir, _, _ = _confirmatory_paths(evidence)
    path = f"{result_dir}/T08_DVE_BASELINES.csv"
    frame = evidence.frame(path)
    values = _column_values(frame, "baseline")
    coherent, snapshots = _confirmatory_coherent(evidence)
    changes: list[dict[str, Any]] = []
    for entry in registry.get("validation_baselines", []):
        if not isinstance(entry, dict):
            continue
        baseline_id = str(entry.get("id"))
        predicates = dict(coherent)
        predicates["measured_baseline_row_present"] = baseline_id in values
        evaluation = _evaluation(
            predicates,
            f"active confirmatory baseline={baseline_id}",
            snapshots + evidence.snapshots(path),
        )
        old = str(entry.get("status", ""))
        _attach(entry, evaluation)
        changes.append(
            {
                "family": "VALIDATION_BASELINES",
                "experiment_id": baseline_id,
                "previous_status": old,
                "status": evaluation.status,
            }
        )
    return changes


def _reconcile_split_policy(registry: dict[str, Any], evidence: _Evidence) -> None:
    split_policy = registry.get("split_policy", {})
    real = split_policy.get("real_data", {})
    methodology_path = "artifacts/calibration/empirical_methodology.json"
    methodology = evidence.object(methodology_path) or {}
    temporal_split = methodology.get("temporal_split", {})
    if isinstance(real, dict) and isinstance(temporal_split, dict):
        counts = {
            str(partition): len(values)
            for partition, values in temporal_split.items()
            if isinstance(values, list)
        }
        real["boundary_status"] = (
            "FROZEN_AND_EXECUTED"
            if {"DEV", "VALIDATION", "LOCKED_TEST"}.issubset(counts)
            else "PARTIAL"
        )
        real["executed_surface_counts"] = counts
        snapshot = evidence.snapshot(methodology_path)
        real["evidence"] = [] if snapshot is None else [asdict(snapshot)]

    synthetic = split_policy.get("synthetic", {})
    result_dir, _, _ = _confirmatory_paths(evidence)
    provenance_path = f"{result_dir}/case_provenance.parquet"
    provenance = evidence.frame(provenance_path)
    if isinstance(synthetic, dict) and provenance is not None and "partition" in provenance:
        synthetic_counts = (
            provenance.groupby("partition")["case_id"].nunique().astype(int).to_dict()
        )
        synthetic["seed_registry_status"] = "FROZEN_BEFORE_LOCKED_EXECUTION"
        synthetic["executed_case_counts"] = {
            str(key): int(value) for key, value in synthetic_counts.items()
        }
        synthetic["confirmatory_design"] = (
            "v6_release_aligned_prequalified_calibration"
            if result_dir.endswith("synthetic_v5")
            else (
                "v5_release_aligned_calibration_unqualified"
                if result_dir.endswith("synthetic_v4")
                else "v4_observable_quote_calibration_unqualified"
            )
        )
        synthetic.pop("executed_seed", None)
        snapshot = evidence.snapshot(provenance_path)
        synthetic["evidence"] = [] if snapshot is None else [asdict(snapshot)]


def reconcile_experiment_registry(root: Path) -> dict[str, Any]:
    """Update the mutable main registry from measured evidence and write a manifest.

    The separately frozen confirmatory hypothesis and experiment registries are
    deliberately outside this function's write set.
    """

    registry_path = root / "experiments/registry.yaml"
    manifest_path = root / "artifacts/experiment_reconciliation.json"
    loaded = yaml.safe_load(registry_path.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        raise ValueError("experiments/registry.yaml must contain a mapping")
    registry: dict[str, Any] = loaded
    before_hash = _sha256_file(registry_path)
    evidence = _Evidence(root)
    changes: list[dict[str, Any]] = []
    families = registry.get("families", {})
    if not isinstance(families, dict):
        raise ValueError("registry families must be a mapping")
    for family, entries in families.items():
        if not isinstance(entries, list):
            continue
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            old = str(entry.get("status", ""))
            evaluation = _evaluate_family(str(family), entry, evidence)
            _attach(entry, evaluation)
            changes.append(
                {
                    "family": str(family),
                    "experiment_id": str(entry.get("id")),
                    "previous_status": old,
                    "status": evaluation.status,
                }
            )
    changes.extend(_reconcile_methods(registry, evidence))
    changes.extend(_reconcile_baselines(registry, evidence))
    _reconcile_split_policy(registry, evidence)
    registry["registry_status"] = "ARTIFACT_DERIVED_RECONCILED"
    registry["scientific_integrity_note"] = (
        "Statuses are derived from hashed, semantically inspected artifacts. COMPLETED requires "
        "non-placeholder measured rows and satisfied predicates; PARTIAL and UNAVAILABLE preserve "
        "attempted limitations. Frozen confirmatory registries are not mutated by reconciliation."
    )
    registry["reconciliation"] = {
        "schema_version": "1.0",
        "manifest": "artifacts/experiment_reconciliation.json",
        "method": "artifact_content_predicates",
    }
    rendered = yaml.safe_dump(registry, sort_keys=False, width=100)
    registry_path.write_text(rendered, encoding="utf-8")
    after_hash = _sha256_file(registry_path)

    summary: dict[str, int] = {}
    family_summary: dict[str, dict[str, int]] = {}
    for change in changes:
        status = str(change["status"])
        family = str(change["family"])
        summary[status] = summary.get(status, 0) + 1
        family_counts = family_summary.setdefault(family, {})
        family_counts[status] = family_counts.get(status, 0) + 1
    manifest = {
        "schema_version": "1.0",
        "status": "RECONCILED_FROM_ARTIFACT_CONTENT",
        "generated_at": datetime.now(UTC).isoformat(),
        "registry_path": "experiments/registry.yaml",
        "registry_sha256_before": before_hash,
        "registry_sha256_after": after_hash,
        "frozen_registries_preserved": [
            "governance/confirmatory_v4_hypothesis_registry.yaml",
            "experiments/confirmatory_v4_registry.yaml",
            "governance/confirmatory_v5_hypothesis_registry.yaml",
            "experiments/confirmatory_v5_registry.yaml",
            "governance/confirmatory_v6_hypothesis_registry.yaml",
            "experiments/confirmatory_v6_registry.yaml",
        ],
        "summary": summary,
        "family_summary": family_summary,
        "experiments": changes,
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest
