"""Evidence-derived phase and acceptance gating.

Expected deliverables are declared here, but status is never assigned from a
phase-ID allowlist.  Each status follows from artifact inspection, semantic
predicates, test evidence, or a separately recorded availability blocker.
"""

from __future__ import annotations

import json
import uuid
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from hashlib import sha256
from pathlib import Path
from typing import Any, Literal

import pandas as pd

ArtifactKind = Literal["file", "json", "csv", "parquet", "markdown"]


@dataclass(frozen=True)
class ArtifactRequirement:
    path: str
    kind: ArtifactKind = "file"
    minimum_rows: int = 0
    required_columns: tuple[str, ...] = ()
    required_json_keys: tuple[str, ...] = ()


@dataclass(frozen=True)
class PhaseRequirement:
    phase_id: str
    name: str
    artifacts: tuple[ArtifactRequirement, ...]
    semantic_check: str | None = None


@dataclass(frozen=True)
class ArtifactEvidence:
    path: str
    passed: bool
    detail: str


@dataclass(frozen=True)
class PhaseEvidence:
    phase_id: str
    name: str
    status: str
    test_status: str
    artifacts: tuple[ArtifactEvidence, ...]
    blocker: str | None
    evaluated_at: str


def _a(
    path: str,
    kind: ArtifactKind = "file",
    rows: int = 0,
    columns: tuple[str, ...] = (),
    keys: tuple[str, ...] = (),
) -> ArtifactRequirement:
    return ArtifactRequirement(path, kind, rows, columns, keys)


PHASE_REQUIREMENTS: tuple[PhaseRequirement, ...] = (
    PhaseRequirement("00", "workspace reconciliation", (_a("AGENTS.md", "markdown"),)),
    PhaseRequirement(
        "01",
        "desktop study",
        (
            _a("desktop_study/DESKTOP_STUDY.md", "markdown"),
            _a("desktop_study/literature.yaml"),
            _a("desktop_study/regulatory_mapping.yaml"),
        ),
    ),
    PhaseRequirement(
        "02",
        "canonical Python environment",
        (_a("artifacts/system/environment.json", "json", keys=("python", "torch", "numpy")),),
    ),
    PhaseRequirement(
        "03",
        "CUDA/GPU validation",
        (
            _a(
                "artifacts/system/environment.json",
                "json",
                keys=("torch_cuda_available", "gpu_name", "gpu_total_memory_bytes"),
            ),
        ),
        "cuda_available",
    ),
    PhaseRequirement(
        "04",
        "GPU and CPU numerical smoke benchmark",
        (_a("results/aggregated/numerical_benchmarks.json", "json", keys=("benchmarks",)),),
        "cpu_gpu_agreement",
    ),
    PhaseRequirement(
        "05",
        "data acquisition",
        (_a("artifacts/data/acquisition_inspection.json", "json"),),
        "public_data_rows",
    ),
    PhaseRequirement(
        "06",
        "data lineage and raw audit",
        (
            _a("artifacts/data/processing_summary.json", "json"),
            _a(
                "data/raw/historicaldata_spx_sample/options_sample_2022H2.zip.manifest.json", "json"
            ),
        ),
    ),
    PhaseRequirement(
        "07", "EDA and data-risk analysis", (_a("results/aggregated/data_eda.json", "json"),)
    ),
    PhaseRequirement(
        "08",
        "preprocessing and settlement handling",
        (_a("artifacts/data/processing_summary.json", "json"),),
    ),
    PhaseRequirement(
        "09",
        "forward/discount estimation",
        (
            _a(
                "results/raw/forward_estimator_comparison.parquet",
                "parquet",
                100,
                ("method", "forward", "discount_factor", "status"),
            ),
        ),
    ),
    PhaseRequirement(
        "10",
        "Black-Scholes foundation",
        (_a("artifacts/tests/final_test_summary.json", "json"),),
        "tests_pass",
    ),
    PhaseRequirement(
        "11", "IV inversion", (_a("artifacts/tests/final_test_summary.json", "json"),), "tests_pass"
    ),
    PhaseRequirement(
        "12", "SVI/SSVI", (_a("artifacts/tests/final_test_summary.json", "json"),), "tests_pass"
    ),
    PhaseRequirement(
        "13",
        "Heston developer CF",
        (_a("results/aggregated/numerical_benchmarks.json", "json"),),
        "cf_oracle",
    ),
    PhaseRequirement(
        "14",
        "independent Heston MC",
        (
            _a(
                "results/aggregated/mc_qe_qem_audit.csv",
                "csv",
                20,
                (
                    "scheme",
                    "steps",
                    "paths",
                    "seed",
                    "price",
                    "standard_error",
                    "bias",
                    "reference_in_95pct_ci",
                ),
            ),
            _a(
                "results/aggregated/numerical_remediation_qualification.json",
                "json",
                keys=("mc_status", "mc_qualification", "mc_result_sha256"),
            ),
        ),
        "mc_remediation",
    ),
    PhaseRequirement(
        "15",
        "independent Heston PDE",
        (_a("results/aggregated/pde_adi_convergence.csv", "csv", 6),),
        "pde_tolerance",
    ),
    PhaseRequirement(
        "16",
        "QuantLib oracle",
        (_a("results/aggregated/numerical_benchmarks.json", "json"),),
        "cf_oracle",
    ),
    PhaseRequirement(
        "17",
        "Local Vol and Bates challengers",
        (
            _a(
                "results/confirmatory/synthetic_v5/truth_qualification.json",
                "file",
            ),
            _a(
                "reports/tables/T06_PRICING_MODEL_COMPARISON.csv",
                "csv",
                2,
                ("model_id", "price_rmse"),
            ),
        ),
        "challengers_qualified",
    ),
    PhaseRequirement(
        "18",
        "calibration objective study",
        (_a("reports/tables/T04_CALIBRATION_OBJECTIVE_COMPARISON.csv", "csv", 5, ("objective",)),),
        "calibration_objectives_complete",
    ),
    PhaseRequirement(
        "19",
        "optimizer study",
        (
            _a(
                "reports/tables/T05_OPTIMIZER_IDENTIFIABILITY.csv",
                "csv",
                4,
                ("optimizer_id", "objective_value"),
            ),
        ),
        "optimizers_complete",
    ),
    PhaseRequirement(
        "20",
        "Feller study",
        (_a("results/aggregated/feller_study.csv", "csv", 3, ("feller_treatment",)),),
        "feller_variants_complete",
    ),
    PhaseRequirement(
        "21",
        "bootstrap and calibration uncertainty",
        # The frozen research profile predeclares eight bid/ask bootstrap
        # recalibrations.  Gate the executed design rather than imposing an
        # unrelated post-hoc row count.
        (_a("results/aggregated/bootstrap_calibration.parquet", "parquet", 8),),
        "empirical_research_complete",
    ),
    PhaseRequirement(
        "22",
        "identifiability",
        (_a("results/aggregated/identifiability_profiles.parquet", "parquet", 10),),
        "empirical_research_complete",
    ),
    PhaseRequirement(
        "23",
        "parameter stability",
        (_a("results/aggregated/parameter_stability.parquet", "parquet", 5),),
        "empirical_research_complete",
    ),
    PhaseRequirement(
        "24",
        "developer release history and corrected freeze",
        (
            _a(
                "artifacts/model_releases/model_release_v1.0.json",
                "json",
                keys=("release_id", "source_sha256", "selection_evidence_hash"),
            ),
            _a(
                "artifacts/model_releases/model_release_v1.1.json",
                "json",
                keys=(
                    "release_id",
                    "source_hashes",
                    "config_sha256",
                    "selection_evidence_hash",
                    "objective",
                    "optimizer",
                ),
            ),
        ),
        "release_hash_current",
    ),
    PhaseRequirement(
        "25",
        "synthetic market lab",
        (
            _a(
                "results/confirmatory/synthetic_v5/truth_qualification.json",
                "file",
            ),
        ),
        "truths_qualified",
    ),
    PhaseRequirement(
        "26",
        "fault-injection lab",
        (
            _a(
                "results/confirmatory/synthetic_v5/fault_coverage.csv",
                "csv",
                17,
                ("fault_id", "implementation_label", "truth_family"),
            ),
        ),
        "all_faults_covered",
    ),
    PhaseRequirement(
        "27",
        "DVE development using DEV only",
        (
            _a(
                "results/confirmatory/synthetic_v5/dve_thresholds.freeze.json",
                "json",
                keys=("status", "design_freeze_hash", "thresholds"),
            ),
        ),
        "threshold_freeze_matches_locked",
    ),
    PhaseRequirement(
        "28",
        "validation-period methodology freeze",
        (
            _a(
                "artifacts/checkpoints/synthetic_confirmatory_v6.freeze.json",
                "json",
                keys=(
                    "freeze_hash",
                    "module_hash",
                    "seed_hash",
                    "schedule_hash",
                    "calibration_design_hash",
                    "hypothesis_registry_hash",
                    "experiment_registry_hash",
                    "developer_release_hash",
                    "release_id",
                ),
            ),
        ),
        "design_freeze_matches_locked",
    ),
    PhaseRequirement(
        "29",
        "real-market experiments",
        (
            _a("reports/tables/T06_PRICING_MODEL_COMPARISON.csv", "csv", 5, ("model_id",)),
            _a(
                "results/aggregated/near_expiry_study.csv",
                "csv",
                1,
                ("experiment", "model_id", "observations"),
            ),
            _a(
                "results/aggregated/liquidity_filter_study.csv",
                "csv",
                1,
                ("experiment", "model_id", "observations"),
            ),
            _a(
                "results/aggregated/am_pm_robustness.csv",
                "csv",
                1,
                ("experiment", "model_id", "observations", "settlement_class"),
            ),
        ),
        "real_models_complete",
    ),
    PhaseRequirement(
        "30", "outcome analysis", (_a("reports/tables/T10_REAL_OOS_OUTCOMES.csv", "csv", 1),)
    ),
    PhaseRequirement(
        "31", "hedging proxy", (_a("results/aggregated/one_day_hedge_proxy.parquet", "parquet", 1),)
    ),
    PhaseRequirement(
        "32",
        "portfolio materiality",
        (
            _a(
                "reports/tables/T11_PORTFOLIO_MATERIALITY.csv",
                "csv",
                6,
                ("portfolio_id", "model_id", "value", "delta", "gamma", "vega"),
            ),
        ),
        "portfolio_set_complete",
    ),
    PhaseRequirement(
        "33",
        "synthetic locked test",
        (
            _a(
                "results/confirmatory/synthetic_v5/confirmatory_metrics.json",
                "json",
                keys=(
                    "status",
                    "baselines",
                    "freeze_hash",
                    "calibration_qualification",
                    "developer_release_id",
                    "developer_release_hash",
                ),
            ),
            _a(
                "results/confirmatory/synthetic_v5/calibration_diagnostics.parquet",
                "parquet",
                102,
                (
                    "case_id",
                    "partition",
                    "draw",
                    "backend",
                    "objective",
                    "success",
                    "objective_value",
                ),
            ),
            _a(
                "results/confirmatory/synthetic_v5/case_provenance.parquet",
                "parquet",
                102,
                ("case_id", "partition", "calibration_inputs", "calibration_seed"),
            ),
        ),
        "confirmatory_locked",
    ),
    PhaseRequirement(
        "34",
        "real locked test",
        (_a("results/aggregated/real_locked_test.parquet", "parquet", 1),),
        "empirical_research_complete",
    ),
    PhaseRequirement(
        "35",
        "DVE baseline comparisons",
        (
            _a(
                "results/confirmatory/synthetic_v5/T08_DVE_BASELINES.csv",
                "csv",
                7,
                ("baseline", "auprc", "recall_at_dev_5pct_fpr"),
            ),
        ),
        "confirmatory_locked",
    ),
    PhaseRequirement(
        "36",
        "DVE ablations",
        (
            _a(
                "results/confirmatory/synthetic_v5/T09_DVE_ABLATION.csv",
                "csv",
                11,
                ("ablation_id", "description", "recall_change"),
            ),
        ),
        "confirmatory_locked",
    ),
    PhaseRequirement(
        "37",
        "developer-method ablations",
        (
            _a(
                "results/aggregated/developer_method_ablation_summary.csv",
                "csv",
                6,
                ("ablation_id",),
            ),
        ),
        "developer_method_ablations_complete",
    ),
    PhaseRequirement(
        "38",
        "statistical inference",
        (
            _a("results/statistics/date_level_inference.csv", "csv", 1),
            _a("results/confirmatory/synthetic_v5/statistical_inference.json", "json"),
        ),
        "empirical_and_confirmatory_complete",
    ),
    PhaseRequirement(
        "39",
        "failure/case analysis",
        (
            _a("results/confirmatory/synthetic_v5/false_positive_cases.parquet", "parquet", 0),
            _a("results/confirmatory/synthetic_v5/false_negative_cases.parquet", "parquet", 0),
        ),
        "confirmatory_locked",
    ),
    PhaseRequirement(
        "40",
        "ongoing-monitoring simulation",
        (_a("results/aggregated/monitoring_simulation.csv", "csv", 1, ("monitoring_status",)),),
    ),
    PhaseRequirement(
        "41",
        "findings and remediation",
        (_a("governance/findings_register.yaml"), _a("governance/remediation_tracker.yaml")),
    ),
    PhaseRequirement(
        "42",
        "final compute benchmark",
        (
            _a(
                "results/aggregated/T12_CPU_GPU_PERFORMANCE.csv",
                "csv",
                4,
                (
                    "workload",
                    "cpu_backend",
                    "gpu_backend",
                    "speedup",
                    "absolute_error",
                    "selected_backend",
                ),
            ),
            _a(
                "results/aggregated/final_compute_benchmarks.csv",
                "csv",
                8,
                ("workload", "backend", "status", "selected_backend"),
            ),
            _a(
                "results/aggregated/final_compute_benchmarks.json",
                "json",
                keys=(
                    "profile",
                    "device",
                    "result_sha256",
                    "canonical_t12_sha256",
                    "all_rows_measured_or_explicitly_unavailable",
                ),
            ),
        ),
        "final_compute_benchmarks_complete",
    ),
    PhaseRequirement(
        "43",
        "independent final review",
        (
            _a("reports/FINAL_REVIEW.md", "markdown"),
            _a(
                "artifacts/review/final_review.json",
                "json",
                keys=(
                    "status",
                    "reviewed_source_sha256",
                    "unresolved_severity_counts",
                    "review_scope",
                ),
            ),
        ),
        "final_review_current",
    ),
    PhaseRequirement(
        "44",
        "reports",
        (
            _a("reports/TECHNICAL_REPORT.md", "markdown"),
            _a("reports/INDEPENDENT_VALIDATION_MEMORANDUM.md", "markdown"),
        ),
    ),
    PhaseRequirement(
        "45", "acceptance audit", (_a("artifacts/acceptance/gate_evidence.json", "json"),)
    ),
)


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("JSON root is not an object")
    return value


def _read_engine_qualification(path: Path) -> list[dict[str, Any]]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(value, dict):
        value = value.get("engines", [])
    if not isinstance(value, list) or not all(isinstance(row, dict) for row in value):
        raise ValueError("truth qualification is not an engine-record list")
    return value


def _sha256_file(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_json(value: Any) -> str:
    return sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def source_tree_sha256(root: Path) -> str:
    """Hash Python source names and bytes for review/test evidence freshness."""

    digest = sha256()
    source_root = root / "src/derivguard"
    for path in sorted(source_root.rglob("*.py"), key=lambda item: item.as_posix()):
        relative = path.relative_to(root).as_posix().encode()
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        payload = path.read_bytes()
        digest.update(len(payload).to_bytes(8, "big"))
        digest.update(payload)
    return digest.hexdigest()


def _active_confirmatory(root: Path) -> tuple[Path, Path, Path]:
    """Locate the newest confirmatory result, design freeze, and source."""

    v6 = root / "results/confirmatory/synthetic_v5"
    if (v6 / "confirmatory_metrics.json").is_file():
        return (
            v6,
            root / "artifacts/checkpoints/synthetic_confirmatory_v6.freeze.json",
            root / "src/derivguard/synthetic/confirmatory_v6.py",
        )
    v5 = root / "results/confirmatory/synthetic_v4"
    if (v5 / "confirmatory_metrics.json").is_file():
        return (
            v5,
            root / "artifacts/checkpoints/synthetic_confirmatory_v5.freeze.json",
            root / "src/derivguard/synthetic/confirmatory_v5.py",
        )
    return (
        root / "results/confirmatory/synthetic_v3",
        root / "artifacts/checkpoints/synthetic_confirmatory_v4.freeze.json",
        root / "src/derivguard/synthetic/confirmatory_v4.py",
    )


def verify_final_evidence(root: Path) -> dict[str, Any]:
    """Validate current structured test and independent-review evidence."""

    current_hash = source_tree_sha256(root)
    tests_path = root / "artifacts/tests/final_test_summary.json"
    review_path = root / "artifacts/review/final_review.json"
    tests = _read_json(tests_path) if tests_path.is_file() else {}
    review = _read_json(review_path) if review_path.is_file() else {}
    counts = review.get("unresolved_severity_counts", {})
    scope = set(map(str, review.get("review_scope", [])))
    required_scope = {
        "developer_validator_independence",
        "heston_cf",
        "mc_qe_qem",
        "pde_adi",
        "calibration_leakage",
        "locked_test_discipline",
        "dve",
        "synthetic_faults",
        "statistics",
        "phase_state",
        "gpu_correctness",
        "report_claims",
    }
    checks = {
        "tests_pass": tests.get("status") == "PASS",
        "tests_cover_current_source": tests.get("source_tree_sha256") == current_hash,
        "review_complete": review.get("status") == "PASS",
        "review_covers_current_source": review.get("reviewed_source_sha256") == current_hash,
        "no_unresolved_critical": int(counts.get("Critical", -1)) == 0,
        "no_unresolved_high": int(counts.get("High", -1)) == 0,
        "required_review_scope": required_scope.issubset(scope),
    }
    return {
        "status": "PASS" if all(checks.values()) else "FAIL",
        "source_tree_sha256": current_hash,
        "checks": checks,
        "tests_path": str(tests_path.relative_to(root)),
        "review_path": str(review_path.relative_to(root)),
    }


def _inspect_artifact(root: Path, requirement: ArtifactRequirement) -> ArtifactEvidence:
    path = root / requirement.path
    if not path.is_file() or path.stat().st_size == 0:
        return ArtifactEvidence(requirement.path, False, "missing or empty")
    try:
        if requirement.kind == "json":
            value = _read_json(path)
            missing = set(requirement.required_json_keys) - set(value)
            if missing:
                return ArtifactEvidence(
                    requirement.path, False, f"missing JSON keys {sorted(missing)}"
                )
        elif requirement.kind in {"csv", "parquet"}:
            frame = pd.read_csv(path) if requirement.kind == "csv" else pd.read_parquet(path)
            if len(frame) < requirement.minimum_rows:
                return ArtifactEvidence(
                    requirement.path, False, f"{len(frame)} rows < {requirement.minimum_rows}"
                )
            missing = set(requirement.required_columns) - set(frame.columns)
            if missing:
                return ArtifactEvidence(
                    requirement.path, False, f"missing columns {sorted(missing)}"
                )
            status_only = set(frame.columns) <= {"execution_status", "measured_value", "reason"}
            if status_only or ("measured_value" in frame and frame["measured_value"].isna().all()):
                return ArtifactEvidence(requirement.path, False, "status-only/NaN placeholder")
        elif requirement.kind == "markdown":
            text = path.read_text(encoding="utf-8")
            if len(text.strip()) < 80:
                return ArtifactEvidence(requirement.path, False, "document is too short")
    except (OSError, ValueError, TypeError, KeyError) as exc:
        return ArtifactEvidence(requirement.path, False, f"inspection error: {exc}")
    return ArtifactEvidence(requirement.path, True, "validated")


def _semantic(root: Path, name: str | None) -> tuple[bool, str]:
    if name is None:
        return True, "not required"
    env_path = root / "artifacts/system/environment.json"
    numerical_path = root / "results/aggregated/numerical_benchmarks.json"
    if name == "cuda_available":
        env = _read_json(env_path)
        ok = bool(env.get("torch_cuda_available")) and "4090" in str(env.get("gpu_name", ""))
        return ok, "CUDA RTX detected" if ok else "CUDA RTX evidence absent"
    if name == "cpu_gpu_agreement":
        numerical = _read_json(numerical_path)
        errors = [
            float(row["max_absolute_error"])
            for row in numerical.get("benchmarks", [])
            if row.get("workload", "").startswith("heston_cf")
        ]
        ok = bool(errors) and max(errors) <= 1.0e-9
        return ok, f"maximum CF CPU/GPU absolute error={max(errors) if errors else 'missing'}"
    if name == "public_data_rows":
        data = _read_json(root / "artifacts/data/acquisition_inspection.json")
        rows = int(data.get("historical_spx_sample", {}).get("vendor_rows_total", 0))
        return rows > 0, f"historical rows={rows}"
    if name == "empirical_research_complete":
        selection_path = root / "results/aggregated/developer_method_selection.json"
        if not selection_path.exists():
            return False, "developer method-selection evidence missing"
        selection = _read_json(selection_path)
        ok = (
            selection.get("status") == "SELECTED_ON_DEV_CONFIRMED_ON_VALIDATION"
            and selection.get("profile") == "research"
            and selection.get("locked_test_used_for_selection") is False
        )
        return ok, (
            f"status={selection.get('status')}; profile={selection.get('profile')}; "
            f"locked used={selection.get('locked_test_used_for_selection')}"
        )
    if name == "empirical_and_confirmatory_complete":
        empirical_ok, empirical_detail = _semantic(root, "empirical_research_complete")
        confirmatory_ok, confirmatory_detail = _semantic(root, "confirmatory_locked")
        return empirical_ok and confirmatory_ok, (
            f"empirical=({empirical_detail}); confirmatory=({confirmatory_detail})"
        )
    if name == "calibration_objectives_complete":
        research_ok, research_detail = _semantic(root, "empirical_research_complete")
        frame = pd.read_csv(root / "reports/tables/T04_CALIBRATION_OBJECTIVE_COMPARISON.csv")
        observed = set(frame["objective"].dropna().astype(str))
        required = {f"C{index:02d}" for index in range(5)}
        ok = research_ok and required.issubset(observed)
        return ok, f"{research_detail}; objectives={sorted(observed)}"
    if name == "optimizers_complete":
        research_ok, research_detail = _semantic(root, "empirical_research_complete")
        frame = pd.read_csv(root / "reports/tables/T05_OPTIMIZER_IDENTIFIABILITY.csv")
        observed = set(frame["optimizer_id"].dropna().astype(str))
        required = {f"O{index:02d}" for index in range(5)}
        ok = research_ok and required.issubset(observed)
        return ok, f"{research_detail}; optimizers={sorted(observed)}"
    if name == "feller_variants_complete":
        research_ok, research_detail = _semantic(root, "empirical_research_complete")
        frame = pd.read_csv(root / "results/aggregated/feller_study.csv")
        observed = set(frame["feller_treatment"].dropna().astype(str))
        required = {"unconstrained", "soft", "hard"}
        ok = research_ok and required.issubset(observed)
        return ok, f"{research_detail}; Feller variants={sorted(observed)}"
    if name == "real_models_complete":
        research_ok, research_detail = _semantic(root, "empirical_research_complete")
        frame = pd.read_csv(root / "reports/tables/T06_PRICING_MODEL_COMPARISON.csv")
        observed = set(frame["model_id"].dropna().astype(str))
        required = {
            "M00_BS_FLAT",
            "M02_SVI",
            "M03_SSVI",
            "M10_HESTON",
            "M20_LOCALVOL_MC",
            "M21_BATES_SENSITIVITY",
        }
        near = pd.read_csv(root / "results/aggregated/near_expiry_study.csv")
        liquidity = pd.read_csv(root / "results/aggregated/liquidity_filter_study.csv")
        settlement = pd.read_csv(root / "results/aggregated/am_pm_robustness.csv")
        supplement_ok = bool(
            set(near["experiment"].astype(str)) == {"near_expiry_1_to_7d"}
            and set(liquidity["experiment"].astype(str)) == {"liquidity_filter"}
            and set(settlement["experiment"].astype(str)) == {"settlement_robustness"}
            and {"AM", "PM"}.issubset(set(settlement["settlement_class"].astype(str)))
        )
        ok = research_ok and required.issubset(observed) and supplement_ok
        return ok, (
            f"{research_detail}; real model IDs={sorted(observed)}; "
            f"near/liquidity/AM-PM supplement complete={supplement_ok}"
        )
    if name == "portfolio_set_complete":
        research_ok, research_detail = _semantic(root, "empirical_research_complete")
        frame = pd.read_csv(root / "reports/tables/T11_PORTFOLIO_MATERIALITY.csv")
        observed = set(frame["portfolio_id"].dropna().astype(str))
        required = {f"P{index:02d}" for index in range(1, 7)}
        metrics_complete = bool(frame[["value", "delta", "gamma", "vega"]].notna().all(axis=None))
        ok = research_ok and required.issubset(observed) and metrics_complete
        return ok, (
            f"{research_detail}; portfolios={sorted(observed)}; "
            f"value/Greek metrics complete={metrics_complete}"
        )
    if name == "developer_method_ablations_complete":
        research_ok, research_detail = _semantic(root, "empirical_research_complete")
        frame = pd.read_csv(root / "results/aggregated/developer_method_ablation_summary.csv")
        observed = set(frame["ablation_id"].dropna().astype(str))
        required = {f"DA{index:02d}" for index in range(1, 7)}
        numeric = frame.select_dtypes(include="number")
        quantitative = not numeric.empty and bool(numeric.notna().any(axis=1).all())
        ok = research_ok and required.issubset(observed) and quantitative
        return ok, (
            f"{research_detail}; ablations={sorted(observed)}; quantitative rows={quantitative}"
        )
    if name == "tests_pass":
        tests = _read_json(root / "artifacts/tests/final_test_summary.json")
        ok = (
            tests.get("pytest_exit_code") == 0
            and tests.get("ruff_exit_code") == 0
            and tests.get("mypy_exit_code") == 0
        )
        return ok, str(tests)
    if name == "cf_oracle":
        error = float(_read_json(numerical_path).get("cf_quantlib_absolute_error", float("inf")))
        return error <= 1.0e-8, f"CF/QuantLib absolute error={error:.6g}"
    if name == "mc_remediation":
        value = _read_json(root / "results/aggregated/numerical_remediation_qualification.json")
        qualification = value.get("mc_qualification", {})
        ok = value.get("mc_status") == "PASS" and all(
            qualification.get(key) is True
            for key in (
                "all_finest_qem_seed_biases_within_3se",
                "minimum_seed_requirement_met",
                "minimum_time_levels_met",
            )
        )
        return ok, f"MC remediation status={value.get('mc_status')}; {qualification}"
    if name == "pde_tolerance":
        path = root / "results/aggregated/pde_adi_convergence.csv"
        if not path.exists():
            return False, "ADI convergence artifact missing"
        frame = pd.read_csv(path)
        required = {"scheme", "case", "study", "level", "absolute_error_cf"}
        if not required.issubset(frame.columns):
            return False, f"ADI convergence columns missing: {sorted(required - set(frame))}"
        convergence = frame.loc[frame["study"] == "combined_grid_convergence"]
        finest = (
            convergence.sort_values(["case", "scheme", "level"]).groupby(["case", "scheme"]).tail(1)
        )
        qualification_path = root / "results/aggregated/numerical_remediation_qualification.json"
        qualification = _read_json(qualification_path) if qualification_path.exists() else {}
        ok = qualification.get("pde_status") == "PASS"
        return (
            ok,
            "PDE status="
            f"{qualification.get('pde_status', 'missing')}; finest case/scheme errors="
            f"{finest[['case', 'scheme', 'absolute_error_cf']].to_dict(orient='records')}",
        )
    if name == "challengers_qualified":
        result_dir, _, _ = _active_confirmatory(root)
        qualification_rows = _read_engine_qualification(result_dir / "truth_qualification.json")
        engines = {str(row.get("engine")): row for row in qualification_rows}
        required_engines = {"BATES_CF", "LOCALVOL_SSVI_MARGINAL"}
        qualified = all(
            engines.get(engine, {}).get("status") == "PASS" for engine in required_engines
        )
        comparison = pd.read_csv(root / "reports/tables/T06_PRICING_MODEL_COMPARISON.csv")
        measured = {"M20_LOCALVOL_MC", "M21_BATES_SENSITIVITY"}.issubset(
            set(comparison.get("model_id", pd.Series(dtype=str)).astype(str))
        )
        research_ok, research_detail = _semantic(root, "empirical_research_complete")
        ok = qualified and measured and research_ok
        return ok, (
            f"truth engines qualified={qualified}; empirical challengers measured={measured}; "
            f"{research_detail}"
        )
    if name == "release_hash_current":
        release_path = root / "artifacts/model_releases/model_release_v1.1.json"
        if not release_path.is_file():
            return False, "corrected Developer Model Release v1.1 is not frozen"
        release = _read_json(release_path)
        source_hashes = release.get("source_hashes", {})
        source_paths = {
            "development/heston_cf.py": root / "src/derivguard/development/heston_cf.py",
            "development/calibration.py": root / "src/derivguard/development/calibration.py",
            "empirical.py": root / "src/derivguard/empirical.py",
        }
        all_source_hashes_current = isinstance(source_hashes, dict) and all(
            path.is_file() and source_hashes.get(name) == _sha256_file(path)
            for name, path in source_paths.items()
        )
        tolerance = root / str(release.get("numerical_tolerances_file", ""))
        selection_file = root / str(release.get("selection_evidence", ""))
        config_current = tolerance.is_file() and release.get("config_sha256") == _sha256_file(
            tolerance
        )
        selection_current = selection_file.is_file() and release.get(
            "selection_evidence_hash"
        ) == _sha256_file(selection_file)
        execution_file = root / str(release.get("execution_evidence", ""))
        execution_current = release.get("release_id") != "Developer Model Release v1.1" or (
            execution_file.is_file()
            and release.get("execution_evidence_hash") == _sha256_file(execution_file)
        )
        prior_release = root / "artifacts/model_releases/model_release_v1.0.json"
        prior_release_current = release.get("release_id") != "Developer Model Release v1.1" or (
            prior_release.is_file()
            and release.get("prior_release_hash") == _sha256_file(prior_release)
        )
        method_frozen = release.get("objective") == "C04" and release.get("optimizer") == "O01"
        release_id = str(release.get("release_id", ""))
        release_versioned = release_id == "Developer Model Release v1.1"
        converged_jobs = int(release.get("real_date_jobs_converged", 0))
        expected_jobs = int(release.get("real_date_jobs_expected", 0))
        convergence_rate = float(release.get("real_date_convergence_rate", 0.0))
        convergence_evidence = (
            expected_jobs > 0
            and 0 < converged_jobs <= expected_jobs
            and abs(convergence_rate - converged_jobs / expected_jobs) <= 1.0e-12
        )
        ok = bool(
            all_source_hashes_current
            and config_current
            and selection_current
            and execution_current
            and prior_release_current
            and method_frozen
            and release_versioned
            and convergence_evidence
        )
        return ok, (
            f"release={release_id}; all source hashes current={all_source_hashes_current}; "
            f"config current={config_current}; selection current={selection_current}; "
            f"execution current={execution_current}; "
            f"prior release current={prior_release_current}; "
            f"C04/O01 frozen={method_frozen}; converged real-date jobs="
            f"{converged_jobs}/{expected_jobs}; convergence evidence={convergence_evidence}"
        )
    if name == "truths_qualified":
        result_dir, _, _ = _active_confirmatory(root)
        qualification_rows = _read_engine_qualification(result_dir / "truth_qualification.json")
        engines = {str(row.get("engine")): row for row in qualification_rows}
        required = {
            "HESTON_FIXED_QUAD",
            "BATES_CF",
            "LOCALVOL_SSVI_MARGINAL",
            "PIECEWISE_REGIME_MARGINAL",
        }
        ok = all(engines.get(engine, {}).get("status") == "PASS" for engine in required)
        return (
            ok,
            "all pre-registered European-marginal controlled DGP constructions passed their "
            "stated numerical checks"
            if ok
            else "one or more controlled DGP constructions failed its stated numerical check",
        )
    if name == "all_faults_covered":
        result_dir, _, _ = _active_confirmatory(root)
        frame = pd.read_csv(result_dir / "fault_coverage.csv")
        observed = set(frame["fault_id"].astype(str))
        required = {f"F{index:02d}" for index in range(17)}
        ok = required.issubset(observed) and bool(frame["implementation_label"].notna().all())
        return ok, f"fault IDs covered={sorted(observed)}"
    if name in {"threshold_freeze_matches_locked", "design_freeze_matches_locked"}:
        result_dir, design_path, source_path = _active_confirmatory(root)
        threshold_path = result_dir / "dve_thresholds.freeze.json"
        locked_path = result_dir / "confirmatory_metrics.json"
        if not all(path.exists() for path in (design_path, threshold_path, locked_path)):
            return False, "design freeze, threshold freeze, or locked metrics missing"
        design = _read_json(design_path)
        threshold = _read_json(threshold_path)
        locked = _read_json(locked_path)
        freeze_hash = design.get("freeze_hash")
        linked_hashes_match = (
            freeze_hash
            and threshold.get("design_freeze_hash") == freeze_hash
            and locked.get("freeze_hash") == freeze_hash
        )
        hypothesis_path = root / str(design.get("hypothesis_registry_path", ""))
        experiment_path = root / str(design.get("experiment_registry_path", ""))
        prequalification_path = root / str(design.get("calibration_prequalification_path", ""))
        design_without_freeze_hash = dict(design)
        design_without_freeze_hash.pop("freeze_hash", None)
        internal_hashes_match = all(
            (
                design.get("config_hash") == _sha256_json(design.get("config")),
                design.get("seed_hash") == _sha256_json(design.get("seed_registry")),
                design.get("schedule_hash") == _sha256_json(design.get("fault_schedule")),
                design.get("calibration_design_hash")
                == _sha256_json(design.get("calibration_design")),
                design.get("dve_formulation_hash") == _sha256_json(design.get("dve_formulation")),
                design.get("freeze_hash") == _sha256_json(design_without_freeze_hash),
            )
        )
        external_hashes_match = (
            source_path.is_file()
            and hypothesis_path.is_file()
            and experiment_path.is_file()
            and design.get("module_hash") == _sha256_file(source_path)
            and design.get("hypothesis_registry_hash") == _sha256_file(hypothesis_path)
            and design.get("experiment_registry_hash") == _sha256_file(experiment_path)
            and prequalification_path.is_file()
            and design.get("calibration_prequalification_hash")
            == _sha256_file(prequalification_path)
            and _read_json(prequalification_path).get("status") == "PASS"
        )
        threshold_frozen = threshold.get("status") == "FROZEN_BEFORE_LOCKED_GENERATION"
        timestamps_ordered = threshold_path.stat().st_mtime_ns < locked_path.stat().st_mtime_ns
        ok = bool(
            linked_hashes_match
            and internal_hashes_match
            and external_hashes_match
            and threshold_frozen
            and timestamps_ordered
        )
        return ok, (
            f"linked freeze hashes match={bool(linked_hashes_match)}; "
            f"internal design hashes match={internal_hashes_match}; "
            f"source/registry hashes current={external_hashes_match}; "
            f"threshold frozen={threshold_frozen}; "
            f"threshold artifact predates locked metrics={timestamps_ordered}"
        )
    if name == "confirmatory_locked":
        result_dir, design_path, _ = _active_confirmatory(root)
        value = _read_json(result_dir / "confirmatory_metrics.json")
        design = _read_json(design_path)
        calibration = pd.read_parquet(result_dir / "calibration_diagnostics.parquet")
        provenance = pd.read_parquet(result_dir / "case_provenance.parquet")
        qualification = _read_json(result_dir / "calibration_qualification.json")
        partitions = {"DEV", "VALIDATION", "LOCKED_TEST", "OOD_LOCKED_TEST"}
        selected = (
            calibration.loc[calibration["start_index"].astype(int) == -1]
            if "start_index" in calibration
            else calibration
        )
        primary = selected.loc[selected["draw"].astype(int) == -1]
        bootstrap = selected.loc[selected["draw"].astype(int) >= 0]
        provenance_case_ids = set(provenance["case_id"].astype(str))
        bootstrap_draws = int(design.get("calibration_design", {}).get("bootstrap_draws", 0))
        optimizer_column = "optimizer_id" if "optimizer_id" in calibration else "optimizer"
        optimizer_values = (
            set(calibration[optimizer_column].astype(str))
            if optimizer_column in calibration
            else set()
        )
        design_release_id = str(
            design.get(
                "release_id",
                design.get("developer_release_id", design.get("developer_model_release_id", "")),
            )
        )
        metrics_release_id = str(
            value.get(
                "release_id",
                value.get("developer_release_id", value.get("developer_model_release_id", "")),
            )
        )
        linked_release_path = next(
            (
                path
                for path in (root / "artifacts/model_releases").glob("model_release_*.json")
                if _read_json(path).get("release_id") == design_release_id
            ),
            None,
        )
        release_hash = None if linked_release_path is None else _sha256_file(linked_release_path)
        linked_release_hash = bool(
            release_hash
            and design.get(
                "release_hash",
                design.get("developer_release_hash", design.get("release_sha256")),
            )
            == release_hash
            and value.get(
                "release_hash", value.get("developer_release_hash", value.get("release_sha256"))
            )
            == release_hash
        )
        observable_calibration = (
            set(calibration["partition"].astype(str)) == partitions
            and not primary.empty
            and not bootstrap.empty
            and calibration["backend"].astype(str).str.len().gt(0).all()
            and calibration["objective"].astype(str).eq("C04").all()
            and optimizer_values == {"O01"}
            and primary["success"].eq(True).all()
            and primary["qualified"].eq(True).all()
            and bootstrap["success"].eq(True).all()
            and bootstrap["qualified"].eq(True).all()
            and set(primary["case_id"].astype(str)) == provenance_case_ids
            and len(bootstrap) == len(provenance_case_ids) * bootstrap_draws
            and provenance["calibration_inputs"].astype(str).eq("observable_quotes_only").all()
            and set(calibration["case_id"].astype(str)) == set(provenance["case_id"].astype(str))
            and qualification.get("status") == "PASS"
            and qualification.get("all_selected_fits_qualified") is True
        )
        freeze_ok, freeze_detail = _semantic(root, "design_freeze_matches_locked")
        ok = bool(
            result_dir.name == "synthetic_v5"
            and value.get("status") == "CONFIRMATORY_LOCKED_COMPLETE"
            and set(value.get("faults_executed", [])) == {f"F{index:02d}" for index in range(17)}
            and set(value.get("baselines", {})) == {f"V{index:02d}" for index in range(7)}
            and observable_calibration
            and bool(design_release_id)
            and design_release_id == metrics_release_id
            and linked_release_hash
            and freeze_ok
        )
        return ok, (
            f"primary v6 result={result_dir.name == 'synthetic_v5'}; "
            f"status={value.get('status')}; faults={value.get('faults_executed')}; "
            f"observable converged C04/O01 calibration evidence={observable_calibration}; "
            f"primary success={int(primary['success'].eq(True).sum())}/{len(primary)}; "
            f"bootstrap success={int(bootstrap['success'].eq(True).sum())}/{len(bootstrap)}; "
            f"release linked={linked_release_hash}; {freeze_detail}"
        )
    if name == "final_compute_benchmarks_complete":
        manifest_path = root / "results/aggregated/final_compute_benchmarks.json"
        manifest = _read_json(manifest_path)
        result_path = root / "results/aggregated/final_compute_benchmarks.csv"
        t12_path = root / "results/aggregated/T12_CPU_GPU_PERFORMANCE.csv"
        frame = pd.read_csv(result_path)
        t12 = pd.read_csv(t12_path)
        required_workloads = {
            "synthetic_heston_truth_and_greeks",
            "heston_population_evaluation",
            "bootstrap_population_calibration",
            "heston_pde_mcs_representative_grid",
        }
        measured_or_explicit = bool(
            frame["status"].isin({"MEASURED", "ATTEMPTED_INCOMPATIBLE", "NOT_RUN_DEVICE_CPU"}).all()
        )
        gpu_priority = {
            "synthetic_heston_truth_and_greeks",
            "heston_population_evaluation",
            "bootstrap_population_calibration",
        }
        gpu_rows = frame.loc[
            frame["backend"].astype(str).str.contains("cuda", case=False, na=False)
            & frame["workload"].astype(str).isin(gpu_priority)
        ]
        gpu_priority_measured = bool(
            set(gpu_rows["workload"].astype(str)) == gpu_priority
            and gpu_rows["status"].astype(str).eq("MEASURED").all()
        )
        gpu_selected = (
            t12["selected_backend"]
            .astype(str)
            .str.contains("cuda|cupyx", case=False, regex=True, na=False)
        )
        selected_gpu_accurate = bool(
            not gpu_selected.any() or t12.loc[gpu_selected, "accuracy_pass"].astype(bool).all()
        )
        ok = bool(
            manifest.get("profile") == "research"
            and manifest.get("device") == "cuda"
            and manifest.get("all_rows_measured_or_explicitly_unavailable") is True
            and manifest.get("result_sha256") == _sha256_file(result_path)
            and manifest.get("canonical_t12_sha256") == _sha256_file(t12_path)
            and required_workloads.issubset(set(frame["workload"].astype(str)))
            and measured_or_explicit
            and gpu_priority_measured
            and selected_gpu_accurate
        )
        return ok, (
            f"profile={manifest.get('profile')}; device={manifest.get('device')}; "
            f"workloads={sorted(set(frame['workload'].astype(str)))}; "
            f"measured/explicit={measured_or_explicit}; "
            f"GPU-priority measured={gpu_priority_measured}; "
            f"selected GPU accurate={selected_gpu_accurate}"
        )
    if name == "final_review_current":
        verification = verify_final_evidence(root)
        return verification["status"] == "PASS", json.dumps(verification["checks"], sort_keys=True)
    return False, f"unknown semantic check {name}"


def _availability(root: Path) -> dict[str, dict[str, Any]]:
    path = root / "results/RESULT_AVAILABILITY.json"
    if not path.exists():
        return {}
    value = _read_json(path)
    entries = value.get("entries", [])
    return {str(entry["phase_id"]): entry for entry in entries if isinstance(entry, dict)}


def evaluate_phases(root: str | Path) -> tuple[PhaseEvidence, ...]:
    root_path = Path(root)
    availability = _availability(root_path)
    evaluated_at = datetime.now(UTC).isoformat()
    output: list[PhaseEvidence] = []
    for phase in PHASE_REQUIREMENTS:
        artifacts = tuple(_inspect_artifact(root_path, item) for item in phase.artifacts)
        if all(item.passed for item in artifacts):
            try:
                semantic_ok, semantic_detail = _semantic(root_path, phase.semantic_check)
            except (OSError, ValueError, TypeError, KeyError) as exc:
                semantic_ok, semantic_detail = False, f"semantic inspection error: {exc}"
        else:
            semantic_ok = False
            semantic_detail = "; ".join(
                f"{item.path}: {item.detail}" for item in artifacts if not item.passed
            )
        all_ok = all(item.passed for item in artifacts) and semantic_ok
        blocker_entry = availability.get(phase.phase_id)
        blocker = None
        if all_ok:
            status = "COMPLETED"
            test_status = "PASS"
        elif blocker_entry:
            status = str(blocker_entry.get("status", "UNAVAILABLE"))
            test_status = "NOT_APPLICABLE" if status == "UNAVAILABLE" else "PARTIAL"
            blocker = str(blocker_entry.get("blocker", "availability blocker recorded"))
        else:
            any_present = any((root_path / item.path).exists() for item in phase.artifacts)
            status = "PARTIAL" if any_present else "PENDING"
            test_status = "PARTIAL" if any_present else "NOT_RUN"
            blocker = semantic_detail if not semantic_ok else "required evidence incomplete"
        output.append(
            PhaseEvidence(
                phase.phase_id,
                phase.name,
                status,
                test_status,
                artifacts,
                blocker,
                evaluated_at,
            )
        )
    return tuple(output)


def write_gate_evidence(root: str | Path) -> dict[str, Any]:
    root_path = Path(root)
    phases = evaluate_phases(root_path)
    payload = {
        "schema_version": "2.0",
        "generated_at": datetime.now(UTC).isoformat(),
        "derivation": "artifact inspection + semantic predicates + explicit availability manifest",
        "phases": [asdict(phase) for phase in phases],
        "summary": {
            status: sum(phase.status == status for phase in phases)
            for status in ("COMPLETED", "PARTIAL", "UNAVAILABLE", "FAILED", "PENDING")
        },
    }
    path = root_path / "artifacts/acceptance/gate_evidence.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    # Phase 45 validates this artifact, so reconcile once after the atomic
    # evidence file exists. No scientific phase is promoted by ID.
    phases = evaluate_phases(root_path)
    payload["phases"] = [asdict(phase) for phase in phases]
    payload["summary"] = {
        status: sum(phase.status == status for phase in phases)
        for status in ("COMPLETED", "PARTIAL", "UNAVAILABLE", "FAILED", "PENDING")
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload


ACCEPTANCE_PHASES: dict[str, tuple[str, ...]] = {
    "ENVIRONMENT": ("02", "03"),
    "DATA": ("05", "06", "07", "08", "09"),
    "GPU": ("03", "04", "14", "42"),
    "NUMERICAL": ("10", "11", "12", "13", "14", "15", "16", "17"),
    "INDEPENDENCE": ("13", "14", "15", "16"),
    "CALIBRATION": ("18", "19", "20", "21", "22", "23"),
    "RELEASE": ("24",),
    "SYNTHETIC": ("25", "26", "33"),
    "DVE": ("27", "28", "35"),
    "REAL-DATA": ("29", "30", "31", "34"),
    "STATISTICAL": ("38",),
    "ABLATION": ("36", "37"),
    "MATERIALITY": ("32",),
    "GOVERNANCE": ("00", "01", "40", "41", "43"),
    "REPRODUCIBILITY": ("02", "24", "27", "28", "42"),
    "REPORTING": ("44", "45"),
}


def acceptance_from_phase_evidence(payload: dict[str, Any]) -> list[dict[str, Any]]:
    """Aggregate acceptance gates without converting unavailable science to PASS."""

    by_id = {str(row["phase_id"]): row for row in payload["phases"]}
    output: list[dict[str, Any]] = []
    for gate, phase_ids in ACCEPTANCE_PHASES.items():
        rows = [by_id[phase_id] for phase_id in phase_ids]
        statuses = [str(row["status"]) for row in rows]
        if any(status == "FAILED" for status in statuses):
            status = "FAIL"
        elif all(status == "COMPLETED" for status in statuses):
            status = "PASS"
        elif all(status == "UNAVAILABLE" for status in statuses):
            status = "NOT APPLICABLE"
        else:
            status = "PARTIAL"
        output.append(
            {
                "gate": gate,
                "status": status,
                "phase_ids": list(phase_ids),
                "phase_statuses": statuses,
                "evidence": [
                    artifact["path"]
                    for row in rows
                    for artifact in row["artifacts"]
                    if artifact["passed"]
                ],
                "blockers": [row["blocker"] for row in rows if row.get("blocker")],
            }
        )
    return output


def reconcile_run_state_from_evidence(root: str | Path, payload: dict[str, Any]) -> dict[str, Any]:
    """Replace legacy declared statuses with evaluated evidence and append a ledger audit."""

    from derivguard.governance.run_state import sha256_manifest

    root_path = Path(root)
    state_path = root_path / "artifacts/run_state.json"
    ledger_path = root_path / "artifacts/execution_ledger.jsonl"
    state = _read_json(state_path)
    evaluated = {str(row["phase_id"]): row for row in payload["phases"]}
    now = datetime.now(UTC).isoformat()
    source_paths = tuple((root_path / "src/derivguard").rglob("*.py"))
    config_paths = tuple((root_path / "configs").rglob("*.*"))
    code_hash = sha256_manifest(source_paths, root=root_path) if source_paths else ""
    config_hash = sha256_manifest(config_paths, root=root_path) if config_paths else ""
    incomplete: list[str] = []
    with ledger_path.open("a", encoding="utf-8", newline="\n") as ledger:
        for phase_id, phase in state["phases"].items():
            row = evaluated[phase_id]
            phase["status"] = row["status"]
            phase["test_status"] = row["test_status"]
            phase["ended_at"] = now
            phase["code_hash"] = code_hash
            phase["config_hash"] = config_hash
            phase["output_artifacts"] = [
                artifact["path"] for artifact in row["artifacts"] if artifact["passed"]
            ]
            phase["errors"] = [row["blocker"]] if row.get("blocker") else []
            phase["resume_point"] = (
                None
                if row["status"] in {"COMPLETED", "UNAVAILABLE"}
                else f"close evidence gap for phase {phase_id}: {row.get('blocker')}"
            )
            if row["status"] not in {"COMPLETED", "UNAVAILABLE"}:
                incomplete.append(phase_id)
            event = {
                "schema_version": "2.0",
                "event_id": str(uuid.uuid4()),
                "timestamp": now,
                "run_id": state.get("run_id"),
                "phase_id": phase_id,
                "event_type": "EVIDENCE_GATE_EVALUATED",
                "status": row["status"],
                "inputs": [],
                "outputs": phase["output_artifacts"],
                "config_hash": config_hash,
                "code_hash": code_hash,
                "test_status": row["test_status"],
                "errors": phase["errors"],
                "message": "status derived by governance.gates; no phase allowlist",
            }
            ledger.write(json.dumps(event, separators=(",", ":")) + "\n")
    state["schema_version"] = "2.0"
    state["updated_at"] = now
    state["current_phase"] = incomplete[0] if incomplete else "45"
    state["resume_phase"] = incomplete[0] if incomplete else None
    state["overall_status"] = "COMPLETE" if not incomplete else "PARTIAL_COMPLETION"
    state["status_derivation"] = "artifacts/acceptance/gate_evidence.json"
    state_path.write_text(json.dumps(state, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return state
