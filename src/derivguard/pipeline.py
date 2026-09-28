"""Dependency-ordered, restartable full-project orchestration.

The orchestrator invokes scientific stages; it does not assign acceptance
status.  Evidence gates evaluate the outputs after execution.
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

from derivguard.compute.benchmarks import run_final_compute_benchmarks
from derivguard.compute.mc_variance_reduction import run_mc_variance_reduction
from derivguard.compute.precision import run_precision_sensitivity
from derivguard.data.processing import process_cboe_marking, process_historical_spx_sample
from derivguard.data.providers import (
    CboeDataShopSampleProvider,
    CboeMarkingProvider,
    MassiveProvider,
    PublicSPXSampleProvider,
)
from derivguard.empirical import run_empirical_program
from derivguard.empirical_supplement import run_empirical_supplement
from derivguard.governance.experiment_reconciliation import reconcile_experiment_registry
from derivguard.governance.gates import verify_final_evidence, write_gate_evidence
from derivguard.reporting.method_ablations import build_developer_method_ablation_summary
from derivguard.research import (
    capture_environment,
    freeze_developer_release_v1_1,
    generate_reports,
    repository_root,
    run_data_eda,
    run_numerical_benchmarks,
)
from derivguard.synthetic.confirmatory_v6 import (
    prequalify_confirmatory_v6_calibration,
    run_confirmatory_v6_study,
)
from derivguard.validation.numerical_remediation import run_numerical_remediation


@dataclass(frozen=True)
class StageResult:
    stage_id: str
    status: str
    started_at: str
    ended_at: str
    runtime_seconds: float
    output_artifacts: tuple[str, ...]
    detail: dict[str, Any]


def _timestamp() -> str:
    return datetime.now(UTC).isoformat()


def _write_registry(path: Path, results: list[StageResult]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": "1.0",
        "generated_at": _timestamp(),
        "stages": [asdict(result) for result in results],
    }
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )


def _run_stage(
    stage_id: str,
    function: Callable[[], dict[str, Any]],
    output_artifacts: tuple[str, ...],
) -> StageResult:
    started_at = _timestamp()
    clock = time.perf_counter()
    try:
        detail = function()
        status = str(detail.get("status", "EXECUTED"))
    except Exception as exc:
        ended_at = _timestamp()
        return StageResult(
            stage_id,
            "FAILED",
            started_at,
            ended_at,
            time.perf_counter() - clock,
            output_artifacts,
            {"exception_type": type(exc).__name__, "error": str(exc)},
        )
    return StageResult(
        stage_id,
        status,
        started_at,
        _timestamp(),
        time.perf_counter() - clock,
        output_artifacts,
        detail,
    )


def run_data_stage(root: Path, *, resume: bool = True) -> dict[str, Any]:
    raw_root = root / "data/raw"
    acquisitions = [
        provider.acquire(raw_root)
        for provider in (
            PublicSPXSampleProvider(),
            CboeMarkingProvider(),
            CboeDataShopSampleProvider(),
            MassiveProvider(),
        )
    ]
    required = acquisitions[:3]
    if any(result.status == "BLOCKED" for result in required):
        raise RuntimeError(f"required public acquisition blocked: {required}")
    historical = required[0].artifact_path
    marking = required[1].artifact_path
    if historical is None or marking is None:
        raise RuntimeError("required cached/acquired artifact path missing")
    historical_result = process_historical_spx_sample(historical, root / "data", resume=resume)
    # The prospective file is small enough to audit again; this also verifies
    # that its current bytes still parse under the adapter.
    marking_result = process_cboe_marking(marking, root / "data")
    return {
        "status": "EXECUTED_OR_RESUMED",
        "acquisitions": [asdict(result) for result in acquisitions],
        "historical_processing": historical_result,
        "marking_processing": marking_result,
    }


def run_full_pipeline(
    profile: str = "research",
    *,
    device: str = "cuda",
    resume: bool = False,
) -> dict[str, Any]:
    """Run every implemented feasible stage, then derive gates from evidence."""

    if profile not in {"research", "audit", "smoke"}:
        raise ValueError("profile must be research, audit, or smoke")
    if device not in {"cuda", "auto"}:
        raise ValueError("device must be cuda or auto")
    if profile == "research" and device != "cuda":
        raise ValueError("research profile requires CUDA; CPU fallback is not permitted")

    root = repository_root()
    results: list[StageResult] = []
    registry_path = root / "artifacts/pipeline_stage_results.json"

    def add(
        stage_id: str,
        function: Callable[[], dict[str, Any]],
        output_artifacts: tuple[str, ...],
    ) -> None:
        result = _run_stage(stage_id, function, output_artifacts)
        results.append(result)
        _write_registry(registry_path, results)
        if result.status == "FAILED":
            raise RuntimeError(f"stage {stage_id} failed: {result.detail}")

    env_path = root / "artifacts/system/environment.json"
    add(
        "environment",
        lambda: (
            json.loads(env_path.read_text(encoding="utf-8"))
            if resume and env_path.exists()
            else capture_environment()
        ),
        ("artifacts/system/environment.json", "artifacts/system/hardware_profile.json"),
    )
    current_environment = json.loads(env_path.read_text(encoding="utf-8"))
    resolved_device = (
        "cuda"
        if device == "cuda" or bool(current_environment.get("torch_cuda_available"))
        else "cpu-bounded-reference"
    )
    add(
        "data_acquire_process",
        lambda: run_data_stage(root, resume=resume),
        ("artifacts/data/acquisition_inspection.json", "artifacts/data/processing_summary.json"),
    )
    eda_path = root / "results/aggregated/data_eda.json"
    add(
        "data_eda",
        lambda: (
            json.loads(eda_path.read_text(encoding="utf-8"))
            if resume and eda_path.exists()
            else run_data_eda()
        ),
        ("results/aggregated/data_eda.json",),
    )
    numerical_path = root / "results/aggregated/numerical_benchmarks.json"
    add(
        "numerical_foundations",
        lambda: (
            json.loads(numerical_path.read_text(encoding="utf-8"))
            if resume
            and numerical_path.exists()
            and json.loads(numerical_path.read_text(encoding="utf-8")).get("profile") == profile
            else run_numerical_benchmarks(profile)
        ),
        ("results/aggregated/numerical_benchmarks.json",),
    )
    add(
        "numerical_remediation",
        lambda: (
            json.loads(
                (root / "results/aggregated/numerical_remediation_qualification.json").read_text(
                    encoding="utf-8"
                )
            )
            if resume
            and (root / "results/aggregated/numerical_remediation_qualification.json").exists()
            and json.loads(
                (root / "results/aggregated/numerical_remediation_qualification.json").read_text(
                    encoding="utf-8"
                )
            ).get("profile")
            == profile
            else run_numerical_remediation(profile, resolved_device)
        ),
        (
            "results/aggregated/mc_qe_qem_audit.csv",
            "results/aggregated/pde_adi_convergence.csv",
            "results/aggregated/numerical_remediation_qualification.json",
        ),
    )
    variance_path = root / "results/aggregated/mc_variance_reduction.json"

    def run_variance_reduction_stage() -> dict[str, Any]:
        if resume and variance_path.exists():
            return cast(dict[str, Any], json.loads(variance_path.read_text(encoding="utf-8")))
        if resolved_device != "cuda":
            return {
                "status": "UNAVAILABLE",
                "reason": "N06 research execution requires the qualified CUDA backend",
            }
        return run_mc_variance_reduction(root, resolved_device)

    add(
        "mc_variance_reduction",
        run_variance_reduction_stage,
        (
            "results/aggregated/mc_variance_reduction.csv",
            "results/aggregated/mc_variance_reduction.json",
        ),
    )
    precision_path = root / "results/aggregated/precision_sensitivity.json"
    add(
        "precision_sensitivity",
        lambda: (
            json.loads(precision_path.read_text(encoding="utf-8"))
            if resume and precision_path.exists()
            else run_precision_sensitivity(root, resolved_device)
        ),
        (
            "results/aggregated/precision_sensitivity.csv",
            "results/aggregated/precision_sensitivity.json",
        ),
    )
    add(
        "empirical_program",
        lambda: run_empirical_program(profile, device=resolved_device, resume=resume, root=root),
        (
            "results/aggregated/developer_method_selection.json",
            "reports/tables/T04_CALIBRATION_OBJECTIVE_COMPARISON.csv",
            "reports/tables/T05_OPTIMIZER_IDENTIFIABILITY.csv",
            "reports/tables/T06_PRICING_MODEL_COMPARISON.csv",
            "reports/tables/T11_PORTFOLIO_MATERIALITY.csv",
        ),
    )
    add(
        "empirical_supplement",
        lambda: run_empirical_supplement(
            root=root,
            device=resolved_device,
            resume=resume,
        ),
        (
            "results/aggregated/near_expiry_study.csv",
            "results/aggregated/liquidity_filter_study.csv",
            "results/aggregated/am_pm_robustness.csv",
        ),
    )
    add(
        "developer_method_ablations",
        lambda: build_developer_method_ablation_summary(root),
        (
            "results/aggregated/developer_method_ablation_summary.csv",
            "results/aggregated/developer_method_ablation_summary.json",
        ),
    )
    add(
        "developer_release",
        freeze_developer_release_v1_1,
        (
            "artifacts/model_releases/model_release_v1.0.json",
            "artifacts/model_releases/model_release_v1.1.json",
        ),
    )
    add(
        "confirmatory_v6_prequalification",
        lambda: prequalify_confirmatory_v6_calibration(
            profile, resolved_device, root=root, resume=resume
        ),
        (
            "artifacts/calibration/confirmatory_v6_prequalification.json",
            "results/aggregated/confirmatory_v6_prequalification.csv",
            "results/raw/confirmatory_v6_prequalification_diagnostics.parquet",
        ),
    )
    add(
        "confirmatory_synthetic",
        lambda: run_confirmatory_v6_study(profile, device=resolved_device, resume=resume),
        (
            "artifacts/checkpoints/synthetic_confirmatory_v6.freeze.json",
            "results/confirmatory/synthetic_v5/confirmatory_metrics.json",
            "results/confirmatory/synthetic_v5/T08_DVE_BASELINES.csv",
            "results/confirmatory/synthetic_v5/T09_DVE_ABLATION.csv",
        ),
    )
    add(
        "final_compute_benchmarks",
        lambda: (
            json.loads(
                (root / "results/aggregated/final_compute_benchmarks.json").read_text(
                    encoding="utf-8"
                )
            )
            if resume
            and (root / "results/aggregated/final_compute_benchmarks.json").exists()
            and json.loads(
                (root / "results/aggregated/final_compute_benchmarks.json").read_text(
                    encoding="utf-8"
                )
            ).get("profile")
            == profile
            else run_final_compute_benchmarks(profile, resolved_device)
        ),
        (
            "results/aggregated/final_compute_benchmarks.json",
            "results/aggregated/T12_CPU_GPU_PERFORMANCE.csv",
        ),
    )
    add(
        "experiment_reconciliation",
        lambda: reconcile_experiment_registry(root),
        ("experiments/registry.yaml", "artifacts/experiment_reconciliation.json"),
    )

    def require_final_evidence() -> dict[str, Any]:
        verification = verify_final_evidence(root)
        if verification["status"] != "PASS":
            raise RuntimeError(f"final test/review evidence is stale or unresolved: {verification}")
        return verification

    add(
        "final_verification_review_precheck",
        require_final_evidence,
        (
            "artifacts/tests/final_test_summary.json",
            "artifacts/review/final_review.json",
            "reports/FINAL_REVIEW.md",
        ),
    )

    def generate_report_stage() -> dict[str, Any]:
        return generate_reports()

    add(
        "reports",
        generate_report_stage,
        ("reports/TECHNICAL_REPORT.md",),
    )
    gate_payload = write_gate_evidence(root)
    _write_registry(registry_path, results)
    return {
        "status": "PIPELINE_EXECUTED",
        "profile": profile,
        "requested_device": device,
        "resolved_device": resolved_device,
        "resume": resume,
        "stage_count": len(results),
        "stage_results": [asdict(result) for result in results],
        "gate_summary": gate_payload["summary"],
    }
