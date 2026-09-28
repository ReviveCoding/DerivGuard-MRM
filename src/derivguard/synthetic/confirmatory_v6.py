"""Confirmatory v6 after the preserved v5 pre-lock bootstrap failure.

V5 stopped on a VALIDATION Bates case before any locked seed was generated.
V6 changes only the bootstrap qualification budget, uses wholly new study
seeds, and retains the same observable C04/O01 developer methodology.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, replace
from datetime import UTC, datetime
from hashlib import sha256
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd

from derivguard.synthetic.confirmatory import (
    ABLATION_SPECS,
    ConfirmatoryRunResult,
    SeedRegistry,
    evaluate_confirmatory_ablations,
    qualify_confirmatory_engines,
)
from derivguard.synthetic.confirmatory_v3 import (
    FAULT_IDS,
    FAULT_SEVERITIES,
    IMPLEMENTATION_LABELS,
    _persist_v3,
    _truth_family,
)
from derivguard.synthetic.confirmatory_v4 import (
    PARAMETER_BOUNDS,
    qualify_v4_calibration_engine,
)
from derivguard.synthetic.confirmatory_v5 import (
    CalibrationUnqualifiedError,
    ConfirmatoryV5Config,
    Partition,
)
from derivguard.synthetic.confirmatory_v5 import (
    _make_case as _make_v5_case,
)
from derivguard.validation.dve import (
    BASELINE_NAMES,
    EvidenceRecord,
    evaluate_locked,
    fit_dev_thresholds,
)


@dataclass(frozen=True)
class ConfirmatoryV6Config(ConfirmatoryV5Config):
    """Frozen v6 design with validation-driven bootstrap remediation."""

    bootstrap_multistarts: int = 5
    bootstrap_max_iterations: int = 160


def _hash(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def _config_hash(config: ConfirmatoryV6Config) -> str:
    return sha256(json.dumps(asdict(config), sort_keys=True).encode()).hexdigest()


def _fault(index: int) -> str:
    return str(FAULT_IDS[index % len(FAULT_IDS)])


def build_v6_seed_registry(config: ConfirmatoryV6Config) -> SeedRegistry:
    registry = SeedRegistry(
        dev=tuple(107_000_001 + index for index in range(config.cases_dev)),
        validation=tuple(108_000_001 + index for index in range(config.cases_validation)),
        locked=tuple(109_000_001 + index for index in range(config.cases_locked)),
        ood=tuple(110_000_001 + index for index in range(config.cases_ood)),
        observation_offset=50_000_000,
    )
    registry.validate()
    return registry


def _make_v6_case(
    partition: Partition,
    index: int,
    seed: int,
    config: ConfirmatoryV6Config,
) -> tuple[EvidenceRecord, dict[str, object], list[dict[str, object]]]:
    try:
        record, provenance, diagnostics = _make_v5_case(partition, index, seed, config)
    except CalibrationUnqualifiedError as exc:
        for item in exc.diagnostics:
            item["case_id"] = str(item.get("case_id", "")).replace("V5-", "V6-", 1)
        raise CalibrationUnqualifiedError(str(exc), exc.diagnostics) from exc
    old_id = record.case_id
    case_id = old_id.replace("V5-", "V6-", 1)
    record = replace(record, case_id=case_id)
    provenance["case_id"] = case_id
    for item in diagnostics:
        if item.get("case_id") == old_id:
            item["case_id"] = case_id
    return record, provenance, diagnostics


def prequalify_confirmatory_v6_calibration(
    profile: str = "research",
    device: str = "cuda",
    *,
    root: Path = Path("."),
    resume: bool = True,
) -> dict[str, Any]:
    """Exercise exact v6 budgets on a seed region disjoint from all studies."""

    config = ConfirmatoryV6Config(profile=profile, device=device)
    config.validate()
    release = root / "artifacts/model_releases/model_release_v1.1.json"
    if not release.is_file():
        raise FileNotFoundError("Developer Model Release v1.1 is required")
    identity = {
        "module_hash": _hash(Path(__file__)),
        "v5_dependency_hash": _hash(Path(__file__).with_name("confirmatory_v5.py")),
        "config_hash": _config_hash(config),
        "developer_release_hash": _hash(release),
    }
    status_path = root / "artifacts/calibration/confirmatory_v6_prequalification.json"
    if resume and status_path.is_file():
        existing = cast(dict[str, Any], json.loads(status_path.read_text(encoding="utf-8")))
        if all(existing.get(key) == value for key, value in identity.items()):
            if existing.get("status") != "PASS":
                raise RuntimeError("existing v6 prequalification did not pass")
            return existing

    tuning_cases = (
        (0, 101_000_001),
        (1, 101_000_002),
        (2, 101_000_003),
        (9, 101_000_010),
        (16, 101_000_017),
    )
    official_registry = build_v6_seed_registry(config)
    official = set(
        official_registry.dev
        + official_registry.validation
        + official_registry.locked
        + official_registry.ood
    )
    if any(seed in official for _, seed in tuning_cases):
        raise RuntimeError("v6 prequalification seed overlaps an official partition")
    started = time.perf_counter()
    rows: list[dict[str, object]] = []
    diagnostics: list[dict[str, object]] = []
    for fault_index, seed in tuning_cases:
        fault_id = _fault(fault_index)
        case_started = time.perf_counter()
        try:
            record, _, case_diagnostics = _make_v6_case("DEV", fault_index, seed, config)
            diagnostics.extend(case_diagnostics)
            rows.append(
                {
                    "seed": seed,
                    "fault_id": fault_id,
                    "truth_family": _truth_family(cast(Any, fault_id)),
                    "status": "PASS",
                    "selected_fit_qualified": True,
                    "runtime_seconds": time.perf_counter() - case_started,
                    "normalized_true_error": record.normalized_true_error,
                }
            )
        except CalibrationUnqualifiedError as exc:
            diagnostics.extend(exc.diagnostics)
            rows.append(
                {
                    "seed": seed,
                    "fault_id": fault_id,
                    "truth_family": _truth_family(cast(Any, fault_id)),
                    "status": "FAIL",
                    "selected_fit_qualified": False,
                    "runtime_seconds": time.perf_counter() - case_started,
                    "reason": str(exc),
                }
            )
    aggregate = root / "results/aggregated/confirmatory_v6_prequalification.csv"
    diagnostics_path = root / "results/raw/confirmatory_v6_prequalification_diagnostics.parquet"
    status_path.parent.mkdir(parents=True, exist_ok=True)
    aggregate.parent.mkdir(parents=True, exist_ok=True)
    diagnostics_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(aggregate, index=False)
    pd.DataFrame(diagnostics).to_parquet(diagnostics_path, index=False)
    passed = bool(rows) and all(row["status"] == "PASS" for row in rows)
    payload: dict[str, Any] = {
        "schema_version": "1.0",
        "status": "PASS" if passed else "FAIL",
        "study_role": "PRE_LOCK_TUNING_QUALIFICATION_NOT_INFERENTIAL",
        **identity,
        "v5_validation_failure_used_for_remediation": True,
        "v5_locked_results_used": False,
        "tuning_seed_region": "101m",
        "official_seed_regions": ["107m", "108m", "109m", "110m"],
        "seed_overlap": False,
        "cases_attempted": len(rows),
        "cases_passed": sum(row["status"] == "PASS" for row in rows),
        "faults_attempted": [str(row["fault_id"]) for row in rows],
        "runtime_seconds": time.perf_counter() - started,
        "summary_path": aggregate.relative_to(root).as_posix(),
        "diagnostics_path": diagnostics_path.relative_to(root).as_posix(),
    }
    status_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if not passed:
        raise RuntimeError("v6 optimizer prequalification failed; design remains unfrozen")
    return payload


def freeze_confirmatory_v6(
    path: Path = Path("artifacts/checkpoints/synthetic_confirmatory_v6.freeze.json"),
    config: ConfirmatoryV6Config | None = None,
    hypothesis_registry: Path = Path("governance/confirmatory_v6_hypothesis_registry.yaml"),
    experiment_registry: Path = Path("experiments/confirmatory_v6_registry.yaml"),
    developer_release: Path = Path("artifacts/model_releases/model_release_v1.1.json"),
    prequalification: Path = Path("artifacts/calibration/confirmatory_v6_prequalification.json"),
) -> dict[str, Any]:
    cfg = config or ConfirmatoryV6Config()
    cfg.validate()
    module_hash = _hash(Path(__file__))
    dependency_hash = _hash(Path(__file__).with_name("confirmatory_v5.py"))
    config_hash = _config_hash(cfg)
    if not developer_release.is_file() or not prequalification.is_file():
        raise FileNotFoundError("v1.1 release and passing v6 prequalification are required")
    prequalification_payload = json.loads(prequalification.read_text(encoding="utf-8"))
    expected_prequalification = {
        "status": "PASS",
        "module_hash": module_hash,
        "v5_dependency_hash": dependency_hash,
        "config_hash": config_hash,
        "developer_release_hash": _hash(developer_release),
    }
    if any(
        prequalification_payload.get(key) != value
        for key, value in expected_prequalification.items()
    ):
        raise ValueError("v6 prequalification is stale, failed, or inconsistent")
    immutable = {
        "hypothesis_registry_hash": _hash(hypothesis_registry),
        "experiment_registry_hash": _hash(experiment_registry),
        "developer_release_hash": _hash(developer_release),
        "calibration_prequalification_hash": _hash(prequalification),
    }
    if path.exists():
        existing = cast(dict[str, Any], json.loads(path.read_text(encoding="utf-8")))
        expected = {
            "module_hash": module_hash,
            "v5_dependency_hash": dependency_hash,
            "config_hash": config_hash,
            **immutable,
        }
        if any(existing.get(key) != value for key, value in expected.items()):
            raise ValueError("v6 source, configuration, release, or registry changed after freeze")
        return existing

    release = json.loads(developer_release.read_text(encoding="utf-8"))
    if (
        release.get("release_id") != "Developer Model Release v1.1"
        or release.get("objective") != "C04"
        or release.get("optimizer") != "O01"
    ):
        raise ValueError("v6 requires frozen v1.1 C04/O01")
    truth_qualification = qualify_confirmatory_engines(cfg.device)
    calibration_qualification = qualify_v4_calibration_engine(cfg.device)
    if any(item.status != "PASS" for item in truth_qualification):
        raise RuntimeError("truth-engine qualification failed")
    if calibration_qualification["status"] != "PASS":
        raise RuntimeError("calibration-pricer qualification failed")

    seeds = build_v6_seed_registry(cfg)
    schedule: list[dict[str, object]] = []
    partitions: tuple[Partition, ...] = (
        "DEV",
        "VALIDATION",
        "LOCKED_TEST",
        "OOD_LOCKED_TEST",
    )
    for partition in partitions:
        for index, seed in enumerate(seeds.for_partition(partition)):
            fault_id = _fault(index)
            schedule.append(
                {
                    "partition": partition,
                    "seed": seed,
                    "observation_seed": seed + seeds.observation_offset,
                    "calibration_seed": seed + 60_000_000,
                    "fault_id": fault_id,
                    "severity": FAULT_SEVERITIES[cast(Any, fault_id)],
                    "truth_family": _truth_family(cast(Any, fault_id)),
                    "implementation_label": IMPLEMENTATION_LABELS[cast(Any, fault_id)],
                    "proxy_fault": fault_id not in {"F00", "F01", "F02", "F03"},
                }
            )
    calibration_design = {
        "developer_release": "Developer Model Release v1.1",
        "objective": "C04",
        "optimizer": "O01 true seeded multistart L-BFGS-B",
        "bounds": PARAMETER_BOUNDS,
        "primary_multistarts": cfg.primary_multistarts,
        "primary_max_iterations": cfg.primary_max_iterations,
        "bootstrap_draws": cfg.bootstrap_draws,
        "bootstrap_multistarts": cfg.bootstrap_multistarts,
        "bootstrap_max_iterations": cfg.bootstrap_max_iterations,
        "qualification": "finite optimizer success; otherwise CALIBRATION_UNQUALIFIED",
        "inputs": "observable bid/mid/ask only",
        "remediation_basis": "v5 VALIDATION bootstrap nonconvergence; no v5 locked seed evaluated",
    }
    dve = {
        "dimensions": [
            "E_data",
            "E_num",
            "E_cal",
            "E_ident",
            "E_param",
            "E_form",
            "E_greek",
            "E_extra",
            "E_outcome",
        ],
        "target_fpr": cfg.target_fpr,
        "materiality_threshold_spread_units": cfg.materiality_threshold,
        "baselines": list(BASELINE_NAMES),
        "ablations": dict(ABLATION_SPECS),
    }
    seed_payload = asdict(seeds)
    payload: dict[str, Any] = {
        "schema_version": "6.0",
        "status": "FROZEN_NOT_EXECUTED",
        "created_utc": datetime.now(UTC).isoformat(),
        "release_id": str(release["release_id"]),
        "module_hash": module_hash,
        "v5_dependency_hash": dependency_hash,
        "config": asdict(cfg),
        "config_hash": config_hash,
        "seed_registry": seed_payload,
        "seed_hash": sha256(json.dumps(seed_payload, sort_keys=True).encode()).hexdigest(),
        "fault_schedule": schedule,
        "schedule_hash": sha256(json.dumps(schedule, sort_keys=True).encode()).hexdigest(),
        "calibration_design": calibration_design,
        "calibration_design_hash": sha256(
            json.dumps(calibration_design, sort_keys=True).encode()
        ).hexdigest(),
        "dve_formulation": dve,
        "dve_formulation_hash": sha256(json.dumps(dve, sort_keys=True).encode()).hexdigest(),
        "hypothesis_registry_path": hypothesis_registry.as_posix(),
        "experiment_registry_path": experiment_registry.as_posix(),
        "developer_release_path": developer_release.as_posix(),
        "calibration_prequalification_path": prequalification.as_posix(),
        **immutable,
        "truth_engine_qualification": [asdict(item) for item in truth_qualification],
        "calibration_engine_qualification": calibration_qualification,
        "prior_v4_or_v5_locked_outcomes_used_for_selection": False,
        "v5_validation_failure_used_for_remediation": True,
        "locked_tuning_prohibited": True,
    }
    payload["freeze_hash"] = sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
    if _hash(Path(__file__)) != module_hash:
        raise RuntimeError("v6 source changed during freeze")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return payload


def run_confirmatory_v6_study(
    profile: str = "research", device: str = "cuda", resume: bool = True
) -> dict[str, Any]:
    output = Path("results/confirmatory/synthetic_v5")
    status_path = output / "confirmatory_metrics.json"
    if resume and status_path.exists():
        return cast(dict[str, Any], json.loads(status_path.read_text(encoding="utf-8")))
    config = ConfirmatoryV6Config(profile=profile, device=device)
    frozen = freeze_confirmatory_v6(config=config)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("v6 output namespace is non-empty")
    output.mkdir(parents=True, exist_ok=True)
    seeds = build_v6_seed_registry(config)
    records: list[EvidenceRecord] = []
    provenance: list[dict[str, object]] = []
    diagnostics: list[dict[str, object]] = []

    def generate(partition: Partition) -> bool:
        for index, seed in enumerate(seeds.for_partition(partition)):
            try:
                record, source, case_diagnostics = _make_v6_case(partition, index, seed, config)
            except CalibrationUnqualifiedError as exc:
                diagnostics.extend(exc.diagnostics)
                pd.DataFrame(diagnostics).to_parquet(
                    output / "calibration_diagnostics.parquet", index=False
                )
                failure = {
                    "status": "CALIBRATION_UNQUALIFIED",
                    "partition": partition,
                    "seed": seed,
                    "fault_id": _fault(index),
                    "reason": str(exc),
                    "freeze_hash": frozen["freeze_hash"],
                    "developer_release_id": frozen["calibration_design"]["developer_release"],
                    "developer_release_hash": frozen["developer_release_hash"],
                    "locked_evaluation_complete": False,
                }
                status_path.write_text(json.dumps(failure, indent=2, sort_keys=True) + "\n")
                return False
            source["design_freeze_hash"] = frozen["freeze_hash"]
            records.append(record)
            provenance.append(source)
            diagnostics.extend(case_diagnostics)
        return True

    if not generate("DEV") or not generate("VALIDATION"):
        return cast(dict[str, Any], json.loads(status_path.read_text(encoding="utf-8")))
    dev = tuple(item for item in records if item.partition == "DEV")
    thresholds = fit_dev_thresholds(
        dev,
        target_fpr=config.target_fpr,
        minimum_regime_negatives=config.minimum_regime_negatives,
    )
    validation = tuple(
        replace(item, partition="LOCKED_TEST") for item in records if item.partition == "VALIDATION"
    )
    validation_metrics = {
        key: asdict(value) for key, value in evaluate_locked(validation, thresholds).items()
    }
    threshold_payload = {
        "status": "FROZEN_BEFORE_LOCKED_GENERATION",
        "frozen_utc": datetime.now(UTC).isoformat(),
        "design_freeze_hash": frozen["freeze_hash"],
        "thresholds": thresholds.to_dict(),
        "validation_diagnostics": validation_metrics,
    }
    with (output / "dve_thresholds.freeze.json").open("x", encoding="utf-8") as handle:
        json.dump(threshold_payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    if not generate("LOCKED_TEST") or not generate("OOD_LOCKED_TEST"):
        return cast(dict[str, Any], json.loads(status_path.read_text(encoding="utf-8")))
    locked = tuple(item for item in records if item.partition in {"LOCKED_TEST", "OOD_LOCKED_TEST"})
    baselines = evaluate_locked(locked, thresholds)
    ablations = evaluate_confirmatory_ablations(dev, locked, thresholds)
    result = ConfirmatoryRunResult(
        run_id=f"SYN-CONFIRMATORY-V6-{str(frozen['freeze_hash'])[:12]}",
        freeze_hash=str(frozen["freeze_hash"]),
        thresholds=thresholds,
        records=tuple(records),
        baselines=baselines,
        ablations=ablations,
        provenance=tuple(provenance),
    )
    _persist_v3(
        output,
        result,
        validation_metrics,
        frozen["fault_schedule"],
        frozen["truth_engine_qualification"],
    )
    diagnostic_frame = pd.DataFrame(diagnostics)
    diagnostic_frame.to_parquet(output / "calibration_diagnostics.parquet", index=False)
    qualification = {
        "status": "PASS",
        "cases": len(records),
        "all_selected_fits_qualified": True,
        "optimizer_start_attempts": int(np.sum(diagnostic_frame["start_index"] >= 0)),
        "converged_start_attempts": int(
            np.sum((diagnostic_frame["start_index"] >= 0) & diagnostic_frame["qualified"])
        ),
    }
    (output / "calibration_qualification.json").write_text(
        json.dumps(qualification, indent=2, sort_keys=True) + "\n"
    )
    status: dict[str, Any] = {
        "status": "CONFIRMATORY_LOCKED_COMPLETE",
        "run_id": result.run_id,
        "freeze_hash": result.freeze_hash,
        "developer_release_id": frozen["calibration_design"]["developer_release"],
        "developer_release_hash": frozen["developer_release_hash"],
        "locked_cases": len(locked),
        "calibration_qualification": "PASS",
        "faults_executed": list(FAULT_IDS),
        "baselines": {key: asdict(value) for key, value in baselines.items()},
        "ablations": {key: asdict(value) for key, value in ablations.items()},
    }
    status_path.write_text(json.dumps(status, indent=2, sort_keys=True) + "\n")
    return status
