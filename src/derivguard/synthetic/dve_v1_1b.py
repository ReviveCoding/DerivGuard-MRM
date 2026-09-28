"""Calibration-qualified remediation of the stopped DVE v1.1 study.

The first v1.1 design stopped on VALIDATION before locked generation.  This
module changes only bootstrap optimizer budget, uses disjoint seeds, and keeps
the scientific hypotheses and transparent aggregation rule unchanged.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from hashlib import sha256
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd

from derivguard.synthetic.confirmatory import SeedRegistry, qualify_confirmatory_engines
from derivguard.synthetic.confirmatory_v3 import (
    FAULT_IDS,
    FAULT_SEVERITIES,
    IMPLEMENTATION_LABELS,
    _truth_family,
)
from derivguard.synthetic.confirmatory_v4 import PARAMETER_BOUNDS, qualify_v4_calibration_engine
from derivguard.synthetic.confirmatory_v5 import CalibrationUnqualifiedError, Partition
from derivguard.synthetic.confirmatory_v6 import ConfirmatoryV6Config, _make_v6_case
from derivguard.synthetic.dve_v1_1 import (
    BASELINES,
    CANDIDATES,
    _bootstrap_metrics,
    _dev_scales,
    _fit_thresholds,
    _flatten,
    _metrics,
    _threshold_series,
    add_scores,
)


@dataclass(frozen=True)
class DVEV11BConfig(ConfirmatoryV6Config):
    profile: str = "research"
    device: str = "cuda"
    target_fpr: float = 0.05
    maximum_validation_fpr: float = 0.075
    cases_dev: int = 34
    cases_validation: int = 17
    cases_locked: int = 34
    cases_ood: int = 17
    bootstrap_multistarts: int = 8
    bootstrap_max_iterations: int = 240
    inference_seed: int = 20261103


def _hash(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, payload: object, *, exclusive: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x" if exclusive else "w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, default=str)
        handle.write("\n")


def _seeds(config: DVEV11BConfig) -> SeedRegistry:
    registry = SeedRegistry(
        dev=tuple(121_000_001 + index for index in range(config.cases_dev)),
        validation=tuple(122_000_001 + index for index in range(config.cases_validation)),
        locked=tuple(123_000_001 + index for index in range(config.cases_locked)),
        ood=tuple(124_000_001 + index for index in range(config.cases_ood)),
        observation_offset=50_000_000,
    )
    registry.validate()
    return registry


def _fault(index: int) -> str:
    return str(FAULT_IDS[index % len(FAULT_IDS)])


def prequalify(root: Path, config: DVEV11BConfig) -> dict[str, Any]:
    path = root / "artifacts/v1_1/dve_v1_1b_prequalification.json"
    identity = {
        "module_hash": _hash(Path(__file__)),
        "config_hash": sha256(json.dumps(asdict(config), sort_keys=True).encode()).hexdigest(),
        "v1_1_failure_hash": _hash(root / "results/v1_1/dve_v1_1/validation_records.failure.json"),
    }
    if path.exists():
        payload = cast(dict[str, Any], json.loads(path.read_text(encoding="utf-8")))
        if payload.get("status") != "PASS" or any(
            payload.get(key) != value for key, value in identity.items()
        ):
            raise RuntimeError("v1.1b prequalification is stale or failed")
        return payload
    cases = (
        (0, 116_000_001),
        (1, 116_000_002),
        (2, 116_000_003),
        (6, 116_000_007),
        (16, 116_000_017),
    )
    rows: list[dict[str, object]] = []
    for index, seed in cases:
        started = time.perf_counter()
        try:
            record, _, _ = _make_v6_case("DEV", index, seed, config)
            rows.append(
                {
                    "seed": seed,
                    "fault_id": _fault(index),
                    "status": "PASS",
                    "normalized_true_error": record.normalized_true_error,
                    "runtime_seconds": time.perf_counter() - started,
                }
            )
        except CalibrationUnqualifiedError as exc:
            rows.append(
                {
                    "seed": seed,
                    "fault_id": _fault(index),
                    "status": "FAIL",
                    "reason": str(exc),
                    "runtime_seconds": time.perf_counter() - started,
                }
            )
    pd.DataFrame(rows).to_csv(root / "results/v1_1/dve_v1_1b_prequalification.csv", index=False)
    payload = {
        "status": "PASS" if all(row["status"] == "PASS" for row in rows) else "FAIL",
        **identity,
        "cases": rows,
        "v1_1_locked_evaluated": False,
    }
    _write_json(path, payload)
    if payload["status"] != "PASS":
        raise RuntimeError("v1.1b optimizer prequalification failed")
    return payload


def freeze_design(root: Path, config: DVEV11BConfig) -> dict[str, Any]:
    path = root / "artifacts/v1_1/dve_v1_1b_design.freeze.json"
    prequalification = prequalify(root, config)
    hypotheses = root / "governance/dve_v1_1b_hypothesis_registry.yaml"
    experiment = root / "experiments/dve_v1_1b_registry.yaml"
    release = root / "artifacts/model_releases/model_release_v1.1.json"
    identity = {
        "module_hash": _hash(Path(__file__)),
        "hypothesis_registry_hash": _hash(hypotheses),
        "experiment_registry_hash": _hash(experiment),
        "developer_release_hash": _hash(release),
        "prequalification_hash": _hash(root / "artifacts/v1_1/dve_v1_1b_prequalification.json"),
        "config_hash": sha256(json.dumps(asdict(config), sort_keys=True).encode()).hexdigest(),
    }
    if path.exists():
        existing = cast(dict[str, Any], json.loads(path.read_text(encoding="utf-8")))
        if any(existing.get(key) != value for key, value in identity.items()):
            raise ValueError("v1.1b source or registry changed after freeze")
        return existing
    truth = qualify_confirmatory_engines(config.device)
    calibration = qualify_v4_calibration_engine(config.device)
    if any(item.status != "PASS" for item in truth) or calibration["status"] != "PASS":
        raise RuntimeError("v1.1b engine qualification failed")
    seeds = _seeds(config)
    schedule = []
    for partition in cast(
        tuple[Partition, ...], ("DEV", "VALIDATION", "LOCKED_TEST", "OOD_LOCKED_TEST")
    ):
        for index, seed in enumerate(seeds.for_partition(partition)):
            fault = _fault(index)
            schedule.append(
                {
                    "partition": partition,
                    "seed": seed,
                    "observation_seed": seed + 50_000_000,
                    "calibration_seed": seed + 60_000_000,
                    "fault_id": fault,
                    "severity": FAULT_SEVERITIES[cast(Any, fault)],
                    "truth_family": _truth_family(cast(Any, fault)),
                    "implementation_label": IMPLEMENTATION_LABELS[cast(Any, fault)],
                }
            )
    payload: dict[str, Any] = {
        "schema_version": "1.1b",
        "status": "FROZEN_BEFORE_NEW_DEV_GENERATION",
        "created_utc": datetime.now(UTC).isoformat(),
        **identity,
        "config": asdict(config),
        "seed_registry": asdict(seeds),
        "fault_schedule": schedule,
        "synthetic_parameter_ranges": {"heston_calibration_bounds": PARAMETER_BOUNDS},
        "materiality_definition": "median absolute truth error / observed spread >= 1.0",
        "aggregation_candidates": list(CANDIDATES),
        "selection_rule": (
            "VALIDATION FPR<=0.075; maximize recall, minimize |FPR-0.05|, "
            "then preregistered candidate order"
        ),
        "remediation_change_only": "bootstrap O01 from 5x160 to 8x240",
        "v1_1_locked_evaluated": False,
        "truth_engine_qualification": [asdict(item) for item in truth],
        "calibration_engine_qualification": calibration,
        "prequalification": prequalification,
    }
    payload["freeze_hash"] = sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
    _write_json(path, payload, exclusive=True)
    return payload


def _generate(
    root: Path, partition: Partition, config: DVEV11BConfig, freeze: dict[str, Any]
) -> pd.DataFrame:
    directory = root / "results/v1_1/dve_v1_1b"
    output = directory / f"{partition.lower()}_records.parquet"
    diagnostics_path = directory / f"{partition.lower()}_calibration_diagnostics.parquet"
    provenance_path = directory / f"{partition.lower()}_provenance.parquet"
    if output.exists() and diagnostics_path.exists() and provenance_path.exists():
        return pd.read_parquet(output)
    directory.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, object]] = []
    diagnostics: list[dict[str, object]] = []
    provenance: list[dict[str, object]] = []
    for index, seed in enumerate(_seeds(config).for_partition(partition)):
        try:
            record, source, case_diagnostics = _make_v6_case(partition, index, seed, config)
        except CalibrationUnqualifiedError as exc:
            diagnostics.extend(exc.diagnostics)
            pd.DataFrame(diagnostics).to_parquet(diagnostics_path, index=False)
            failure = {
                "status": "CALIBRATION_UNQUALIFIED",
                "partition": partition,
                "seed": seed,
                "fault_id": _fault(index),
                "reason": str(exc),
                "freeze_hash": freeze["freeze_hash"],
            }
            _write_json(output.with_suffix(".failure.json"), failure)
            raise RuntimeError(f"v1.1b calibration unqualified: {failure}") from exc
        flat = _flatten(record)
        flat["case_id"] = str(flat["case_id"]).replace("V1_1-", "V1_1B-", 1)
        source["case_id"] = flat["case_id"]
        source["design_freeze_hash"] = freeze["freeze_hash"]
        for item in case_diagnostics:
            item["case_id"] = flat["case_id"]
        records.append(flat)
        diagnostics.extend(case_diagnostics)
        provenance.append(source)
        pd.DataFrame(records).to_parquet(output.with_suffix(".partial.parquet"), index=False)
    pd.DataFrame(records).to_parquet(output, index=False)
    pd.DataFrame(diagnostics).to_parquet(diagnostics_path, index=False)
    pd.DataFrame(provenance).to_parquet(provenance_path, index=False)
    output.with_suffix(".partial.parquet").unlink(missing_ok=True)
    return pd.DataFrame(records)


def _select(
    root: Path,
    dev_raw: pd.DataFrame,
    validation_raw: pd.DataFrame,
    config: DVEV11BConfig,
    design: dict[str, Any],
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
    path = root / "artifacts/v1_1/dve_v1_1b_aggregation.freeze.json"
    if path.exists():
        frozen = cast(dict[str, Any], json.loads(path.read_text(encoding="utf-8")))
        scales = cast(dict[str, float], frozen["evidence_scales"])
        chosen = str(frozen["selected_candidate"])
        return (
            frozen,
            add_scores(dev_raw, scales, chosen),
            add_scores(validation_raw, scales, chosen),
        )
    scales = _dev_scales(dev_raw)
    dev = add_scores(dev_raw, scales)
    validation = add_scores(validation_raw, scales)
    candidate_thresholds = _fit_thresholds(dev, CANDIDATES, config.target_fpr)
    diagnostics = pd.DataFrame(
        [
            {"candidate": name, **_metrics(validation, name, candidate_thresholds[name])}
            for name in CANDIDATES
        ]
    )
    eligible = diagnostics[diagnostics["false_positive_rate"] <= config.maximum_validation_fpr]
    pool = eligible if not eligible.empty else diagnostics
    pool = pool.assign(
        fpr_distance=(pool["false_positive_rate"] - config.target_fpr).abs(),
        priority=pool["candidate"].map({name: index for index, name in enumerate(CANDIDATES)}),
    )
    sort_columns = (
        ["recall", "fpr_distance", "priority"]
        if not eligible.empty
        else ["fpr_distance", "recall", "priority"]
    )
    ascending = [False, True, True] if not eligible.empty else [True, False, True]
    chosen = str(pool.sort_values(sort_columns, ascending=ascending).iloc[0]["candidate"])
    dev = add_scores(dev_raw, scales, chosen)
    validation = add_scores(validation_raw, scales, chosen)
    simple = ("FitOnly", "ChallengerOnly", "BootstrapOnly", "NumericalOnly", "NonRegimeDVE")
    baseline_thresholds = _fit_thresholds(dev, simple, config.target_fpr)
    global_full = float(candidate_thresholds[cast(Any, chosen)])
    negative = dev[~dev["material_error"]]
    regime_thresholds = {}
    for regime in sorted(dev["regime"].unique()):
        values = negative[negative["regime"] == regime]["FullV1_1DVE"]
        regime_thresholds[str(regime)] = (
            float(np.quantile(values, 0.95, method="higher"))
            if len(values) >= config.minimum_regime_negatives
            else global_full
        )
    payload: dict[str, Any] = {
        "status": "FROZEN_BEFORE_FRESH_LOCKED_GENERATION",
        "created_utc": datetime.now(UTC).isoformat(),
        "design_freeze_hash": design["freeze_hash"],
        "evidence_scales": scales,
        "candidate_thresholds": candidate_thresholds,
        "validation_diagnostics": diagnostics.to_dict("records"),
        "selected_candidate": chosen,
        "baseline_thresholds": baseline_thresholds,
        "full_global_threshold": global_full,
        "full_regime_thresholds": regime_thresholds,
        "locked_results_used": False,
    }
    payload["freeze_hash"] = sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
    _write_json(path, payload, exclusive=True)
    diagnostics.to_csv(
        root / "results/v1_1/dve_v1_1b/validation_candidate_selection.csv", index=False
    )
    return payload, dev, validation


def run(root: Path = Path("."), config: DVEV11BConfig | None = None) -> dict[str, Any]:
    cfg = config or DVEV11BConfig()
    status_path = root / "results/v1_1/dve_v1_1b/confirmatory_metrics.json"
    if status_path.exists():
        return cast(dict[str, Any], json.loads(status_path.read_text(encoding="utf-8")))
    design = freeze_design(root, cfg)
    dev_raw = _generate(root, "DEV", cfg, design)
    validation_raw = _generate(root, "VALIDATION", cfg, design)
    aggregation, dev, validation = _select(root, dev_raw, validation_raw, cfg, design)
    pd.DataFrame(
        [
            {
                "partition": "VALIDATION",
                **_metrics(validation, name, _threshold_series(validation, aggregation, name)),
            }
            for name in BASELINES
        ]
    ).to_csv(root / "results/v1_1/dve_v1_1b/validation_baselines.csv", index=False)
    locked_raw = _generate(root, "LOCKED_TEST", cfg, design)
    ood_raw = _generate(root, "OOD_LOCKED_TEST", cfg, design)
    selected = str(aggregation["selected_candidate"])
    scales = cast(dict[str, float], aggregation["evidence_scales"])
    locked, ood = add_scores(locked_raw, scales, selected), add_scores(ood_raw, scales, selected)
    locked.to_parquet(root / "results/v1_1/dve_v1_1b/locked_scored.parquet", index=False)
    ood.to_parquet(root / "results/v1_1/dve_v1_1b/ood_locked_scored.parquet", index=False)
    rows: list[dict[str, object]] = []
    for partition, frame in (
        ("LOCKED_TEST", locked),
        ("OOD_LOCKED_TEST", ood),
        ("COMBINED", pd.concat([locked, ood], ignore_index=True)),
    ):
        for index, name in enumerate(BASELINES):
            threshold = _threshold_series(frame, aggregation, name)
            rows.append(
                {
                    "partition": partition,
                    **_metrics(frame, name, threshold),
                    **_bootstrap_metrics(
                        frame.reset_index(drop=True),
                        name,
                        threshold.reset_index(drop=True),
                        cfg.inference_seed + index + len(rows) * 31,
                    ),
                }
            )
    metrics = pd.DataFrame(rows)
    metrics.to_csv(root / "results/v1_1/dve_v1_1b/locked_baseline_metrics.csv", index=False)
    full_threshold = _threshold_series(locked, aggregation, "FullV1_1DVE")
    prediction = locked["FullV1_1DVE"].to_numpy() > full_threshold.to_numpy()
    locked[prediction & ~locked["material_error"]].to_parquet(
        root / "results/v1_1/dve_v1_1b/false_positive_cases.parquet", index=False
    )
    locked[~prediction & locked["material_error"]].to_parquet(
        root / "results/v1_1/dve_v1_1b/false_negative_cases.parquet", index=False
    )
    payload = {
        "status": "CONFIRMATORY_LOCKED_COMPLETE",
        "created_utc": datetime.now(UTC).isoformat(),
        "design_freeze_hash": design["freeze_hash"],
        "aggregation_freeze_hash": aggregation["freeze_hash"],
        "selected_transparent_aggregation": selected,
        "fresh_seed_regions": ["121m", "122m", "123m", "124m"],
        "v1_or_v1_1_locked_observations_reused": False,
        "case_counts": {
            "DEV": len(dev),
            "VALIDATION": len(validation),
            "LOCKED_TEST": len(locked),
            "OOD_LOCKED_TEST": len(ood),
        },
        "combined_locked_metrics": metrics[metrics["partition"] == "COMBINED"].to_dict("records"),
    }
    _write_json(status_path, payload)
    return payload
