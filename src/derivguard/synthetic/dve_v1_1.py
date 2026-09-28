"""Fresh, preregistered transparent DVE v1.1 calibration and locked study."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from hashlib import sha256
from pathlib import Path
from typing import Any, Literal, cast

import numpy as np
import pandas as pd
from numpy.typing import NDArray

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
from derivguard.validation.dve import EVIDENCE_NAMES, EvidenceRecord

FloatArray = NDArray[np.float64]
CandidateName = Literal[
    "maximum_evidence_percentile",
    "top_2_evidence_mean",
    "top_3_evidence_mean",
    "logical_two_high_components",
    "materiality_gated_top_2",
]

CANDIDATES: tuple[CandidateName, ...] = (
    "maximum_evidence_percentile",
    "top_2_evidence_mean",
    "top_3_evidence_mean",
    "logical_two_high_components",
    "materiality_gated_top_2",
)
BASELINES = (
    "FitOnly",
    "ChallengerOnly",
    "BootstrapOnly",
    "NumericalOnly",
    "NonRegimeDVE",
    "FullV1_1DVE",
)


@dataclass(frozen=True)
class DVEV11Config(ConfirmatoryV6Config):
    """Frozen v1.1 sample sizes and transparent aggregation controls."""

    profile: str = "research"
    device: str = "cuda"
    target_fpr: float = 0.05
    maximum_validation_fpr: float = 0.075
    cases_dev: int = 34
    cases_validation: int = 17
    cases_locked: int = 34
    cases_ood: int = 17
    inference_seed: int = 20261102


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


def _hash(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, payload: object, *, exclusive: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    mode = "x" if exclusive else "w"
    with path.open(mode, encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, default=str)
        handle.write("\n")


def build_seed_registry(config: DVEV11Config) -> SeedRegistry:
    registry = SeedRegistry(
        dev=tuple(117_000_001 + index for index in range(config.cases_dev)),
        validation=tuple(118_000_001 + index for index in range(config.cases_validation)),
        locked=tuple(119_000_001 + index for index in range(config.cases_locked)),
        ood=tuple(120_000_001 + index for index in range(config.cases_ood)),
        observation_offset=50_000_000,
    )
    registry.validate()
    return registry


def _fault(index: int) -> str:
    return str(FAULT_IDS[index % len(FAULT_IDS)])


def freeze_design(root: Path, config: DVEV11Config) -> dict[str, Any]:
    """Freeze hypotheses, experiment, seeds, ranges, and engines before generation."""

    path = root / "artifacts/v1_1/dve_v1_1_design.freeze.json"
    module = Path(__file__)
    hypotheses = root / "governance/dve_v1_1_hypothesis_registry.yaml"
    experiment = root / "experiments/dve_v1_1_registry.yaml"
    release = root / "artifacts/model_releases/model_release_v1.1.json"
    config.validate()
    identity = {
        "module_hash": _hash(module),
        "hypothesis_registry_hash": _hash(hypotheses),
        "experiment_registry_hash": _hash(experiment),
        "developer_release_hash": _hash(release),
        "config_hash": sha256(json.dumps(asdict(config), sort_keys=True).encode()).hexdigest(),
    }
    if path.exists():
        existing = cast(dict[str, Any], json.loads(path.read_text(encoding="utf-8")))
        if any(existing.get(key) != value for key, value in identity.items()):
            raise ValueError("DVE v1.1 source, registry, release, or config changed after freeze")
        return existing
    truth_qualification = qualify_confirmatory_engines(config.device)
    calibration_qualification = qualify_v4_calibration_engine(config.device)
    if any(item.status != "PASS" for item in truth_qualification):
        raise RuntimeError("fresh-study truth engine qualification failed")
    if calibration_qualification["status"] != "PASS":
        raise RuntimeError("fresh-study calibration engine qualification failed")
    seeds = build_seed_registry(config)
    old_seed_floor, old_seed_ceiling = 101_000_000, 111_000_000
    all_seeds = (*seeds.dev, *seeds.validation, *seeds.locked, *seeds.ood)
    if any(old_seed_floor <= seed < old_seed_ceiling for seed in all_seeds):
        raise RuntimeError("fresh v1.1 seed overlaps the preserved v1/v6 seed regions")
    schedule = []
    for partition in cast(
        tuple[Partition, ...], ("DEV", "VALIDATION", "LOCKED_TEST", "OOD_LOCKED_TEST")
    ):
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
                }
            )
    payload: dict[str, Any] = {
        "schema_version": "1.1",
        "status": "FROZEN_BEFORE_NEW_DEV_GENERATION",
        "created_utc": _utc_now(),
        **identity,
        "config": asdict(config),
        "seed_registry": asdict(seeds),
        "seed_overlap_with_v1": False,
        "fault_schedule": schedule,
        "synthetic_parameter_ranges": {
            "heston_calibration_bounds": PARAMETER_BOUNDS,
            "source": "inherited qualified confirmatory generator; unchanged",
        },
        "materiality_definition": "median absolute truth error / observed bid-ask spread >= 1.0",
        "aggregation_candidates": list(CANDIDATES),
        "selection_rule": (
            "On VALIDATION, retain candidates with FPR<=0.075; maximize recall, then "
            "minimize |FPR-0.05|, then preregistered order. If none qualify, minimize "
            "|FPR-0.05|, then maximize recall, then preregistered order."
        ),
        "truth_engine_qualification": [asdict(item) for item in truth_qualification],
        "calibration_engine_qualification": calibration_qualification,
        "locked_tuning_prohibited": True,
    }
    payload["freeze_hash"] = sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
    _write_json(path, payload, exclusive=True)
    return payload


def _flatten(record: EvidenceRecord) -> dict[str, object]:
    return {
        "case_id": record.case_id.replace("V6-", "V1_1-", 1),
        "partition": record.partition,
        "regime": record.regime,
        "fit_error": record.fit_error,
        "oos_error": record.oos_error,
        "material_error": record.material_error,
        "normalized_true_error": record.normalized_true_error,
        "fault_id": record.fault_id,
        "proxy_fault": record.proxy_fault,
        **record.evidence.as_dict(),
    }


def _generate_partition(
    root: Path,
    partition: Partition,
    config: DVEV11Config,
    frozen: dict[str, Any],
) -> pd.DataFrame:
    output = root / f"results/v1_1/dve_v1_1/{partition.lower()}_records.parquet"
    provenance_path = output.with_name(f"{partition.lower()}_provenance.parquet")
    diagnostics_path = output.with_name(f"{partition.lower()}_calibration_diagnostics.parquet")
    if output.exists() and provenance_path.exists() and diagnostics_path.exists():
        return pd.read_parquet(output)
    records: list[dict[str, object]] = []
    provenance: list[dict[str, object]] = []
    diagnostics: list[dict[str, object]] = []
    seeds = build_seed_registry(config).for_partition(partition)
    output.parent.mkdir(parents=True, exist_ok=True)
    for index, seed in enumerate(seeds):
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
                "design_freeze_hash": frozen["freeze_hash"],
            }
            _write_json(output.with_suffix(".failure.json"), failure)
            raise RuntimeError(f"fresh DVE v1.1 calibration unqualified: {failure}") from exc
        flat = _flatten(record)
        source["case_id"] = flat["case_id"]
        source["design_freeze_hash"] = frozen["freeze_hash"]
        for item in case_diagnostics:
            item["case_id"] = flat["case_id"]
        records.append(flat)
        provenance.append(source)
        diagnostics.extend(case_diagnostics)
        # Durable per-case progress: an interrupted partition can be audited even
        # though official resume restarts that incomplete partition.
        pd.DataFrame(records).to_parquet(output.with_suffix(".partial.parquet"), index=False)
    pd.DataFrame(records).to_parquet(output, index=False)
    pd.DataFrame(provenance).to_parquet(provenance_path, index=False)
    pd.DataFrame(diagnostics).to_parquet(diagnostics_path, index=False)
    output.with_suffix(".partial.parquet").unlink(missing_ok=True)
    return pd.DataFrame(records)


def _quantile_higher(values: pd.Series, probability: float) -> float:
    if values.empty:
        raise ValueError("threshold requires non-material DEV examples")
    return float(np.quantile(values.to_numpy(dtype=float), probability, method="higher"))


def _dev_scales(dev: pd.DataFrame) -> dict[str, float]:
    negative = dev[~dev["material_error"]]
    return {name: max(_quantile_higher(negative[name], 0.95), 1.0e-12) for name in EVIDENCE_NAMES}


def add_scores(
    frame: pd.DataFrame, scales: dict[str, float], selected: str | None = None
) -> pd.DataFrame:
    result = frame.copy()
    contributions = np.column_stack(
        [result[name].to_numpy(dtype=float) / scales[name] for name in EVIDENCE_NAMES]
    )
    ordered = np.sort(contributions, axis=1)[:, ::-1]
    result["maximum_evidence_percentile"] = ordered[:, 0]
    result["top_2_evidence_mean"] = ordered[:, :2].mean(axis=1)
    result["top_3_evidence_mean"] = ordered[:, :3].mean(axis=1)
    result["logical_two_high_components"] = np.sum(contributions >= 1.0, axis=1).astype(float)
    material_gate = (
        np.maximum(
            contributions[:, EVIDENCE_NAMES.index("E_form")],
            contributions[:, EVIDENCE_NAMES.index("E_greek")],
        )
        >= 1.0
    )
    result["materiality_gated_top_2"] = result["top_2_evidence_mean"] * material_gate
    result["FitOnly"] = result["fit_error"]
    result["ChallengerOnly"] = contributions[:, EVIDENCE_NAMES.index("E_form")]
    result["BootstrapOnly"] = contributions[:, EVIDENCE_NAMES.index("E_cal")]
    result["NumericalOnly"] = contributions[:, EVIDENCE_NAMES.index("E_num")]
    result["NonRegimeDVE"] = contributions.mean(axis=1)
    if selected is not None:
        result["FullV1_1DVE"] = result[selected]
    return result


def _auc(labels: FloatArray, scores: FloatArray) -> float:
    positive = labels == 1.0
    negative = ~positive
    if not np.any(positive) or not np.any(negative):
        return np.nan
    comparisons = scores[positive][:, None] - scores[negative][None, :]
    return float(np.mean((comparisons > 0.0) + 0.5 * (comparisons == 0.0)))


def _average_precision(labels: FloatArray, scores: FloatArray) -> float:
    positives = int(np.sum(labels))
    if positives == 0:
        return np.nan
    order = np.argsort(-scores, kind="mergesort")
    ranked = labels[order]
    precision = np.cumsum(ranked) / np.arange(1, len(ranked) + 1)
    return float(np.sum(precision * ranked) / positives)


def _metrics(
    frame: pd.DataFrame, score: str, threshold: float | pd.Series
) -> dict[str, float | int | str]:
    labels = frame["material_error"].to_numpy(dtype=bool)
    scores = frame[score].to_numpy(dtype=float)
    raw_threshold = (
        threshold.to_numpy(dtype=float) if isinstance(threshold, pd.Series) else threshold
    )
    predicted = scores > raw_threshold
    positives, negatives = int(labels.sum()), int((~labels).sum())
    tp, fp = int(np.sum(predicted & labels)), int(np.sum(predicted & ~labels))
    return {
        "baseline": score,
        "observations": len(frame),
        "positives": positives,
        "negatives": negatives,
        "recall": tp / positives if positives else np.nan,
        "false_positive_rate": fp / negatives if negatives else np.nan,
        "material_risk_miss_rate": 1.0 - tp / positives if positives else np.nan,
        "auroc": _auc(labels.astype(float), scores),
        "auprc": _average_precision(labels.astype(float), scores),
    }


def _fit_thresholds(
    dev: pd.DataFrame, names: tuple[str, ...], target_fpr: float
) -> dict[str, float]:
    negative = dev[~dev["material_error"]]
    return {name: _quantile_higher(negative[name], 1.0 - target_fpr) for name in names}


def select_aggregation(
    root: Path,
    dev_raw: pd.DataFrame,
    validation_raw: pd.DataFrame,
    config: DVEV11Config,
    design: dict[str, Any],
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
    path = root / "artifacts/v1_1/dve_v1_1_aggregation.freeze.json"
    if path.exists():
        frozen = cast(dict[str, Any], json.loads(path.read_text(encoding="utf-8")))
        scales = cast(dict[str, float], frozen["evidence_scales"])
        return (
            frozen,
            add_scores(dev_raw, scales, str(frozen["selected_candidate"])),
            add_scores(validation_raw, scales, str(frozen["selected_candidate"])),
        )
    scales = _dev_scales(dev_raw)
    dev = add_scores(dev_raw, scales)
    validation = add_scores(validation_raw, scales)
    thresholds = _fit_thresholds(dev, CANDIDATES, config.target_fpr)
    validation_rows = [
        {"candidate": name, **_metrics(validation, name, thresholds[name])} for name in CANDIDATES
    ]
    diagnostics = pd.DataFrame(validation_rows)
    eligible = diagnostics[diagnostics["false_positive_rate"] <= config.maximum_validation_fpr]
    if not eligible.empty:
        chosen = (
            eligible.assign(
                fpr_distance=(eligible["false_positive_rate"] - config.target_fpr).abs(),
                priority=eligible["candidate"].map(
                    {name: index for index, name in enumerate(CANDIDATES)}
                ),
            )
            .sort_values(["recall", "fpr_distance", "priority"], ascending=[False, True, True])
            .iloc[0]
        )
    else:
        chosen = (
            diagnostics.assign(
                fpr_distance=(diagnostics["false_positive_rate"] - config.target_fpr).abs(),
                priority=diagnostics["candidate"].map(
                    {name: index for index, name in enumerate(CANDIDATES)}
                ),
            )
            .sort_values(["fpr_distance", "recall", "priority"], ascending=[True, False, True])
            .iloc[0]
        )
    selected = str(chosen["candidate"])
    dev = add_scores(dev_raw, scales, selected)
    validation = add_scores(validation_raw, scales, selected)
    baseline_names = ("FitOnly", "ChallengerOnly", "BootstrapOnly", "NumericalOnly", "NonRegimeDVE")
    baseline_thresholds = _fit_thresholds(dev, baseline_names, config.target_fpr)
    global_full = float(thresholds[cast(CandidateName, selected)])
    regime_thresholds: dict[str, float] = {}
    negative = dev[~dev["material_error"]]
    for regime in sorted(dev["regime"].unique()):
        values = negative[negative["regime"] == regime]["FullV1_1DVE"]
        regime_thresholds[str(regime)] = (
            _quantile_higher(values, 1.0 - config.target_fpr)
            if len(values) >= config.minimum_regime_negatives
            else global_full
        )
    payload: dict[str, Any] = {
        "schema_version": "1.0",
        "status": "FROZEN_BEFORE_FRESH_LOCKED_GENERATION",
        "created_utc": _utc_now(),
        "design_freeze_hash": design["freeze_hash"],
        "evidence_scales": scales,
        "candidate_thresholds": thresholds,
        "validation_diagnostics": validation_rows,
        "selected_candidate": selected,
        "baseline_thresholds": baseline_thresholds,
        "full_global_threshold": global_full,
        "full_regime_thresholds": regime_thresholds,
        "selection_rule": design["selection_rule"],
        "locked_results_used": False,
    }
    payload["freeze_hash"] = sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
    _write_json(path, payload, exclusive=True)
    diagnostics.to_csv(
        root / "results/v1_1/dve_v1_1/validation_candidate_selection.csv", index=False
    )
    return payload, dev, validation


def _threshold_series(frame: pd.DataFrame, frozen: dict[str, Any], baseline: str) -> pd.Series:
    if baseline == "FullV1_1DVE":
        global_threshold = float(frozen["full_global_threshold"])
        regimes = cast(dict[str, float], frozen["full_regime_thresholds"])
        return frame["regime"].map(regimes).fillna(global_threshold).astype(float)
    value = float(cast(dict[str, float], frozen["baseline_thresholds"])[baseline])
    return pd.Series(value, index=frame.index, dtype=float)


def _bootstrap_metrics(
    frame: pd.DataFrame, baseline: str, threshold: pd.Series, seed: int, draws: int = 1000
) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    recall = np.empty(draws)
    fpr = np.empty(draws)
    for draw in range(draws):
        index = rng.integers(0, len(frame), size=len(frame))
        sample = frame.iloc[index]
        metrics = _metrics(sample, baseline, threshold.iloc[index].reset_index(drop=True))
        recall[draw] = float(metrics["recall"])
        fpr[draw] = float(metrics["false_positive_rate"])
    return {
        "recall_ci_low": float(np.nanquantile(recall, 0.025)),
        "recall_ci_high": float(np.nanquantile(recall, 0.975)),
        "fpr_ci_low": float(np.nanquantile(fpr, 0.025)),
        "fpr_ci_high": float(np.nanquantile(fpr, 0.975)),
    }


def run_fresh_dve_v1_1(
    root: Path = Path("."), config: DVEV11Config | None = None
) -> dict[str, Any]:
    cfg = config or DVEV11Config()
    status_path = root / "results/v1_1/dve_v1_1/confirmatory_metrics.json"
    if status_path.exists():
        return cast(dict[str, Any], json.loads(status_path.read_text(encoding="utf-8")))
    design = freeze_design(root, cfg)
    dev_raw = _generate_partition(root, "DEV", cfg, design)
    validation_raw = _generate_partition(root, "VALIDATION", cfg, design)
    aggregation, dev, validation = select_aggregation(root, dev_raw, validation_raw, cfg, design)
    validation_rows = []
    for baseline in BASELINES:
        threshold = _threshold_series(validation, aggregation, baseline)
        validation_rows.append(
            {"partition": "VALIDATION", **_metrics(validation, baseline, threshold)}
        )
    pd.DataFrame(validation_rows).to_csv(
        root / "results/v1_1/dve_v1_1/validation_baselines.csv", index=False
    )
    locked_raw = _generate_partition(root, "LOCKED_TEST", cfg, design)
    ood_raw = _generate_partition(root, "OOD_LOCKED_TEST", cfg, design)
    selected = str(aggregation["selected_candidate"])
    locked = add_scores(
        locked_raw, cast(dict[str, float], aggregation["evidence_scales"]), selected
    )
    ood = add_scores(ood_raw, cast(dict[str, float], aggregation["evidence_scales"]), selected)
    locked.to_parquet(root / "results/v1_1/dve_v1_1/locked_scored.parquet", index=False)
    ood.to_parquet(root / "results/v1_1/dve_v1_1/ood_locked_scored.parquet", index=False)
    metrics_rows: list[dict[str, object]] = []
    for partition, frame in (
        ("LOCKED_TEST", locked),
        ("OOD_LOCKED_TEST", ood),
        ("COMBINED", pd.concat([locked, ood], ignore_index=True)),
    ):
        for index, baseline in enumerate(BASELINES):
            threshold = _threshold_series(frame, aggregation, baseline)
            metrics_rows.append(
                {
                    "partition": partition,
                    **_metrics(frame, baseline, threshold),
                    **_bootstrap_metrics(
                        frame.reset_index(drop=True),
                        baseline,
                        threshold.reset_index(drop=True),
                        cfg.inference_seed + index + len(metrics_rows) * 31,
                    ),
                }
            )
    metrics = pd.DataFrame(metrics_rows)
    metrics.to_csv(root / "results/v1_1/dve_v1_1/locked_baseline_metrics.csv", index=False)
    full_threshold = _threshold_series(locked, aggregation, "FullV1_1DVE")
    prediction = locked["FullV1_1DVE"].to_numpy() > full_threshold.to_numpy()
    locked[prediction & ~locked["material_error"]].to_parquet(
        root / "results/v1_1/dve_v1_1/false_positive_cases.parquet", index=False
    )
    locked[~prediction & locked["material_error"]].to_parquet(
        root / "results/v1_1/dve_v1_1/false_negative_cases.parquet", index=False
    )
    payload = {
        "status": "CONFIRMATORY_LOCKED_COMPLETE",
        "created_utc": _utc_now(),
        "design_freeze_hash": design["freeze_hash"],
        "aggregation_freeze_hash": aggregation["freeze_hash"],
        "selected_transparent_aggregation": selected,
        "fresh_seed_regions": ["117m", "118m", "119m", "120m"],
        "v1_locked_observations_reused": False,
        "case_counts": {
            "DEV": len(dev),
            "VALIDATION": len(validation),
            "LOCKED_TEST": len(locked),
            "OOD_LOCKED_TEST": len(ood),
        },
        "combined_locked_metrics": metrics[metrics["partition"] == "COMBINED"].to_dict("records"),
        "locked_tuning_performed": False,
    }
    _write_json(status_path, payload)
    return payload


def diagnose_v1_dve(root: Path = Path(".")) -> dict[str, Any]:
    """Diagnose preserved v1 Full-DVE failures without tuning against them."""

    source = root / "results/confirmatory/synthetic_v5/evidence_score_records.parquet"
    frame = pd.read_parquet(source)
    locked = frame[frame["partition"].isin(["LOCKED_TEST", "OOD_LOCKED_TEST"])].copy()
    locked["prediction"] = locked["V06_prediction"].fillna(False).astype(bool)
    locked["outcome"] = np.select(
        [
            locked["prediction"] & ~locked["material_error"],
            ~locked["prediction"] & locked["material_error"],
            locked["prediction"] & locked["material_error"],
        ],
        ["FALSE_POSITIVE", "FALSE_NEGATIVE", "TRUE_POSITIVE"],
        default="TRUE_NEGATIVE",
    )
    regime = locked["regime"].str.split("|", expand=True)
    locked["maturity_regime"] = regime[0]
    locked["moneyness_regime"] = regime[1]
    locked["liquidity_regime"] = regime[2]
    locked["noise_proxy_regime"] = pd.qcut(
        locked["E_data"].rank(method="first"), 3, labels=["LOW", "MEDIUM", "HIGH"]
    ).astype(str)
    output = root / "results/v1_1/dve_failure"
    output.mkdir(parents=True, exist_ok=True)
    locked.to_parquet(output / "v1_locked_failure_cases.parquet", index=False)
    dimensions = [
        "fault_id",
        "truth_family",
        "moneyness_regime",
        "maturity_regime",
        "liquidity_regime",
        "noise_proxy_regime",
    ]
    summaries = []
    for dimension in dimensions:
        item = locked.groupby([dimension, "outcome"], as_index=False).size()
        item["dimension"] = dimension
        item = item.rename(columns={dimension: "value", "size": "count"})
        summaries.append(item)
    summary = pd.concat(summaries, ignore_index=True)
    summary.to_csv(output / "failure_slices.csv", index=False)
    conditional = []
    for evidence in EVIDENCE_NAMES:
        for material, group in locked.groupby("material_error"):
            conditional.append(
                {
                    "evidence": evidence,
                    "material_error": bool(material),
                    "count": len(group),
                    "mean": float(group[evidence].mean()),
                    "median": float(group[evidence].median()),
                    "p90": float(group[evidence].quantile(0.90)),
                }
            )
    conditional_frame = pd.DataFrame(conditional)
    conditional_frame.to_csv(output / "conditional_evidence_signal.csv", index=False)
    false_positive = locked[locked["outcome"] == "FALSE_POSITIVE"]
    payload = {
        "status": "DESCRIPTIVE_DIAGNOSIS_OF_IMMUTABLE_V1_RESULT",
        "source_hash": _hash(source),
        "locked_cases": len(locked),
        "false_positives": len(false_positive),
        "false_negatives": int((locked["outcome"] == "FALSE_NEGATIVE").sum()),
        "false_positive_concentration": false_positive.groupby("regime")
        .size()
        .sort_values(ascending=False)
        .to_dict(),
        "tuning_use": "PROHIBITED",
    }
    _write_json(output / "diagnosis_summary.json", payload)
    return payload


def write_dve_reports(root: Path = Path(".")) -> None:
    """Render concise reports strictly from persisted v1/v1.1 result tables."""

    report_dir = root / "reports/v1_1"
    report_dir.mkdir(parents=True, exist_ok=True)
    diagnosis = cast(
        dict[str, Any],
        json.loads(
            (root / "results/v1_1/dve_failure/diagnosis_summary.json").read_text(encoding="utf-8")
        ),
    )
    slices = pd.read_csv(root / "results/v1_1/dve_failure/failure_slices.csv")
    evidence = pd.read_csv(root / "results/v1_1/dve_failure/conditional_evidence_signal.csv")

    def markdown(frame: pd.DataFrame, rows: int = 20) -> str:
        if frame.empty:
            return "No rows available."
        chosen = frame.head(rows)
        columns = list(chosen.columns)
        header = "| " + " | ".join(columns) + " |"
        separator = "| " + " | ".join("---" for _ in columns) + " |"
        body = [
            "| " + " | ".join(str(value).replace("|", "\\|") for value in row) + " |"
            for row in chosen.itertuples(index=False, name=None)
        ]
        return "\n".join([header, separator, *body])

    failure_text = (
        "# DVE v1 Failure Analysis\n\n"
        "This is a descriptive diagnosis of immutable v1 evidence. It was not used to tune "
        "v1.1.\n\n"
        f"The preserved locked/OOD evidence contains {diagnosis['locked_cases']} cases, "
        f"{diagnosis['false_positives']} Full-DVE false positives, and "
        f"{diagnosis['false_negatives']} false negatives. False-positive concentration: "
        f"`{json.dumps(diagnosis['false_positive_concentration'], sort_keys=True)}`.\n\n"
        "## Failure slices\n\n"
        + markdown(slices.sort_values("count", ascending=False))
        + "\n\n## Conditional evidence components\n\n"
        + markdown(evidence)
        + "\n"
    )
    (report_dir / "DVE_FAILURE_ANALYSIS.md").write_text(failure_text, encoding="utf-8")

    status_path = root / "results/v1_1/dve_v1_1/confirmatory_metrics.json"
    if not status_path.exists():
        return
    status = cast(dict[str, Any], json.loads(status_path.read_text(encoding="utf-8")))
    metrics = pd.read_csv(root / "results/v1_1/dve_v1_1/locked_baseline_metrics.csv")
    validation = pd.read_csv(root / "results/v1_1/dve_v1_1/validation_candidate_selection.csv")
    confirmatory_text = (
        "# DVE v1.1 Confirmatory Study\n\n"
        "This is new confirmatory evidence generated from fresh 117m/118m/119m/120m "
        "seed regions. No v1 locked observation was reused for selection.\n\n"
        f"The selected transparent aggregation was "
        f"`{status['selected_transparent_aggregation']}`. Results are retained whether "
        "favorable or unfavorable.\n\n"
        "## VALIDATION candidate selection\n\n"
        + markdown(validation)
        + "\n\n## Fresh locked and OOD results\n\n"
        + markdown(metrics)
        + "\n"
    )
    (report_dir / "DVE_V1_1_CONFIRMATORY.md").write_text(confirmatory_text, encoding="utf-8")
