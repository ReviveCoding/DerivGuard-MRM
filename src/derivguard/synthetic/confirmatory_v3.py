"""Versioned F00--F16 confirmatory study built without changing v2 artifacts.

The full seed/fault/severity schedule and DVE formulation are frozen before
execution. DEV and VALIDATION precede a second threshold freeze; LOCKED/OOD
generation begins only afterward. No prior exploratory or v2 outcome is read.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import asdict, dataclass, replace
from datetime import UTC, datetime
from hashlib import sha256
from pathlib import Path
from typing import Any, Literal, cast

import numpy as np
import pandas as pd

from derivguard.synthetic.confirmatory import (
    ABLATION_SPECS,
    ConfirmatoryConfig,
    ConfirmatoryRunResult,
    SeedRegistry,
    _surface_design,
    build_confirmatory_case,
    evaluate_confirmatory_ablations,
    generate_confirmatory_truth,
    qualify_confirmatory_engines,
)
from derivguard.synthetic.faults import FaultId, inject_fault
from derivguard.validation.dve import (
    BASELINE_NAMES,
    EvidenceRecord,
    baseline_scores,
    evaluate_locked,
    fit_dev_thresholds,
    make_record,
)

Partition = Literal["DEV", "VALIDATION", "LOCKED_TEST", "OOD_LOCKED_TEST"]

FAULT_IDS: tuple[FaultId, ...] = tuple(cast(FaultId, f"F{index:02d}") for index in range(17))
FAULT_SEVERITIES: dict[FaultId, float] = {
    "F00": 0.0,
    "F01": 2.0,
    "F02": 2.0,
    "F03": 2.0,
    "F04": 2.0,
    "F05": 2.0,
    "F06": 2.0,
    "F07": 2.0,
    "F08": 2.0,
    "F09": 2.0,
    "F10": 2.0,
    "F11": 2.0,
    "F12": 2.0,
    "F13": 2.0,
    "F14": 2.0,
    "F15": 2.0,
    "F16": 2.0,
}

IMPLEMENTATION_LABELS: dict[FaultId, str] = {
    "F00": "canonical Heston truth; no injected fault",
    "F01": "real Heston-on-canonical-Bates truth",
    "F02": "real Heston-on-SSVI-marginal Dupire LocalVol truth",
    "F03": "real Heston-on-piecewise two-regime risk-neutral mixture truth",
    "F04": "controlled proxy: isolated poor-integration-resolution operator",
    "F05": "controlled proxy: isolated coarse-PDE-disagreement operator",
    "F06": "controlled proxy: isolated insufficient-Monte-Carlo operator",
    "F07": "controlled proxy: isolated calibration-local-minimum operator",
    "F08": "controlled proxy: isolated sparse-calibration-support operator",
    "F09": "controlled proxy: isolated corrupt-quote data-evidence operator",
    "F10": "controlled proxy: isolated stale/asynchronous observation operator",
    "F11": "controlled proxy: isolated incorrect-rate/forward operator",
    "F12": "controlled proxy: isolated wing-extrapolation operator",
    "F13": "controlled proxy: isolated maturity-extrapolation operator",
    "F14": "controlled proxy: isolated parameter-instability/identifiability operator",
    "F15": "controlled proxy: isolated biased-Greeks operator",
    "F16": "controlled proxy: combined F04+F09+F14 multiple-mechanism operator",
}


@dataclass(frozen=True)
class ConfirmatoryV3Config:
    profile: str = "research"
    device: str = "cuda"
    cases_dev: int = 34
    cases_validation: int = 17
    cases_locked: int = 34
    cases_ood: int = 17
    contracts_per_case: int = 25
    materiality_threshold: float = 1.0
    target_fpr: float = 0.05
    minimum_regime_negatives: int = 5
    heston_nodes: int = 96
    bates_nodes: int = 128

    def validate(self) -> None:
        if self.profile not in {"research", "audit"}:
            raise ValueError("profile must be research or audit")
        if self.device not in {"cuda", "cpu-bounded-reference"}:
            raise ValueError("invalid device")
        counts = (self.cases_dev, self.cases_validation, self.cases_locked, self.cases_ood)
        if any(count < 17 for count in counts):
            raise ValueError("every partition must cover all 17 faults")
        if self.contracts_per_case < 3 or self.materiality_threshold <= 0.0:
            raise ValueError("invalid case size or materiality threshold")

    def engine_config(self) -> ConfirmatoryConfig:
        return ConfirmatoryConfig(
            cases_dev=self.cases_dev,
            cases_validation=self.cases_validation,
            cases_locked=self.cases_locked,
            cases_ood=self.cases_ood,
            contracts_per_case=self.contracts_per_case,
            materiality_threshold_spread_units=self.materiality_threshold,
            target_fpr=self.target_fpr,
            minimum_regime_negatives=self.minimum_regime_negatives,
            heston_nodes=self.heston_nodes,
            bates_nodes=self.bates_nodes,
            device=self.device,
            max_cpu_contracts=256,
        )


def build_v3_seed_registry(config: ConfirmatoryV3Config) -> SeedRegistry:
    registry = SeedRegistry(
        dev=tuple(83_000_001 + index for index in range(config.cases_dev)),
        validation=tuple(84_000_001 + index for index in range(config.cases_validation)),
        locked=tuple(85_000_001 + index for index in range(config.cases_locked)),
        ood=tuple(86_000_001 + index for index in range(config.cases_ood)),
        observation_offset=20_000_000,
    )
    registry.validate()
    return registry


def _fault_for_index(index: int) -> FaultId:
    return FAULT_IDS[index % len(FAULT_IDS)]


def _truth_family(fault_id: FaultId) -> str:
    return {"F01": "BATES", "F02": "LOCALVOL_SSVI", "F03": "PIECEWISE_REGIME"}.get(
        fault_id, "HESTON"
    )


def freeze_confirmatory_v3(
    path: Path = Path("artifacts/checkpoints/synthetic_confirmatory_v3.freeze.json"),
    config: ConfirmatoryV3Config | None = None,
) -> dict[str, Any]:
    cfg = config or ConfirmatoryV3Config()
    cfg.validate()
    if path.exists():
        loaded: object = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(loaded, dict):
            raise ValueError("invalid v3 freeze payload")
        loaded_payload = cast(dict[str, Any], loaded)
        expected = sha256(json.dumps(asdict(cfg), sort_keys=True).encode()).hexdigest()
        if loaded_payload.get("config_hash") != expected:
            raise ValueError("existing v3 freeze differs from requested configuration")
        current_module_hash = sha256(Path(__file__).read_bytes()).hexdigest()
        if loaded_payload.get("module_hash") != current_module_hash:
            raise ValueError("v3 source changed after immutable design freeze")
        return loaded_payload
    seeds = build_v3_seed_registry(cfg)
    qualification = qualify_confirmatory_engines(cfg.device)
    if any(item.status != "PASS" for item in qualification):
        raise RuntimeError("v3 truth engine qualification failed")
    schedule: list[dict[str, object]] = []
    partitions: tuple[Partition, ...] = (
        "DEV",
        "VALIDATION",
        "LOCKED_TEST",
        "OOD_LOCKED_TEST",
    )
    for partition in partitions:
        for index, seed in enumerate(seeds.for_partition(partition)):
            fault = _fault_for_index(index)
            schedule.append(
                {
                    "partition": partition,
                    "seed": seed,
                    "fault_id": fault,
                    "severity": FAULT_SEVERITIES[fault],
                    "truth_family": _truth_family(fault),
                    "implementation_label": IMPLEMENTATION_LABELS[fault],
                }
            )
    module_hash = sha256(Path(__file__).read_bytes()).hexdigest()
    seed_payload = asdict(seeds)
    seed_hash = sha256(json.dumps(seed_payload, sort_keys=True).encode()).hexdigest()
    schedule_hash = sha256(json.dumps(schedule, sort_keys=True).encode()).hexdigest()
    dve_payload = {
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
        "locked_tuning_prohibited": True,
    }
    payload: dict[str, Any] = {
        "schema_version": "3.0",
        "status": "FROZEN_NOT_EXECUTED",
        "created_utc": datetime.now(UTC).isoformat(),
        "config": asdict(cfg),
        "config_hash": sha256(json.dumps(asdict(cfg), sort_keys=True).encode()).hexdigest(),
        "module_hash": module_hash,
        "seed_registry": seed_payload,
        "seed_hash": seed_hash,
        "fault_schedule": schedule,
        "schedule_hash": schedule_hash,
        "fault_severities": FAULT_SEVERITIES,
        "implementation_labels": IMPLEMENTATION_LABELS,
        "baseline_registry": list(BASELINE_NAMES),
        "ablation_registry": dict(ABLATION_SPECS),
        "dve_formulation": dve_payload,
        "dve_formulation_hash": sha256(
            json.dumps(dve_payload, sort_keys=True).encode()
        ).hexdigest(),
        "engine_qualification": [asdict(item) for item in qualification],
        "prior_v2_outcomes_read": False,
    }
    payload["freeze_hash"] = sha256(
        json.dumps(payload, sort_keys=True, default=str).encode()
    ).hexdigest()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return payload


def _make_v3_case(
    partition: Partition,
    index: int,
    seed: int,
    config: ConfirmatoryV3Config,
) -> tuple[EvidenceRecord, dict[str, object]]:
    fault = _fault_for_index(index)
    family = _truth_family(fault)
    engine = config.engine_config()
    design = _surface_design(seed, config.contracts_per_case, ood=partition == "OOD_LOCKED_TEST")
    truth = generate_confirmatory_truth(design, cast(Any, family), engine, seed=seed)
    observation_seed = seed + 20_000_000
    base = build_confirmatory_case(
        truth, cast(Any, family), engine, observation_seed=observation_seed
    )
    if fault not in {"F00", "F01", "F02", "F03"}:
        base = inject_fault(base, fault, severity=FAULT_SEVERITIES[fault])
        # These are predeclared deterministic evidence perturbations applied to
        # genuine qualified Heston cases, rather than complete degraded-engine
        # reruns. Preserve that distinction in every result row.
        base = replace(base, proxy_fault=True)
    case_id = f"V3-{partition}-{fault}-{seed}"
    record = make_record(
        case_id,
        partition,
        base,
        materiality_threshold=config.materiality_threshold,
    )
    provenance = {
        "case_id": case_id,
        "partition": partition,
        "fault_id": fault,
        "severity": FAULT_SEVERITIES[fault],
        "truth_family": family,
        "implementation_label": IMPLEMENTATION_LABELS[fault],
        "proxy_fault": base.proxy_fault,
        "truth_seed": seed,
        "observation_seed": observation_seed,
        "backend": truth.backend,
        "precision": truth.precision,
        "generated_utc": datetime.now(UTC).isoformat(),
    }
    return record, provenance


def _persist_v3(
    output: Path,
    result: ConfirmatoryRunResult,
    validation_metrics: Mapping[str, object],
    fault_schedule: list[dict[str, object]],
    qualification: list[dict[str, object]],
) -> None:
    output.mkdir(parents=True, exist_ok=True)
    provenance_by_case = {str(row["case_id"]): row for row in result.provenance}
    rows: list[dict[str, object]] = []
    fp: list[dict[str, object]] = []
    fn: list[dict[str, object]] = []
    for record in result.records:
        scores = baseline_scores(record, result.thresholds.evidence_scales)
        row = {
            "case_id": record.case_id,
            "partition": record.partition,
            "regime": record.regime,
            "fault_id": record.fault_id,
            "material_error": record.material_error,
            "normalized_true_error": record.normalized_true_error,
            "fit_error": record.fit_error,
            **record.evidence.as_dict(),
            **{f"score_{key}": value for key, value in scores.items()},
            **provenance_by_case[record.case_id],
        }
        if record.partition in {"LOCKED_TEST", "OOD_LOCKED_TEST"}:
            predicted = scores["V06"] > result.thresholds.threshold("V06", record.regime)
            row["V06_prediction"] = predicted
            if predicted and not record.material_error:
                fp.append(row.copy())
            if not predicted and record.material_error:
                fn.append(row.copy())
        rows.append(row)
    frame = pd.DataFrame(rows)
    frame.to_parquet(output / "evidence_score_records.parquet", index=False)
    pd.DataFrame(result.provenance).to_parquet(output / "case_provenance.parquet", index=False)
    pd.DataFrame(fp, columns=frame.columns).to_parquet(output / "false_positive_cases.parquet")
    pd.DataFrame(fn, columns=frame.columns).to_parquet(output / "false_negative_cases.parquet")
    pd.DataFrame([asdict(item) for item in result.baselines.values()]).to_csv(
        output / "T08_DVE_BASELINES.csv", index=False
    )
    pd.DataFrame(
        [
            {
                "ablation_id": key,
                "description": value.description,
                **asdict(value.metrics),
                "recall_change": value.recall_change,
                "auprc_change": value.auprc_change,
                "false_alarm_change": value.false_alarm_change,
                "miss_rate_change": value.miss_rate_change,
            }
            for key, value in result.ablations.items()
        ]
    ).to_csv(output / "T09_DVE_ABLATION.csv", index=False)
    locked = tuple(
        item for item in result.records if item.partition in {"LOCKED_TEST", "OOD_LOCKED_TEST"}
    )
    t07: list[dict[str, object]] = []
    for fault in FAULT_IDS:
        subset = tuple(item for item in locked if item.fault_id == fault)
        for fault_metric in evaluate_locked(subset, result.thresholds).values():
            t07.append({"fault_id": fault, **asdict(fault_metric)})
    pd.DataFrame(t07).to_csv(output / "T07_SYNTHETIC_FAULT_DETECTION.csv", index=False)
    locked_frame = frame[frame["partition"].isin(["LOCKED_TEST", "OOD_LOCKED_TEST"])].assign(
        maturity_regime=lambda value: value["regime"].str.split("|").str[0],
        liquidity_regime=lambda value: value["regime"].str.split("|").str[-1],
    )
    stratified_rows: list[dict[str, object]] = []
    for dimension in (
        "truth_family",
        "fault_id",
        "regime",
        "liquidity_regime",
        "maturity_regime",
    ):
        for group, subset_frame in locked_frame.groupby(dimension):
            case_ids = set(subset_frame["case_id"])
            subset_records = tuple(item for item in locked if item.case_id in case_ids)
            metrics = evaluate_locked(subset_records, result.thresholds)["V06"]
            stratified_rows.append(
                {
                    "dimension": dimension,
                    "stratum": group,
                    **asdict(metrics),
                    "material_rate": float(subset_frame["material_error"].mean()),
                    "median_error": float(subset_frame["normalized_true_error"].median()),
                    "median_score": float(subset_frame["score_V06"].median()),
                }
            )
    pd.DataFrame(stratified_rows).to_csv(output / "stratified_metrics.csv", index=False)
    rng = np.random.default_rng(89_000_001)
    inference: dict[str, object] = {}
    for baseline in BASELINE_NAMES[:-1]:
        recall: list[float] = []
        auprc: list[float] = []
        for _ in range(250):
            sample = tuple(locked[i] for i in rng.integers(0, len(locked), len(locked)))
            bootstrap_metrics = evaluate_locked(sample, result.thresholds)
            recall.append(
                bootstrap_metrics["V06"].recall_at_dev_5pct_fpr
                - bootstrap_metrics[baseline].recall_at_dev_5pct_fpr
            )
            auprc.append(bootstrap_metrics["V06"].auprc - bootstrap_metrics[baseline].auprc)
        inference[baseline] = {
            "recall_difference_ci95": np.nanpercentile(recall, [2.5, 97.5]).tolist(),
            "auprc_difference_ci95": np.nanpercentile(auprc, [2.5, 97.5]).tolist(),
        }
    (output / "statistical_inference.json").write_text(
        json.dumps(
            {
                "method": "paired case-level bootstrap over locked synthetic cases",
                "bootstrap_seed": 89_000_001,
                "replicates": 250,
                "paired_bootstrap": inference,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    pd.DataFrame(fault_schedule).to_csv(output / "fault_coverage.csv", index=False)
    (output / "truth_qualification.json").write_text(
        json.dumps(qualification, indent=2, sort_keys=True) + "\n"
    )
    (output / "validation_diagnostics.json").write_text(
        json.dumps(validation_metrics, indent=2, sort_keys=True) + "\n"
    )


def run_confirmatory_v3_study(
    profile: str = "research", device: str = "cuda", resume: bool = True
) -> dict[str, Any]:
    """Run frozen F00--F16 study; COMPLETE is returned only after locked output."""

    output = Path("results/confirmatory/synthetic_v2")
    metrics_path = output / "confirmatory_metrics.json"
    if resume and metrics_path.exists():
        return cast(dict[str, Any], json.loads(metrics_path.read_text(encoding="utf-8")))
    config = ConfirmatoryV3Config(profile=profile, device=device)
    freeze_path = Path("artifacts/checkpoints/synthetic_confirmatory_v3.freeze.json")
    frozen = freeze_confirmatory_v3(freeze_path, config)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("v3 output namespace is non-empty; refusing mixed execution")
    seeds = build_v3_seed_registry(config)
    records: list[EvidenceRecord] = []
    provenance: list[dict[str, object]] = []

    def generate(partition: Partition) -> None:
        for index, seed in enumerate(seeds.for_partition(partition)):
            record, source = _make_v3_case(partition, index, seed, config)
            source["design_freeze_hash"] = frozen["freeze_hash"]
            records.append(record)
            provenance.append(source)

    generate("DEV")
    generate("VALIDATION")
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
    output.mkdir(parents=True, exist_ok=True)
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
    generate("LOCKED_TEST")
    generate("OOD_LOCKED_TEST")
    locked = tuple(item for item in records if item.partition in {"LOCKED_TEST", "OOD_LOCKED_TEST"})
    baselines = evaluate_locked(locked, thresholds)
    ablations = evaluate_confirmatory_ablations(dev, locked, thresholds)
    result = ConfirmatoryRunResult(
        run_id=f"SYN-CONFIRMATORY-V3-{str(frozen['freeze_hash'])[:12]}",
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
        frozen["engine_qualification"],
    )
    status = {
        "status": "CONFIRMATORY_LOCKED_COMPLETE",
        "run_id": result.run_id,
        "freeze_hash": result.freeze_hash,
        "locked_cases": len(locked),
        "faults_executed": list(FAULT_IDS),
        "config_hash": frozen["config_hash"],
        "module_hash": frozen["module_hash"],
        "seed_hash": frozen["seed_hash"],
        "schedule_hash": frozen["schedule_hash"],
        "dve_formulation_hash": frozen["dve_formulation_hash"],
        "baselines": {key: asdict(value) for key, value in baselines.items()},
        "ablations": {key: asdict(value) for key, value in ablations.items()},
    }
    metrics_path.write_text(json.dumps(status, indent=2, sort_keys=True) + "\n")
    return status
