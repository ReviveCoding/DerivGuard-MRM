"""Transparent DerivGuard Validation Evidence construction and freezing.

Evidence dimensions remain separately available.  Composite baseline scores
are simple equal-weight means of DEV-scaled dimensions and are never exposed
as probabilities of failure.  Thresholds are fitted exclusively from DEV
records and then frozen for locked evaluation.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from hashlib import sha256
from typing import Literal

import numpy as np
from numpy.typing import NDArray

from derivguard.synthetic.faults import ValidationCase

FloatArray = NDArray[np.float64]
Partition = Literal["DEV", "VALIDATION", "LOCKED_TEST", "OOD_LOCKED_TEST"]

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
BASELINE_NAMES = ("V00", "V01", "V02", "V03", "V04", "V05", "V06")


@dataclass(frozen=True)
class EvidenceVector:
    E_data: float
    E_num: float
    E_cal: float
    E_ident: float
    E_param: float
    E_form: float
    E_greek: float
    E_extra: float
    E_outcome: float

    def as_dict(self) -> dict[str, float]:
        return {name: float(getattr(self, name)) for name in EVIDENCE_NAMES}

    def selected(self, dimensions: Sequence[str]) -> FloatArray:
        unknown = set(dimensions) - set(EVIDENCE_NAMES)
        if unknown:
            raise ValueError(f"unknown evidence dimensions: {sorted(unknown)}")
        return np.asarray([getattr(self, name) for name in dimensions], dtype=np.float64)


@dataclass(frozen=True)
class EvidenceRecord:
    case_id: str
    partition: Partition
    regime: str
    evidence: EvidenceVector
    fit_error: float
    oos_error: float
    material_error: bool
    normalized_true_error: float
    fault_id: str
    proxy_fault: bool = False


@dataclass(frozen=True)
class FrozenDVE:
    """DEV-fitted scales and thresholds safe for locked-test application."""

    evidence_dimensions: tuple[str, ...]
    evidence_scales: Mapping[str, float]
    global_thresholds: Mapping[str, float]
    regime_thresholds: Mapping[str, Mapping[str, float]]
    regime_fallbacks: Mapping[str, tuple[str, ...]]
    target_fpr: float
    dev_case_ids: tuple[str, ...]
    config_hash: str

    def threshold(self, baseline: str, regime: str) -> float:
        if baseline == "V06":
            return float(
                self.regime_thresholds.get(baseline, {}).get(
                    regime, self.global_thresholds[baseline]
                )
            )
        return float(self.global_thresholds[baseline])

    def to_dict(self) -> dict[str, object]:
        return {
            "evidence_dimensions": list(self.evidence_dimensions),
            "evidence_scales": dict(self.evidence_scales),
            "global_thresholds": dict(self.global_thresholds),
            "regime_thresholds": {
                key: dict(value) for key, value in self.regime_thresholds.items()
            },
            "regime_fallbacks": {key: list(value) for key, value in self.regime_fallbacks.items()},
            "target_fpr": self.target_fpr,
            "dev_case_ids": list(self.dev_case_ids),
            "config_hash": self.config_hash,
        }


@dataclass(frozen=True)
class DetectionMetrics:
    baseline: str
    observations: int
    positives: int
    negatives: int
    recall_at_dev_5pct_fpr: float
    locked_false_positive_rate: float
    auprc: float
    auroc: float
    false_alarm_rate: float
    material_risk_miss_rate: float


@dataclass(frozen=True)
class AblationResult:
    removed_dimension: str
    full: DetectionMetrics
    ablated: DetectionMetrics
    recall_change: float
    auprc_change: float
    false_alarm_change: float
    miss_rate_change: float


def _finite_median(values: FloatArray, default: float = 0.0) -> float:
    finite = values[np.isfinite(values)]
    return float(np.median(finite)) if finite.size else default


def _validate_case_shapes(case: ValidationCase) -> None:
    observations = len(case.surface)
    vectors = (
        case.developer_price,
        case.developer_delta,
        case.developer_gamma,
        case.developer_vega,
        case.support_distance,
        case.materiality_scale,
    )
    if any(value.shape != (observations,) for value in vectors):
        raise ValueError("validation-case vectors must match the surface length")
    matrices = (
        case.numerical_prices,
        case.calibration_prices,
        case.challenger_prices,
        case.challenger_delta,
        case.challenger_gamma,
        case.challenger_vega,
    )
    if any(value.ndim != 2 or value.shape[0] != observations for value in matrices):
        raise ValueError("validation-case matrices must have one row per surface observation")
    if any(value.shape[1] < 2 for value in matrices):
        raise ValueError("at least two independent values/draws are required per evidence matrix")
    if np.any(~np.isfinite(case.materiality_scale)) or np.any(case.materiality_scale <= 0.0):
        raise ValueError("materiality scale must be finite and strictly positive")


def compute_evidence(case: ValidationCase) -> EvidenceVector:
    """Aggregate one surface-level evidence vector without using latent labels."""

    _validate_case_shapes(case)
    observed = case.surface.observed
    scale = np.maximum(case.materiality_scale, 1.0e-12)
    valid = observed & np.isfinite(case.surface.mid)
    mask = valid if np.any(valid) else np.ones(len(scale), dtype=bool)
    num_range = np.ptp(case.numerical_prices, axis=1) / scale
    cal_dispersion = np.std(case.calibration_prices, axis=1, ddof=1) / scale
    form_disagreement = np.median(
        np.abs(case.challenger_prices - case.developer_price[:, None]) / scale[:, None], axis=1
    )
    delta_scale = np.maximum(np.abs(case.developer_delta), 0.05)
    gamma_scale = np.maximum(np.abs(case.developer_gamma), 1.0e-4)
    vega_scale = np.maximum(np.abs(case.developer_vega), 0.10)
    greek_disagreement = np.median(
        np.column_stack(
            (
                np.median(np.abs(case.challenger_delta - case.developer_delta[:, None]), axis=1)
                / delta_scale,
                np.median(np.abs(case.challenger_gamma - case.developer_gamma[:, None]), axis=1)
                / gamma_scale,
                np.median(np.abs(case.challenger_vega - case.developer_vega[:, None]), axis=1)
                / vega_scale,
            )
        ),
        axis=1,
    )
    # This is observable data evidence. Latent synthetic truth must not enter
    # the validator score, even as a scale.
    relative_spread = case.surface.spread / np.maximum(np.abs(case.surface.mid), 0.25)
    data_evidence = case.data_quality_score + _finite_median(relative_spread[mask])
    values = EvidenceVector(
        E_data=data_evidence,
        E_num=_finite_median(num_range[mask]),
        E_cal=_finite_median(cal_dispersion[mask]),
        E_ident=max(float(case.identifiability_score), 0.0),
        E_param=float(np.linalg.norm(case.parameter_drift)),
        E_form=_finite_median(form_disagreement[mask]),
        E_greek=_finite_median(greek_disagreement[mask]),
        E_extra=_finite_median(np.maximum(case.support_distance[mask], 0.0)),
        E_outcome=max(float(case.prior_outcome_score), 0.0),
    )
    if not np.all(np.isfinite(list(values.as_dict().values()))):
        raise ValueError("evidence vector contains non-finite values")
    return values


def make_record(
    case_id: str,
    partition: Partition,
    case: ValidationCase,
    *,
    materiality_threshold: float,
) -> EvidenceRecord:
    """Build a record using a pre-registered normalized error threshold."""

    if materiality_threshold <= 0.0:
        raise ValueError("materiality_threshold must be positive")
    normalized = case.normalized_true_error
    fit_mask = case.surface.observed & np.isfinite(case.surface.mid)
    fit = (
        _finite_median(
            np.abs(case.developer_price[fit_mask] - case.surface.mid[fit_mask])
            / np.maximum(case.materiality_scale[fit_mask], 1.0e-12)
        )
        if np.any(fit_mask)
        else float("inf")
    )
    error = _finite_median(normalized)
    return EvidenceRecord(
        case_id=case_id,
        partition=partition,
        regime=case.regime,
        evidence=compute_evidence(case),
        fit_error=fit,
        oos_error=max(case.prior_outcome_score, 0.0),
        material_error=error >= materiality_threshold,
        normalized_true_error=error,
        fault_id=case.fault_id,
        proxy_fault=case.proxy_fault,
    )


def _quantile_higher(values: FloatArray, q: float) -> float:
    if values.size == 0:
        raise ValueError("cannot fit a threshold without negative DEV examples")
    return float(np.quantile(values, q, method="higher"))


def _evidence_scales(
    records: Sequence[EvidenceRecord], dimensions: Sequence[str]
) -> dict[str, float]:
    negatives = [record for record in records if not record.material_error]
    if not negatives:
        raise ValueError("DEV requires at least one non-material example")
    result: dict[str, float] = {}
    for name in dimensions:
        values = np.asarray([getattr(record.evidence, name) for record in negatives])
        # A 95th-percentile scale is interpretable: contribution=1 is a high
        # but non-material DEV observation. Degenerate dimensions use 1.
        result[name] = max(_quantile_higher(values, 0.95), 1.0e-12)
    return result


def baseline_scores(
    record: EvidenceRecord,
    scales: Mapping[str, float],
    dimensions: Sequence[str] = EVIDENCE_NAMES,
) -> dict[str, float]:
    contributions = np.asarray(
        [getattr(record.evidence, name) / scales[name] for name in dimensions], dtype=np.float64
    )
    composite = float(np.mean(contributions))
    return {
        "V00": record.fit_error,
        "V01": record.oos_error,
        "V02": record.evidence.E_form / scales.get("E_form", 1.0),
        "V03": record.evidence.E_cal / scales.get("E_cal", 1.0),
        "V04": record.evidence.E_num / scales.get("E_num", 1.0),
        "V05": composite,
        "V06": composite,
    }


def fit_dev_thresholds(
    records: Iterable[EvidenceRecord],
    *,
    target_fpr: float = 0.05,
    dimensions: Sequence[str] = EVIDENCE_NAMES,
    minimum_regime_negatives: int = 10,
) -> FrozenDVE:
    """Fit and freeze all thresholds, refusing non-DEV input."""

    rows = tuple(records)
    if not rows:
        raise ValueError("at least one DEV record is required")
    if any(row.partition != "DEV" for row in rows):
        raise ValueError("threshold fitting is restricted to DEV records")
    if len({row.case_id for row in rows}) != len(rows):
        raise ValueError("DEV case identifiers must be unique")
    if not 0.0 < target_fpr < 1.0:
        raise ValueError("target_fpr must lie in (0,1)")
    chosen = tuple(dimensions)
    if not chosen or set(chosen) - set(EVIDENCE_NAMES):
        raise ValueError("dimensions must be a non-empty subset of evidence dimensions")
    scales = _evidence_scales(rows, chosen)
    negatives = tuple(row for row in rows if not row.material_error)
    scores = {
        name: np.asarray([baseline_scores(row, scales, chosen)[name] for row in negatives])
        for name in BASELINE_NAMES
    }
    globals_ = {name: _quantile_higher(value, 1.0 - target_fpr) for name, value in scores.items()}
    regime_thresholds: dict[str, dict[str, float]] = {"V06": {}}
    fallbacks: list[str] = []
    for regime in sorted({row.regime for row in rows}):
        regime_negatives = tuple(row for row in negatives if row.regime == regime)
        if len(regime_negatives) < minimum_regime_negatives:
            regime_thresholds["V06"][regime] = globals_["V06"]
            fallbacks.append(regime)
        else:
            values = np.asarray(
                [baseline_scores(row, scales, chosen)["V06"] for row in regime_negatives]
            )
            regime_thresholds["V06"][regime] = _quantile_higher(values, 1.0 - target_fpr)
    payload = {
        "dimensions": chosen,
        "scales": scales,
        "global_thresholds": globals_,
        "regime_thresholds": regime_thresholds,
        "target_fpr": target_fpr,
        "minimum_regime_negatives": minimum_regime_negatives,
        "dev_case_ids": sorted(row.case_id for row in rows),
    }
    config_hash = sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
    return FrozenDVE(
        chosen,
        scales,
        globals_,
        regime_thresholds,
        {"V06": tuple(fallbacks)},
        target_fpr,
        tuple(sorted(row.case_id for row in rows)),
        config_hash,
    )


def _auroc(labels: NDArray[np.bool_], scores: FloatArray) -> float:
    positives = int(np.sum(labels))
    negatives = labels.size - positives
    if positives == 0 or negatives == 0:
        return float("nan")
    order = np.argsort(scores, kind="mergesort")
    sorted_scores = scores[order]
    ranks = np.empty(scores.size, dtype=np.float64)
    start = 0
    while start < scores.size:
        end = start + 1
        while end < scores.size and sorted_scores[end] == sorted_scores[start]:
            end += 1
        ranks[order[start:end]] = 0.5 * (start + 1 + end)
        start = end
    rank_sum = float(np.sum(ranks[labels]))
    return (rank_sum - positives * (positives + 1) / 2.0) / (positives * negatives)


def _average_precision(labels: NDArray[np.bool_], scores: FloatArray) -> float:
    positives = int(np.sum(labels))
    if positives == 0:
        return float("nan")
    order = np.argsort(-scores, kind="mergesort")
    sorted_labels = labels[order]
    sorted_scores = scores[order]
    true_positives = 0
    observations = 0
    previous_recall = 0.0
    average_precision = 0.0
    start = 0
    while start < scores.size:
        end = start + 1
        while end < scores.size and sorted_scores[end] == sorted_scores[start]:
            end += 1
        true_positives += int(np.sum(sorted_labels[start:end]))
        observations += end - start
        recall = true_positives / positives
        precision = true_positives / observations
        average_precision += (recall - previous_recall) * precision
        previous_recall = recall
        start = end
    return average_precision


def evaluate_locked(
    records: Iterable[EvidenceRecord], frozen: FrozenDVE
) -> dict[str, DetectionMetrics]:
    """Apply frozen DEV thresholds; no fitting occurs in this function."""

    rows = tuple(records)
    if not rows:
        raise ValueError("at least one locked-test record is required")
    if any(row.partition not in {"LOCKED_TEST", "OOD_LOCKED_TEST"} for row in rows):
        raise ValueError("locked evaluation accepts locked partitions only")
    if any(row.case_id in frozen.dev_case_ids for row in rows):
        raise ValueError("DEV/locked case identifier overlap detected")
    labels = np.asarray([row.material_error for row in rows], dtype=bool)
    positives = int(np.sum(labels))
    negatives = labels.size - positives
    results: dict[str, DetectionMetrics] = {}
    for name in BASELINE_NAMES:
        scores = np.asarray(
            [
                baseline_scores(row, frozen.evidence_scales, frozen.evidence_dimensions)[name]
                for row in rows
            ]
        )
        thresholds = np.asarray([frozen.threshold(name, row.regime) for row in rows])
        predicted = scores > thresholds
        tp = int(np.sum(predicted & labels))
        fp = int(np.sum(predicted & ~labels))
        recall = tp / positives if positives else float("nan")
        fpr = fp / negatives if negatives else float("nan")
        results[name] = DetectionMetrics(
            baseline=name,
            observations=len(rows),
            positives=positives,
            negatives=negatives,
            recall_at_dev_5pct_fpr=recall,
            locked_false_positive_rate=fpr,
            auprc=_average_precision(labels, scores),
            auroc=_auroc(labels, scores),
            false_alarm_rate=fpr,
            material_risk_miss_rate=(1.0 - recall) if positives else float("nan"),
        )
    return results


def run_component_ablations(
    dev_records: Iterable[EvidenceRecord], locked_records: Iterable[EvidenceRecord]
) -> dict[str, AblationResult]:
    """Fit each ablation on DEV, then compare it with Full DVE on LOCKED TEST."""

    dev = tuple(dev_records)
    locked = tuple(locked_records)
    full_model = fit_dev_thresholds(dev)
    full = evaluate_locked(locked, full_model)["V06"]
    output: dict[str, AblationResult] = {}
    for removed in EVIDENCE_NAMES:
        dimensions = tuple(name for name in EVIDENCE_NAMES if name != removed)
        ablated_model = fit_dev_thresholds(dev, dimensions=dimensions)
        ablated = evaluate_locked(locked, ablated_model)["V06"]
        output[removed] = AblationResult(
            removed,
            full,
            ablated,
            ablated.recall_at_dev_5pct_fpr - full.recall_at_dev_5pct_fpr,
            ablated.auprc - full.auprc,
            ablated.false_alarm_rate - full.false_alarm_rate,
            ablated.material_risk_miss_rate - full.material_risk_miss_rate,
        )
    return output
