from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from derivguard.synthetic import (
    clean_validation_case,
    generate_truth,
    inject_fault,
    observe_truth,
    stratified_design,
)
from derivguard.validation import (
    BASELINE_NAMES,
    EVIDENCE_NAMES,
    evaluate_locked,
    fit_dev_thresholds,
    make_record,
    run_component_ablations,
)


def _records(partition: str, prefix: str, seed_offset: int):
    records = []
    # Each record is an independently observed surface.  Half are clean and
    # half contain a material integration-resolution fault.
    for index in range(24):
        design = stratified_design(10, seed=seed_offset + index)
        surface = observe_truth(generate_truth(design, "SYN-BS"), seed=seed_offset + 100 + index)
        case = clean_validation_case(surface)
        if index >= 12:
            case = inject_fault(case, "F04", severity=8.0)
        records.append(
            make_record(
                f"{prefix}-{index}",
                partition,  # type: ignore[arg-type]
                case,
                materiality_threshold=0.50,
            )
        )
    return records


def test_dev_freeze_locked_evaluation_and_all_baselines() -> None:
    dev = _records("DEV", "DEV", 1000)
    locked = _records("LOCKED_TEST", "LOCK", 2000)
    frozen = fit_dev_thresholds(dev, minimum_regime_negatives=5)
    first = evaluate_locked(locked, frozen)
    second = evaluate_locked(locked, frozen)

    assert set(first) == set(BASELINE_NAMES)
    assert first == second
    assert len(frozen.config_hash) == 64
    assert first["V06"].observations == 24
    assert first["V06"].positives > 0
    assert first["V06"].negatives > 0
    assert 0.0 <= first["V06"].auroc <= 1.0
    assert 0.0 <= first["V06"].auprc <= 1.0


def test_partition_and_identifier_leakage_guards() -> None:
    dev = _records("DEV", "DEV", 3000)
    locked = _records("LOCKED_TEST", "LOCK", 4000)
    with pytest.raises(ValueError, match="DEV records"):
        fit_dev_thresholds(locked)
    frozen = fit_dev_thresholds(dev)
    overlapping = [locked[0].__class__(**{**locked[0].__dict__, "case_id": dev[0].case_id})]
    with pytest.raises(ValueError, match="overlap"):
        evaluate_locked(overlapping, frozen)
    with pytest.raises(ValueError, match="locked partitions"):
        evaluate_locked(dev, frozen)


def test_component_ablations_report_locked_metric_deltas() -> None:
    dev = _records("DEV", "DEV", 5000)
    locked = _records("LOCKED_TEST", "LOCK", 6000)
    results = run_component_ablations(dev, locked)

    assert set(results) == set(EVIDENCE_NAMES)
    for result in results.values():
        assert result.ablated.observations == len(locked)
        assert np.isfinite(result.recall_change)
        assert np.isfinite(result.auprc_change)


def test_regime_threshold_falls_back_when_dev_support_is_insufficient() -> None:
    dev = _records("DEV", "DEV", 7000)
    # Add one stress-regime non-material example; minimum is deliberately 3.
    surface = observe_truth(generate_truth(stratified_design(8, seed=7999), "SYN-BS"), seed=8000)
    stress = inject_fault(clean_validation_case(surface), "F03", severity=0.0)
    dev.append(make_record("DEV-stress", "DEV", stress, materiality_threshold=0.5))
    frozen = fit_dev_thresholds(dev, minimum_regime_negatives=3)

    # Regimes are derived from observable maturity/moneyness/liquidity inputs,
    # never from the injected fault label itself.
    assert frozen.regime_fallbacks["V06"]
    assert all("F03" not in regime for regime in frozen.regime_fallbacks["V06"])
    assert frozen.threshold("V06", "STRESS|MID|LIQUID") == frozen.global_thresholds["V06"]


def test_malformed_evidence_matrices_are_rejected() -> None:
    surface = observe_truth(generate_truth(stratified_design(4, seed=9000), "SYN-BS"), seed=9001)
    malformed = replace(clean_validation_case(surface), numerical_prices=np.zeros((3, 2)))
    with pytest.raises(ValueError, match="one row per surface"):
        make_record("BAD", "DEV", malformed, materiality_threshold=0.5)
