from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from derivguard.synthetic.confirmatory import (
    ABLATION_SPECS,
    ConfirmatoryConfig,
    _surface_design,
    build_confirmatory_case,
    build_seed_registry,
    create_freeze_manifest,
    generate_confirmatory_truth,
    qualify_confirmatory_engines,
    run_confirmatory,
    run_confirmatory_study,
)
from derivguard.synthetic.observation import ObservationConfig
from derivguard.validation.dve import BASELINE_NAMES


def _small_config() -> ConfirmatoryConfig:
    return ConfirmatoryConfig(
        cases_dev=4,
        cases_validation=1,
        cases_locked=4,
        cases_ood=1,
        contracts_per_case=3,
        minimum_regime_negatives=1,
        heston_nodes=32,
        bates_nodes=32,
        max_cpu_contracts=64,
        observation=ObservationConfig(
            base_missing_probability=0.0,
            wing_missing_probability=0.0,
            sparse_keep_fraction=1.0,
        ),
    )


def test_qualified_engines_and_partition_seeds_are_disjoint() -> None:
    qualifications = qualify_confirmatory_engines()
    assert {item.engine for item in qualifications} == {
        "HESTON_FIXED_QUAD",
        "BATES_CF",
        "LOCALVOL_SSVI_MARGINAL",
        "PIECEWISE_REGIME_MARGINAL",
    }
    assert all(item.status == "PASS" for item in qualifications)
    registry = build_seed_registry(_small_config())
    truth_seeds = registry.dev + registry.validation + registry.locked + registry.ood
    observation_seeds = tuple(seed + registry.observation_offset for seed in truth_seeds)
    assert len(set(truth_seeds)) == len(truth_seeds)
    assert not set(truth_seeds) & set(observation_seeds)


@pytest.mark.parametrize(
    ("family", "fault_id"),
    [("BATES", "F01"), ("LOCALVOL_SSVI", "F02")],
)
def test_f01_f02_are_real_model_form_cases(family: str, fault_id: str) -> None:
    config = _small_config()
    design = _surface_design(123_456, config.contracts_per_case, ood=False)
    truth = generate_confirmatory_truth(design, family, config, seed=123_456)  # type: ignore[arg-type]
    case = build_confirmatory_case(
        truth,
        family,  # type: ignore[arg-type]
        config,
        observation_seed=456_789,
    )
    assert case.fault_id == fault_id
    assert not case.proxy_fault
    assert np.max(np.abs(case.developer_price - truth.true_price)) > 1.0e-8
    assert not np.array_equal(case.challenger_prices[:, 0], truth.true_price)
    assert np.ptp(case.calibration_prices, axis=1).max() > 0.0
    assert np.ptp(case.numerical_prices, axis=1).max() > 0.0


def test_freeze_is_immutable_and_precedes_bounded_execution(tmp_path: Path) -> None:
    config = _small_config()
    manifest = tmp_path / "confirmatory.freeze.json"
    payload = create_freeze_manifest(manifest, config)
    assert payload["status"] == "FROZEN_NOT_EXECUTED"
    assert payload["baseline_registry"] == list(BASELINE_NAMES)
    assert set(payload["ablation_registry"]) == set(ABLATION_SPECS)
    with pytest.raises(FileExistsError):
        create_freeze_manifest(manifest, config)

    result = run_confirmatory(manifest, config)
    assert set(result.baselines) == set(BASELINE_NAMES)
    assert set(result.ablations) == set(ABLATION_SPECS)
    assert all(not record.proxy_fault for record in result.records)
    assert {item["freeze_hash"] for item in result.provenance} == {payload["freeze_hash"]}


def test_research_entrypoint_freezes_before_executor_is_called(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import derivguard.synthetic.confirmatory as confirmatory

    monkeypatch.chdir(tmp_path)

    class ExecutorReached(RuntimeError):
        pass

    def check_frozen(manifest_path: Path, config: ConfirmatoryConfig, **kwargs: object) -> None:
        del config, kwargs
        assert manifest_path.exists()
        raise ExecutorReached

    monkeypatch.setattr(confirmatory, "run_confirmatory", check_frozen)
    with pytest.raises(ExecutorReached):
        run_confirmatory_study()
    assert (tmp_path / "artifacts/checkpoints/synthetic_confirmatory_v2.freeze.json").exists()
