import json
from dataclasses import asdict
from hashlib import sha256
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import derivguard.synthetic.confirmatory_v5 as confirmatory_v5_module
from derivguard.synthetic.confirmatory import _surface_design, generate_confirmatory_truth
from derivguard.synthetic.confirmatory_v5 import (
    CalibrationUnqualifiedError,
    ConfirmatoryV5Config,
    build_v5_seed_registry,
    calibrate_o01_observable_heston,
    freeze_confirmatory_v5,
)
from derivguard.synthetic.observation import observe_truth


def _audit_config() -> ConfirmatoryV5Config:
    return ConfirmatoryV5Config(
        profile="audit",
        device="cpu-bounded-reference",
        cases_dev=17,
        cases_validation=17,
        cases_locked=17,
        cases_ood=17,
        contracts_per_case=4,
        truth_nodes=32,
        bates_nodes=32,
        calibration_nodes=16,
        primary_multistarts=2,
        primary_max_iterations=80,
        bootstrap_draws=2,
        bootstrap_multistarts=2,
        bootstrap_max_iterations=40,
    )


def _release(path: Path) -> Path:
    path.write_text(
        json.dumps(
            {
                "release_id": "Developer Model Release v1.1",
                "objective": "C04",
                "optimizer": "O01",
                "status": "FROZEN_BEFORE_CONFIRMATORY_V5",
            }
        ),
        encoding="utf-8",
    )
    return path


def _prequalification(path: Path, config: ConfirmatoryV5Config, release: Path) -> Path:
    path.write_text(
        json.dumps(
            {
                "status": "PASS",
                "module_hash": sha256(
                    Path(confirmatory_v5_module.__file__).read_bytes()
                ).hexdigest(),
                "config_hash": sha256(
                    json.dumps(asdict(config), sort_keys=True).encode()
                ).hexdigest(),
                "developer_release_hash": sha256(release.read_bytes()).hexdigest(),
            }
        ),
        encoding="utf-8",
    )
    return path


def test_v5_freezes_new_seeds_release_and_registries(tmp_path: Path) -> None:
    config = _audit_config()
    seeds = build_v5_seed_registry(config)
    assert min(seeds.dev) == 103_000_001
    release = _release(tmp_path / "model_release_v1.1.json")
    payload = freeze_confirmatory_v5(
        tmp_path / "v5.freeze.json",
        config,
        developer_release=release,
        prequalification=_prequalification(tmp_path / "prequalification.json", config, release),
    )
    assert payload["calibration_design"]["objective"] == "C04"
    assert payload["calibration_design"]["optimizer"].startswith("O01")
    assert payload["developer_release_hash"]
    assert payload["hypothesis_registry_hash"]
    assert {row["fault_id"] for row in payload["fault_schedule"]} == {
        f"F{index:02d}" for index in range(17)
    }


def test_nonconverged_multistarts_are_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    config = _audit_config()
    design = _surface_design(102_345_678, 4, ood=False)
    truth = generate_confirmatory_truth(
        design, "HESTON", config.engine_config().truth_config(), seed=102_345_678
    )
    surface = observe_truth(truth, seed=152_345_678)

    def failed_minimize(*args: object, **kwargs: object) -> SimpleNamespace:
        return SimpleNamespace(
            success=False,
            message="iteration limit",
            x=np.asarray([0.04, 2.0, 0.04, 0.5, -0.7]),
            fun=1.0,
            nit=80,
            jac=np.ones(5),
        )

    monkeypatch.setattr("derivguard.synthetic.confirmatory_v5.minimize", failed_minimize)
    with pytest.raises(CalibrationUnqualifiedError, match="no converged C04/O01"):
        calibrate_o01_observable_heston(surface, config, seed=162_345_678)
