import json
from dataclasses import asdict
from hashlib import sha256
from pathlib import Path

import derivguard.synthetic.confirmatory_v6 as confirmatory_v6_module
from derivguard.synthetic.confirmatory_v6 import (
    ConfirmatoryV6Config,
    build_v6_seed_registry,
    freeze_confirmatory_v6,
)


def _audit_config() -> ConfirmatoryV6Config:
    return ConfirmatoryV6Config(
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
        bootstrap_multistarts=5,
        bootstrap_max_iterations=160,
    )


def test_v6_freezes_new_seeds_and_strengthened_bootstrap(tmp_path: Path) -> None:
    config = _audit_config()
    release = tmp_path / "model_release_v1.1.json"
    release.write_text(
        json.dumps(
            {
                "release_id": "Developer Model Release v1.1",
                "objective": "C04",
                "optimizer": "O01",
            }
        ),
        encoding="utf-8",
    )
    prequalification = tmp_path / "prequalification.json"
    prequalification.write_text(
        json.dumps(
            {
                "status": "PASS",
                "module_hash": sha256(
                    Path(confirmatory_v6_module.__file__).read_bytes()
                ).hexdigest(),
                "v5_dependency_hash": sha256(
                    Path(confirmatory_v6_module.__file__)
                    .with_name("confirmatory_v5.py")
                    .read_bytes()
                ).hexdigest(),
                "config_hash": sha256(
                    json.dumps(asdict(config), sort_keys=True).encode()
                ).hexdigest(),
                "developer_release_hash": sha256(release.read_bytes()).hexdigest(),
            }
        ),
        encoding="utf-8",
    )
    seeds = build_v6_seed_registry(config)
    assert min(seeds.dev) == 107_000_001
    assert max(seeds.ood) < 111_000_000

    frozen = freeze_confirmatory_v6(
        tmp_path / "v6.freeze.json",
        config,
        developer_release=release,
        prequalification=prequalification,
    )

    assert frozen["schema_version"] == "6.0"
    assert frozen["calibration_design"]["bootstrap_multistarts"] == 5
    assert frozen["calibration_design"]["bootstrap_max_iterations"] == 160
    assert frozen["v5_validation_failure_used_for_remediation"] is True
    assert frozen["prior_v4_or_v5_locked_outcomes_used_for_selection"] is False
