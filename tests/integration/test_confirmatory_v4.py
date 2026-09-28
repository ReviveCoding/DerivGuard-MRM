from dataclasses import replace
from pathlib import Path

import numpy as np

from derivguard.synthetic.confirmatory import _surface_design, generate_confirmatory_truth
from derivguard.synthetic.confirmatory_v4 import (
    ConfirmatoryV4Config,
    build_v4_seed_registry,
    calibrate_observable_heston,
    freeze_confirmatory_v4,
)
from derivguard.synthetic.observation import observe_truth


def _audit_config() -> ConfirmatoryV4Config:
    return ConfirmatoryV4Config(
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
        population_candidates=4,
        local_iterations=1,
        bootstrap_draws=2,
        bootstrap_candidates=4,
        bootstrap_local_iterations=1,
    )


def test_v4_freezes_registry_hashes_and_new_seed_regions(tmp_path: Path) -> None:
    config = _audit_config()
    seeds = build_v4_seed_registry(config)
    assert min(seeds.dev) == 93_000_001
    payload = freeze_confirmatory_v4(tmp_path / "v4.freeze.json", config)
    assert payload["hypothesis_registry_hash"]
    assert payload["experiment_registry_hash"]
    assert payload["calibration_design"]["inputs"] == "observable bid/mid/ask only"
    assert {row["fault_id"] for row in payload["fault_schedule"]} == {
        f"F{index:02d}" for index in range(17)
    }


def test_calibration_is_invariant_to_hidden_heston_parameter_arrays() -> None:
    config = _audit_config()
    design = _surface_design(91_234_567, 4, ood=False)
    truth = generate_confirmatory_truth(design, "HESTON", config.truth_config(), seed=91_234_567)
    surface = observe_truth(truth, seed=121_234_567)
    first, _ = calibrate_observable_heston(surface, config, seed=131_234_567)
    altered_design = replace(
        design,
        heston_v0=np.full(4, 0.45),
        heston_kappa=np.full(4, 9.0),
        heston_theta=np.full(4, 0.40),
        heston_sigma_v=np.full(4, 2.0),
        heston_rho=np.full(4, 0.8),
    )
    altered_surface = replace(surface, truth=replace(truth, design=altered_design))
    second, _ = calibrate_observable_heston(altered_surface, config, seed=131_234_567)
    assert np.allclose(
        [first.v0, first.kappa, first.theta, first.sigma_v, first.rho],
        [second.v0, second.kappa, second.theta, second.sigma_v, second.rho],
        rtol=0.0,
        atol=0.0,
    )
