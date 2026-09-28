from pathlib import Path

from derivguard.synthetic.confirmatory_v3 import (
    FAULT_IDS,
    ConfirmatoryV3Config,
    build_v3_seed_registry,
    freeze_confirmatory_v3,
)


def test_v3_freezes_all_faults_with_new_disjoint_seeds(tmp_path: Path) -> None:
    config = ConfirmatoryV3Config(
        profile="audit",
        device="cpu-bounded-reference",
        cases_dev=17,
        cases_validation=17,
        cases_locked=17,
        cases_ood=17,
        contracts_per_case=3,
        heston_nodes=32,
        bates_nodes=32,
    )
    seeds = build_v3_seed_registry(config)
    assert min(seeds.dev) >= 83_000_001
    payload = freeze_confirmatory_v3(tmp_path / "v3.freeze.json", config)
    assert {row["fault_id"] for row in payload["fault_schedule"]} == set(FAULT_IDS)
    assert payload["prior_v2_outcomes_read"] is False
