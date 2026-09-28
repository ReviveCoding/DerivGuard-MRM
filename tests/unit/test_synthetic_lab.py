from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from derivguard.synthetic import (
    FAULT_DESCRIPTIONS,
    ObservationConfig,
    UnsupportedTruthModelError,
    clean_validation_case,
    generate_truth,
    inject_fault,
    observe_truth,
    stratified_design,
    truth_capabilities,
)


def test_latin_hypercube_design_is_reproducible_and_stratified() -> None:
    first = stratified_design(16, seed=41)
    second = stratified_design(16, seed=41)
    different = stratified_design(16, seed=42)

    np.testing.assert_array_equal(first.spot, second.spot)
    assert not np.array_equal(first.spot, different.spot)
    assert np.all(first.spot > 0.0)
    assert np.all(first.strike > 0.0)
    assert len(np.unique(first.scenario_id)) == 16


def test_observation_model_preserves_latent_truth_and_audits_exclusions() -> None:
    truth = generate_truth(stratified_design(50, seed=10), "SYN-BS")
    original = truth.true_price.copy()
    config = ObservationConfig(
        base_missing_probability=0.20,
        wing_missing_probability=0.30,
        sparse_keep_fraction=0.80,
    )
    observed = observe_truth(truth, config, seed=11)

    np.testing.assert_array_equal(truth.true_price, original)
    assert np.all(observed.ask[observed.observed] >= observed.bid[observed.observed])
    assert np.all(np.isnan(observed.mid[~observed.observed]))
    assert set(observed.exclusion_code) <= {
        "INCLUDED",
        "SYNTHETIC_MISSING",
        "SYNTHETIC_SPARSITY",
    }


def test_unqualified_truth_models_are_explicitly_unsupported() -> None:
    capabilities = {item.model_id: item for item in truth_capabilities()}
    assert capabilities["SYN-BATES"].status == "UNSUPPORTED"
    assert capabilities["SYN-LOCALVOL"].status == "UNSUPPORTED"
    with pytest.raises(UnsupportedTruthModelError):
        generate_truth(stratified_design(2), "SYN-BATES")


def test_research_scale_heston_does_not_silently_run_on_cpu() -> None:
    with pytest.raises(RuntimeError, match="qualified GPU runner"):
        generate_truth(stratified_design(3), "SYN-HESTON", max_cpu_heston_scenarios=2)


def test_all_faults_are_isolated_and_canonical_case_is_unchanged() -> None:
    surface = observe_truth(generate_truth(stratified_design(12, seed=90), "SYN-BS"), seed=91)
    clean = clean_validation_case(surface)
    clean_price = clean.developer_price.copy()
    clean_numerical = clean.numerical_prices.copy()

    for fault_id in FAULT_DESCRIPTIONS:
        faulted = inject_fault(clean, fault_id, severity=2.0)
        assert faulted.fault_id == fault_id
        assert np.all(np.isfinite(faulted.developer_price))
        assert np.all(np.isfinite(faulted.numerical_prices))

    np.testing.assert_array_equal(clean.developer_price, clean_price)
    np.testing.assert_array_equal(clean.numerical_prices, clean_numerical)
    assert inject_fault(clean, "F01").proxy_fault


def test_invalid_observation_and_fault_parameters_are_rejected() -> None:
    truth = generate_truth(stratified_design(2), "SYN-BS")
    with pytest.raises(ValueError):
        observe_truth(truth, replace(ObservationConfig(), sparse_keep_fraction=1.1))
    case = clean_validation_case(observe_truth(truth))
    with pytest.raises(ValueError):
        inject_fault(case, "F04", severity=-1.0)
