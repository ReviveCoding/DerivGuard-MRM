from __future__ import annotations

import pandas as pd
import pytest

from derivguard.empirical_supplement import filter_quote_subset, validate_selection_payload


def _qualified_payload() -> dict[str, object]:
    return {
        "status": "SELECTED_ON_DEV_CONFIRMED_ON_VALIDATION",
        "profile": "research",
        "selected_objective": "C04",
        "selected_optimizer": "O03",
        "selected_feller_treatment": "unconstrained",
        "selected_parameters": {
            "v0": 0.04,
            "kappa": 1.5,
            "theta": 0.05,
            "sigma_v": 0.4,
            "rho": -0.7,
        },
        "config_hash": "abc123",
        "locked_test_used_for_selection": False,
    }


def test_selection_guard_rejects_smoke_and_locked_leakage() -> None:
    qualified = _qualified_payload()
    validate_selection_payload(qualified)
    smoke = {**qualified, "profile": "smoke"}
    with pytest.raises(RuntimeError, match="research selection profile"):
        validate_selection_payload(smoke)
    leaked = {**qualified, "locked_test_used_for_selection": True}
    with pytest.raises(RuntimeError, match="locked-test isolation"):
        validate_selection_payload(leaked)


def test_liquidity_subsets_are_explicit_and_nested() -> None:
    frame = pd.DataFrame(
        {
            "volume": [0, 1, 0, 2],
            "open_interest": [0, 0, 3, 4],
            "relative_spread": [0.05, 0.20, 0.08, 0.09],
        }
    )
    assert len(filter_quote_subset(frame, "ALL_USABLE")) == 4
    assert filter_quote_subset(frame, "LIQUID").index.tolist() == [1, 2, 3]
    assert filter_quote_subset(frame, "NARROW_SPREAD").index.tolist() == [0, 2, 3]
    with pytest.raises(ValueError, match="unknown quote subset"):
        filter_quote_subset(frame, "HIDDEN_DROP")
