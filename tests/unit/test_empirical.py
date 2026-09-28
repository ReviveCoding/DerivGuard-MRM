from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from derivguard.development.calibration import CalibrationData, CalibrationResult
from derivguard.development.heston_cf import HestonParameters
from derivguard.empirical import (
    EmpiricalConfig,
    PreparedSurface,
    _calibration_from_payload,
    _calibration_payload,
    _universe_diagnostics,
    chronological_split,
    real_calibration_checkpoint_key,
    released_optimizer_name,
    representative_files,
)


def test_complete_date_split_and_representatives() -> None:
    files = [Path(f"2022-01-{index:02d}_options.parquet") for index in range(1, 11)]
    split = chronological_split(files)
    assert [len(split[name]) for name in ("DEV", "VALIDATION", "LOCKED_TEST")] == [6, 2, 2]
    selected = representative_files(split["DEV"], 3)
    assert selected == [files[0], files[2], files[5]]


def test_universe_diagnostics_measures_timing_and_near_expiry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    frame = pd.DataFrame(
        {
            "quote_date": ["2022-07-01"] * 4,
            "expiration": ["2022-07-01", "2022-07-08", "2022-08-01", "2022-08-01"],
            "settlement_class": ["PM", "PM", "AM", "AM"],
            "relative_spread": [0.05, 0.20, 0.08, 0.09],
            "volume": [1, 0, 0, 2],
            "open_interest": [0, 1, 0, 0],
            "quote_datetime": [np.nan] * 4,
            "inclusion_status": ["INCLUDED"] * 4,
            "iv": [0.20, 0.21, 0.22, 0.23],
        }
    )
    monkeypatch.setattr(pd, "read_parquet", lambda *_args, **_kwargs: frame.copy())
    result = _universe_diagnostics([Path("2022-07-01_options.parquet")]).set_index(
        "settlement_class"
    )
    assert result.loc["PM", "near_expiry_0_to_7d_quotes"] == 2
    assert result.loc["PM", "zero_dte_quotes"] == 1
    assert result.loc["PM", "timestamp_coverage"] == 0.0
    assert result.loc["AM", "narrow_spread_quotes"] == 2


def _small_surface() -> PreparedSurface:
    values = np.asarray([100.0, 105.0])
    data = CalibrationData(
        spot=np.asarray([100.0, 100.0]),
        strike=values,
        maturity=np.asarray([0.25, 0.25]),
        rate=np.asarray([0.02, 0.02]),
        dividend_yield=np.asarray([0.01, 0.01]),
        option_type=np.asarray(["call", "call"]),
        market_price=np.asarray([5.0, 2.5]),
        market_iv=np.asarray([0.20, 0.21]),
        vega=np.asarray([18.0, 17.0]),
        bid=np.asarray([4.9, 2.4]),
        ask=np.asarray([5.1, 2.6]),
    )
    return PreparedSurface(
        "2022-07-01",
        "DEV",
        "PM",
        data,
        np.asarray([100.2, 100.2]),
        np.log(values / 100.2),
        np.asarray(["2022-10-01", "2022-10-01"]),
        np.asarray(["row-a", "row-b"]),
    )


def test_frozen_o01_mapping_and_checkpoint_identity() -> None:
    assert released_optimizer_name("O01") == "multistart"
    with pytest.raises(ValueError, match="not a CPU"):
        released_optimizer_name("O04")
    surface = _small_surface()
    config = EmpiricalConfig()
    initial = HestonParameters(0.04, 1.5, 0.05, 0.4, -0.7)
    first = real_calibration_checkpoint_key(
        surface,
        config,
        objective="C04",
        optimizer_id="O01",
        feller="unconstrained",
        initial=initial,
        seed=20260927,
    )
    repeat = real_calibration_checkpoint_key(
        surface,
        config,
        objective="C04",
        optimizer_id="O01",
        feller="unconstrained",
        initial=initial,
        seed=20260927,
    )
    changed_seed = real_calibration_checkpoint_key(
        surface,
        config,
        objective="C04",
        optimizer_id="O01",
        feller="unconstrained",
        initial=initial,
        seed=20260928,
    )
    assert first == repeat
    assert first != changed_seed


def test_calibration_checkpoint_roundtrip() -> None:
    original = CalibrationResult(
        HestonParameters(0.04, 1.5, 0.05, 0.4, -0.7),
        1.25,
        "C04",
        "multistart",
        "unconstrained",
        True,
        "converged",
        42,
        8,
        2,
    )
    restored = _calibration_from_payload(_calibration_payload(original))
    assert restored == original
