import numpy as np
import pandas as pd
import pytest

from derivguard.data.forward import (
    compare_forward_estimators,
    estimate_forward_discount,
    estimate_grouped_forwards,
    matched_put_call_pairs,
)


def _pairs() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    strikes = np.linspace(80.0, 120.0, 17)
    discount = 0.97
    forward = 103.0
    differences = discount * (forward - strikes)
    puts = np.maximum(strikes - forward, 0.0) * discount + 4.0
    calls = puts + differences
    return strikes, calls, puts


@pytest.mark.parametrize("method", ["F0", "F1", "F2", "F3"])
def test_all_forward_estimators_recover_exact_parity(method: str) -> None:
    strikes, calls, puts = _pairs()
    kwargs = {}
    if method in {"F1", "F3"}:
        kwargs = {"call_spreads": np.full(17, 0.2), "put_spreads": np.full(17, 0.3)}
    result = estimate_forward_discount(strikes, calls, puts, method=method, **kwargs)  # type: ignore[arg-type]
    assert result.forward == pytest.approx(103.0, abs=1.0e-10)
    assert result.discount_factor == pytest.approx(0.97, abs=1.0e-12)
    assert result.converged


def test_huber_reduces_single_outlier_influence() -> None:
    strikes, calls, puts = _pairs()
    calls[0] += 20.0
    ols = estimate_forward_discount(strikes, calls, puts, method="F0")
    robust = estimate_forward_discount(strikes, calls, puts, method="F2")
    assert abs(robust.forward - 103.0) < abs(ols.forward - 103.0)
    assert robust.converged


def test_weighted_estimators_require_spreads() -> None:
    strikes, calls, puts = _pairs()
    with pytest.raises(ValueError, match="requires"):
        estimate_forward_discount(strikes, calls, puts, method="F1")


def test_matched_pairs_choose_narrowest_duplicate_and_compare() -> None:
    strikes, calls, puts = _pairs()
    rows = []
    for strike, call, put in zip(strikes, calls, puts, strict=True):
        rows.extend(
            [
                {"strike": strike, "option_type": "call", "bid": call - 0.1, "ask": call + 0.1},
                {"strike": strike, "option_type": "put", "bid": put - 0.1, "ask": put + 0.1},
            ]
        )
    rows.append(
        {"strike": strikes[0], "option_type": "call", "bid": calls[0] - 2, "ask": calls[0] + 2}
    )
    frame = pd.DataFrame(rows)
    pairs = matched_put_call_pairs(frame)
    assert len(pairs) == len(strikes)
    assert len(compare_forward_estimators(frame)) == 4

    frame["quote_date"] = "2026-09-01"
    frame["expiration"] = "2026-10-01"
    frame["settlement_class"] = "PM"
    grouped = estimate_grouped_forwards(frame)
    assert len(grouped) == 4
    assert set(grouped["status"]) == {"SUCCESS"}
