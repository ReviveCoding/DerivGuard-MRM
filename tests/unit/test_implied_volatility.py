import numpy as np
import pytest

from derivguard.market.black_scholes import option_price
from derivguard.market.implied_volatility import implied_volatility, no_arbitrage_bounds


@pytest.mark.parametrize("option_type", ["call", "put"])
@pytest.mark.parametrize("maturity", [1.0 / 365.0, 0.25, 2.0, 10.0])
@pytest.mark.parametrize("strike", [50.0, 80.0, 100.0, 125.0, 200.0])
@pytest.mark.parametrize("volatility", [0.05, 0.2, 0.8, 2.0])
def test_iv_price_roundtrip(
    option_type: str, maturity: float, strike: float, volatility: float
) -> None:
    price = float(option_price(100.0, strike, maturity, volatility, 0.03, 0.01, option_type))  # type: ignore[arg-type]
    result = implied_volatility(price, 100.0, strike, maturity, 0.03, 0.01, option_type)  # type: ignore[arg-type]
    repriced = float(
        option_price(100.0, strike, maturity, result.volatility, 0.03, 0.01, option_type)
    )  # type: ignore[arg-type]
    assert result.converged
    assert repriced == pytest.approx(price, abs=2.0e-8)
    # A nearly intrinsic deep option may not numerically identify its source IV;
    # price roundtrip, not parameter recovery, is the correct acceptance test.
    if price - result.lower_bound > 1.0e-7:
        assert result.volatility == pytest.approx(volatility, abs=2.0e-7)


def test_no_arbitrage_bound_failures_are_diagnostic() -> None:
    lower, upper = no_arbitrage_bounds(100.0, 90.0, 1.0, 0.02, 0.01, "call")
    below = implied_volatility(lower - 1.0, 100.0, 90.0, 1.0, 0.02, 0.01, "call")
    above = implied_volatility(upper + 1.0, 100.0, 90.0, 1.0, 0.02, 0.01, "call")
    assert not below.converged and below.status == "below_no_arbitrage_bound"
    assert not above.converged and above.status == "above_no_arbitrage_bound"
    assert np.isnan(below.volatility)


def test_expiry_handling() -> None:
    result = implied_volatility(10.0, 110.0, 100.0, 0.0)
    assert result.converged and result.status == "expiry_price" and result.volatility == 0.0
