import numpy as np
import pytest

from derivguard.market.black_scholes import call_price, greeks, option_price, put_price


def test_known_atm_value_and_put_call_parity() -> None:
    call = call_price(100.0, 100.0, 1.0, 0.2, 0.05, 0.02)
    put = put_price(100.0, 100.0, 1.0, 0.2, 0.05, 0.02)
    assert call == pytest.approx(9.227005508154036, abs=1.0e-12)
    assert call - put == pytest.approx(100.0 * np.exp(-0.02) - 100.0 * np.exp(-0.05), abs=1.0e-12)


def test_bounds_monotonicity_and_convexity() -> None:
    strikes = np.linspace(60.0, 140.0, 161)
    calls = np.asarray(call_price(100.0, strikes, 0.75, 0.3, 0.03, 0.01))
    lower = np.maximum(100.0 * np.exp(-0.01 * 0.75) - strikes * np.exp(-0.03 * 0.75), 0.0)
    assert np.all(calls >= lower - 1.0e-12)
    assert np.all(calls <= 100.0 * np.exp(-0.01 * 0.75) + 1.0e-12)
    assert np.all(np.diff(calls) <= 0.0)
    assert np.all(np.diff(calls, n=2) >= -1.0e-12)


def test_expiry_and_zero_volatility_limits() -> None:
    assert call_price(110.0, 100.0, 0.0, 0.2) == pytest.approx(10.0)
    assert put_price(90.0, 100.0, 0.0, 0.2) == pytest.approx(10.0)
    deterministic = call_price(100.0, 95.0, 2.0, 0.0, 0.04, 0.01)
    expected = max(100.0 * np.exp(-0.02) - 95.0 * np.exp(-0.08), 0.0)
    assert deterministic == pytest.approx(expected)
    deterministic_greeks = greeks(100.0, 95.0, 2.0, 0.0, 0.04, 0.01, "call")
    expected_theta = 0.01 * 100.0 * np.exp(-0.02) - 0.04 * 95.0 * np.exp(-0.08)
    assert deterministic_greeks.delta == pytest.approx(np.exp(-0.02))
    assert deterministic_greeks.gamma == 0.0
    assert deterministic_greeks.vega == 0.0
    assert deterministic_greeks.theta == pytest.approx(expected_theta)


@pytest.mark.parametrize("option_type", ["call", "put"])
def test_analytic_greeks_match_centered_finite_differences(option_type: str) -> None:
    s, k, t, sigma, r, q = 103.0, 97.0, 1.4, 0.27, 0.035, 0.012
    analytic = greeks(s, k, t, sigma, r, q, option_type)  # type: ignore[arg-type]
    h_s = 1.0e-3
    h_v = 1.0e-5
    h_t = 1.0e-5
    base = option_price(s, k, t, sigma, r, q, option_type)  # type: ignore[arg-type]
    up = option_price(s + h_s, k, t, sigma, r, q, option_type)  # type: ignore[arg-type]
    down = option_price(s - h_s, k, t, sigma, r, q, option_type)  # type: ignore[arg-type]
    fd_delta = (up - down) / (2.0 * h_s)
    fd_gamma = (up - 2.0 * base + down) / (h_s * h_s)
    fd_vega = (
        option_price(s, k, t, sigma + h_v, r, q, option_type)
        - option_price(s, k, t, sigma - h_v, r, q, option_type)
    ) / (2.0 * h_v)
    # Calendar theta is -dV/dT.
    fd_theta = -(
        option_price(s, k, t + h_t, sigma, r, q, option_type)
        - option_price(s, k, t - h_t, sigma, r, q, option_type)
    ) / (2.0 * h_t)
    assert analytic.delta == pytest.approx(fd_delta, rel=1.0e-8)
    assert analytic.gamma == pytest.approx(fd_gamma, rel=2.0e-5)
    assert analytic.vega == pytest.approx(fd_vega, rel=1.0e-8)
    assert analytic.theta == pytest.approx(fd_theta, rel=1.0e-8)


def test_invalid_inputs_are_rejected() -> None:
    with pytest.raises(ValueError):
        call_price(-1.0, 100.0, 1.0, 0.2)
    with pytest.raises(ValueError):
        option_price(100.0, 100.0, 1.0, 0.2, option_type="other")  # type: ignore[arg-type]
