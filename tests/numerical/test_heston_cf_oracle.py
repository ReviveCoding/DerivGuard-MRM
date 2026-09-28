import importlib.util

import numpy as np
import pytest

from derivguard.development.heston_cf import (
    HestonParameters,
    heston_price_adaptive,
    heston_price_fixed_quad,
)
from derivguard.validation.quantlib_oracle import (
    QuantLibHestonParameters,
    quantlib_heston_price,
)


@pytest.mark.numerical
def test_fixed_quadrature_agrees_with_adaptive_reference() -> None:
    params = HestonParameters(v0=0.04, kappa=2.0, theta=0.04, sigma_v=0.5, rho=-0.7)
    strikes = np.asarray([70.0, 85.0, 100.0, 115.0, 140.0])
    adaptive = np.asarray(heston_price_adaptive(100.0, strikes, 1.0, params, 0.03, 0.01))
    fixed = np.asarray(
        heston_price_fixed_quad(
            100.0, strikes, 1.0, params, 0.03, 0.01, nodes=160, upper_bound=220.0
        )
    )
    assert fixed == pytest.approx(adaptive, abs=2.0e-7, rel=2.0e-7)


@pytest.mark.numerical
@pytest.mark.parametrize(
    "params,maturity",
    [
        (HestonParameters(0.04, 1.5, 0.04, 0.9, -0.95), 0.05),
        (HestonParameters(0.09, 0.5, 0.06, 1.0, 0.9), 5.0),
        # Deliberate Feller violation is a supported numerical regime.
        (HestonParameters(0.02, 0.8, 0.03, 0.8, -0.5), 1.5),
    ],
)
def test_stress_prices_are_finite_and_respect_bounds(
    params: HestonParameters, maturity: float
) -> None:
    strikes = np.asarray([40.0, 100.0, 220.0])
    calls = np.asarray(
        heston_price_fixed_quad(
            100.0, strikes, maturity, params, 0.025, 0.01, nodes=512, upper_bound=600.0
        )
    )
    lower = np.maximum(100.0 * np.exp(-0.01 * maturity) - strikes * np.exp(-0.025 * maturity), 0.0)
    assert np.all(np.isfinite(calls))
    assert np.all(calls >= lower - 2.0e-5)
    assert np.all(calls <= 100.0 * np.exp(-0.01 * maturity) + 2.0e-5)


@pytest.mark.numerical
@pytest.mark.skipif(importlib.util.find_spec("QuantLib") is None, reason="QuantLib unavailable")
def test_developer_cf_agrees_with_quantlib_external_oracle() -> None:
    p = HestonParameters(0.04, 2.0, 0.04, 0.5, -0.7)
    qlp = QuantLibHestonParameters(0.04, 2.0, 0.04, 0.5, -0.7)
    for strike in (80.0, 100.0, 120.0):
        ours = heston_price_adaptive(100.0, strike, 1.0, p, 0.03, 0.01)
        oracle = quantlib_heston_price(100.0, strike, 1.0, qlp, 0.03, 0.01)
        assert oracle.effective_maturity == 1.0
        assert ours == pytest.approx(oracle.price, abs=2.0e-6, rel=2.0e-6)


def test_put_call_parity() -> None:
    p = HestonParameters(0.04, 2.0, 0.04, 0.5, -0.7)
    call = heston_price_fixed_quad(100.0, 105.0, 0.75, p, 0.02, 0.01, "call")
    put = heston_price_fixed_quad(100.0, 105.0, 0.75, p, 0.02, 0.01, "put")
    expected = 100.0 * np.exp(-0.01 * 0.75) - 105.0 * np.exp(-0.02 * 0.75)
    assert call - put == pytest.approx(expected, abs=1.0e-10)
