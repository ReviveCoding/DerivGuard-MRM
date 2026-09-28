from __future__ import annotations

import math

import pytest

from derivguard.development.heston_cf import HestonParameters, heston_price_fixed_quad
from derivguard.market.black_scholes import call_price, put_price


def test_heston_reference_case_is_stable() -> None:
    value = heston_price_fixed_quad(
        100.0,
        100.0,
        1.0,
        HestonParameters(0.04, 2.0, 0.04, 0.5, -0.7),
        0.03,
        0.01,
    )
    assert value == pytest.approx(8.252848982579188, abs=2.0e-10)


def test_black_scholes_reference_and_parity_are_stable() -> None:
    call = float(call_price(100.0, 105.0, 0.75, 0.22, 0.03, 0.01))
    put = float(put_price(100.0, 105.0, 0.75, 0.22, 0.03, 0.01))
    parity = 100.0 * math.exp(-0.01 * 0.75) - 105.0 * math.exp(-0.03 * 0.75)
    assert call == pytest.approx(6.076395208001763, abs=1.0e-12)
    assert call - put == pytest.approx(parity, abs=1.0e-12)
