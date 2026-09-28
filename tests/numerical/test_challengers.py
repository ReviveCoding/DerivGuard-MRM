import numpy as np
import pytest

from derivguard.challengers.bates import BatesParameters, bates_price
from derivguard.challengers.local_vol import (
    dupire_local_variance,
    local_vol_mc_price,
    ssvi_surface_price,
)
from derivguard.development.heston_cf import HestonParameters, heston_price_fixed_quad
from derivguard.market.svi import SSVIParameters, SSVISurface


@pytest.mark.numerical
def test_bates_zero_jump_intensity_matches_heston() -> None:
    bates = BatesParameters(0.04, 2.0, 0.04, 0.5, -0.7, 0.0, -0.1, 0.2)
    heston = HestonParameters(0.04, 2.0, 0.04, 0.5, -0.7)
    strikes = np.array([80.0, 100.0, 120.0])
    actual = np.asarray(bates_price(100.0, strikes, 1.0, bates, 0.03, 0.01))
    expected = np.asarray(
        heston_price_fixed_quad(
            100.0, strikes, 1.0, heston, 0.03, 0.01, nodes=160, upper_bound=240.0
        )
    )
    np.testing.assert_allclose(actual, expected, atol=2.0e-10, rtol=2.0e-10)


@pytest.mark.numerical
def test_bates_prices_respect_parity_with_jumps() -> None:
    p = BatesParameters(0.04, 1.5, 0.05, 0.6, -0.65, 0.4, -0.08, 0.18)
    call = bates_price(100.0, 105.0, 0.75, p, 0.025, 0.01, "call")
    put = bates_price(100.0, 105.0, 0.75, p, 0.025, 0.01, "put")
    parity = 100.0 * np.exp(-0.01 * 0.75) - 105.0 * np.exp(-0.025 * 0.75)
    assert call - put == pytest.approx(parity, abs=1.0e-10)


def _surface() -> SSVISurface:
    return SSVISurface(
        np.array([0.05, 0.25, 0.5, 1.0]),
        np.array([0.002, 0.01, 0.02, 0.04]),
        SSVIParameters(-0.3, 0.25, 0.5),
    )


@pytest.mark.numerical
def test_dupire_local_variance_is_positive_on_valid_ssvi_surface() -> None:
    surface = _surface()
    k = np.linspace(-0.5, 0.5, 51)
    variance = np.asarray(dupire_local_variance(surface, k, 0.5))
    assert np.all(np.isfinite(variance))
    assert np.all(variance > 0.0)


@pytest.mark.numerical
def test_local_vol_mc_is_consistent_with_source_surface_within_mc_error() -> None:
    surface = _surface()
    reference = float(ssvi_surface_price(surface, 100.0, 100.0, 0.5, 0.02, 0.01))
    estimate = local_vol_mc_price(
        surface, 100.0, 100.0, 0.5, 0.02, 0.01, paths=20_000, steps=64, seed=101
    )
    # Euler bias and first-slice clamping are included in this bounded test.
    assert abs(estimate.price - reference) <= 5.0 * estimate.standard_error + 0.12
    assert estimate.clamped_time_steps > 0
