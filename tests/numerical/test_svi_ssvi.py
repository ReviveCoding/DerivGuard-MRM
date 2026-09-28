import numpy as np
import pytest

from derivguard.market.svi import (
    SSVIParameters,
    SSVISurface,
    SVIParameters,
    check_svi_slice,
    ssvi_derivatives,
    ssvi_total_variance,
    svi_derivatives,
    svi_total_variance,
)


def test_svi_analytic_derivatives() -> None:
    parameters = SVIParameters(a=0.02, b=0.15, rho=-0.35, m=0.03, sigma=0.25)
    k = np.linspace(-1.0, 1.0, 101)
    h = 1.0e-5
    first, second = svi_derivatives(k, parameters)
    numeric_first = (
        np.asarray(svi_total_variance(k + h, parameters))
        - np.asarray(svi_total_variance(k - h, parameters))
    ) / (2.0 * h)
    numeric_second = (
        np.asarray(svi_total_variance(k + h, parameters))
        - 2.0 * np.asarray(svi_total_variance(k, parameters))
        + np.asarray(svi_total_variance(k - h, parameters))
    ) / (h * h)
    np.testing.assert_allclose(first, numeric_first, atol=2.0e-11)
    np.testing.assert_allclose(second, numeric_second, atol=2.0e-6)


def test_svi_slice_diagnostics_accepts_regular_slice() -> None:
    diagnostics = check_svi_slice(SVIParameters(a=0.025, b=0.08, rho=-0.25, m=0.0, sigma=0.3))
    assert diagnostics.valid, diagnostics.messages
    assert diagnostics.minimum_total_variance > 0.0
    assert diagnostics.minimum_density_factor >= 0.0


def test_invalid_svi_parameters_rejected() -> None:
    with pytest.raises(ValueError):
        svi_total_variance(0.0, SVIParameters(0.01, -0.2, 0.0, 0.0, 0.1))


def test_ssvi_atm_variance_and_surface_calendar_behavior() -> None:
    parameters = SSVIParameters(rho=-0.3, eta=0.35, gamma=0.5)
    maturities = np.array([0.1, 0.25, 0.5, 1.0, 2.0])
    theta = np.array([0.004, 0.01, 0.02, 0.04, 0.08])
    surface = SSVISurface(maturities, theta, parameters)
    np.testing.assert_allclose(surface.total_variance(0.0, maturities), theta)
    assert surface.diagnostics().valid
    grid = np.linspace(-1.0, 1.0, 101)
    short = np.asarray(surface.total_variance(grid, 0.25))
    long = np.asarray(surface.total_variance(grid, 1.0))
    assert np.all(long >= short)


def test_ssvi_sufficient_condition_violation_is_reported() -> None:
    surface = SSVISurface(
        np.array([0.1, 1.0]),
        np.array([0.02, 0.2]),
        SSVIParameters(rho=0.8, eta=5.0, gamma=0.0),
    )
    diagnostics = surface.diagnostics()
    assert not diagnostics.valid
    assert any("condition exceeded" in message for message in diagnostics.messages)


def test_ssvi_refuses_extrapolation() -> None:
    surface = SSVISurface(
        np.array([0.25, 1.0]),
        np.array([0.01, 0.04]),
        SSVIParameters(rho=-0.2, eta=0.3, gamma=0.5),
    )
    with pytest.raises(ValueError, match="extrapolation"):
        surface.total_variance(0.0, 2.0)


def test_ssvi_vectorized_formula_is_positive() -> None:
    k = np.linspace(-3.0, 3.0, 301)
    values = np.asarray(ssvi_total_variance(k, 0.04, SSVIParameters(-0.4, 0.4, 0.5)))
    assert np.all(np.isfinite(values))
    assert np.all(values > 0.0)


def test_ssvi_analytic_derivatives_and_density_diagnostic() -> None:
    parameters = SSVIParameters(-0.35, 0.4, 0.5)
    k = np.linspace(-1.5, 1.5, 101)
    theta = 0.04
    h = 1.0e-5
    first, second = ssvi_derivatives(k, theta, parameters)
    center = np.asarray(ssvi_total_variance(k, theta, parameters))
    up = np.asarray(ssvi_total_variance(k + h, theta, parameters))
    down = np.asarray(ssvi_total_variance(k - h, theta, parameters))
    np.testing.assert_allclose(first, (up - down) / (2.0 * h), atol=2.0e-11)
    np.testing.assert_allclose(second, (up - 2.0 * center + down) / (h * h), atol=2.0e-6)
    surface = SSVISurface(np.array([0.25, 1.0]), np.array([0.01, theta]), parameters)
    diagnostics = surface.diagnostics(log_moneyness_grid=k)
    assert diagnostics.valid
    assert diagnostics.minimum_density_factor >= 0.0
