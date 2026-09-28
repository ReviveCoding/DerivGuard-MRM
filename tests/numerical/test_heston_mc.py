import math

import pytest

from derivguard.development.heston_cf import HestonParameters, heston_price_adaptive
from derivguard.validation.heston_mc import ValidationHestonParameters, heston_mc_numpy


@pytest.mark.numerical
@pytest.mark.parametrize("scheme", ["full_truncation", "qe", "qe_m"])
def test_independent_mc_is_uncertainty_consistent_with_cf(scheme: str) -> None:
    # This is a bounded numerical test.  Research acceptance additionally
    # requires multiple seeds and at least three path/time-step levels.
    validation_params = ValidationHestonParameters(0.04, 2.0, 0.04, 0.5, -0.7)
    reference = float(
        heston_price_adaptive(
            100.0, 100.0, 1.0, HestonParameters(0.04, 2.0, 0.04, 0.5, -0.7), 0.03, 0.01
        )
    )
    estimate = heston_mc_numpy(
        100.0,
        100.0,
        1.0,
        validation_params,
        0.03,
        0.01,
        paths=60_000,
        steps=192,
        scheme=scheme,  # type: ignore[arg-type]
        seed=9821,
        antithetic=True,
        control_variate=True,
    )
    # Sampling uncertainty plus a small bounded discretization allowance.
    assert abs(estimate.price - reference) <= 3.0 * estimate.standard_error + 0.035
    assert estimate.confidence_low < estimate.confidence_high


def test_antithetic_seed_is_reproducible() -> None:
    p = ValidationHestonParameters(0.04, 2.0, 0.04, 0.5, -0.7)
    kwargs = dict(paths=2_000, steps=8, seed=123, antithetic=True, control_variate=False)
    first = heston_mc_numpy(100.0, 100.0, 0.5, p, **kwargs)
    second = heston_mc_numpy(100.0, 100.0, 0.5, p, **kwargs)
    assert first.price == second.price
    assert first.standard_error == second.standard_error


def test_qem_common_random_number_call_put_parity_is_sampling_consistent() -> None:
    """QE-M's discrete martingale correction is visible without a control variate."""

    p = ValidationHestonParameters(0.04, 2.0, 0.04, 0.8, -0.7)
    common = dict(
        paths=80_000,
        steps=16,
        scheme="qe_m",
        seed=1741,
        antithetic=True,
        control_variate=False,
    )
    call = heston_mc_numpy(100.0, 100.0, 1.0, p, 0.03, 0.01, option_type="call", **common)
    put = heston_mc_numpy(100.0, 100.0, 1.0, p, 0.03, 0.01, option_type="put", **common)
    parity_target = 100.0 * math.exp(-0.01) - 100.0 * math.exp(-0.03)
    # Common paths make C-P the discounted terminal-stock estimator.  Use a
    # conservative bound because payoff SEs are individually estimated and
    # their positive covariance is intentionally not assumed here.
    combined_se = (call.standard_error**2 + put.standard_error**2) ** 0.5
    assert abs((call.price - put.price) - parity_target) <= 3.0 * combined_se
