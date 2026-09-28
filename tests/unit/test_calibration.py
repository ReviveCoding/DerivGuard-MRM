import numpy as np
import pytest
from scipy.optimize import OptimizeResult

import derivguard.development.calibration as calibration_module
from derivguard.development.calibration import (
    CalibrationData,
    CalibrationResult,
    bootstrap_calibration,
    calibrate_heston,
    calibration_loss,
    heston_objective,
    identifiability_diagnostics,
    profile_objective,
    rolling_stability,
)
from derivguard.development.heston_cf import HestonParameters, heston_price_fixed_quad
from derivguard.market.black_scholes import call_price, greeks
from derivguard.market.implied_volatility import implied_volatility


def _data() -> CalibrationData:
    spot = np.full(3, 100.0)
    strikes = np.array([90.0, 100.0, 110.0])
    maturity = np.full(3, 0.5)
    rate = np.full(3, 0.02)
    dividend = np.full(3, 0.01)
    iv = np.full(3, 0.2)
    price = np.asarray(call_price(spot, strikes, maturity, iv, rate, dividend))
    vega = np.asarray(greeks(spot, strikes, maturity, iv, rate, dividend).vega)
    return CalibrationData(
        spot,
        strikes,
        maturity,
        rate,
        dividend,
        np.array(["call"] * 3),
        price,
        iv,
        vega,
        np.maximum(price - 0.05, 0.0),
        price + 0.05,
    )


def test_calibration_objective_formulas_are_explicit() -> None:
    market = np.array([1.0, 2.0])
    model = np.array([2.0, 4.0])
    assert calibration_loss("C00", model, market) == pytest.approx(np.sqrt(2.5))
    assert calibration_loss(
        "C01", model, market, model_iv=np.array([0.3, 0.5]), market_iv=np.array([0.2, 0.3])
    ) == pytest.approx(np.sqrt(0.025))
    assert calibration_loss(
        "C02",
        model,
        market,
        model_iv=np.array([0.3, 0.5]),
        market_iv=np.array([0.2, 0.3]),
        vega=np.array([1.0, 3.0]),
    ) == pytest.approx(np.sqrt((0.01 + 3.0 * 0.04) / 4.0))
    assert calibration_loss("C03", model, market, spread=np.array([1.0, 2.0])) == 1.0
    assert (
        calibration_loss("C04", model, market, spread=np.array([1.0, 2.0]), huber_delta=1.5) == 0.5
    )


def test_hard_and_soft_feller_treatments() -> None:
    data = _data()
    violating = np.array([0.04, 0.5, 0.04, 1.0, -0.5])
    hard = heston_objective(violating, data, "C00", feller_treatment="hard", nodes=32)
    unconstrained = heston_objective(
        violating, data, "C00", feller_treatment="unconstrained", nodes=32
    )
    soft = heston_objective(
        violating, data, "C00", feller_treatment="soft", feller_penalty=10.0, nodes=32
    )
    assert np.isinf(hard)
    assert np.isfinite(unconstrained)
    assert soft > unconstrained


def test_quote_bootstrap_records_failures_without_fabricating_draws() -> None:
    data = _data()

    def calibrator(sample: CalibrationData, seed: int) -> CalibrationResult:
        p = HestonParameters(0.04 + (seed % 3) * 0.001, 2.0, 0.04, 0.4, -0.6)
        return CalibrationResult(p, 0.1, "C00", "local", "unconstrained", True, "ok", 1, 1, 0)

    result = bootstrap_calibration(data, calibrator, draws=5, seed=11)
    assert np.all(result.successful)
    assert result.parameter_draws.shape == (5, 5)
    assert result.covariance.shape == (5, 5)


def test_identifiability_profile_and_stability_diagnostics() -> None:
    results = [
        CalibrationResult(
            HestonParameters(
                0.03 + i * 0.005, 1.0 + 0.2 * i, 0.04 + i * 0.002, 0.4 + i * 0.03, -0.7 + i * 0.04
            ),
            0.1,
            "C00",
            "local",
            "unconstrained",
            True,
            "ok",
            1,
            1,
            i,
        )
        for i in range(4)
    ]
    diagnostics = identifiability_diagnostics(results)
    assert diagnostics.successful_solutions == 4
    assert diagnostics.parameter_correlation.shape == (5, 5)
    profiles = profile_objective(
        lambda x: float(np.sum((x - np.array([0.04, 2.0, 0.04, 0.5, -0.6])) ** 2)),
        "rho",
        [-0.8, -0.6, -0.4],
        [(0.01, 0.2), (0.1, 5.0), (0.01, 0.2), (0.1, 1.5), (-0.95, 0.0)],
        [0.04, 2.0, 0.04, 0.5, -0.6],
    )
    assert min(profiles, key=lambda point: point.objective_value).fixed_value == -0.6
    history = np.array(
        [
            [0.04, 2.0, 0.04, 0.5, -0.6],
            [0.041, 2.1, 0.041, 0.51, -0.59],
            [0.042, 2.2, 0.042, 0.52, -0.58],
            [0.043, 2.3, 0.043, 0.53, -0.57],
            [0.08, 4.0, 0.08, 1.0, -0.2],
        ]
    )
    stability = rolling_stability(history)
    assert stability.jump_score[-1] > stability.jump_score[1]


def test_calibration_data_prices_recover_known_heston_surface() -> None:
    params = HestonParameters(0.04, 2.0, 0.04, 0.4, -0.6)
    data = _data()
    prices = np.asarray(
        heston_price_fixed_quad(100.0, data.strike, 0.5, params, 0.02, 0.01, nodes=64)
    )
    ivs = np.array(
        [
            implied_volatility(float(p), 100.0, float(k), 0.5, 0.02, 0.01).volatility
            for p, k in zip(prices, data.strike, strict=True)
        ]
    )
    surface = CalibrationData(
        data.spot,
        data.strike,
        data.maturity,
        data.rate,
        data.dividend_yield,
        data.option_type,
        prices,
        ivs,
        data.vega,
        np.maximum(prices - 0.02, 0.0),
        prices + 0.02,
    )
    loss = heston_objective(np.array([0.04, 2.0, 0.04, 0.4, -0.6]), surface, "C00", nodes=64)
    assert loss < 1.0e-11


def test_multistart_prefers_converged_solution_over_lower_unconverged_objective(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    outcomes = iter(
        (
            OptimizeResult(
                x=np.array([0.04, 2.0, 0.04, 0.4, -0.6]),
                fun=0.01,
                success=False,
                message="iteration limit",
                nfev=20,
                nit=2,
            ),
            OptimizeResult(
                x=np.array([0.05, 1.8, 0.05, 0.35, -0.5]),
                fun=0.02,
                success=True,
                message="converged",
                nfev=12,
                nit=1,
            ),
        )
    )
    monkeypatch.setattr(calibration_module, "minimize", lambda *args, **kwargs: next(outcomes))

    result = calibrate_heston(
        _data(), optimizer="multistart", multistarts=2, max_iterations=2, nodes=16
    )

    assert result.success is True
    assert result.objective_value == pytest.approx(0.02)
    assert result.start_index == 1
