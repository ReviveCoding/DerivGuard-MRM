"""Heston calibration objectives, optimizer orchestration, and diagnostics.

This module contains developer-model calibration only.  Independent validator
engines must not import it.  The objective labels follow the project registry:
``C00`` price RMSE, ``C01`` IV RMSE, ``C02`` vega-weighted IV RMSE, ``C03``
spread-normalized price RMSE, and ``C04`` Huber loss on spread-scaled prices.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
from typing import Any, Literal, cast

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import differential_evolution, minimize

from derivguard.development.heston_cf import HestonParameters, heston_price_fixed_quad
from derivguard.market.implied_volatility import implied_volatility

FloatArray = NDArray[np.float64]
ObjectiveName = Literal["C00", "C01", "C02", "C03", "C04"]
OptimizerName = Literal["local", "multistart", "de", "de_local"]
FellerTreatment = Literal["unconstrained", "soft", "hard"]
PARAMETER_NAMES = ("v0", "kappa", "theta", "sigma_v", "rho")


@dataclass(frozen=True)
class CalibrationData:
    """One calibration surface with all quote-dependent scales retained."""

    spot: FloatArray
    strike: FloatArray
    maturity: FloatArray
    rate: FloatArray
    dividend_yield: FloatArray
    option_type: NDArray[np.str_]
    market_price: FloatArray
    market_iv: FloatArray
    vega: FloatArray
    bid: FloatArray
    ask: FloatArray

    def __post_init__(self) -> None:
        numeric_names = (
            "spot",
            "strike",
            "maturity",
            "rate",
            "dividend_yield",
            "market_price",
            "market_iv",
            "vega",
            "bid",
            "ask",
        )
        arrays = {name: np.asarray(getattr(self, name), dtype=np.float64) for name in numeric_names}
        kinds = np.asarray(self.option_type, dtype=np.str_)
        sizes = {value.size for value in (*arrays.values(), kinds)}
        if len(sizes) != 1 or not sizes or next(iter(sizes)) == 0:
            raise ValueError("all calibration fields must be non-empty equal-length vectors")
        if any(value.ndim != 1 for value in (*arrays.values(), kinds)):
            raise ValueError("calibration fields must be one-dimensional")
        if any(np.any(~np.isfinite(value)) for value in arrays.values()):
            raise ValueError("calibration numeric fields must be finite")
        if np.any(arrays["spot"] <= 0.0) or np.any(arrays["strike"] <= 0.0):
            raise ValueError("spot and strike must be positive")
        if np.any(arrays["maturity"] <= 0.0):
            raise ValueError("calibration maturities must be positive")
        if np.any(arrays["market_price"] < 0.0) or np.any(arrays["market_iv"] < 0.0):
            raise ValueError("market price and IV must be non-negative")
        if np.any(arrays["vega"] < 0.0):
            raise ValueError("vega must be non-negative")
        if np.any(arrays["bid"] < 0.0) or np.any(arrays["ask"] < arrays["bid"]):
            raise ValueError("quotes require 0 <= bid <= ask")
        kinds = np.char.lower(kinds)
        if not np.all(np.isin(kinds, ("call", "put"))):
            raise ValueError("option_type values must be call or put")
        for name, value in arrays.items():
            object.__setattr__(self, name, value.copy())
        object.__setattr__(self, "option_type", kinds.copy())

    def with_market_prices(self, prices: FloatArray) -> CalibrationData:
        value = np.asarray(prices, dtype=np.float64)
        if value.shape != self.market_price.shape:
            raise ValueError("replacement prices must match calibration data")
        iv = np.asarray(
            [
                implied_volatility(
                    float(price),
                    float(self.spot[i]),
                    float(self.strike[i]),
                    float(self.maturity[i]),
                    float(self.rate[i]),
                    float(self.dividend_yield[i]),
                    cast(Literal["call", "put"], str(self.option_type[i])),
                ).volatility
                for i, price in enumerate(value)
            ],
            dtype=np.float64,
        )
        if np.any(~np.isfinite(iv)):
            raise ValueError("bootstrap price draw does not admit finite implied volatility")
        return replace(self, market_price=value, market_iv=iv)


@dataclass(frozen=True)
class CalibrationResult:
    parameters: HestonParameters
    objective_value: float
    objective_name: ObjectiveName
    optimizer: OptimizerName
    feller_treatment: FellerTreatment
    success: bool
    message: str
    evaluations: int
    iterations: int
    start_index: int


@dataclass(frozen=True)
class BootstrapResult:
    parameter_draws: FloatArray
    successful: NDArray[np.bool_]
    covariance: FloatArray
    objective_values: FloatArray
    prediction_draws: FloatArray | None
    seed: int


@dataclass(frozen=True)
class IdentifiabilityDiagnostics:
    parameter_dispersion: FloatArray
    parameter_correlation: FloatArray
    successful_solutions: int
    condition_number: float


@dataclass(frozen=True)
class ProfilePoint:
    parameter: str
    fixed_value: float
    objective_value: float
    nuisance_parameters: FloatArray
    success: bool


@dataclass(frozen=True)
class StabilityDiagnostics:
    parameter_names: tuple[str, ...]
    levels: FloatArray
    changes: FloatArray
    standardized_changes: FloatArray
    jump_score: FloatArray


def calibration_loss(
    objective: ObjectiveName,
    model_price: FloatArray,
    market_price: FloatArray,
    *,
    model_iv: FloatArray | None = None,
    market_iv: FloatArray | None = None,
    vega: FloatArray | None = None,
    spread: FloatArray | None = None,
    spread_floor: float = 0.01,
    huber_delta: float = 1.5,
) -> float:
    """Calculate one documented calibration objective.

    C02 uses ``sqrt(sum(vega * dIV^2) / sum(vega))``. C03 divides price
    residuals by ``max(ask-bid, spread_floor)``. C04 is the mean Huber loss of
    the same dimensionless residual, rather than an RMSE with a hidden scale.
    """

    model = np.asarray(model_price, dtype=np.float64)
    market = np.asarray(market_price, dtype=np.float64)
    if model.shape != market.shape or model.ndim != 1 or model.size == 0:
        raise ValueError("model and market prices must be equal non-empty vectors")
    if np.any(~np.isfinite(model)) or np.any(~np.isfinite(market)):
        return float("inf")
    price_residual = model - market
    if objective == "C00":
        return float(np.sqrt(np.mean(price_residual * price_residual)))
    if objective in {"C01", "C02"}:
        if model_iv is None or market_iv is None:
            raise ValueError(f"{objective} requires model_iv and market_iv")
        model_iv_array = np.asarray(model_iv, dtype=np.float64)
        market_iv_array = np.asarray(market_iv, dtype=np.float64)
        if model_iv_array.shape != model.shape or market_iv_array.shape != model.shape:
            raise ValueError("IV vectors must match price vectors")
        if np.any(~np.isfinite(model_iv_array)) or np.any(~np.isfinite(market_iv_array)):
            return float("inf")
        residual = model_iv_array - market_iv_array
        if objective == "C01":
            return float(np.sqrt(np.mean(residual * residual)))
        if vega is None:
            raise ValueError("C02 requires vega")
        weights = np.asarray(vega, dtype=np.float64)
        if weights.shape != model.shape or np.any(~np.isfinite(weights)) or np.any(weights < 0.0):
            raise ValueError("vega weights must be finite, non-negative, and shape-compatible")
        if np.sum(weights) <= 0.0:
            raise ValueError("C02 requires at least one positive vega")
        return float(np.sqrt(np.sum(weights * residual * residual) / np.sum(weights)))
    if objective not in {"C03", "C04"}:
        raise ValueError(f"unknown calibration objective {objective}")
    if spread is None or spread_floor <= 0.0:
        raise ValueError(f"{objective} requires spread and a positive spread_floor")
    scales = np.maximum(np.asarray(spread, dtype=np.float64), spread_floor)
    if scales.shape != model.shape or np.any(~np.isfinite(scales)):
        raise ValueError("spread must be finite and shape-compatible")
    residual = price_residual / scales
    if objective == "C03":
        return float(np.sqrt(np.mean(residual * residual)))
    if huber_delta <= 0.0:
        raise ValueError("huber_delta must be positive")
    absolute = np.abs(residual)
    huber = np.where(
        absolute <= huber_delta,
        0.5 * residual * residual,
        huber_delta * (absolute - 0.5 * huber_delta),
    )
    return float(np.mean(huber))


def parameter_vector(parameters: HestonParameters) -> FloatArray:
    return np.asarray(
        [parameters.v0, parameters.kappa, parameters.theta, parameters.sigma_v, parameters.rho],
        dtype=np.float64,
    )


def parameters_from_vector(value: Sequence[float]) -> HestonParameters:
    vector = np.asarray(value, dtype=np.float64)
    if vector.shape != (5,):
        raise ValueError("Heston parameter vector must have length five")
    result = HestonParameters(*map(float, vector))
    result.validate()
    return result


def price_calibration_data(
    data: CalibrationData,
    parameters: HestonParameters,
    *,
    nodes: int = 96,
    upper_bound: float = 200.0,
) -> FloatArray:
    """Price a heterogeneous surface, vectorizing each convention group."""

    output = np.empty(data.market_price.size, dtype=np.float64)
    # Use tuple grouping because mixed numeric/string NumPy matrices coerce all
    # values to strings and can merge distinct floating-point conventions.
    groups: dict[tuple[float, float, float, float, str], list[int]] = {}
    for i in range(data.market_price.size):
        key = (
            float(data.spot[i]),
            float(data.maturity[i]),
            float(data.rate[i]),
            float(data.dividend_yield[i]),
            str(data.option_type[i]),
        )
        groups.setdefault(key, []).append(i)
    for (spot, maturity, rate, dividend, kind), indices in groups.items():
        idx = np.asarray(indices, dtype=np.int64)
        output[idx] = np.asarray(
            heston_price_fixed_quad(
                spot,
                data.strike[idx],
                maturity,
                parameters,
                rate,
                dividend,
                cast(Literal["call", "put"], kind),
                nodes=nodes,
                upper_bound=upper_bound,
            ),
            dtype=np.float64,
        )
    return output


def _model_ivs(data: CalibrationData, prices: FloatArray) -> FloatArray:
    return np.asarray(
        [
            implied_volatility(
                float(prices[i]),
                float(data.spot[i]),
                float(data.strike[i]),
                float(data.maturity[i]),
                float(data.rate[i]),
                float(data.dividend_yield[i]),
                cast(Literal["call", "put"], str(data.option_type[i])),
            ).volatility
            for i in range(prices.size)
        ],
        dtype=np.float64,
    )


def heston_objective(
    vector: Sequence[float],
    data: CalibrationData,
    objective: ObjectiveName,
    *,
    feller_treatment: FellerTreatment = "unconstrained",
    feller_penalty: float = 10.0,
    spread_floor: float = 0.01,
    huber_delta: float = 1.5,
    nodes: int = 96,
    upper_bound: float = 200.0,
) -> float:
    try:
        parameters = parameters_from_vector(vector)
    except ValueError:
        return float("inf")
    violation = max(parameters.sigma_v**2 - 2.0 * parameters.kappa * parameters.theta, 0.0)
    if feller_treatment == "hard" and violation > 1.0e-12:
        return float("inf")
    if feller_treatment not in {"unconstrained", "soft", "hard"}:
        raise ValueError("unknown Feller treatment")
    prices = price_calibration_data(data, parameters, nodes=nodes, upper_bound=upper_bound)
    ivs = _model_ivs(data, prices) if objective in {"C01", "C02"} else None
    loss = calibration_loss(
        objective,
        prices,
        data.market_price,
        model_iv=ivs,
        market_iv=data.market_iv,
        vega=data.vega,
        spread=data.ask - data.bid,
        spread_floor=spread_floor,
        huber_delta=huber_delta,
    )
    if feller_treatment == "soft":
        if feller_penalty < 0.0:
            raise ValueError("feller_penalty must be non-negative")
        scale = max(parameters.sigma_v**2, 1.0e-12)
        loss += feller_penalty * (violation / scale) ** 2
    return float(loss)


def calibrate_heston(
    data: CalibrationData,
    *,
    objective: ObjectiveName = "C04",
    optimizer: OptimizerName = "de_local",
    bounds: Sequence[tuple[float, float]] = (
        (0.0025, 0.5),
        (0.05, 10.0),
        (0.0025, 0.5),
        (0.05, 2.5),
        (-0.999, 0.999),
    ),
    initial: HestonParameters | None = None,
    feller_treatment: FellerTreatment = "unconstrained",
    multistarts: int = 8,
    seed: int = 20260927,
    max_iterations: int = 250,
    de_population: int = 10,
    nodes: int = 96,
    upper_bound: float = 200.0,
    spread_floor: float = 0.01,
) -> CalibrationResult:
    """Run bounded local, multi-start, DE, or DE followed by local refinement."""

    bounds_tuple = tuple((float(low), float(high)) for low, high in bounds)
    if len(bounds_tuple) != 5 or any(low >= high for low, high in bounds_tuple):
        raise ValueError("five ordered parameter bounds are required")
    if multistarts < 1 or max_iterations < 1 or de_population < 5:
        raise ValueError("invalid optimizer controls")
    objective_function = lambda x: heston_objective(  # noqa: E731
        x,
        data,
        objective,
        feller_treatment=feller_treatment,
        nodes=nodes,
        upper_bound=upper_bound,
        spread_floor=spread_floor,
    )
    local_objective = lambda x: heston_objective(  # noqa: E731
        x,
        data,
        objective,
        # SLSQP enforces the explicit nonlinear inequality. Returning infinity
        # for its small infeasible trial steps would make finite differencing
        # unreliable, so only the local objective evaluation is unconstrained.
        feller_treatment="unconstrained" if feller_treatment == "hard" else feller_treatment,
        nodes=nodes,
        upper_bound=upper_bound,
        spread_floor=spread_floor,
    )
    rng = np.random.default_rng(seed)
    midpoint = np.asarray([(low + high) / 2.0 for low, high in bounds_tuple])
    initial_vector = parameter_vector(initial) if initial is not None else midpoint
    if np.any(initial_vector < np.asarray(bounds_tuple)[:, 0]) or np.any(
        initial_vector > np.asarray(bounds_tuple)[:, 1]
    ):
        raise ValueError("initial parameters lie outside bounds")

    def local(start: FloatArray) -> object:
        method = "SLSQP" if feller_treatment == "hard" else "L-BFGS-B"
        constraints: Any = ()
        local_start = start.copy()
        if feller_treatment == "hard":
            # An infinite hard-constraint objective at an infeasible initial
            # point prevents SLSQP from estimating a useful first gradient.
            # Move only the starting point to a demonstrably feasible bounded
            # point; the optimizer constraint remains the actual hard gate.
            if 2.0 * local_start[1] * local_start[2] < local_start[3] ** 2:
                local_start[1] = bounds_tuple[1][1]
                local_start[2] = bounds_tuple[2][1]
                local_start[3] = bounds_tuple[3][0]
            if 2.0 * local_start[1] * local_start[2] < local_start[3] ** 2:
                raise ValueError("parameter bounds contain no Feller-feasible starting point")
            constraints = (
                {
                    "type": "ineq",
                    "fun": lambda x: 2.0 * x[1] * x[2] - x[3] * x[3],
                },
            )
        return minimize(  # type: ignore[call-overload]
            local_objective,
            local_start,
            method=method,
            bounds=bounds_tuple,
            constraints=constraints,
            options={"maxiter": max_iterations, "ftol": 1.0e-10},
        )

    results: list[tuple[object, int]] = []
    if optimizer in {"local", "multistart"}:
        starts = [initial_vector]
        if optimizer == "multistart":
            for _ in range(multistarts - 1):
                starts.append(np.asarray([rng.uniform(low, high) for low, high in bounds_tuple]))
        results.extend((local(start), i) for i, start in enumerate(starts))
    elif optimizer in {"de", "de_local"}:
        de_result = differential_evolution(
            objective_function,
            bounds_tuple,
            seed=seed,
            maxiter=max_iterations,
            popsize=de_population,
            polish=False,
            updating="immediate",
        )
        results.append((de_result, 0))
        if optimizer == "de_local":
            results.append((local(np.asarray(de_result.x, dtype=np.float64)), 1))
    else:
        raise ValueError("unknown optimizer")
    # A finite low objective from an optimizer that exhausted its iteration
    # budget is useful diagnostic evidence, but it is not preferable to a
    # genuinely converged solution.  Selecting across all starts solely by
    # objective previously allowed a non-converged start to mask a converged
    # one.  Prefer successful finite candidates and fall back to the best
    # finite diagnostic candidate only when no start converged.
    finite_results = [
        pair
        for pair in results
        if np.isfinite(float(pair[0].fun))  # type: ignore[attr-defined]
        and np.all(np.isfinite(np.asarray(pair[0].x, dtype=np.float64)))  # type: ignore[attr-defined]
    ]
    if not finite_results:
        raise RuntimeError("all calibration optimizer starts returned non-finite results")
    successful_results = [
        pair
        for pair in finite_results
        if bool(pair[0].success)  # type: ignore[attr-defined]
    ]
    eligible_results = successful_results or finite_results
    best_result, start_index = min(
        eligible_results,
        key=lambda pair: float(pair[0].fun),  # type: ignore[attr-defined]
    )
    best_parameters = parameters_from_vector(best_result.x)  # type: ignore[attr-defined]
    return CalibrationResult(
        parameters=best_parameters,
        objective_value=float(best_result.fun),  # type: ignore[attr-defined]
        objective_name=objective,
        optimizer=optimizer,
        feller_treatment=feller_treatment,
        success=bool(best_result.success),  # type: ignore[attr-defined]
        message=str(best_result.message),  # type: ignore[attr-defined]
        evaluations=int(best_result.nfev),  # type: ignore[attr-defined]
        iterations=int(getattr(best_result, "nit", 0)),
        start_index=start_index,
    )


def bootstrap_calibration(
    data: CalibrationData,
    calibrator: Callable[[CalibrationData, int], CalibrationResult],
    *,
    draws: int,
    seed: int = 20260927,
    predictor: Callable[[HestonParameters], FloatArray] | None = None,
) -> BootstrapResult:
    """Uniform bid/ask quote bootstrap with deterministic per-draw seeds."""

    if draws < 2:
        raise ValueError("at least two bootstrap draws are required")
    rng = np.random.default_rng(seed)
    parameters = np.full((draws, 5), np.nan, dtype=np.float64)
    objectives = np.full(draws, np.nan, dtype=np.float64)
    successes = np.zeros(draws, dtype=bool)
    predictions: list[FloatArray] = []
    prediction_shape: tuple[int, ...] | None = None
    for index in range(draws):
        prices = rng.uniform(data.bid, data.ask)
        try:
            sampled = data.with_market_prices(np.asarray(prices, dtype=np.float64))
            result = calibrator(sampled, seed + index)
            parameters[index] = parameter_vector(result.parameters)
            objectives[index] = result.objective_value
            successes[index] = result.success and np.all(np.isfinite(parameters[index]))
        except (RuntimeError, ValueError, FloatingPointError):
            continue
        if predictor is not None and successes[index]:
            # Predictor shape/value errors are programming or contract errors,
            # not failed calibrations, and must not be silently reclassified.
            prediction = np.asarray(predictor(result.parameters), dtype=np.float64)
            if np.any(~np.isfinite(prediction)):
                raise ValueError("predictor returned non-finite values")
            if prediction_shape is None:
                prediction_shape = prediction.shape
            if prediction.shape != prediction_shape:
                raise ValueError("predictor returned inconsistent shapes")
            predictions.append(prediction)
    valid = parameters[successes]
    covariance = (
        np.asarray(np.cov(valid, rowvar=False), dtype=np.float64)
        if valid.shape[0] >= 2
        else np.full((5, 5), np.nan, dtype=np.float64)
    )
    prediction_draws = np.stack(predictions) if predictions else None
    return BootstrapResult(parameters, successes, covariance, objectives, prediction_draws, seed)


def identifiability_diagnostics(results: Sequence[CalibrationResult]) -> IdentifiabilityDiagnostics:
    successful = [result for result in results if result.success]
    if len(successful) < 2:
        raise ValueError("at least two successful calibration solutions are required")
    matrix = np.vstack([parameter_vector(result.parameters) for result in successful])
    dispersion = np.std(matrix, axis=0, ddof=1)
    correlation = np.asarray(np.corrcoef(matrix, rowvar=False), dtype=np.float64)
    covariance = np.cov(matrix, rowvar=False)
    condition = float(np.linalg.cond(covariance))
    return IdentifiabilityDiagnostics(dispersion, correlation, len(successful), condition)


def profile_objective(
    objective_function: Callable[[FloatArray], float],
    parameter: str,
    grid: Sequence[float],
    bounds: Sequence[tuple[float, float]],
    initial: Sequence[float],
    *,
    max_iterations: int = 200,
) -> tuple[ProfilePoint, ...]:
    """Minimize over nuisance parameters for each fixed profile-grid value."""

    if parameter not in PARAMETER_NAMES:
        raise ValueError("unknown Heston parameter")
    index = PARAMETER_NAMES.index(parameter)
    all_bounds = tuple(bounds)
    start = np.asarray(initial, dtype=np.float64)
    if len(all_bounds) != 5 or start.shape != (5,):
        raise ValueError("profile inputs require five parameters")
    nuisance_indices = [i for i in range(5) if i != index]
    points: list[ProfilePoint] = []
    nuisance_start = start[nuisance_indices]
    nuisance_bounds = [all_bounds[i] for i in nuisance_indices]
    for fixed in grid:
        if not all_bounds[index][0] <= fixed <= all_bounds[index][1]:
            raise ValueError("profile grid lies outside parameter bounds")

        fixed_value = float(fixed)

        def reduced(value: FloatArray, fixed_profile_value: float = fixed_value) -> float:
            full = start.copy()
            full[index] = fixed_profile_value
            full[nuisance_indices] = value
            return objective_function(full)

        result = minimize(
            reduced,
            nuisance_start,
            method="L-BFGS-B",
            bounds=nuisance_bounds,
            options={"maxiter": max_iterations, "ftol": 1.0e-10},
        )
        nuisance_start = np.asarray(result.x, dtype=np.float64)
        points.append(
            ProfilePoint(
                parameter,
                fixed_value,
                float(result.fun),
                nuisance_start.copy(),
                bool(result.success),
            )
        )
    return tuple(points)


def rolling_stability(parameter_history: FloatArray) -> StabilityDiagnostics:
    """Return robust standardized parameter changes and an aggregate jump score."""

    levels = np.asarray(parameter_history, dtype=np.float64)
    if levels.ndim != 2 or levels.shape[1] != 5 or levels.shape[0] < 2:
        raise ValueError("parameter history must have shape [dates>=2, 5]")
    if np.any(~np.isfinite(levels)):
        raise ValueError("parameter history must be finite")
    changes = np.diff(levels, axis=0, prepend=levels[[0]])
    historical = changes[1:]
    median = np.median(historical, axis=0)
    mad = np.median(np.abs(historical - median), axis=0)
    robust_scale = np.maximum(1.4826 * mad, np.finfo(np.float64).eps)
    standardized = (changes - median) / robust_scale
    standardized[0] = 0.0
    jump_score = np.sqrt(np.mean(standardized * standardized, axis=1))
    return StabilityDiagnostics(PARAMETER_NAMES, levels.copy(), changes, standardized, jump_score)
