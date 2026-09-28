"""Safeguarded implied-volatility inversion for European options."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from .black_scholes import option_price

OptionType = Literal["call", "put"]


@dataclass(frozen=True)
class IVResult:
    """Result and diagnostics from a scalar bracketed IV solve."""

    volatility: float
    converged: bool
    iterations: int
    price_residual: float
    status: str
    lower_bound: float
    upper_bound: float


def no_arbitrage_bounds(
    spot: float,
    strike: float,
    maturity: float,
    rate: float = 0.0,
    dividend_yield: float = 0.0,
    option_type: OptionType = "call",
) -> tuple[float, float]:
    """Return European price bounds under deterministic discounting."""

    if not all(np.isfinite(x) for x in (spot, strike, maturity, rate, dividend_yield)):
        raise ValueError("bound inputs must be finite")
    if spot <= 0.0 or strike <= 0.0 or maturity < 0.0:
        raise ValueError("spot and strike must be positive; maturity must be non-negative")
    discounted_spot = spot * np.exp(-dividend_yield * maturity)
    discounted_strike = strike * np.exp(-rate * maturity)
    if option_type == "call":
        return max(discounted_spot - discounted_strike, 0.0), discounted_spot
    if option_type == "put":
        return max(discounted_strike - discounted_spot, 0.0), discounted_strike
    raise ValueError("option_type must be 'call' or 'put'")


def implied_volatility(
    price: float,
    spot: float,
    strike: float,
    maturity: float,
    rate: float = 0.0,
    dividend_yield: float = 0.0,
    option_type: OptionType = "call",
    *,
    price_tolerance: float = 1.0e-10,
    volatility_tolerance: float = 1.0e-12,
    initial_upper_volatility: float = 1.0,
    maximum_volatility: float = 10.0,
    max_iterations: int = 200,
) -> IVResult:
    """Invert a scalar option price using adaptive bracketing and bisection.

    Invalid prices return a diagnostic result rather than a fabricated
    volatility.  Programming errors such as non-positive spot/strike or invalid
    tolerances raise ``ValueError``.
    """

    lower_price, upper_price = no_arbitrage_bounds(
        spot, strike, maturity, rate, dividend_yield, option_type
    )
    if not np.isfinite(price):
        return IVResult(np.nan, False, 0, np.nan, "non_finite_price", lower_price, upper_price)
    if price_tolerance <= 0.0 or volatility_tolerance <= 0.0:
        raise ValueError("solver tolerances must be positive")
    if initial_upper_volatility <= 0.0 or maximum_volatility < initial_upper_volatility:
        raise ValueError("invalid volatility bracket configuration")
    if max_iterations < 1:
        raise ValueError("max_iterations must be positive")

    scale_tolerance = price_tolerance * max(1.0, abs(price), spot, strike)
    if price < lower_price - scale_tolerance:
        return IVResult(
            np.nan,
            False,
            0,
            price - lower_price,
            "below_no_arbitrage_bound",
            lower_price,
            upper_price,
        )
    if price > upper_price + scale_tolerance:
        return IVResult(
            np.nan,
            False,
            0,
            price - upper_price,
            "above_no_arbitrage_bound",
            lower_price,
            upper_price,
        )
    if maturity == 0.0:
        residual = price - lower_price
        status = "expiry_price" if abs(residual) <= scale_tolerance else "no_iv_at_expiry"
        return IVResult(
            0.0 if status == "expiry_price" else np.nan,
            status == "expiry_price",
            0,
            residual,
            status,
            lower_price,
            upper_price,
        )
    if abs(price - lower_price) <= scale_tolerance:
        return IVResult(
            0.0, True, 0, lower_price - price, "at_lower_bound", lower_price, upper_price
        )
    # The finite-volatility price approaches but does not attain the upper
    # bound.  Flag a boundary price instead of returning an arbitrary huge IV.
    if upper_price - price <= scale_tolerance:
        return IVResult(
            np.inf, False, 0, upper_price - price, "at_upper_bound", lower_price, upper_price
        )

    low_vol = 0.0
    high_vol = initial_upper_volatility
    high_price = float(
        option_price(spot, strike, maturity, high_vol, rate, dividend_yield, option_type)
    )
    while high_price < price and high_vol < maximum_volatility:
        high_vol = min(maximum_volatility, 2.0 * high_vol)
        high_price = float(
            option_price(spot, strike, maturity, high_vol, rate, dividend_yield, option_type)
        )
    if high_price < price:
        return IVResult(
            np.nan, False, 0, high_price - price, "not_bracketed", lower_price, upper_price
        )

    residual = high_price - price
    midpoint = high_vol
    for iteration in range(1, max_iterations + 1):
        midpoint = 0.5 * (low_vol + high_vol)
        model_price = float(
            option_price(spot, strike, maturity, midpoint, rate, dividend_yield, option_type)
        )
        residual = model_price - price
        if abs(residual) <= scale_tolerance or high_vol - low_vol <= volatility_tolerance:
            return IVResult(
                midpoint, True, iteration, residual, "converged", lower_price, upper_price
            )
        if residual < 0.0:
            low_vol = midpoint
        else:
            high_vol = midpoint
    return IVResult(
        midpoint, False, max_iterations, residual, "maximum_iterations", lower_price, upper_price
    )
