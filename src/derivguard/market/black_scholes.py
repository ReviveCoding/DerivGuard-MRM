"""Black--Scholes prices and analytic Greeks.

The implementation uses continuously compounded rates and a continuous
dividend yield.  ``theta`` is the calendar-time derivative (value change per
year as expiry approaches), matching the convention commonly quoted by risk
systems.  At expiry gamma, vega, and theta are reported as zero.  For
deterministic zero-volatility contracts, delta and theta use their limiting
values away from the forward exercise boundary; delta uses half-delta and
theta uses zero exactly at the kink.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import erf
from typing import Literal, cast

import numpy as np
from numpy.typing import ArrayLike, NDArray

OptionType = Literal["call", "put"]
FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class BlackScholesGreeks:
    """Analytic first- and second-order Black--Scholes sensitivities."""

    delta: float | FloatArray
    gamma: float | FloatArray
    vega: float | FloatArray
    theta: float | FloatArray


def _normal_cdf(x: FloatArray) -> FloatArray:
    # NumPy does not expose erf in every supported version.  The fallback is
    # deterministic and is adequate for the bounded foundational workload.
    try:
        from scipy.special import ndtr

        return cast(FloatArray, np.asarray(ndtr(x), dtype=np.float64))
    except ImportError:  # pragma: no cover - SciPy is a required dependency
        vec_erf = np.vectorize(erf, otypes=[np.float64])
        return cast(FloatArray, 0.5 * (1.0 + vec_erf(x / np.sqrt(2.0))))


def _normal_pdf(x: FloatArray) -> FloatArray:
    return cast(FloatArray, np.exp(-0.5 * x * x) / np.sqrt(2.0 * np.pi))


def _inputs(
    spot: ArrayLike,
    strike: ArrayLike,
    maturity: ArrayLike,
    volatility: ArrayLike,
    rate: ArrayLike,
    dividend_yield: ArrayLike,
) -> tuple[FloatArray, FloatArray, FloatArray, FloatArray, FloatArray, FloatArray]:
    values = tuple(
        np.asarray(x, dtype=np.float64)
        for x in (spot, strike, maturity, volatility, rate, dividend_yield)
    )
    s, k, t, sigma, r, q = np.broadcast_arrays(*values)
    if not all(np.all(np.isfinite(x)) for x in (s, k, t, sigma, r, q)):
        raise ValueError("Black-Scholes inputs must be finite")
    if np.any(s <= 0.0):
        raise ValueError("spot must be strictly positive")
    if np.any(k <= 0.0):
        raise ValueError("strike must be strictly positive")
    if np.any(t < 0.0):
        raise ValueError("maturity must be non-negative")
    if np.any(sigma < 0.0):
        raise ValueError("volatility must be non-negative")
    return tuple(np.asarray(x, dtype=np.float64) for x in (s, k, t, sigma, r, q))  # type: ignore[return-value]


def _scalarize(value: FloatArray) -> float | FloatArray:
    return float(value) if value.ndim == 0 else value


def _validate_option_type(option_type: str) -> OptionType:
    normalized = option_type.lower()
    if normalized not in {"call", "put"}:
        raise ValueError("option_type must be 'call' or 'put'")
    return normalized  # type: ignore[return-value]


def option_price(
    spot: ArrayLike,
    strike: ArrayLike,
    maturity: ArrayLike,
    volatility: ArrayLike,
    rate: ArrayLike = 0.0,
    dividend_yield: ArrayLike = 0.0,
    option_type: OptionType = "call",
) -> float | FloatArray:
    """Return a European option price, broadcasting numeric inputs.

    The expiry and zero-volatility branches are evaluated analytically and do
    not rely on divisions by a small artificial epsilon.
    """

    kind = _validate_option_type(option_type)
    s, k, t, sigma, r, q = _inputs(spot, strike, maturity, volatility, rate, dividend_yield)
    discount_r = np.exp(-r * t)
    discount_q = np.exp(-q * t)
    discounted_spot = s * discount_q
    discounted_strike = k * discount_r
    deterministic_call = np.maximum(discounted_spot - discounted_strike, 0.0)

    regular = (t > 0.0) & (sigma > 0.0)
    sqrt_t = np.sqrt(np.where(regular, t, 1.0))
    sigma_safe = np.where(regular, sigma, 1.0)
    d1 = (np.log(s / k) + (r - q + 0.5 * sigma_safe * sigma_safe) * t) / (sigma_safe * sqrt_t)
    d2 = d1 - sigma_safe * sqrt_t
    regular_call = discounted_spot * _normal_cdf(d1) - discounted_strike * _normal_cdf(d2)
    call = np.where(regular, regular_call, deterministic_call)
    if kind == "call":
        return _scalarize(call)
    # Direct evaluation avoids losing a small put value by subtracting two
    # nearly equal, deep-in-the-money call/parity terms.
    deterministic_put = np.maximum(discounted_strike - discounted_spot, 0.0)
    regular_put = discounted_strike * _normal_cdf(-d2) - discounted_spot * _normal_cdf(-d1)
    put = np.where(regular, regular_put, deterministic_put)
    return _scalarize(put)


def call_price(
    spot: ArrayLike,
    strike: ArrayLike,
    maturity: ArrayLike,
    volatility: ArrayLike,
    rate: ArrayLike = 0.0,
    dividend_yield: ArrayLike = 0.0,
) -> float | FloatArray:
    return option_price(spot, strike, maturity, volatility, rate, dividend_yield, "call")


def put_price(
    spot: ArrayLike,
    strike: ArrayLike,
    maturity: ArrayLike,
    volatility: ArrayLike,
    rate: ArrayLike = 0.0,
    dividend_yield: ArrayLike = 0.0,
) -> float | FloatArray:
    return option_price(spot, strike, maturity, volatility, rate, dividend_yield, "put")


def greeks(
    spot: ArrayLike,
    strike: ArrayLike,
    maturity: ArrayLike,
    volatility: ArrayLike,
    rate: ArrayLike = 0.0,
    dividend_yield: ArrayLike = 0.0,
    option_type: OptionType = "call",
) -> BlackScholesGreeks:
    """Return analytic delta, gamma, vega, and annualized theta."""

    kind = _validate_option_type(option_type)
    s, k, t, sigma, r, q = _inputs(spot, strike, maturity, volatility, rate, dividend_yield)
    discount_r = np.exp(-r * t)
    discount_q = np.exp(-q * t)
    regular = (t > 0.0) & (sigma > 0.0)
    sqrt_t = np.sqrt(np.where(regular, t, 1.0))
    sigma_safe = np.where(regular, sigma, 1.0)
    d1 = (np.log(s / k) + (r - q + 0.5 * sigma_safe * sigma_safe) * t) / (sigma_safe * sqrt_t)
    d2 = d1 - sigma_safe * sqrt_t
    pdf_d1 = _normal_pdf(d1)

    boundary_gap = s * discount_q - k * discount_r
    step = np.where(boundary_gap > 0.0, 1.0, np.where(boundary_gap < 0.0, 0.0, 0.5))
    if kind == "call":
        delta_regular = discount_q * _normal_cdf(d1)
        delta_limit = discount_q * step
        theta_regular = (
            -s * discount_q * pdf_d1 * sigma / (2.0 * sqrt_t)
            + q * s * discount_q * _normal_cdf(d1)
            - r * k * discount_r * _normal_cdf(d2)
        )
        theta_deterministic = np.where(
            boundary_gap > 0.0,
            q * s * discount_q - r * k * discount_r,
            0.0,
        )
    else:
        delta_regular = discount_q * (_normal_cdf(d1) - 1.0)
        delta_limit = discount_q * (step - 1.0)
        theta_regular = (
            -s * discount_q * pdf_d1 * sigma / (2.0 * sqrt_t)
            - q * s * discount_q * _normal_cdf(-d1)
            + r * k * discount_r * _normal_cdf(-d2)
        )
        theta_deterministic = np.where(
            boundary_gap < 0.0,
            r * k * discount_r - q * s * discount_q,
            0.0,
        )

    gamma_regular = discount_q * pdf_d1 / (s * sigma_safe * sqrt_t)
    vega_regular = s * discount_q * pdf_d1 * sqrt_t
    zeros = np.zeros_like(s)
    return BlackScholesGreeks(
        delta=_scalarize(np.where(regular, delta_regular, delta_limit)),
        gamma=_scalarize(np.where(regular, gamma_regular, zeros)),
        vega=_scalarize(np.where(regular, vega_regular, zeros)),
        theta=_scalarize(
            np.where(regular, theta_regular, np.where(t > 0.0, theta_deterministic, zeros))
        ),
    )
