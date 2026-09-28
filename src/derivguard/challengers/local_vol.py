"""Dupire local-volatility challenger derived from a smooth SSVI surface."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray

from derivguard.market.black_scholes import option_price
from derivguard.market.svi import SSVISurface, density_factor, ssvi_derivatives

FloatArray = NDArray[np.float64]
OptionType = Literal["call", "put"]


@dataclass(frozen=True)
class LocalVolMonteCarloEstimate:
    price: float
    standard_error: float
    paths: int
    steps: int
    seed: int
    antithetic: bool
    clamped_time_steps: int


def dupire_local_variance(
    surface: SSVISurface,
    log_forward_moneyness: ArrayLike,
    maturity: ArrayLike,
    *,
    time_bump: float = 1.0e-4,
    density_floor: float = 1.0e-8,
) -> float | FloatArray:
    """Return Dupire local variance ``d_t w / g(k)``.

    The derivative is central in the interior and one-sided at the calibrated
    term-structure endpoints.  Extrapolation is refused rather than silently
    extending the SSVI surface.
    """

    k, t = np.broadcast_arrays(
        np.asarray(log_forward_moneyness, dtype=np.float64),
        np.asarray(maturity, dtype=np.float64),
    )
    if time_bump <= 0.0 or density_floor <= 0.0:
        raise ValueError("time_bump and density_floor must be positive")
    low = float(surface.maturities[0])
    high = float(surface.maturities[-1])
    if np.any(~np.isfinite(t)) or np.any(t < low) or np.any(t > high):
        raise ValueError("local volatility is defined only inside the SSVI maturity support")
    before = np.maximum(t - time_bump, low)
    after = np.minimum(t + time_bump, high)
    width = after - before
    if np.any(width <= 0.0):
        raise ValueError("SSVI maturity support is too narrow for time differentiation")
    w_before = np.asarray(surface.total_variance(k, before), dtype=np.float64)
    w_after = np.asarray(surface.total_variance(k, after), dtype=np.float64)
    time_derivative = (w_after - w_before) / width
    theta = surface.theta(t)
    w = np.asarray(surface.total_variance(k, t), dtype=np.float64)
    first, second = ssvi_derivatives(k, theta, surface.parameters)
    denominator = np.asarray(density_factor(k, w, first, second), dtype=np.float64)
    if np.any(denominator <= density_floor):
        raise ValueError("Dupire density denominator is non-positive or numerically singular")
    variance = time_derivative / denominator
    if np.any(~np.isfinite(variance)) or np.any(variance < 0.0):
        raise ValueError("SSVI surface implies invalid local variance")
    return float(variance) if variance.ndim == 0 else variance


def dupire_local_volatility(
    surface: SSVISurface,
    log_forward_moneyness: ArrayLike,
    maturity: ArrayLike,
    **kwargs: float,
) -> float | FloatArray:
    value = np.sqrt(
        np.asarray(
            dupire_local_variance(surface, log_forward_moneyness, maturity, **kwargs),
            dtype=np.float64,
        )
    )
    return float(value) if value.ndim == 0 else value


def ssvi_surface_price(
    surface: SSVISurface,
    spot: float,
    strike: ArrayLike,
    maturity: float,
    rate: float = 0.0,
    dividend_yield: float = 0.0,
    option_type: OptionType = "call",
) -> float | FloatArray:
    """Return the smooth SSVI reference value used to construct LocalVol."""

    strikes = np.asarray(strike, dtype=np.float64)
    forward = spot * np.exp((rate - dividend_yield) * maturity)
    k = np.log(strikes / forward)
    volatility = surface.implied_volatility(k, maturity)
    return option_price(spot, strikes, maturity, volatility, rate, dividend_yield, option_type)


def local_vol_mc_price(
    surface: SSVISurface,
    spot: float,
    strike: float,
    maturity: float,
    rate: float = 0.0,
    dividend_yield: float = 0.0,
    option_type: OptionType = "call",
    *,
    paths: int = 100_000,
    steps: int = 256,
    seed: int = 20260927,
    antithetic: bool = True,
) -> LocalVolMonteCarloEstimate:
    """Bounded independent Euler simulation under the SSVI Dupire diffusion.

    Times earlier than the first calibrated SSVI maturity use the first slice;
    this count is returned explicitly because it is a surface-support
    limitation, not an exact Dupire extrapolation.
    """

    if spot <= 0.0 or strike <= 0.0 or maturity <= 0.0:
        raise ValueError("spot, strike, and maturity must be positive")
    if paths < 2 or steps < 1:
        raise ValueError("paths >=2 and steps >=1 are required")
    if maturity > surface.maturities[-1]:
        raise ValueError("maturity exceeds SSVI support")
    if option_type not in {"call", "put"}:
        raise ValueError("option_type must be call or put")
    generated = (paths + 1) // 2 if antithetic else paths
    rng = np.random.default_rng(seed)
    log_spot = np.full(paths, np.log(spot), dtype=np.float64)
    dt = maturity / steps
    clamped = 0
    for step in range(steps):
        time = (step + 0.5) * dt
        supported_time = max(time, float(surface.maturities[0]))
        clamped += int(time < surface.maturities[0])
        z = rng.standard_normal(generated)
        if antithetic:
            z = np.concatenate((z, -z))[:paths]
        current = np.exp(log_spot)
        forward = spot * np.exp((rate - dividend_yield) * supported_time)
        k = np.log(current / forward)
        variance = np.asarray(dupire_local_variance(surface, k, supported_time), dtype=np.float64)
        log_spot += (rate - dividend_yield - 0.5 * variance) * dt + np.sqrt(variance * dt) * z
    terminal = np.exp(log_spot)
    payoff = (
        np.maximum(terminal - strike, 0.0)
        if option_type == "call"
        else np.maximum(strike - terminal, 0.0)
    )
    discounted = np.exp(-rate * maturity) * payoff
    return LocalVolMonteCarloEstimate(
        float(np.mean(discounted)),
        float(np.std(discounted, ddof=1) / np.sqrt(paths)),
        paths,
        steps,
        seed,
        antithetic,
        clamped,
    )
