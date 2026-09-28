"""Bates stochastic-volatility-with-lognormal-jumps challenger.

The characteristic function is implemented locally rather than wrapping the
developer Heston pricer.  Jump sizes are normal in log space and the stock
drift includes the risk-neutral compound-Poisson compensator.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray

FloatArray = NDArray[np.float64]
OptionType = Literal["call", "put"]


@dataclass(frozen=True)
class BatesParameters:
    v0: float
    kappa: float
    theta: float
    sigma_v: float
    rho: float
    jump_intensity: float
    mean_log_jump: float
    jump_volatility: float

    def validate(self) -> None:
        values = np.asarray(
            (
                self.v0,
                self.kappa,
                self.theta,
                self.sigma_v,
                self.rho,
                self.jump_intensity,
                self.mean_log_jump,
                self.jump_volatility,
            )
        )
        if np.any(~np.isfinite(values)):
            raise ValueError("Bates parameters must be finite")
        if self.v0 < 0.0 or self.theta < 0.0 or self.kappa <= 0.0 or self.sigma_v <= 0.0:
            raise ValueError("invalid variance-process parameters")
        if abs(self.rho) > 1.0:
            raise ValueError("rho must lie in [-1,1]")
        if self.jump_intensity < 0.0 or self.jump_volatility < 0.0:
            raise ValueError("jump intensity and volatility must be non-negative")

    @property
    def mean_relative_jump(self) -> float:
        return float(np.exp(self.mean_log_jump + 0.5 * self.jump_volatility**2) - 1.0)


def _bates_cf(
    u: NDArray[np.complex128],
    spot: float,
    maturity: float,
    rate: float,
    dividend_yield: float,
    p: BatesParameters,
) -> NDArray[np.complex128]:
    """Risk-neutral Bates characteristic function with a decaying branch."""

    iu = 1j * u
    beta = p.kappa - p.rho * p.sigma_v * iu
    root = np.sqrt(beta * beta + p.sigma_v**2 * (u * u + iu))
    root = np.where(np.real(root) < 0.0, -root, root)
    ratio = (beta - root) / (beta + root)
    decay = np.exp(-root * maturity)
    log_ratio = np.log1p(-ratio * decay) - np.log1p(-ratio)
    compensated_drift = rate - dividend_yield - p.jump_intensity * p.mean_relative_jump
    c = iu * (np.log(spot) + compensated_drift * maturity) + (p.kappa * p.theta / p.sigma_v**2) * (
        (beta - root) * maturity - 2.0 * log_ratio
    )
    d = (beta - root) / p.sigma_v**2 * (1.0 - decay) / (1.0 - ratio * decay)
    jump_cf = np.exp(
        p.jump_intensity
        * maturity
        * (np.exp(iu * p.mean_log_jump - 0.5 * p.jump_volatility**2 * u * u) - 1.0)
    )
    return np.asarray(np.exp(c + d * p.v0) * jump_cf, dtype=np.complex128)


def bates_price(
    spot: float,
    strike: ArrayLike,
    maturity: float,
    parameters: BatesParameters,
    rate: float = 0.0,
    dividend_yield: float = 0.0,
    option_type: OptionType = "call",
    *,
    nodes: int = 160,
    upper_bound: float = 240.0,
) -> float | FloatArray:
    """Price European options with fixed Gauss--Legendre integration."""

    parameters.validate()
    strikes = np.asarray(strike, dtype=np.float64)
    if spot <= 0.0 or maturity < 0.0 or np.any(strikes <= 0.0):
        raise ValueError("spot/strikes must be positive and maturity non-negative")
    if option_type not in {"call", "put"}:
        raise ValueError("option_type must be call or put")
    if nodes < 16 or upper_bound <= 0.0:
        raise ValueError("nodes must be >=16 and upper_bound positive")
    if maturity == 0.0:
        value = np.maximum(spot - strikes, 0.0)
        if option_type == "put":
            value = np.maximum(strikes - spot, 0.0)
        return float(value) if value.ndim == 0 else value
    roots, weights = np.polynomial.legendre.leggauss(nodes)
    u = np.asarray(0.5 * upper_bound * (roots + 1.0), dtype=np.complex128)
    weights = 0.5 * upper_bound * weights
    flat = strikes.reshape(-1)
    phase = np.exp(-1j * np.log(flat[:, None]) * u[None, :])
    phi = _bates_cf(u, spot, maturity, rate, dividend_yield, parameters)
    phi_shift = _bates_cf(u - 1j, spot, maturity, rate, dividend_yield, parameters)
    # E[S_T] is fixed by the compensator, irrespective of jump parameters.
    first_moment = spot * np.exp((rate - dividend_yield) * maturity)
    p1 = (
        0.5
        + np.sum(
            weights[None, :] * np.real(phase * phi_shift[None, :] / (1j * u * first_moment)),
            axis=1,
        )
        / np.pi
    )
    p2 = 0.5 + np.sum(weights[None, :] * np.real(phase * phi[None, :] / (1j * u)), axis=1) / np.pi
    calls = spot * np.exp(-dividend_yield * maturity) * p1 - flat * np.exp(-rate * maturity) * p2
    if option_type == "put":
        calls = calls - spot * np.exp(-dividend_yield * maturity) + flat * np.exp(-rate * maturity)
    result = np.asarray(calls.reshape(strikes.shape), dtype=np.float64)
    return float(result) if result.ndim == 0 else result
