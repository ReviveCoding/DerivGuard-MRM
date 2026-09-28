"""Developer Heston characteristic-function pricing engines.

The CPU reference uses adaptive quadrature.  The fixed Gauss--Legendre path
has both NumPy and lazy-imported Torch implementations and is intended for
large, vectorized CUDA batches.  The characteristic function is the stable
``exp(-dT)`` ("Little Heston Trap") representation; no complex logarithm of
an exponentially growing term is taken.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, cast

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.integrate import quad

OptionType = Literal["call", "put"]
type FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class HestonParameters:
    v0: float
    kappa: float
    theta: float
    sigma_v: float
    rho: float

    def validate(self) -> None:
        values = np.asarray((self.v0, self.kappa, self.theta, self.sigma_v, self.rho))
        if not np.all(np.isfinite(values)):
            raise ValueError("Heston parameters must be finite")
        if self.v0 < 0.0 or self.kappa <= 0.0 or self.theta < 0.0 or self.sigma_v <= 0.0:
            raise ValueError("v0/theta must be non-negative and kappa/sigma_v positive")
        if not -1.0 <= self.rho <= 1.0:
            raise ValueError("rho must lie in [-1, 1]")

    @property
    def feller_ratio(self) -> float:
        return 2.0 * self.kappa * self.theta / (self.sigma_v * self.sigma_v)


def _validate_market(spot: float, strike: ArrayLike, maturity: float) -> FloatArray:
    strikes = np.asarray(strike, dtype=np.float64)
    if not np.isfinite(spot) or spot <= 0.0:
        raise ValueError("spot must be finite and positive")
    if not np.all(np.isfinite(strikes)) or np.any(strikes <= 0.0):
        raise ValueError("strikes must be finite and positive")
    if not np.isfinite(maturity) or maturity < 0.0:
        raise ValueError("maturity must be finite and non-negative")
    return strikes


def _characteristic_function(
    u: np.complex128 | NDArray[np.complex128],
    spot: float,
    maturity: float,
    rate: float,
    dividend_yield: float,
    p: HestonParameters,
) -> np.complex128 | NDArray[np.complex128]:
    """Return E[exp(i*u*log(S_T))] using a branch-stable formulation."""

    z = np.asarray(u, dtype=np.complex128)
    iu = 1j * z
    beta = p.kappa - p.rho * p.sigma_v * iu
    d = np.sqrt(beta * beta + p.sigma_v * p.sigma_v * (z * z + iu))
    # Select the decaying square-root branch.  This keeps exp(-dT) bounded.
    d = np.where(np.real(d) < 0.0, -d, d)
    g = (beta - d) / (beta + d)
    exp_minus_dt = np.exp(-d * maturity)
    log_term = np.log1p(-g * exp_minus_dt) - np.log1p(-g)
    c = iu * (np.log(spot) + (rate - dividend_yield) * maturity) + (
        p.kappa * p.theta / (p.sigma_v * p.sigma_v)
    ) * ((beta - d) * maturity - 2.0 * log_term)
    dcoef = ((beta - d) / (p.sigma_v * p.sigma_v)) * (
        (1.0 - exp_minus_dt) / (1.0 - g * exp_minus_dt)
    )
    result = np.exp(c + dcoef * p.v0)
    if result.ndim == 0:
        return np.complex128(result)
    return cast(np.complex128 | NDArray[np.complex128], result)


def _call_from_probabilities(
    spot: float,
    strikes: FloatArray,
    maturity: float,
    rate: float,
    dividend_yield: float,
    p1_integrals: FloatArray,
    p2_integrals: FloatArray,
) -> FloatArray:
    prob1 = 0.5 + p1_integrals / np.pi
    prob2 = 0.5 + p2_integrals / np.pi
    return cast(
        FloatArray,
        spot * np.exp(-dividend_yield * maturity) * prob1
        - strikes * np.exp(-rate * maturity) * prob2,
    )


def heston_price_adaptive(
    spot: float,
    strike: ArrayLike,
    maturity: float,
    params: HestonParameters,
    rate: float = 0.0,
    dividend_yield: float = 0.0,
    option_type: OptionType = "call",
    *,
    absolute_tolerance: float = 1.0e-9,
    integration_limit: int = 300,
) -> float | FloatArray:
    """Price European options using independent adaptive integrations.

    This is deliberately a bounded CPU reference, not the research-scale
    calibration engine.  Each strike receives its own error-controlled SciPy
    integration over ``[0, infinity)``.
    """

    params.validate()
    strikes = _validate_market(spot, strike, maturity)
    if option_type not in {"call", "put"}:
        raise ValueError("option_type must be 'call' or 'put'")
    if maturity == 0.0:
        payoff = np.maximum(spot - strikes, 0.0)
        if option_type == "put":
            payoff = np.maximum(strikes - spot, 0.0)
        return float(payoff) if payoff.ndim == 0 else payoff

    # The first moment is known analytically.  Evaluating the generic formula
    # at u=-i can create a removable 0/0 in g for extreme positive rho.
    phi_minus_i = spot * np.exp((rate - dividend_yield) * maturity)

    def one_price(k: float) -> float:
        log_k = np.log(k)

        def integrand_1(u: float) -> float:
            numerator = np.exp(-1j * u * log_k) * _characteristic_function(
                np.complex128(u - 1j), spot, maturity, rate, dividend_yield, params
            )
            return float(np.real(numerator / (1j * u * phi_minus_i)))

        def integrand_2(u: float) -> float:
            numerator = np.exp(-1j * u * log_k) * _characteristic_function(
                np.complex128(u), spot, maturity, rate, dividend_yield, params
            )
            return float(np.real(numerator / (1j * u)))

        i1 = quad(
            integrand_1,
            0.0,
            np.inf,
            epsabs=absolute_tolerance,
            epsrel=absolute_tolerance,
            limit=integration_limit,
        )[0]
        i2 = quad(
            integrand_2,
            0.0,
            np.inf,
            epsabs=absolute_tolerance,
            epsrel=absolute_tolerance,
            limit=integration_limit,
        )[0]
        call = float(
            _call_from_probabilities(
                spot,
                np.asarray(k),
                maturity,
                rate,
                dividend_yield,
                np.asarray(i1),
                np.asarray(i2),
            )
        )
        if option_type == "call":
            return call
        return float(
            call - spot * np.exp(-dividend_yield * maturity) + k * np.exp(-rate * maturity)
        )

    flat = np.asarray([one_price(float(k)) for k in strikes.reshape(-1)]).reshape(strikes.shape)
    return float(flat) if flat.ndim == 0 else flat


def heston_price_fixed_quad(
    spot: float,
    strike: ArrayLike,
    maturity: float,
    params: HestonParameters,
    rate: float = 0.0,
    dividend_yield: float = 0.0,
    option_type: OptionType = "call",
    *,
    nodes: int = 128,
    upper_bound: float = 200.0,
) -> float | FloatArray:
    """Vectorized FP64 Gauss--Legendre pricing on a finite integration range."""

    params.validate()
    strikes = _validate_market(spot, strike, maturity)
    if nodes < 16 or upper_bound <= 0.0:
        raise ValueError("nodes must be >= 16 and upper_bound positive")
    if maturity == 0.0:
        payoff = np.maximum(spot - strikes, 0.0)
        if option_type == "put":
            payoff = np.maximum(strikes - spot, 0.0)
        return float(payoff) if payoff.ndim == 0 else payoff
    roots, raw_weights = np.polynomial.legendre.leggauss(nodes)
    u = 0.5 * upper_bound * (roots + 1.0)
    weights = 0.5 * upper_bound * raw_weights
    phase = np.exp(-1j * np.log(strikes.reshape(-1, 1)) * u.reshape(1, -1))
    phi_mi = spot * np.exp((rate - dividend_yield) * maturity)
    phi_1 = np.asarray(
        _characteristic_function(
            np.asarray(u - 1j, dtype=np.complex128),
            spot,
            maturity,
            rate,
            dividend_yield,
            params,
        )
    )
    phi_2 = np.asarray(
        _characteristic_function(
            np.asarray(u, dtype=np.complex128),
            spot,
            maturity,
            rate,
            dividend_yield,
            params,
        )
    )
    i1 = np.sum(weights * np.real(phase * phi_1 / (1j * u * phi_mi)), axis=1)
    i2 = np.sum(weights * np.real(phase * phi_2 / (1j * u)), axis=1)
    calls = _call_from_probabilities(
        spot, strikes.reshape(-1), maturity, rate, dividend_yield, i1, i2
    )
    if option_type == "put":
        calls = (
            calls
            - spot * np.exp(-dividend_yield * maturity)
            + strikes.reshape(-1) * np.exp(-rate * maturity)
        )
    elif option_type != "call":
        raise ValueError("option_type must be 'call' or 'put'")
    result = calls.reshape(strikes.shape)
    return float(result) if result.ndim == 0 else result


def heston_call_price_torch(
    spot: float,
    strikes: Any,
    maturity: float,
    parameter_matrix: Any,
    rate: float = 0.0,
    dividend_yield: float = 0.0,
    *,
    nodes: int = 128,
    upper_bound: float = 200.0,
) -> Any:
    """Return a ``[parameter_batch, strike_batch]`` Torch FP64 call matrix.

    ``parameter_matrix`` columns are ``v0,kappa,theta,sigma_v,rho``.  Inputs
    remain on their existing Torch device, allowing a single long-lived CUDA
    executor to batch population calibration without host round trips.
    """

    try:
        import torch
    except ImportError as exc:  # pragma: no cover - depends on optional stack
        raise RuntimeError("Torch is required for the vectorized backend") from exc

    if not isinstance(strikes, torch.Tensor) or not isinstance(parameter_matrix, torch.Tensor):
        raise TypeError("strikes and parameter_matrix must be Torch tensors")
    if parameter_matrix.ndim != 2 or parameter_matrix.shape[1] != 5:
        raise ValueError("parameter_matrix must have shape [batch, 5]")
    if strikes.ndim != 1 or torch.any(strikes <= 0.0):
        raise ValueError("strikes must be a positive one-dimensional tensor")
    if parameter_matrix.device != strikes.device:
        raise ValueError("parameters and strikes must be on the same device")
    dtype = torch.float64
    device = strikes.device
    p = parameter_matrix.to(dtype=dtype)
    if torch.any(p[:, :4] <= 0.0) or torch.any(torch.abs(p[:, 4]) > 1.0):
        raise ValueError("invalid Heston parameter matrix")
    roots_np, weights_np = np.polynomial.legendre.leggauss(nodes)
    u = torch.as_tensor(0.5 * upper_bound * (roots_np + 1.0), dtype=dtype, device=device)
    weights = torch.as_tensor(0.5 * upper_bound * weights_np, dtype=dtype, device=device)
    cdtype = torch.complex128
    uc = u.to(cdtype)[None, :]
    v0, kappa, theta, sigma, rho = (p[:, i : i + 1] for i in range(5))

    def cf(z: Any) -> Any:
        iu = 1j * z
        beta = kappa - rho * sigma * iu
        d = torch.sqrt(beta * beta + sigma * sigma * (z * z + iu))
        d = torch.where(torch.real(d) < 0.0, -d, d)
        g = (beta - d) / (beta + d)
        exp_dt = torch.exp(-d * maturity)
        log_term = torch.log1p(-g * exp_dt) - torch.log1p(-g)
        c = iu * (np.log(spot) + (rate - dividend_yield) * maturity) + (
            kappa * theta / (sigma * sigma)
        ) * ((beta - d) * maturity - 2.0 * log_term)
        dc = (beta - d) / (sigma * sigma) * (1.0 - exp_dt) / (1.0 - g * exp_dt)
        return torch.exp(c + dc * v0)

    phi_mi = torch.full(
        (p.shape[0], 1),
        spot * np.exp((rate - dividend_yield) * maturity),
        dtype=cdtype,
        device=device,
    )
    phi1 = cf(uc - 1j)
    phi2 = cf(uc)
    phase = torch.exp(
        -1j * torch.log(strikes.to(dtype=dtype))[:, None].to(cdtype) * uc
    )  # [strike,node]
    denom = 1j * uc
    i1 = torch.sum(
        weights[None, None, :]
        * torch.real(
            phase[None, :, :] * phi1[:, None, :] / (denom[:, None, :] * phi_mi[:, :, None])
        ),
        dim=-1,
    )
    i2 = torch.sum(
        weights[None, None, :]
        * torch.real(phase[None, :, :] * phi2[:, None, :] / denom[:, None, :]),
        dim=-1,
    )
    return spot * np.exp(-dividend_yield * maturity) * (0.5 + i1 / np.pi) - strikes.to(dtype=dtype)[
        None, :
    ] * np.exp(-rate * maturity) * (0.5 + i2 / np.pi)
