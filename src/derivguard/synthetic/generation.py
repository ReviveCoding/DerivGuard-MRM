"""Reproducible synthetic ground-truth designs.

This module keeps latent truth separate from subsequently observed quotes.  It
provides an analytic Black--Scholes generator and a bounded CPU Heston
generator suitable for tests and reference designs.  Research-scale Heston
generation is deliberately delegated to the qualified GPU runner; calling the
CPU implementation with a large design is rejected rather than silently
running an expensive CPU workload.

Bates and LocalVol are declared explicitly unsupported here until their
canonical challenger engines have been independently qualified.  A caller
can therefore register an unavailable experiment without substituting a
different data-generating process under the requested name.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import NDArray

from derivguard.development.heston_cf import HestonParameters, heston_price_fixed_quad
from derivguard.market.black_scholes import call_price, greeks

FloatArray = NDArray[np.float64]
TruthModel = Literal["SYN-BS", "SYN-HESTON", "SYN-BATES", "SYN-LOCALVOL"]


class UnsupportedTruthModelError(NotImplementedError):
    """Raised instead of silently replacing an unavailable truth model."""


@dataclass(frozen=True)
class TruthCapability:
    model_id: TruthModel
    supported: bool
    status: str
    reason: str | None = None


@dataclass(frozen=True)
class SyntheticDesign:
    """Stratified scenario design, before any values have been generated."""

    scenario_id: NDArray[np.str_]
    spot: FloatArray
    strike: FloatArray
    maturity: FloatArray
    rate: FloatArray
    dividend_yield: FloatArray
    volatility: FloatArray
    heston_v0: FloatArray
    heston_kappa: FloatArray
    heston_theta: FloatArray
    heston_sigma_v: FloatArray
    heston_rho: FloatArray

    def __len__(self) -> int:
        return int(self.spot.size)


@dataclass(frozen=True)
class SyntheticTruth:
    """Latent values that must never be overwritten by observation noise."""

    design: SyntheticDesign
    model_id: TruthModel
    true_price: FloatArray
    true_delta: FloatArray
    true_gamma: FloatArray
    true_vega: FloatArray
    backend: str
    precision: str

    def __len__(self) -> int:
        return len(self.design)


def truth_capabilities() -> tuple[TruthCapability, ...]:
    return (
        TruthCapability("SYN-BS", True, "AVAILABLE"),
        TruthCapability("SYN-HESTON", True, "AVAILABLE_BOUNDED_CPU_REFERENCE"),
        TruthCapability(
            "SYN-BATES",
            False,
            "UNSUPPORTED",
            "Canonical Bates truth engine has not yet passed numerical qualification.",
        ),
        TruthCapability(
            "SYN-LOCALVOL",
            False,
            "UNSUPPORTED",
            "Canonical LocalVol truth engine has not yet passed numerical qualification.",
        ),
    )


def _latin_hypercube(n: int, dimensions: int, seed: int) -> FloatArray:
    if n < 1 or dimensions < 1:
        raise ValueError("n and dimensions must be positive")
    rng = np.random.default_rng(seed)
    result = np.empty((n, dimensions), dtype=np.float64)
    for column in range(dimensions):
        result[:, column] = (rng.permutation(n) + rng.random(n)) / n
    return result


def stratified_design(n: int, *, seed: int = 20260927) -> SyntheticDesign:
    """Return a reproducible Latin-hypercube design over documented ranges.

    Ranges are intentionally broad enough to cover ordinary and stressed
    European index-option conditions; they are design inputs, not empirical
    estimates.  Strike is derived from log-moneyness and spot.
    """

    unit = _latin_hypercube(n, 11, seed)

    def scale(column: int, low: float, high: float) -> FloatArray:
        return np.asarray(low + (high - low) * unit[:, column], dtype=np.float64)

    spot = scale(0, 80.0, 120.0)
    log_moneyness = scale(1, -0.35, 0.35)
    maturity = scale(2, 7.0 / 365.0, 2.0)
    rate = scale(3, 0.0, 0.08)
    dividend = scale(4, 0.0, 0.04)
    volatility = scale(5, 0.08, 0.65)
    v0 = scale(6, 0.01, 0.25)
    kappa = scale(7, 0.25, 6.0)
    theta = scale(8, 0.01, 0.20)
    sigma_v = scale(9, 0.10, 1.50)
    rho = scale(10, -0.95, 0.20)
    return SyntheticDesign(
        scenario_id=np.asarray([f"SYN-{seed}-{i:06d}" for i in range(n)]),
        spot=spot,
        strike=spot * np.exp(log_moneyness),
        maturity=maturity,
        rate=rate,
        dividend_yield=dividend,
        volatility=volatility,
        heston_v0=v0,
        heston_kappa=kappa,
        heston_theta=theta,
        heston_sigma_v=sigma_v,
        heston_rho=rho,
    )


def _finite_difference_heston_greeks(
    design: SyntheticDesign, prices: FloatArray, *, nodes: int, upper_bound: float
) -> tuple[FloatArray, FloatArray, FloatArray]:
    delta = np.empty(len(design), dtype=np.float64)
    gamma = np.empty(len(design), dtype=np.float64)
    vega = np.empty(len(design), dtype=np.float64)
    for index in range(len(design)):
        params = HestonParameters(
            design.heston_v0[index],
            design.heston_kappa[index],
            design.heston_theta[index],
            design.heston_sigma_v[index],
            design.heston_rho[index],
        )
        s = design.spot[index]
        ds = max(1.0e-3 * s, 1.0e-4)
        common = dict(
            strike=design.strike[index],
            maturity=design.maturity[index],
            params=params,
            rate=design.rate[index],
            dividend_yield=design.dividend_yield[index],
            nodes=nodes,
            upper_bound=upper_bound,
        )
        up = float(heston_price_fixed_quad(s + ds, **common))
        down = float(heston_price_fixed_quad(s - ds, **common))
        delta[index] = (up - down) / (2.0 * ds)
        gamma[index] = (up - 2.0 * prices[index] + down) / (ds * ds)
        dv = max(1.0e-3 * params.v0, 1.0e-5)
        upper_params = HestonParameters(
            params.v0 + dv, params.kappa, params.theta, params.sigma_v, params.rho
        )
        lower_params = HestonParameters(
            max(params.v0 - dv, 1.0e-10),
            params.kappa,
            params.theta,
            params.sigma_v,
            params.rho,
        )
        upper = float(
            heston_price_fixed_quad(
                s, params=upper_params, **{k: v for k, v in common.items() if k != "params"}
            )
        )
        lower = float(
            heston_price_fixed_quad(
                s, params=lower_params, **{k: v for k, v in common.items() if k != "params"}
            )
        )
        # Sensitivity to volatility sqrt(v0), comparable in units to BS vega.
        sigma_up = np.sqrt(params.v0 + dv)
        sigma_down = np.sqrt(max(params.v0 - dv, 1.0e-10))
        vega[index] = (upper - lower) / (sigma_up - sigma_down)
    return delta, gamma, vega


def generate_truth(
    design: SyntheticDesign,
    model_id: TruthModel,
    *,
    heston_nodes: int = 96,
    heston_upper_bound: float = 180.0,
    max_cpu_heston_scenarios: int = 128,
) -> SyntheticTruth:
    """Generate latent truth without observation noise or missingness."""

    if model_id in {"SYN-BATES", "SYN-LOCALVOL"}:
        capability = next(x for x in truth_capabilities() if x.model_id == model_id)
        raise UnsupportedTruthModelError(capability.reason)
    if model_id == "SYN-BS":
        price = np.asarray(
            call_price(
                design.spot,
                design.strike,
                design.maturity,
                design.volatility,
                design.rate,
                design.dividend_yield,
            ),
            dtype=np.float64,
        )
        risk = greeks(
            design.spot,
            design.strike,
            design.maturity,
            design.volatility,
            design.rate,
            design.dividend_yield,
        )
        return SyntheticTruth(
            design,
            model_id,
            price,
            np.asarray(risk.delta),
            np.asarray(risk.gamma),
            np.asarray(risk.vega),
            "numpy-cpu-analytic",
            "float64",
        )
    if model_id != "SYN-HESTON":
        raise ValueError(f"unknown truth model: {model_id}")
    if len(design) > max_cpu_heston_scenarios:
        raise RuntimeError(
            "research-scale Heston truth must use the qualified GPU runner; "
            f"bounded CPU limit is {max_cpu_heston_scenarios}"
        )
    prices = np.empty(len(design), dtype=np.float64)
    for index in range(len(design)):
        params = HestonParameters(
            design.heston_v0[index],
            design.heston_kappa[index],
            design.heston_theta[index],
            design.heston_sigma_v[index],
            design.heston_rho[index],
        )
        prices[index] = float(
            heston_price_fixed_quad(
                design.spot[index],
                design.strike[index],
                design.maturity[index],
                params,
                design.rate[index],
                design.dividend_yield[index],
                nodes=heston_nodes,
                upper_bound=heston_upper_bound,
            )
        )
    delta, gamma, vega = _finite_difference_heston_greeks(
        design, prices, nodes=heston_nodes, upper_bound=heston_upper_bound
    )
    return SyntheticTruth(
        design,
        model_id,
        prices,
        delta,
        gamma,
        vega,
        "numpy-cpu-bounded-reference",
        "float64",
    )
