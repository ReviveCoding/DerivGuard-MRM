"""Calibrated-developer confirmatory synthetic study (version 4).

Unlike v3, every developer Heston parameter vector is estimated solely from
the observable synthetic bid/mid/ask surface. Latent truth is retained only by
``make_record`` for material-error labels and locked evaluation.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, replace
from datetime import UTC, datetime
from hashlib import sha256
from pathlib import Path
from typing import Any, Literal, cast

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import minimize

from derivguard.development.heston_cf import HestonParameters, heston_price_fixed_quad
from derivguard.market.black_scholes import call_price, greeks
from derivguard.market.implied_volatility import implied_volatility
from derivguard.market.svi import SSVIParameters, SSVISurface
from derivguard.synthetic.confirmatory import (
    ABLATION_SPECS,
    ConfirmatoryConfig,
    ConfirmatoryRunResult,
    SeedRegistry,
    _surface_design,
    evaluate_confirmatory_ablations,
    generate_confirmatory_truth,
    qualify_confirmatory_engines,
)
from derivguard.synthetic.confirmatory_v3 import (
    FAULT_IDS,
    FAULT_SEVERITIES,
    IMPLEMENTATION_LABELS,
    _persist_v3,
    _truth_family,
)
from derivguard.synthetic.faults import FaultId, ValidationCase, clean_validation_case, inject_fault
from derivguard.synthetic.generation import FloatArray, SyntheticDesign
from derivguard.synthetic.observation import ObservedSurface, observe_truth
from derivguard.validation.dve import (
    BASELINE_NAMES,
    EvidenceRecord,
    evaluate_locked,
    fit_dev_thresholds,
    make_record,
)
from derivguard.validation.quantlib_oracle import (
    QuantLibHestonParameters,
    quantlib_heston_price,
)

Partition = Literal["DEV", "VALIDATION", "LOCKED_TEST", "OOD_LOCKED_TEST"]
ParameterArray = NDArray[np.float64]

PARAMETER_BOUNDS: tuple[tuple[float, float], ...] = (
    (0.0025, 0.5),
    (0.05, 10.0),
    (0.0025, 0.5),
    (0.05, 2.5),
    (-0.999, 0.999),
)


@dataclass(frozen=True)
class ConfirmatoryV4Config:
    profile: str = "research"
    device: str = "cuda"
    cases_dev: int = 34
    cases_validation: int = 17
    cases_locked: int = 34
    cases_ood: int = 17
    contracts_per_case: int = 25
    materiality_threshold: float = 1.0
    target_fpr: float = 0.05
    minimum_regime_negatives: int = 5
    truth_nodes: int = 96
    bates_nodes: int = 128
    calibration_nodes: int = 64
    population_candidates: int = 64
    local_iterations: int = 40
    bootstrap_draws: int = 4
    bootstrap_candidates: int = 24
    bootstrap_local_iterations: int = 12

    def validate(self) -> None:
        if self.profile not in {"research", "audit"}:
            raise ValueError("profile must be research or audit")
        if self.device not in {"cuda", "cpu-bounded-reference"}:
            raise ValueError("invalid device")
        if any(
            count < 17
            for count in (
                self.cases_dev,
                self.cases_validation,
                self.cases_locked,
                self.cases_ood,
            )
        ):
            raise ValueError("each partition must cover F00--F16")
        if self.contracts_per_case < 3 or self.calibration_nodes < 16:
            raise ValueError("invalid surface size or quadrature")
        if self.population_candidates < 4 or self.bootstrap_candidates < 4:
            raise ValueError("candidate populations are too small")
        if self.bootstrap_draws < 2 or self.materiality_threshold <= 0.0:
            raise ValueError("invalid bootstrap or materiality controls")

    def truth_config(self) -> ConfirmatoryConfig:
        return ConfirmatoryConfig(
            cases_dev=self.cases_dev,
            cases_validation=self.cases_validation,
            cases_locked=self.cases_locked,
            cases_ood=self.cases_ood,
            contracts_per_case=self.contracts_per_case,
            materiality_threshold_spread_units=self.materiality_threshold,
            target_fpr=self.target_fpr,
            minimum_regime_negatives=self.minimum_regime_negatives,
            heston_nodes=self.truth_nodes,
            bates_nodes=self.bates_nodes,
            device=self.device,
        )


@dataclass(frozen=True)
class CalibrationDiagnostic:
    case_id: str
    partition: str
    draw: int
    seed: int
    backend: str
    objective: str
    success: bool
    message: str
    population_candidates: int
    local_iterations: int
    evaluations: int
    objective_value: float
    runtime_seconds: float
    v0: float
    kappa: float
    theta: float
    sigma_v: float
    rho: float
    observed_quotes: int


def build_v4_seed_registry(config: ConfirmatoryV4Config) -> SeedRegistry:
    registry = SeedRegistry(
        dev=tuple(93_000_001 + index for index in range(config.cases_dev)),
        validation=tuple(94_000_001 + index for index in range(config.cases_validation)),
        locked=tuple(95_000_001 + index for index in range(config.cases_locked)),
        ood=tuple(96_000_001 + index for index in range(config.cases_ood)),
        observation_offset=30_000_000,
    )
    registry.validate()
    return registry


def _file_hash(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def _fault_for_index(index: int) -> FaultId:
    return FAULT_IDS[index % len(FAULT_IDS)]


def qualify_v4_calibration_engine(device: str) -> dict[str, object]:
    """Qualify the batched calibration pricer against an independent CPU calculation."""

    design = _surface_design(92_999_991, 3, ood=False)
    population = np.asarray(
        [[0.04, 2.0, 0.05, 0.45, -0.70], [0.08, 1.1, 0.07, 0.75, -0.35]],
        dtype=np.float64,
    )
    reference = _cpu_price_matrix(design, population, nodes=64)
    if device == "cuda":
        candidate = _cuda_price_matrix(design, population, nodes=64)
        backend = "torch-cuda-fp64"
    else:
        candidate = _cpu_price_matrix(design, population, nodes=64)
        backend = "numpy-cpu-fp64"
    error = float(np.max(np.abs(reference - candidate)))
    tolerance = 5.0e-7
    return {
        "engine": "v4_batched_heston_calibration_pricer",
        "backend": backend,
        "status": "PASS" if error <= tolerance else "FAIL",
        "maximum_absolute_error": error,
        "tolerance": tolerance,
        "reference": "independent per-contract public heston_price_fixed_quad",
    }


def freeze_confirmatory_v4(
    path: Path = Path("artifacts/checkpoints/synthetic_confirmatory_v4.freeze.json"),
    config: ConfirmatoryV4Config | None = None,
    hypothesis_registry: Path = Path("governance/confirmatory_v4_hypothesis_registry.yaml"),
    experiment_registry: Path = Path("experiments/confirmatory_v4_registry.yaml"),
) -> dict[str, Any]:
    """Freeze registries, design, calibration, seeds, faults, and DVE pre-generation."""

    cfg = config or ConfirmatoryV4Config()
    cfg.validate()
    module_hash = _file_hash(Path(__file__))
    if path.exists():
        loaded: object = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(loaded, dict):
            raise ValueError("invalid v4 freeze")
        payload = cast(dict[str, Any], loaded)
        config_hash = sha256(json.dumps(asdict(cfg), sort_keys=True).encode()).hexdigest()
        if payload.get("config_hash") != config_hash or payload.get("module_hash") != module_hash:
            raise ValueError("v4 source or configuration differs from immutable freeze")
        for registry, key in (
            (hypothesis_registry, "hypothesis_registry_hash"),
            (experiment_registry, "experiment_registry_hash"),
        ):
            if payload.get(key) != _file_hash(registry):
                raise ValueError("v4 registry differs from immutable freeze")
        return payload
    seeds = build_v4_seed_registry(cfg)
    qualifications = qualify_confirmatory_engines(cfg.device)
    if any(item.status != "PASS" for item in qualifications):
        raise RuntimeError("truth engine qualification failed")
    calibration_qualification = qualify_v4_calibration_engine(cfg.device)
    if calibration_qualification["status"] != "PASS":
        raise RuntimeError("v4 calibration pricing engine qualification failed")
    schedule: list[dict[str, object]] = []
    partitions: tuple[Partition, ...] = (
        "DEV",
        "VALIDATION",
        "LOCKED_TEST",
        "OOD_LOCKED_TEST",
    )
    for partition in partitions:
        for index, seed in enumerate(seeds.for_partition(partition)):
            fault = _fault_for_index(index)
            schedule.append(
                {
                    "partition": partition,
                    "seed": seed,
                    "observation_seed": seed + seeds.observation_offset,
                    "fault_id": fault,
                    "severity": FAULT_SEVERITIES[fault],
                    "truth_family": _truth_family(fault),
                    "implementation_label": IMPLEMENTATION_LABELS[fault],
                    "proxy_fault": fault not in {"F00", "F01", "F02", "F03"},
                }
            )
    seed_payload = asdict(seeds)
    calibration_design = {
        "inputs": "observable bid/mid/ask only",
        "objective": "C04 Huber spread-normalized price",
        "bounds": PARAMETER_BOUNDS,
        "candidate_design": "seeded independent uniform bounded population",
        "population_candidates": cfg.population_candidates,
        "local_method": "L-BFGS-B",
        "local_iterations": cfg.local_iterations,
        "nodes": cfg.calibration_nodes,
        "bootstrap_draws": cfg.bootstrap_draws,
        "bootstrap_candidates": cfg.bootstrap_candidates,
        "bootstrap_local_iterations": cfg.bootstrap_local_iterations,
    }
    dve = {
        "dimensions": [
            "E_data",
            "E_num",
            "E_cal",
            "E_ident",
            "E_param",
            "E_form",
            "E_greek",
            "E_extra",
            "E_outcome",
        ],
        "target_fpr": cfg.target_fpr,
        "materiality_threshold_spread_units": cfg.materiality_threshold,
        "baselines": list(BASELINE_NAMES),
        "ablations": dict(ABLATION_SPECS),
    }
    payload = {
        "schema_version": "4.0",
        "status": "FROZEN_NOT_EXECUTED",
        "created_utc": datetime.now(UTC).isoformat(),
        "module_hash": module_hash,
        "config": asdict(cfg),
        "config_hash": sha256(json.dumps(asdict(cfg), sort_keys=True).encode()).hexdigest(),
        "seed_registry": seed_payload,
        "seed_hash": sha256(json.dumps(seed_payload, sort_keys=True).encode()).hexdigest(),
        "fault_schedule": schedule,
        "schedule_hash": sha256(json.dumps(schedule, sort_keys=True).encode()).hexdigest(),
        "calibration_design": calibration_design,
        "calibration_design_hash": sha256(
            json.dumps(calibration_design, sort_keys=True).encode()
        ).hexdigest(),
        "dve_formulation": dve,
        "dve_formulation_hash": sha256(json.dumps(dve, sort_keys=True).encode()).hexdigest(),
        "hypothesis_registry_path": hypothesis_registry.as_posix(),
        "hypothesis_registry_hash": _file_hash(hypothesis_registry),
        "experiment_registry_path": experiment_registry.as_posix(),
        "experiment_registry_hash": _file_hash(experiment_registry),
        "engine_qualification": [asdict(item) for item in qualifications],
        "calibration_engine_qualification": calibration_qualification,
        "prior_v2_v3_outcomes_read": False,
        "locked_tuning_prohibited": True,
    }
    payload["freeze_hash"] = sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
    if _file_hash(Path(__file__)) != module_hash:
        raise RuntimeError("v4 source changed during freeze construction")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return payload


def _cpu_price_matrix(
    design: SyntheticDesign, population: ParameterArray, *, nodes: int, spot_scale: float = 1.0
) -> ParameterArray:
    output = np.empty((population.shape[0], len(design)), dtype=np.float64)
    for row, raw in enumerate(population):
        params = HestonParameters(*map(float, raw))
        for index in range(len(design)):
            output[row, index] = float(
                heston_price_fixed_quad(
                    float(design.spot[index] * spot_scale),
                    float(design.strike[index]),
                    float(design.maturity[index]),
                    params,
                    float(design.rate[index]),
                    float(design.dividend_yield[index]),
                    nodes=nodes,
                )
            )
    return output


def _cuda_price_matrix(
    design: SyntheticDesign, population: ParameterArray, *, nodes: int, spot_scale: float = 1.0
) -> ParameterArray:
    """Price a candidate-by-contract Heston matrix in one CUDA FP64 graph."""

    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("v4 research calibration requires CUDA")
    dtype, cdtype = torch.float64, torch.complex128
    device = torch.device("cuda")
    p = torch.as_tensor(population, dtype=dtype, device=device)
    spot = torch.as_tensor(design.spot * spot_scale, dtype=dtype, device=device)[None, :]
    strike = torch.as_tensor(design.strike, dtype=dtype, device=device)[None, :]
    maturity = torch.as_tensor(design.maturity, dtype=dtype, device=device)[None, :]
    rate = torch.as_tensor(design.rate, dtype=dtype, device=device)[None, :]
    dividend = torch.as_tensor(design.dividend_yield, dtype=dtype, device=device)[None, :]
    roots, raw_weights = np.polynomial.legendre.leggauss(nodes)
    u = torch.as_tensor(0.5 * 200.0 * (roots + 1.0), dtype=dtype, device=device)
    weights = torch.as_tensor(0.5 * 200.0 * raw_weights, dtype=dtype, device=device)
    z = u.to(cdtype)[None, None, :]
    v0, kappa, theta, sigma, rho = (p[:, index][:, None, None] for index in range(5))

    def cf(argument: Any) -> Any:
        iu = 1j * argument
        beta = kappa - rho * sigma * iu
        root = torch.sqrt(beta * beta + sigma * sigma * (argument * argument + iu))
        root = torch.where(torch.real(root) < 0.0, -root, root)
        ratio = (beta - root) / (beta + root)
        decay = torch.exp(-root * maturity[:, :, None])
        log_ratio = torch.log1p(-ratio * decay) - torch.log1p(-ratio)
        c = iu * (
            torch.log(spot)[:, :, None] + (rate - dividend)[:, :, None] * maturity[:, :, None]
        ) + (kappa * theta / (sigma * sigma)) * (
            (beta - root) * maturity[:, :, None] - 2.0 * log_ratio
        )
        d = (beta - root) / (sigma * sigma) * (1.0 - decay) / (1.0 - ratio * decay)
        return torch.exp(c + d * v0)

    phase = torch.exp(-1j * torch.log(strike)[:, :, None].to(cdtype) * z)
    denominator = 1j * z
    phi_mi = spot * torch.exp((rate - dividend) * maturity)
    first = torch.sum(
        weights[None, None, :]
        * torch.real(phase * cf(z - 1j) / (denominator * phi_mi[:, :, None])),
        dim=-1,
    )
    second = torch.sum(weights[None, None, :] * torch.real(phase * cf(z) / denominator), dim=-1)
    prices = spot * torch.exp(-dividend * maturity) * (0.5 + first / np.pi) - strike * torch.exp(
        -rate * maturity
    ) * (0.5 + second / np.pi)
    return np.asarray(prices.detach().cpu().numpy(), dtype=np.float64)


def _price_matrix(
    design: SyntheticDesign,
    population: ParameterArray,
    config: ConfirmatoryV4Config,
    *,
    spot_scale: float = 1.0,
) -> ParameterArray:
    if config.device == "cuda":
        return _cuda_price_matrix(
            design, population, nodes=config.calibration_nodes, spot_scale=spot_scale
        )
    return _cpu_price_matrix(
        design, population, nodes=config.calibration_nodes, spot_scale=spot_scale
    )


def _losses(prices: ParameterArray, target: FloatArray, spread: FloatArray) -> FloatArray:
    scaled = (prices - target[None, :]) / np.maximum(spread[None, :], 0.01)
    absolute = np.abs(scaled)
    huber = np.where(absolute <= 1.5, 0.5 * scaled * scaled, 1.5 * (absolute - 0.75))
    return np.asarray(np.mean(huber, axis=1), dtype=np.float64)


def calibrate_observable_heston(
    surface: ObservedSurface,
    config: ConfirmatoryV4Config,
    *,
    seed: int,
    market_prices: FloatArray | None = None,
    candidates: int | None = None,
    local_iterations: int | None = None,
) -> tuple[HestonParameters, dict[str, object]]:
    """Calibrate without reading any latent Heston parameter or true price."""

    start_time = time.perf_counter()
    design = surface.truth.design
    observed = surface.observed & np.isfinite(surface.mid)
    if np.sum(observed) < 3:
        raise ValueError("at least three observable quotes are required")
    target_all = surface.mid if market_prices is None else np.asarray(market_prices)
    target = np.asarray(target_all[observed], dtype=np.float64)
    spread = np.asarray(surface.spread[observed], dtype=np.float64)
    observable_design = replace(
        design,
        scenario_id=design.scenario_id[observed],
        spot=design.spot[observed],
        strike=design.strike[observed],
        maturity=design.maturity[observed],
        rate=design.rate[observed],
        dividend_yield=design.dividend_yield[observed],
        # Not used by Heston pricing; zeroed to make the observable-only data
        # boundary mechanically evident alongside the latent Heston arrays.
        volatility=np.zeros(np.sum(observed)),
        heston_v0=np.zeros(np.sum(observed)),
        heston_kappa=np.zeros(np.sum(observed)),
        heston_theta=np.zeros(np.sum(observed)),
        heston_sigma_v=np.zeros(np.sum(observed)),
        heston_rho=np.zeros(np.sum(observed)),
    )
    population_size = candidates or config.population_candidates
    maximum_iterations = local_iterations or config.local_iterations
    rng = np.random.default_rng(seed)
    bounds = np.asarray(PARAMETER_BOUNDS, dtype=np.float64)
    population = rng.uniform(bounds[:, 0], bounds[:, 1], size=(population_size, 5))
    population_prices = _price_matrix(observable_design, population, config)
    population_losses = _losses(population_prices, target, spread)
    initial = population[int(np.argmin(population_losses))]
    evaluations = population_size

    def objective(raw: ParameterArray) -> float:
        nonlocal evaluations
        evaluations += 1
        prices = _price_matrix(observable_design, np.asarray(raw)[None, :], config)[0]
        return float(_losses(prices[None, :], target, spread)[0])

    result = minimize(
        objective,
        initial,
        method="L-BFGS-B",
        bounds=PARAMETER_BOUNDS,
        options={"maxiter": maximum_iterations, "ftol": 1.0e-9},
    )
    raw = np.asarray(result.x, dtype=np.float64)
    if raw.shape != (5,) or np.any(~np.isfinite(raw)) or not np.isfinite(result.fun):
        raise RuntimeError("observable Heston calibration produced a non-finite solution")
    params = HestonParameters(*map(float, raw))
    params.validate()
    diagnostic: dict[str, object] = {
        "seed": seed,
        "backend": "torch-cuda-fp64" if config.device == "cuda" else "numpy-cpu-fp64",
        "objective": "C04",
        "success": bool(result.success),
        "message": str(result.message),
        "population_candidates": population_size,
        "local_iterations": int(getattr(result, "nit", 0)),
        "evaluations": evaluations,
        "objective_value": float(result.fun),
        "runtime_seconds": time.perf_counter() - start_time,
        "parameters": raw.tolist(),
        "observed_quotes": int(np.sum(observed)),
    }
    return params, diagnostic


def _observable_challengers(
    surface: ObservedSurface,
) -> tuple[tuple[ParameterArray, ...], dict[str, object]]:
    design = surface.truth.design
    observed = surface.observed & np.isfinite(surface.mid)

    def flat_objective(raw: ParameterArray) -> float:
        prices = np.asarray(
            call_price(
                design.spot[observed],
                design.strike[observed],
                design.maturity[observed],
                float(raw[0]),
                design.rate[observed],
                design.dividend_yield[observed],
            )
        )
        return float(np.mean(np.square(prices - surface.mid[observed])))

    result = minimize(
        flat_objective,
        np.asarray([0.25]),
        method="L-BFGS-B",
        bounds=((0.03, 2.0),),
    )
    flat_vol = float(result.x[0])
    observed_iv = np.full(len(design), np.nan)
    for index in np.flatnonzero(observed):
        inversion = implied_volatility(
            float(surface.mid[index]),
            float(design.spot[index]),
            float(design.strike[index]),
            float(design.maturity[index]),
            float(design.rate[index]),
            float(design.dividend_yield[index]),
        )
        if inversion.converged:
            observed_iv[index] = inversion.volatility
    mask = observed & np.isfinite(observed_iv)
    forward = design.spot * np.exp((design.rate - design.dividend_yield) * design.maturity)
    log_moneyness = np.log(design.strike / forward)

    def ssvi_objective(raw: ParameterArray) -> float:
        variance_rate, rho, eta, gamma = map(float, raw)
        try:
            maturities = np.unique(design.maturity[mask])
            surface_fit = SSVISurface(
                maturities,
                variance_rate * maturities,
                SSVIParameters(rho, eta, gamma),
            )
            variance = np.asarray(
                surface_fit.total_variance(log_moneyness[mask], design.maturity[mask])
            )
            fitted_iv = np.sqrt(np.maximum(variance / design.maturity[mask], 0.0))
            return float(np.mean(np.square(fitted_iv - observed_iv[mask])))
        except ValueError:
            return 1.0e6

    ssvi_result = minimize(
        ssvi_objective,
        np.asarray([flat_vol * flat_vol, -0.3, 0.4, 0.5]),
        method="L-BFGS-B",
        bounds=((0.0009, 1.0), (-0.95, 0.95), (0.01, 3.0), (0.0, 1.0)),
        options={"maxiter": 100, "ftol": 1.0e-12},
    )
    if np.sum(mask) >= 4 and ssvi_result.success and np.isfinite(ssvi_result.fun):
        variance_rate, rho, eta, gamma = map(float, ssvi_result.x)
        maturities = np.unique(design.maturity)
        surface_fit = SSVISurface(
            maturities,
            variance_rate * maturities,
            SSVIParameters(rho, eta, gamma),
        )
        diagnostic = surface_fit.diagnostics()
        fallback = not diagnostic.valid
        smile_vol = (
            np.full(len(design), flat_vol)
            if fallback
            else np.asarray(surface_fit.implied_volatility(log_moneyness, design.maturity))
        )
        qualification: dict[str, object] = {
            "status": "FALLBACK_STATIC_ARBITRAGE_FAILURE" if fallback else "PASS",
            "optimizer_success": True,
            "objective_value": float(ssvi_result.fun),
            "static_arbitrage_valid": diagnostic.valid,
            "diagnostic_messages": list(diagnostic.messages),
            "parameters": {
                "atm_variance_rate": variance_rate,
                "rho": rho,
                "eta": eta,
                "gamma": gamma,
            },
            "fallback_to_flat_bs": fallback,
            "observed_iv_count": int(np.sum(mask)),
        }
    else:
        smile_vol = np.full(len(design), flat_vol)
        qualification = {
            "status": "FALLBACK_FIT_FAILURE",
            "optimizer_success": bool(ssvi_result.success),
            "objective_value": float(ssvi_result.fun),
            "static_arbitrage_valid": False,
            "diagnostic_messages": [str(ssvi_result.message)],
            "parameters": None,
            "fallback_to_flat_bs": True,
            "observed_iv_count": int(np.sum(mask)),
        }
    flat_vector = np.full(len(design), flat_vol)
    flat_prices = np.asarray(
        call_price(
            design.spot,
            design.strike,
            design.maturity,
            flat_vector,
            design.rate,
            design.dividend_yield,
        )
    )
    smile_prices = np.asarray(
        call_price(
            design.spot,
            design.strike,
            design.maturity,
            smile_vol,
            design.rate,
            design.dividend_yield,
        )
    )
    flat_risk = greeks(
        design.spot,
        design.strike,
        design.maturity,
        flat_vector,
        design.rate,
        design.dividend_yield,
    )
    smile_risk = greeks(
        design.spot,
        design.strike,
        design.maturity,
        smile_vol,
        design.rate,
        design.dividend_yield,
    )
    return (
        (
            np.column_stack((flat_prices, smile_prices)),
            np.column_stack((flat_risk.delta, smile_risk.delta)),
            np.column_stack((flat_risk.gamma, smile_risk.gamma)),
            np.column_stack((flat_risk.vega, smile_risk.vega)),
        ),
        qualification,
    )


def _developer_values(
    design: SyntheticDesign, params: HestonParameters, config: ConfirmatoryV4Config
) -> tuple[FloatArray, FloatArray, FloatArray, FloatArray]:
    raw = np.asarray([[params.v0, params.kappa, params.theta, params.sigma_v, params.rho]])
    price = _price_matrix(design, raw, config)[0]
    bump = 1.0e-4
    up = _price_matrix(design, raw, config, spot_scale=1.0 + bump)[0]
    down = _price_matrix(design, raw, config, spot_scale=1.0 - bump)[0]
    spot = design.spot
    delta = (up - down) / (2.0 * bump * spot)
    gamma = (up - 2.0 * price + down) / np.square(bump * spot)
    vol_bump = 1.0e-4
    root_v0, root_theta = np.sqrt(params.v0), np.sqrt(params.theta)
    upper = raw.copy()
    lower = raw.copy()
    upper[0, 0], upper[0, 2] = (root_v0 + vol_bump) ** 2, (root_theta + vol_bump) ** 2
    lower[0, 0], lower[0, 2] = (
        max(root_v0 - vol_bump, 1.0e-6) ** 2,
        max(root_theta - vol_bump, 1.0e-6) ** 2,
    )
    vega = (_price_matrix(design, upper, config)[0] - _price_matrix(design, lower, config)[0]) / (
        2.0 * vol_bump
    )
    return price, delta, gamma, vega


def build_calibrated_case(
    surface: ObservedSurface,
    config: ConfirmatoryV4Config,
    *,
    calibration_seed: int,
) -> tuple[ValidationCase, list[dict[str, object]], dict[str, object]]:
    """Construct all DVE inputs from observables and fitted parameters."""

    params, primary = calibrate_observable_heston(surface, config, seed=calibration_seed)
    design = surface.truth.design
    developer_price, developer_delta, developer_gamma, developer_vega = _developer_values(
        design, params, config
    )
    ql_params = QuantLibHestonParameters(
        params.v0, params.kappa, params.theta, params.sigma_v, params.rho
    )
    oracle = np.asarray(
        [
            quantlib_heston_price(
                float(design.spot[index]),
                float(design.strike[index]),
                float(design.maturity[index]),
                ql_params,
                float(design.rate[index]),
                float(design.dividend_yield[index]),
            ).price
            for index in range(len(design))
        ]
    )
    diagnostics = [primary]
    calibration_prices = np.empty((len(design), config.bootstrap_draws), dtype=np.float64)
    rng = np.random.default_rng(calibration_seed + 1_000_000)
    observed = surface.observed & np.isfinite(surface.mid)
    for draw in range(config.bootstrap_draws):
        sampled = surface.mid.copy()
        sampled[observed] = surface.bid[observed] + rng.random(np.sum(observed)) * (
            surface.ask[observed] - surface.bid[observed]
        )
        draw_params, diagnostic = calibrate_observable_heston(
            surface,
            config,
            seed=calibration_seed + 10_000 + draw,
            market_prices=sampled,
            candidates=config.bootstrap_candidates,
            local_iterations=config.bootstrap_local_iterations,
        )
        diagnostics.append(diagnostic)
        calibration_prices[:, draw] = _developer_values(design, draw_params, config)[0]
    challenger_values, ssvi_qualification = _observable_challengers(surface)
    challenger_price, challenger_delta, challenger_gamma, challenger_vega = challenger_values
    clean = clean_validation_case(surface, bootstrap_draws=config.bootstrap_draws)
    case = replace(
        clean,
        developer_price=developer_price,
        developer_delta=developer_delta,
        developer_gamma=developer_gamma,
        developer_vega=developer_vega,
        numerical_prices=np.column_stack((developer_price, oracle)),
        calibration_prices=calibration_prices,
        challenger_prices=challenger_price,
        challenger_delta=challenger_delta,
        challenger_gamma=challenger_gamma,
        challenger_vega=challenger_vega,
        proxy_fault=False,
    )
    return case, diagnostics, ssvi_qualification


def _make_v4_case(
    partition: Partition,
    index: int,
    seed: int,
    config: ConfirmatoryV4Config,
) -> tuple[EvidenceRecord, dict[str, object], list[CalibrationDiagnostic]]:
    fault = _fault_for_index(index)
    family = _truth_family(fault)
    design = _surface_design(seed, config.contracts_per_case, ood=partition == "OOD_LOCKED_TEST")
    truth = generate_confirmatory_truth(design, cast(Any, family), config.truth_config(), seed=seed)
    observation_seed = seed + 30_000_000
    surface = observe_truth(truth, config.truth_config().observation, seed=observation_seed)
    case, raw_diagnostics, ssvi_qualification = build_calibrated_case(
        surface, config, calibration_seed=seed + 40_000_000
    )
    case = replace(case, fault_id=fault)
    if fault not in {"F00", "F01", "F02", "F03"}:
        case = replace(
            inject_fault(case, fault, severity=FAULT_SEVERITIES[fault]), proxy_fault=True
        )
    case_id = f"V4-{partition}-{fault}-{seed}"
    record = make_record(
        case_id, partition, case, materiality_threshold=config.materiality_threshold
    )
    diagnostics: list[CalibrationDiagnostic] = []
    for draw, item in enumerate(raw_diagnostics, start=-1):
        parameters = cast(list[float], item["parameters"])
        diagnostics.append(
            CalibrationDiagnostic(
                case_id=case_id,
                partition=partition,
                draw=draw,
                seed=cast(int, item["seed"]),
                backend=str(item["backend"]),
                objective=str(item["objective"]),
                success=bool(item["success"]),
                message=str(item["message"]),
                population_candidates=cast(int, item["population_candidates"]),
                local_iterations=cast(int, item["local_iterations"]),
                evaluations=cast(int, item["evaluations"]),
                objective_value=cast(float, item["objective_value"]),
                runtime_seconds=cast(float, item["runtime_seconds"]),
                v0=parameters[0],
                kappa=parameters[1],
                theta=parameters[2],
                sigma_v=parameters[3],
                rho=parameters[4],
                observed_quotes=cast(int, item["observed_quotes"]),
            )
        )
    provenance = {
        "case_id": case_id,
        "partition": partition,
        "fault_id": fault,
        "severity": FAULT_SEVERITIES[fault],
        "truth_family": family,
        "implementation_label": IMPLEMENTATION_LABELS[fault],
        "proxy_fault": case.proxy_fault,
        "truth_seed": seed,
        "observation_seed": observation_seed,
        "calibration_seed": seed + 40_000_000,
        "calibration_inputs": "observable_quotes_only",
        "ssvi_qualification": json.dumps(ssvi_qualification, sort_keys=True),
        "ssvi_status": ssvi_qualification["status"],
        "ssvi_static_arbitrage_valid": ssvi_qualification["static_arbitrage_valid"],
        "ssvi_fallback_to_flat_bs": ssvi_qualification["fallback_to_flat_bs"],
        "backend": truth.backend,
        "precision": truth.precision,
        "generated_utc": datetime.now(UTC).isoformat(),
    }
    return record, provenance, diagnostics


def run_confirmatory_v4_study(
    profile: str = "research", device: str = "cuda", resume: bool = True
) -> dict[str, Any]:
    """Run v4 after its immutable design freeze; never reads prior outcomes."""

    output = Path("results/confirmatory/synthetic_v3")
    metrics_path = output / "confirmatory_metrics.json"
    if resume and metrics_path.exists():
        return cast(dict[str, Any], json.loads(metrics_path.read_text(encoding="utf-8")))
    config = ConfirmatoryV4Config(profile=profile, device=device)
    frozen = freeze_confirmatory_v4(config=config)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("v4 namespace is non-empty; refusing mixed execution")
    seeds = build_v4_seed_registry(config)
    records: list[EvidenceRecord] = []
    provenance: list[dict[str, object]] = []
    diagnostics: list[CalibrationDiagnostic] = []

    def generate(partition: Partition) -> None:
        for index, seed in enumerate(seeds.for_partition(partition)):
            record, source, case_diagnostics = _make_v4_case(partition, index, seed, config)
            source["design_freeze_hash"] = frozen["freeze_hash"]
            records.append(record)
            provenance.append(source)
            diagnostics.extend(case_diagnostics)

    generate("DEV")
    generate("VALIDATION")
    dev = tuple(item for item in records if item.partition == "DEV")
    thresholds = fit_dev_thresholds(
        dev,
        target_fpr=config.target_fpr,
        minimum_regime_negatives=config.minimum_regime_negatives,
    )
    validation = tuple(
        replace(item, partition="LOCKED_TEST") for item in records if item.partition == "VALIDATION"
    )
    validation_metrics = {
        key: asdict(value) for key, value in evaluate_locked(validation, thresholds).items()
    }
    output.mkdir(parents=True, exist_ok=True)
    threshold_payload = {
        "status": "FROZEN_BEFORE_LOCKED_GENERATION",
        "frozen_utc": datetime.now(UTC).isoformat(),
        "design_freeze_hash": frozen["freeze_hash"],
        "thresholds": thresholds.to_dict(),
        "validation_diagnostics": validation_metrics,
        "validation_selects_locked_method": False,
    }
    with (output / "dve_thresholds.freeze.json").open("x", encoding="utf-8") as handle:
        json.dump(threshold_payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    generate("LOCKED_TEST")
    generate("OOD_LOCKED_TEST")
    locked = tuple(item for item in records if item.partition in {"LOCKED_TEST", "OOD_LOCKED_TEST"})
    baselines = evaluate_locked(locked, thresholds)
    ablations = evaluate_confirmatory_ablations(dev, locked, thresholds)
    result = ConfirmatoryRunResult(
        run_id=f"SYN-CONFIRMATORY-V4-{str(frozen['freeze_hash'])[:12]}",
        freeze_hash=str(frozen["freeze_hash"]),
        thresholds=thresholds,
        records=tuple(records),
        baselines=baselines,
        ablations=ablations,
        provenance=tuple(provenance),
    )
    _persist_v3(
        output,
        result,
        validation_metrics,
        frozen["fault_schedule"],
        frozen["engine_qualification"],
    )
    import pandas as pd

    pd.DataFrame([asdict(item) for item in diagnostics]).to_parquet(
        output / "calibration_diagnostics.parquet", index=False
    )
    status = {
        "status": "CONFIRMATORY_LOCKED_COMPLETE",
        "run_id": result.run_id,
        "freeze_hash": result.freeze_hash,
        "locked_cases": len(locked),
        "calibrations": len(diagnostics),
        "calibration_successes": sum(item.success for item in diagnostics),
        "faults_executed": list(FAULT_IDS),
        "baselines": {key: asdict(value) for key, value in baselines.items()},
        "ablations": {key: asdict(value) for key, value in ablations.items()},
    }
    metrics_path.write_text(json.dumps(status, indent=2, sort_keys=True) + "\n")
    return status
