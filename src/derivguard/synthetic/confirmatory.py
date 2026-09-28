"""Frozen, provenance-rich confirmatory synthetic validation framework.

This module is deliberately separate from the original exploratory synthetic
lab.  It never rewrites exploratory artifacts and never represents the proxy
F01/F02 fault operators as model-form experiments.  Confirmatory F01 and F02
are constructed from canonical Bates and LocalVol/SSVI-marginal truth while
the developer response is priced by Heston.

Only bounded CPU execution is implemented here.  A configuration that exceeds
its explicit CPU ceiling is rejected instead of silently launching a large
CPU job; the research-scale runner must supply a qualified coordinated GPU
executor in a later integration step.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, field, replace
from datetime import UTC, datetime
from hashlib import sha256
from pathlib import Path
from typing import Any, Literal, cast

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from scipy.optimize import minimize_scalar

from derivguard.challengers.bates import BatesParameters, bates_price
from derivguard.development.heston_cf import (
    HestonParameters,
    heston_price_adaptive,
    heston_price_fixed_quad,
)
from derivguard.market.black_scholes import call_price, greeks
from derivguard.market.implied_volatility import implied_volatility
from derivguard.market.svi import SSVIParameters, SSVISurface, ssvi_total_variance
from derivguard.synthetic.faults import ValidationCase, clean_validation_case
from derivguard.synthetic.generation import SyntheticDesign, SyntheticTruth
from derivguard.synthetic.observation import ObservationConfig, observe_truth
from derivguard.validation.dve import (
    BASELINE_NAMES,
    EVIDENCE_NAMES,
    DetectionMetrics,
    EvidenceRecord,
    FrozenDVE,
    baseline_scores,
    evaluate_locked,
    fit_dev_thresholds,
    make_record,
)
from derivguard.validation.quantlib_oracle import (
    QuantLibHestonParameters,
    quantlib_heston_price,
)

FloatArray = NDArray[np.float64]
Partition = Literal["DEV", "VALIDATION", "LOCKED_TEST", "OOD_LOCKED_TEST"]
TruthFamily = Literal["HESTON", "BATES", "LOCALVOL_SSVI", "PIECEWISE_REGIME"]

TRUTH_FAMILIES: tuple[TruthFamily, ...] = (
    "HESTON",
    "BATES",
    "LOCALVOL_SSVI",
    "PIECEWISE_REGIME",
)

ABLATION_SPECS: Mapping[str, str] = {
    "A01": "remove E_data",
    "A02": "remove E_num",
    "A03": "remove E_cal",
    "A04": "remove E_ident",
    "A05": "remove E_param",
    "A06": "remove E_form",
    "A07": "remove E_greek",
    "A08": "remove E_extra",
    "A09": "remove regime conditioning",
    "A10": "remove E_outcome",
    "A11": "remove economic-materiality prioritization",
}

_DIMENSION_ABLATIONS: Mapping[str, str] = {
    "A01": "E_data",
    "A02": "E_num",
    "A03": "E_cal",
    "A04": "E_ident",
    "A05": "E_param",
    "A06": "E_form",
    "A07": "E_greek",
    "A08": "E_extra",
    "A10": "E_outcome",
}


@dataclass(frozen=True)
class ConfirmatoryConfig:
    schema_version: str = "1.0"
    experiment_id: str = "SYN-CONFIRMATORY-V1"
    cases_dev: int = 8
    cases_validation: int = 4
    cases_locked: int = 8
    cases_ood: int = 4
    contracts_per_case: int = 8
    materiality_threshold_spread_units: float = 1.0
    target_fpr: float = 0.05
    minimum_regime_negatives: int = 2
    heston_nodes: int = 96
    bates_nodes: int = 128
    integration_upper_bound: float = 200.0
    device: str = "cpu-bounded-reference"
    precision: str = "float64"
    max_cpu_contracts: int = 256
    truth_families: tuple[TruthFamily, ...] = TRUTH_FAMILIES
    observation: ObservationConfig = field(default_factory=ObservationConfig)

    def validate(self) -> None:
        counts = (self.cases_dev, self.cases_validation, self.cases_locked, self.cases_ood)
        if any(value < 1 for value in counts) or self.contracts_per_case < 3:
            raise ValueError("all partitions require cases and at least three contracts per case")
        if not 0.0 < self.target_fpr < 1.0:
            raise ValueError("target_fpr must lie in (0,1)")
        if self.materiality_threshold_spread_units <= 0.0:
            raise ValueError("materiality threshold must be positive")
        if self.heston_nodes < 32 or self.bates_nodes < 32:
            raise ValueError("confirmatory quadrature requires at least 32 nodes")
        if self.integration_upper_bound <= 0.0 or self.max_cpu_contracts < 1:
            raise ValueError("integration bound and CPU ceiling must be positive")
        if not self.truth_families or set(self.truth_families) - set(TRUTH_FAMILIES):
            raise ValueError("unknown or empty truth-family selection")
        if self.device not in {"cpu-bounded-reference", "cuda"}:
            raise ValueError("device must be cpu-bounded-reference or cuda")
        self.observation.validate()

    @property
    def total_cases(self) -> int:
        return self.cases_dev + self.cases_validation + self.cases_locked + self.cases_ood

    @property
    def total_contracts(self) -> int:
        return self.total_cases * self.contracts_per_case


@dataclass(frozen=True)
class SeedRegistry:
    dev: tuple[int, ...]
    validation: tuple[int, ...]
    locked: tuple[int, ...]
    ood: tuple[int, ...]
    observation_offset: int = 10_000_000

    def validate(self) -> None:
        groups = (self.dev, self.validation, self.locked, self.ood)
        all_seeds = [seed for group in groups for seed in group]
        if any(seed < 0 for seed in all_seeds) or len(set(all_seeds)) != len(all_seeds):
            raise ValueError("partition seeds must be nonnegative and globally disjoint")
        observation_seeds = [seed + self.observation_offset for seed in all_seeds]
        if set(all_seeds) & set(observation_seeds):
            raise ValueError("truth and observation seed spaces overlap")

    def for_partition(self, partition: Partition) -> tuple[int, ...]:
        return {
            "DEV": self.dev,
            "VALIDATION": self.validation,
            "LOCKED_TEST": self.locked,
            "OOD_LOCKED_TEST": self.ood,
        }[partition]


@dataclass(frozen=True)
class EngineQualification:
    engine: str
    status: str
    measured_error: float
    tolerance: float
    evidence: str


@dataclass(frozen=True)
class ConfirmatoryAblation:
    ablation_id: str
    description: str
    metrics: DetectionMetrics
    recall_change: float
    auprc_change: float
    false_alarm_change: float
    miss_rate_change: float


@dataclass(frozen=True)
class ConfirmatoryRunResult:
    run_id: str
    freeze_hash: str
    thresholds: FrozenDVE
    records: tuple[EvidenceRecord, ...]
    baselines: Mapping[str, DetectionMetrics]
    ablations: Mapping[str, ConfirmatoryAblation]
    provenance: tuple[Mapping[str, object], ...]


def _config_payload(config: ConfirmatoryConfig) -> dict[str, Any]:
    payload = asdict(config)
    payload["truth_families"] = list(config.truth_families)
    return payload


def _config_hash(config: ConfirmatoryConfig) -> str:
    return sha256(json.dumps(_config_payload(config), sort_keys=True).encode()).hexdigest()


def build_seed_registry(config: ConfirmatoryConfig) -> SeedRegistry:
    """Create explicitly disjoint, new seed regions for each partition."""

    registry = SeedRegistry(
        dev=tuple(73_000_001 + index for index in range(config.cases_dev)),
        validation=tuple(74_000_001 + index for index in range(config.cases_validation)),
        locked=tuple(75_000_001 + index for index in range(config.cases_locked)),
        ood=tuple(76_000_001 + index for index in range(config.cases_ood)),
    )
    registry.validate()
    return registry


def qualify_confirmatory_engines(
    device: str = "cpu-bounded-reference",
) -> tuple[EngineQualification, ...]:
    """Run inexpensive deterministic checks required before a design is frozen."""

    heston = HestonParameters(0.04, 2.0, 0.04, 0.5, -0.7)
    fixed = float(heston_price_fixed_quad(100.0, 105.0, 0.75, heston, 0.03, 0.01))
    adaptive = float(heston_price_adaptive(100.0, 105.0, 0.75, heston, 0.03, 0.01))
    heston_error = abs(fixed - adaptive)
    bates_zero = BatesParameters(0.04, 2.0, 0.04, 0.5, -0.7, 0.0, -0.08, 0.20)
    bates_value = float(bates_price(100.0, 105.0, 0.75, bates_zero, 0.03, 0.01))
    bates_error = abs(bates_value - fixed)
    surface = SSVISurface(
        np.asarray([0.25, 0.75, 1.5]),
        np.asarray([0.012, 0.035, 0.065]),
        SSVIParameters(-0.35, 0.45, 0.5),
    )
    diagnostics = surface.diagnostics(log_moneyness_grid=np.linspace(-1.0, 1.0, 101))
    local_error = 0.0 if diagnostics.valid else float("inf")
    # The piecewise engine is a convex mixture of two risk-neutral BS
    # marginals. Check its call lies inside elementary European bounds.
    low = float(call_price(100.0, 105.0, 0.75, 0.18, 0.03, 0.01))
    high = float(call_price(100.0, 105.0, 0.75, 0.42, 0.03, 0.01))
    mixture = 0.65 * low + 0.35 * high
    lower = max(100.0 * np.exp(-0.01 * 0.75) - 105.0 * np.exp(-0.03 * 0.75), 0.0)
    upper = 100.0 * np.exp(-0.01 * 0.75)
    regime_error = max(lower - mixture, mixture - upper, 0.0)
    checks: tuple[EngineQualification, ...] = (
        EngineQualification(
            "HESTON_FIXED_QUAD",
            "PASS" if heston_error <= 2.0e-6 else "FAIL",
            heston_error,
            2.0e-6,
            "fixed Gauss-Legendre versus adaptive integration",
        ),
        EngineQualification(
            "BATES_CF",
            "PASS" if bates_error <= 2.0e-6 else "FAIL",
            bates_error,
            2.0e-6,
            "zero jump intensity versus qualified Heston CF",
        ),
        EngineQualification(
            "LOCALVOL_SSVI_MARGINAL",
            "PASS" if local_error == 0.0 else "FAIL",
            local_error,
            0.0,
            "SSVI positivity, calendar, density, and wing diagnostics",
        ),
        EngineQualification(
            "PIECEWISE_REGIME_MARGINAL",
            "PASS" if regime_error == 0.0 else "FAIL",
            regime_error,
            0.0,
            "convex risk-neutral mixture respects European call bounds",
        ),
    )
    if device == "cuda":
        design = _surface_design(88_000_001, 4, ood=False)
        cuda_checks: list[EngineQualification] = []
        cpu_config = ConfirmatoryConfig(
            cases_dev=1,
            cases_validation=1,
            cases_locked=1,
            cases_ood=1,
            contracts_per_case=4,
            heston_nodes=64,
            bates_nodes=64,
            device="cpu-bounded-reference",
        )
        gpu_config = replace(cpu_config, device="cuda")
        cpu_functions: Mapping[
            str, Callable[[], tuple[FloatArray, FloatArray, FloatArray, FloatArray]]
        ] = {
            "HESTON_FIXED_QUAD": lambda: _heston_values(design, cpu_config),
            "BATES_CF": lambda: _bates_values(design, cpu_config, 88_000_001),
            "LOCALVOL_SSVI_MARGINAL": lambda: _localvol_ssvi_values(design, 88_000_001),
            "PIECEWISE_REGIME_MARGINAL": lambda: _piecewise_regime_values(design, 88_000_001),
        }
        family_by_engine: Mapping[str, TruthFamily] = {
            "HESTON_FIXED_QUAD": "HESTON",
            "BATES_CF": "BATES",
            "LOCALVOL_SSVI_MARGINAL": "LOCALVOL_SSVI",
            "PIECEWISE_REGIME_MARGINAL": "PIECEWISE_REGIME",
        }
        for engine, cpu_fn in cpu_functions.items():
            cpu_values = cpu_fn()[0]
            gpu_values = _cuda_family_values(
                design, family_by_engine[engine], gpu_config, 88_000_001
            )[0]
            error = float(np.max(np.abs(cpu_values - gpu_values)))
            tolerance = 5.0e-7
            cuda_checks.append(
                EngineQualification(
                    engine,
                    "PASS" if error <= tolerance else "FAIL",
                    error,
                    tolerance,
                    "Torch CUDA FP64 versus bounded NumPy FP64 spot check",
                )
            )
        checks = tuple(cuda_checks)
    return checks


def create_freeze_manifest(
    path: Path,
    config: ConfirmatoryConfig,
    *,
    source_paths: Sequence[Path] = (),
) -> dict[str, Any]:
    """Create the immutable pre-execution design manifest.

    The function refuses replacement, including an apparently identical
    manifest. This makes the freeze event unambiguous and auditable.
    """

    config.validate()
    seeds = build_seed_registry(config)
    qualifications = qualify_confirmatory_engines(config.device)
    failed = [item.engine for item in qualifications if item.status != "PASS"]
    if failed:
        raise RuntimeError(f"cannot freeze with unqualified engines: {failed}")
    content_hashes: dict[str, str] = {}
    for source_path in source_paths:
        content_hashes[source_path.as_posix()] = sha256(source_path.read_bytes()).hexdigest()
    payload: dict[str, Any] = {
        "schema_version": "1.0",
        "status": "FROZEN_NOT_EXECUTED",
        "created_utc": datetime.now(UTC).isoformat(),
        "experiment_id": config.experiment_id,
        "config": _config_payload(config),
        "config_hash": _config_hash(config),
        "seeds": asdict(seeds),
        "engine_qualification": [asdict(item) for item in qualifications],
        "content_hashes": content_hashes,
        "baseline_registry": list(BASELINE_NAMES),
        "ablation_registry": dict(ABLATION_SPECS),
        "fault_registry": {
            "HESTON": "F00",
            "BATES": "F01_REAL_HESTON_ON_BATES",
            "LOCALVOL_SSVI": "F02_REAL_HESTON_ON_LOCALVOL_SSVI_MARGINAL",
            "PIECEWISE_REGIME": "F03_REAL_REGIME_TRUTH",
        },
        "exploratory_artifacts_are_inputs": False,
        "locked_test_tuning_prohibited": True,
    }
    canonical = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    payload["freeze_hash"] = sha256(canonical.encode()).hexdigest()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return payload


def load_freeze_manifest(path: Path, config: ConfirmatoryConfig) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("status") != "FROZEN_NOT_EXECUTED":
        raise ValueError("confirmatory manifest is not in the pre-execution frozen state")
    if payload.get("config_hash") != _config_hash(config):
        raise ValueError("runtime configuration differs from the frozen design")
    if any(item.get("status") != "PASS" for item in payload.get("engine_qualification", [])):
        raise ValueError("frozen manifest contains an unqualified engine")
    return cast(dict[str, Any], payload)


def _surface_design(seed: int, contracts: int, *, ood: bool) -> SyntheticDesign:
    rng = np.random.default_rng(seed)
    spot = float(rng.uniform(85.0, 115.0))
    log_limit = 0.55 if ood else 0.30
    log_moneyness = np.linspace(-log_limit, log_limit, contracts)
    rng.shuffle(log_moneyness)
    maturity_low, maturity_high = (1.5, 3.0) if ood else (0.08, 1.5)
    maturity = rng.uniform(maturity_low, maturity_high, contracts)
    rate = np.full(contracts, rng.uniform(0.005, 0.07))
    dividend = np.full(contracts, rng.uniform(0.005, 0.035))
    theta = rng.uniform(0.025, 0.10) if not ood else rng.uniform(0.10, 0.20)
    v0 = rng.uniform(0.7 * theta, 1.3 * theta)
    kappa = rng.uniform(0.6, 4.0) if not ood else rng.uniform(0.2, 0.7)
    sigma_v = rng.uniform(0.25, 0.9) if not ood else rng.uniform(0.9, 1.4)
    rho = rng.uniform(-0.85, -0.25) if not ood else rng.uniform(-0.98, -0.85)

    def repeated(value: float) -> FloatArray:
        return np.full(contracts, value, dtype=np.float64)

    return SyntheticDesign(
        scenario_id=np.asarray([f"CONF-{seed}-{index:04d}" for index in range(contracts)]),
        spot=repeated(spot),
        strike=spot * np.exp(log_moneyness),
        maturity=np.asarray(maturity, dtype=np.float64),
        rate=rate,
        dividend_yield=dividend,
        volatility=repeated(np.sqrt(theta)),
        heston_v0=repeated(v0),
        heston_kappa=repeated(kappa),
        heston_theta=repeated(theta),
        heston_sigma_v=repeated(sigma_v),
        heston_rho=repeated(rho),
    )


def _heston_parameters(design: SyntheticDesign, index: int = 0) -> HestonParameters:
    return HestonParameters(
        float(design.heston_v0[index]),
        float(design.heston_kappa[index]),
        float(design.heston_theta[index]),
        float(design.heston_sigma_v[index]),
        float(design.heston_rho[index]),
    )


def _generic_greeks(
    design: SyntheticDesign,
    pricer: Callable[[int, float, float], float],
) -> tuple[FloatArray, FloatArray, FloatArray, FloatArray]:
    prices = np.empty(len(design), dtype=np.float64)
    delta = np.empty(len(design), dtype=np.float64)
    gamma = np.empty(len(design), dtype=np.float64)
    vega = np.empty(len(design), dtype=np.float64)
    for index in range(len(design)):
        spot = float(design.spot[index])
        ds = max(spot * 1.0e-3, 1.0e-4)
        variance_scale = 1.0e-3
        base = pricer(index, spot, 1.0)
        up = pricer(index, spot + ds, 1.0)
        down = pricer(index, spot - ds, 1.0)
        prices[index] = base
        delta[index] = (up - down) / (2.0 * ds)
        gamma[index] = (up - 2.0 * base + down) / (ds * ds)
        vol_up = pricer(index, spot, 1.0 + variance_scale)
        vol_down = pricer(index, spot, 1.0 - variance_scale)
        base_vol = np.sqrt(float(design.heston_theta[index]))
        vega[index] = (vol_up - vol_down) / (2.0 * variance_scale * base_vol)
    return prices, delta, gamma, vega


def _heston_values(
    design: SyntheticDesign, config: ConfirmatoryConfig
) -> tuple[FloatArray, FloatArray, FloatArray, FloatArray]:
    base = _heston_parameters(design)

    def price(index: int, spot: float, variance_scale: float) -> float:
        parameters = HestonParameters(
            max(base.v0 * variance_scale * variance_scale, 1.0e-8),
            base.kappa,
            max(base.theta * variance_scale * variance_scale, 1.0e-8),
            base.sigma_v,
            base.rho,
        )
        return float(
            heston_price_fixed_quad(
                spot,
                float(design.strike[index]),
                float(design.maturity[index]),
                parameters,
                float(design.rate[index]),
                float(design.dividend_yield[index]),
                nodes=config.heston_nodes,
                upper_bound=config.integration_upper_bound,
            )
        )

    return _generic_greeks(design, price)


def _bates_values(
    design: SyntheticDesign, config: ConfirmatoryConfig, seed: int
) -> tuple[FloatArray, FloatArray, FloatArray, FloatArray]:
    rng = np.random.default_rng(seed + 1_000_000)
    heston = _heston_parameters(design)
    jump_intensity = float(rng.uniform(0.15, 0.8))
    mean_jump = float(rng.uniform(-0.18, -0.03))
    jump_volatility = float(rng.uniform(0.10, 0.30))

    def price(index: int, spot: float, variance_scale: float) -> float:
        parameters = BatesParameters(
            max(heston.v0 * variance_scale * variance_scale, 1.0e-8),
            heston.kappa,
            max(heston.theta * variance_scale * variance_scale, 1.0e-8),
            heston.sigma_v,
            heston.rho,
            jump_intensity,
            mean_jump,
            jump_volatility,
        )
        return float(
            bates_price(
                spot,
                float(design.strike[index]),
                float(design.maturity[index]),
                parameters,
                float(design.rate[index]),
                float(design.dividend_yield[index]),
                nodes=config.bates_nodes,
                upper_bound=config.integration_upper_bound,
            )
        )

    return _generic_greeks(design, price)


def _localvol_ssvi_values(
    design: SyntheticDesign, seed: int
) -> tuple[FloatArray, FloatArray, FloatArray, FloatArray]:
    rng = np.random.default_rng(seed + 2_000_000)
    parameters = SSVIParameters(
        rho=float(rng.uniform(-0.65, -0.20)),
        eta=float(rng.uniform(0.25, 0.60)),
        gamma=float(rng.uniform(0.35, 0.65)),
    )
    theta_rate = float(design.heston_theta[0])

    def price(index: int, spot: float, variance_scale: float) -> float:
        maturity = float(design.maturity[index])
        forward = spot * np.exp(
            (float(design.rate[index]) - float(design.dividend_yield[index])) * maturity
        )
        k = np.log(float(design.strike[index]) / forward)
        theta = theta_rate * variance_scale * variance_scale * maturity
        total_variance = float(ssvi_total_variance(k, theta, parameters))
        volatility = np.sqrt(total_variance / maturity)
        return float(
            call_price(
                spot,
                float(design.strike[index]),
                maturity,
                volatility,
                float(design.rate[index]),
                float(design.dividend_yield[index]),
            )
        )

    return _generic_greeks(design, price)


def _piecewise_regime_values(
    design: SyntheticDesign, seed: int
) -> tuple[FloatArray, FloatArray, FloatArray, FloatArray]:
    rng = np.random.default_rng(seed + 3_000_000)
    switch = float(rng.uniform(0.25, 0.75))
    weight = float(rng.uniform(0.45, 0.75))
    low = float(rng.uniform(0.12, 0.22))
    high = float(rng.uniform(0.35, 0.60))

    def integrated_vol(maturity: float, first: float, second: float) -> float:
        variance = first * first * min(maturity, switch)
        variance += second * second * max(maturity - switch, 0.0)
        return float(np.sqrt(variance / maturity))

    def price(index: int, spot: float, variance_scale: float) -> float:
        maturity = float(design.maturity[index])
        low_high = integrated_vol(maturity, low, high) * variance_scale
        high_low = integrated_vol(maturity, high, low) * variance_scale
        common = (
            spot,
            float(design.strike[index]),
            maturity,
        )
        rates = (float(design.rate[index]), float(design.dividend_yield[index]))
        first = float(call_price(*common, low_high, *rates))
        second = float(call_price(*common, high_low, *rates))
        return weight * first + (1.0 - weight) * second

    return _generic_greeks(design, price)


def _cuda_family_values(
    design: SyntheticDesign,
    family: TruthFamily,
    config: ConfirmatoryConfig,
    seed: int,
) -> tuple[FloatArray, FloatArray, FloatArray, FloatArray]:
    """Vectorized Torch FP64 truth and finite-difference Greeks on one CUDA device."""

    try:
        import torch
    except ImportError as exc:  # pragma: no cover - optional environment
        raise RuntimeError("Torch is required for confirmatory CUDA execution") from exc
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; no CPU research fallback is permitted")
    device = torch.device("cuda")
    dtype = torch.float64
    cdtype = torch.complex128

    def tensor(values: FloatArray) -> Any:
        return torch.as_tensor(values, dtype=dtype, device=device)

    strike = tensor(design.strike)
    maturity = tensor(design.maturity)
    rate = tensor(design.rate)
    dividend = tensor(design.dividend_yield)
    base_spot = tensor(design.spot)
    v0 = tensor(design.heston_v0)
    kappa = tensor(design.heston_kappa)
    theta = tensor(design.heston_theta)
    sigma = tensor(design.heston_sigma_v)
    rho = tensor(design.heston_rho)

    rng = np.random.default_rng(seed + 1_000_000)
    jump_intensity = float(rng.uniform(0.15, 0.8))
    mean_jump = float(rng.uniform(-0.18, -0.03))
    jump_vol = float(rng.uniform(0.10, 0.30))
    ssvi_rng = np.random.default_rng(seed + 2_000_000)
    ssvi_rho = float(ssvi_rng.uniform(-0.65, -0.20))
    ssvi_eta = float(ssvi_rng.uniform(0.25, 0.60))
    ssvi_gamma = float(ssvi_rng.uniform(0.35, 0.65))
    regime_rng = np.random.default_rng(seed + 3_000_000)
    switch = float(regime_rng.uniform(0.25, 0.75))
    weight = float(regime_rng.uniform(0.45, 0.75))
    low = float(regime_rng.uniform(0.12, 0.22))
    high = float(regime_rng.uniform(0.35, 0.60))

    def bs(spot: Any, volatility: Any) -> Any:
        root_t = torch.sqrt(maturity)
        d1 = (
            torch.log(spot / strike) + (rate - dividend + 0.5 * volatility * volatility) * maturity
        ) / (volatility * root_t)
        d2 = d1 - volatility * root_t

        def normal(value: Any) -> Any:
            return 0.5 * (1.0 + torch.erf(value / np.sqrt(2.0)))

        return spot * torch.exp(-dividend * maturity) * normal(d1) - strike * torch.exp(
            -rate * maturity
        ) * normal(d2)

    def price(spot: Any, variance_scale: float) -> Any:
        if family in {"HESTON", "BATES"}:
            nodes = config.heston_nodes if family == "HESTON" else config.bates_nodes
            roots_np, weights_np = np.polynomial.legendre.leggauss(nodes)
            u = torch.as_tensor(
                0.5 * config.integration_upper_bound * (roots_np + 1.0),
                dtype=dtype,
                device=device,
            )
            weights = torch.as_tensor(
                0.5 * config.integration_upper_bound * weights_np,
                dtype=dtype,
                device=device,
            )
            scaled_v0 = v0 * variance_scale * variance_scale
            scaled_theta = theta * variance_scale * variance_scale

            def cf(z: Any) -> Any:
                z = z[None, :]
                iu = 1j * z
                beta = kappa[:, None] - rho[:, None] * sigma[:, None] * iu
                root = torch.sqrt(beta * beta + sigma[:, None] ** 2 * (z * z + iu))
                root = torch.where(torch.real(root) < 0.0, -root, root)
                ratio = (beta - root) / (beta + root)
                decay = torch.exp(-root * maturity[:, None])
                log_ratio = torch.log1p(-ratio * decay) - torch.log1p(-ratio)
                compensated = rate - dividend
                if family == "BATES":
                    mean_relative = np.exp(mean_jump + 0.5 * jump_vol**2) - 1.0
                    compensated = compensated - jump_intensity * mean_relative
                c = iu * (torch.log(spot)[:, None] + compensated[:, None] * maturity[:, None]) + (
                    kappa * scaled_theta / (sigma * sigma)
                )[:, None] * ((beta - root) * maturity[:, None] - 2.0 * log_ratio)
                d = (beta - root) / sigma[:, None] ** 2 * (1.0 - decay) / (1.0 - ratio * decay)
                value = torch.exp(c + d * scaled_v0[:, None])
                if family == "BATES":
                    jump = torch.exp(
                        jump_intensity
                        * maturity[:, None]
                        * (torch.exp(iu * mean_jump - 0.5 * jump_vol**2 * z * z) - 1.0)
                    )
                    value = value * jump
                return value

            uc = u.to(cdtype)
            phase = torch.exp(-1j * torch.log(strike)[:, None].to(cdtype) * uc[None, :])
            phi = cf(uc)
            shifted = cf(uc - 1j)
            first_moment = spot * torch.exp((rate - dividend) * maturity)
            i1 = torch.sum(
                weights[None, :]
                * torch.real(phase * shifted / (1j * uc[None, :] * first_moment[:, None])),
                dim=1,
            )
            i2 = torch.sum(weights[None, :] * torch.real(phase * phi / (1j * uc[None, :])), dim=1)
            return spot * torch.exp(-dividend * maturity) * (0.5 + i1 / np.pi) - strike * torch.exp(
                -rate * maturity
            ) * (0.5 + i2 / np.pi)
        if family == "LOCALVOL_SSVI":
            forward = spot * torch.exp((rate - dividend) * maturity)
            log_moneyness = torch.log(strike / forward)
            theta_total = theta * variance_scale * variance_scale * maturity
            phi = ssvi_eta / (theta_total**ssvi_gamma * (1.0 + theta_total) ** (1.0 - ssvi_gamma))
            shifted = phi * log_moneyness
            total = (
                0.5
                * theta_total
                * (
                    1.0
                    + ssvi_rho * shifted
                    + torch.sqrt((shifted + ssvi_rho) ** 2 + 1.0 - ssvi_rho**2)
                )
            )
            return bs(spot, torch.sqrt(total / maturity))
        first_variance = low**2 * torch.minimum(maturity, torch.tensor(switch, device=device))
        first_variance += high**2 * torch.clamp(maturity - switch, min=0.0)
        second_variance = high**2 * torch.minimum(maturity, torch.tensor(switch, device=device))
        second_variance += low**2 * torch.clamp(maturity - switch, min=0.0)
        first_vol = torch.sqrt(first_variance / maturity) * variance_scale
        second_vol = torch.sqrt(second_variance / maturity) * variance_scale
        return weight * bs(spot, first_vol) + (1.0 - weight) * bs(spot, second_vol)

    ds = torch.clamp(base_spot * 1.0e-3, min=1.0e-4)
    epsilon = 1.0e-3
    with torch.no_grad():
        base = price(base_spot, 1.0)
        up = price(base_spot + ds, 1.0)
        down = price(base_spot - ds, 1.0)
        vol_up = price(base_spot, 1.0 + epsilon)
        vol_down = price(base_spot, 1.0 - epsilon)
        delta = (up - down) / (2.0 * ds)
        gamma = (up - 2.0 * base + down) / (ds * ds)
        vega = (vol_up - vol_down) / (2.0 * epsilon * torch.sqrt(theta))

    def numpy(value: Any) -> FloatArray:
        return np.asarray(value.detach().cpu().numpy(), dtype=np.float64)

    return numpy(base), numpy(delta), numpy(gamma), numpy(vega)


def generate_confirmatory_truth(
    design: SyntheticDesign,
    family: TruthFamily,
    config: ConfirmatoryConfig,
    *,
    seed: int,
) -> SyntheticTruth:
    """Generate qualified latent values; no proxy truth models are permitted."""

    if config.device == "cuda":
        values = _cuda_family_values(design, family, config, seed)
        model_id = {
            "HESTON": "SYN-HESTON",
            "BATES": "SYN-BATES",
            "LOCALVOL_SSVI": "SYN-LOCALVOL",
            "PIECEWISE_REGIME": "SYN-REGIME",
        }[family]
        backend = f"torch-cuda-fp64-{family.lower()}"
    elif family == "HESTON":
        values = _heston_values(design, config)
        model_id = "SYN-HESTON"
        backend = "canonical-heston-cf-fp64"
    elif family == "BATES":
        values = _bates_values(design, config, seed)
        model_id = "SYN-BATES"
        backend = "canonical-bates-cf-fp64"
    elif family == "LOCALVOL_SSVI":
        values = _localvol_ssvi_values(design, seed)
        model_id = "SYN-LOCALVOL"
        backend = "qualified-ssvi-marginal-of-dupire-localvol-fp64"
    elif family == "PIECEWISE_REGIME":
        values = _piecewise_regime_values(design, seed)
        model_id = "SYN-REGIME"
        backend = "piecewise-two-regime-risk-neutral-mixture-fp64"
    else:
        raise ValueError(f"unknown confirmatory truth family: {family}")
    prices, delta, gamma, vega = values
    if any(np.any(~np.isfinite(value)) for value in values):
        raise FloatingPointError(f"non-finite {family} truth value")
    return SyntheticTruth(
        design=design,
        model_id=cast(Any, model_id),
        true_price=prices,
        true_delta=delta,
        true_gamma=gamma,
        true_vega=vega,
        backend=backend,
        precision="float64",
    )


def build_confirmatory_case(
    truth: SyntheticTruth,
    family: TruthFamily,
    config: ConfirmatoryConfig,
    *,
    observation_seed: int,
) -> ValidationCase:
    """Build genuine Heston-on-truth cases for F01/F02 and regime truth."""

    surface = observe_truth(truth, config.observation, seed=observation_seed)
    clean = clean_validation_case(surface)
    if config.device == "cuda":
        developer_values = _cuda_family_values(
            truth.design, "HESTON", config, observation_seed - 10_000_000
        )
    else:
        developer_values = _heston_values(truth.design, config)
    developer_price, developer_delta, developer_gamma, developer_vega = developer_values
    if family == "HESTON":
        fault_id = "F00"
    elif family == "BATES":
        fault_id = "F01"
    elif family == "LOCALVOL_SSVI":
        fault_id = "F02"
    else:
        fault_id = "F03"
    # Evidence models are fitted only to observable synthetic quotes. Latent
    # truth is used by make_record for material-error labels, never as a DVE input.
    design = truth.design
    observed = surface.observed & np.isfinite(surface.mid)
    if np.sum(observed) < 3:
        raise ValueError("confirmatory surface has fewer than three observable quotes")

    def fit_flat(target: FloatArray) -> float:
        def objective(volatility: float) -> float:
            values = np.asarray(
                call_price(
                    design.spot[observed],
                    design.strike[observed],
                    design.maturity[observed],
                    volatility,
                    design.rate[observed],
                    design.dividend_yield[observed],
                )
            )
            return float(np.mean(np.square(values - target[observed])))

        result = minimize_scalar(objective, bounds=(0.03, 2.0), method="bounded")
        if not result.success:
            raise RuntimeError("observable flat-vol challenger fit failed")
        return float(result.x)

    flat_vol = fit_flat(surface.mid)
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
    iv_mask = observed & np.isfinite(observed_iv)
    log_moneyness = np.log(design.strike / design.spot)
    if np.sum(iv_mask) >= 3:
        coefficients = np.polyfit(log_moneyness[iv_mask], observed_iv[iv_mask], deg=2)
        skew_vol = np.clip(np.polyval(coefficients, log_moneyness), 0.03, 2.0)
    else:
        skew_vol = np.full(len(design), flat_vol)
    flat_vector = np.full(len(design), flat_vol)
    flat_price = np.asarray(
        call_price(
            design.spot,
            design.strike,
            design.maturity,
            flat_vector,
            design.rate,
            design.dividend_yield,
        )
    )
    skew_price = np.asarray(
        call_price(
            design.spot,
            design.strike,
            design.maturity,
            skew_vol,
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
    skew_risk = greeks(
        design.spot,
        design.strike,
        design.maturity,
        skew_vol,
        design.rate,
        design.dividend_yield,
    )
    calibration = np.empty((len(design), 8), dtype=np.float64)
    rng = np.random.default_rng(observation_seed + 5_000_000)
    for draw in range(calibration.shape[1]):
        sampled = surface.bid + rng.random(len(design)) * (surface.ask - surface.bid)
        sampled = np.where(observed, sampled, surface.mid)
        draw_vol = fit_flat(sampled)
        calibration[:, draw] = np.asarray(
            call_price(
                design.spot,
                design.strike,
                design.maturity,
                draw_vol,
                design.rate,
                design.dividend_yield,
            )
        )
    ql_parameters = QuantLibHestonParameters(
        float(design.heston_v0[0]),
        float(design.heston_kappa[0]),
        float(design.heston_theta[0]),
        float(design.heston_sigma_v[0]),
        float(design.heston_rho[0]),
    )
    oracle = np.asarray(
        [
            quantlib_heston_price(
                float(design.spot[index]),
                float(design.strike[index]),
                float(design.maturity[index]),
                ql_parameters,
                float(design.rate[index]),
                float(design.dividend_yield[index]),
            ).price
            for index in range(len(design))
        ]
    )
    return replace(
        clean,
        fault_id=cast(Any, fault_id),
        developer_price=developer_price,
        developer_delta=developer_delta,
        developer_gamma=developer_gamma,
        developer_vega=developer_vega,
        numerical_prices=np.column_stack((developer_price, oracle)),
        calibration_prices=calibration,
        challenger_prices=np.column_stack((flat_price, skew_price)),
        challenger_delta=np.column_stack((flat_risk.delta, skew_risk.delta)),
        challenger_gamma=np.column_stack((flat_risk.gamma, skew_risk.gamma)),
        challenger_vega=np.column_stack((flat_risk.vega, skew_risk.vega)),
        proxy_fault=False,
    )


def _global_only(model: FrozenDVE) -> FrozenDVE:
    return replace(model, regime_thresholds={"V06": {}}, regime_fallbacks={"V06": ()})


def evaluate_confirmatory_ablations(
    dev_records: Sequence[EvidenceRecord],
    locked_records: Sequence[EvidenceRecord],
    full_model: FrozenDVE,
) -> dict[str, ConfirmatoryAblation]:
    full = evaluate_locked(locked_records, full_model)["V06"]
    output: dict[str, ConfirmatoryAblation] = {}

    def store(ablation_id: str, metric: DetectionMetrics) -> None:
        output[ablation_id] = ConfirmatoryAblation(
            ablation_id,
            ABLATION_SPECS[ablation_id],
            metric,
            metric.recall_at_dev_5pct_fpr - full.recall_at_dev_5pct_fpr,
            metric.auprc - full.auprc,
            metric.false_alarm_rate - full.false_alarm_rate,
            metric.material_risk_miss_rate - full.material_risk_miss_rate,
        )

    for ablation_id, removed in _DIMENSION_ABLATIONS.items():
        dimensions = tuple(name for name in EVIDENCE_NAMES if name != removed)
        model = fit_dev_thresholds(
            dev_records,
            target_fpr=full_model.target_fpr,
            dimensions=dimensions,
            minimum_regime_negatives=2,
        )
        store(ablation_id, evaluate_locked(locked_records, model)["V06"])
    store("A09", evaluate_locked(locked_records, _global_only(full_model))["V06"])
    # Without materiality prioritization, every injected mechanism is treated
    # as equally positive. This changes the target explicitly and is reported
    # as an ablation, never mixed into the primary material-error endpoint.
    dev_unprioritized = tuple(
        replace(record, material_error=record.fault_id != "F00") for record in dev_records
    )
    locked_unprioritized = tuple(
        replace(record, material_error=record.fault_id != "F00") for record in locked_records
    )
    no_materiality = fit_dev_thresholds(
        dev_unprioritized,
        target_fpr=full_model.target_fpr,
        minimum_regime_negatives=2,
    )
    store("A11", evaluate_locked(locked_unprioritized, no_materiality)["V06"])
    if set(output) != set(ABLATION_SPECS):
        raise AssertionError("confirmatory ablation coverage is incomplete")
    return output


def run_confirmatory(
    manifest_path: Path,
    config: ConfirmatoryConfig,
    *,
    output_root: Path | None = None,
) -> ConfirmatoryRunResult:
    """Execute a bounded frozen design and optionally persist new artifacts."""

    config.validate()
    manifest = load_freeze_manifest(manifest_path, config)
    if config.device != "cuda" and config.total_contracts > config.max_cpu_contracts:
        raise RuntimeError(
            "confirmatory design exceeds bounded CPU ceiling; "
            "use a qualified coordinated GPU runner"
        )
    if output_root is not None and "exploratory" in {part.lower() for part in output_root.parts}:
        raise ValueError("confirmatory outputs may not overwrite the exploratory namespace")
    seeds = SeedRegistry(**manifest["seeds"])
    seeds.validate()
    if output_root is not None:
        if output_root.exists() and any(output_root.iterdir()):
            raise FileExistsError("confirmatory output directory is non-empty; refusing overwrite")
        output_root.mkdir(parents=True, exist_ok=True)
    records: list[EvidenceRecord] = []
    provenance: list[Mapping[str, object]] = []
    run_id = f"{config.experiment_id}-{str(manifest['freeze_hash'])[:12]}"

    def generate_partition(partition: Partition) -> None:
        for index, seed in enumerate(seeds.for_partition(partition)):
            family = config.truth_families[index % len(config.truth_families)]
            design = _surface_design(
                seed,
                config.contracts_per_case,
                ood=partition == "OOD_LOCKED_TEST",
            )
            truth = generate_confirmatory_truth(design, family, config, seed=seed)
            observation_seed = seed + seeds.observation_offset
            case = build_confirmatory_case(
                truth,
                family,
                config,
                observation_seed=observation_seed,
            )
            case_id = f"{partition}-{family}-{seed}"
            record = make_record(
                case_id,
                partition,
                case,
                materiality_threshold=config.materiality_threshold_spread_units,
            )
            records.append(record)
            provenance.append(
                {
                    "run_id": run_id,
                    "case_id": case_id,
                    "partition": partition,
                    "truth_family": family,
                    "fault_id": record.fault_id,
                    "proxy_fault": record.proxy_fault,
                    "truth_seed": seed,
                    "observation_seed": observation_seed,
                    "backend": truth.backend,
                    "precision": truth.precision,
                    "config_hash": manifest["config_hash"],
                    "freeze_hash": manifest["freeze_hash"],
                    "generated_utc": datetime.now(UTC).isoformat(),
                }
            )

    # DEV and VALIDATION are generated first. Locked seed generation is not
    # entered until the DVE threshold artifact is durably frozen.
    generate_partition("DEV")
    generate_partition("VALIDATION")
    dev = tuple(record for record in records if record.partition == "DEV")
    model = fit_dev_thresholds(
        dev,
        target_fpr=config.target_fpr,
        minimum_regime_negatives=config.minimum_regime_negatives,
    )
    validation = tuple(record for record in records if record.partition == "VALIDATION")
    validation_as_locked = tuple(replace(record, partition="LOCKED_TEST") for record in validation)
    validation_metrics = evaluate_locked(validation_as_locked, model)
    threshold_payload = {
        "status": "FROZEN_BEFORE_LOCKED_GENERATION",
        "frozen_utc": datetime.now(UTC).isoformat(),
        "design_freeze_hash": manifest["freeze_hash"],
        "thresholds": model.to_dict(),
        "validation_diagnostics": {
            name: asdict(value) for name, value in validation_metrics.items()
        },
        "validation_role": "confirmation only; no locked-test selection",
    }
    threshold_path = (
        output_root / "dve_thresholds.freeze.json"
        if output_root is not None
        else manifest_path.with_suffix(".dve_thresholds.json")
    )
    with threshold_path.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(threshold_payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    generate_partition("LOCKED_TEST")
    generate_partition("OOD_LOCKED_TEST")
    locked = tuple(
        record for record in records if record.partition in {"LOCKED_TEST", "OOD_LOCKED_TEST"}
    )
    baselines = evaluate_locked(locked, model)
    if set(baselines) != set(BASELINE_NAMES):
        raise AssertionError("V00-V06 baseline coverage is incomplete")
    ablations = evaluate_confirmatory_ablations(dev, locked, model)
    result = ConfirmatoryRunResult(
        run_id,
        str(manifest["freeze_hash"]),
        model,
        tuple(records),
        baselines,
        ablations,
        tuple(provenance),
    )
    if output_root is not None:
        pd.DataFrame(provenance).to_parquet(output_root / "case_provenance.parquet", index=False)
        provenance_by_case = {str(row["case_id"]): row for row in provenance}
        record_rows: list[dict[str, object]] = []
        false_positives: list[dict[str, object]] = []
        false_negatives: list[dict[str, object]] = []
        for record in records:
            scores = baseline_scores(record, model.evidence_scales, model.evidence_dimensions)
            row: dict[str, object] = {
                "case_id": record.case_id,
                "partition": record.partition,
                "regime": record.regime,
                "fault_id": record.fault_id,
                "proxy_fault": record.proxy_fault,
                "fit_error": record.fit_error,
                "oos_error": record.oos_error,
                "material_error": record.material_error,
                "normalized_true_error": record.normalized_true_error,
                **record.evidence.as_dict(),
                **{f"score_{name}": value for name, value in scores.items()},
                "truth_family": provenance_by_case[record.case_id]["truth_family"],
            }
            if record.partition in {"LOCKED_TEST", "OOD_LOCKED_TEST"}:
                prediction = scores["V06"] > model.threshold("V06", record.regime)
                row["V06_prediction"] = prediction
                if prediction and not record.material_error:
                    false_positives.append(row.copy())
                if not prediction and record.material_error:
                    false_negatives.append(row.copy())
            record_rows.append(row)
        evidence_frame = pd.DataFrame(record_rows)
        evidence_frame.to_parquet(output_root / "evidence_score_records.parquet", index=False)
        pd.DataFrame(false_positives).to_parquet(
            output_root / "false_positive_cases.parquet", index=False
        )
        pd.DataFrame(false_negatives).to_parquet(
            output_root / "false_negative_cases.parquet", index=False
        )
        pd.DataFrame([asdict(value) for value in baselines.values()]).to_csv(
            output_root / "T08_DVE_BASELINES.csv", index=False
        )
        pd.DataFrame(
            [
                {
                    "ablation_id": name,
                    "description": value.description,
                    **asdict(value.metrics),
                    "recall_change": value.recall_change,
                    "auprc_change": value.auprc_change,
                    "false_alarm_change": value.false_alarm_change,
                    "miss_rate_change": value.miss_rate_change,
                }
                for name, value in ablations.items()
            ]
        ).to_csv(output_root / "T09_DVE_ABLATION.csv", index=False)
        t07_rows: list[dict[str, object]] = []
        for fault in sorted({record.fault_id for record in locked}):
            subset = tuple(record for record in locked if record.fault_id == fault)
            for _baseline, metric in evaluate_locked(subset, model).items():
                t07_rows.append({"fault_id": fault, **asdict(metric)})
        pd.DataFrame(t07_rows).to_csv(
            output_root / "T07_SYNTHETIC_FAULT_DETECTION.csv", index=False
        )
        stratified = (
            evidence_frame[evidence_frame["partition"].isin(["LOCKED_TEST", "OOD_LOCKED_TEST"])]
            .assign(
                maturity_regime=lambda value: value["regime"].str.split("|").str[0],
                liquidity_regime=lambda value: value["regime"].str.split("|").str[-1],
            )
            .groupby(
                ["truth_family", "fault_id", "regime", "liquidity_regime", "maturity_regime"],
                dropna=False,
            )
            .agg(
                observations=("case_id", "size"),
                material_rate=("material_error", "mean"),
                median_normalized_error=("normalized_true_error", "median"),
                median_V06_score=("score_V06", "median"),
            )
            .reset_index()
        )
        stratified.to_csv(output_root / "stratified_metrics.csv", index=False)
        rng = np.random.default_rng(79_000_001)
        bootstrap: dict[str, object] = {}
        for baseline in BASELINE_NAMES[:-1]:
            recall_delta: list[float] = []
            auprc_delta: list[float] = []
            for _ in range(250):
                sample = tuple(locked[index] for index in rng.integers(0, len(locked), len(locked)))
                sampled = evaluate_locked(sample, model)
                recall_delta.append(
                    sampled["V06"].recall_at_dev_5pct_fpr - sampled[baseline].recall_at_dev_5pct_fpr
                )
                auprc_delta.append(sampled["V06"].auprc - sampled[baseline].auprc)
            bootstrap[baseline] = {
                "recall_difference_V06_minus_baseline_ci95": np.nanpercentile(
                    recall_delta, [2.5, 97.5]
                ).tolist(),
                "auprc_difference_V06_minus_baseline_ci95": np.nanpercentile(
                    auprc_delta, [2.5, 97.5]
                ).tolist(),
                "bootstrap_replicates": 250,
                "seed": 79_000_001,
            }
        (output_root / "statistical_inference.json").write_text(
            json.dumps({"paired_case_bootstrap": bootstrap}, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        qualification = {
            "freeze_hash": manifest["freeze_hash"],
            "engines": manifest["engine_qualification"],
        }
        (output_root / "truth_qualification.json").write_text(
            json.dumps(qualification, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        coverage = []
        for index in range(17):
            fault = f"F{index:02d}"
            executed = fault in {record.fault_id for record in records}
            coverage.append(
                {
                    "fault_id": fault,
                    "status": "EXECUTED_REAL" if executed else "UNAVAILABLE_CONFIRMATORY_V1",
                    "reason": ""
                    if executed
                    else "isolated confirmatory mechanism not yet implemented",
                }
            )
        pd.DataFrame(coverage).to_csv(output_root / "fault_coverage.csv", index=False)
        metrics = {
            "run_id": run_id,
            "freeze_hash": result.freeze_hash,
            "baselines": {name: asdict(value) for name, value in baselines.items()},
            "ablations": {name: asdict(value) for name, value in ablations.items()},
            "thresholds": model.to_dict(),
        }
        (output_root / "confirmatory_metrics.json").write_text(
            json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    return result


def freeze_confirmatory_registries(
    manifest_path: Path | None = None,
    config: ConfirmatoryConfig | None = None,
) -> dict[str, Any]:
    """Freeze hypotheses/seeds/baselines/ablations before any locked generation.

    If an existing manifest matches the supplied configuration it is returned
    unchanged. A mismatched existing freeze remains an error.
    """

    cfg = config or ConfirmatoryConfig()
    target = manifest_path or Path("artifacts/checkpoints/synthetic_confirmatory_v2.freeze.json")
    if target.exists():
        return load_freeze_manifest(target, cfg)
    module_path = Path(__file__).resolve()
    return create_freeze_manifest(target, cfg, source_paths=(module_path,))


def _profile_config(profile: str, device: str) -> ConfirmatoryConfig:
    normalized = profile.strip().lower()
    normalized_device = device.strip().lower()
    if normalized == "research":
        if normalized_device != "cuda":
            raise ValueError("research confirmatory profile requires device='cuda'")
        return ConfirmatoryConfig(
            cases_dev=32,
            cases_validation=16,
            cases_locked=32,
            cases_ood=16,
            contracts_per_case=25,
            device="cuda",
            max_cpu_contracts=256,
            minimum_regime_negatives=5,
        )
    if normalized in {"audit", "smoke"}:
        if normalized_device not in {"cpu", "cpu-bounded-reference", "auto"}:
            raise ValueError("bounded audit profile requires cpu or auto device")
        return ConfirmatoryConfig(device="cpu-bounded-reference")
    raise ValueError("profile must be research, audit, or smoke")


def run_confirmatory_study(
    profile: str = "research",
    device: str = "cuda",
    resume: bool = True,
) -> dict[str, Any]:
    """Primary callable entry point with freeze-before-generation enforcement.

    The research profile currently returns a scientifically explicit blocked
    status after freezing, because the coordinated GPU implementation for all
    truth families is not yet qualified. The bounded audit profile executes
    the exact engines and is intended for integration verification, not as a
    substitute confirmatory result.
    """

    config = _profile_config(profile, device)
    root = Path("results/confirmatory/synthetic_v1")
    manifest_path = Path("artifacts/checkpoints/synthetic_confirmatory_v2.freeze.json")
    manifest = freeze_confirmatory_registries(manifest_path, config)
    metrics_path = root / "confirmatory_metrics.json"
    if resume and metrics_path.exists():
        payload = json.loads(metrics_path.read_text(encoding="utf-8"))
        if payload.get("freeze_hash") != manifest.get("freeze_hash"):
            raise ValueError("existing confirmatory results do not match the frozen manifest")
        return cast(dict[str, Any], payload)
    result = run_confirmatory(manifest_path, config, output_root=root)
    return {
        "status": "COMPLETE_BOUNDED_AUDIT",
        "profile": profile,
        "device": device,
        "run_id": result.run_id,
        "freeze_hash": result.freeze_hash,
        "records": len(result.records),
        "baselines": {name: asdict(value) for name, value in result.baselines.items()},
        "ablations": {name: asdict(value) for name, value in result.ablations.items()},
    }
