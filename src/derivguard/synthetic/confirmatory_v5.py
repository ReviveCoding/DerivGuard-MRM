"""Release-aligned, convergence-gated confirmatory synthetic study v5."""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, replace
from datetime import UTC, datetime
from hashlib import sha256
from pathlib import Path
from typing import Any, Literal, cast

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from scipy.optimize import minimize

from derivguard.development.heston_cf import HestonParameters
from derivguard.synthetic.confirmatory import (
    ABLATION_SPECS,
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
from derivguard.synthetic.confirmatory_v4 import (
    PARAMETER_BOUNDS,
    ConfirmatoryV4Config,
    _developer_values,
    _observable_challengers,
    _price_matrix,
    qualify_v4_calibration_engine,
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


class CalibrationUnqualifiedError(RuntimeError):
    """Raised before DVE construction when no converged release-aligned fit exists."""

    def __init__(self, message: str, diagnostics: list[dict[str, object]]) -> None:
        super().__init__(message)
        self.diagnostics = diagnostics


@dataclass(frozen=True)
class ConfirmatoryV5Config:
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
    primary_multistarts: int = 5
    primary_max_iterations: int = 160
    bootstrap_draws: int = 4
    bootstrap_multistarts: int = 3
    bootstrap_max_iterations: int = 80
    finite_difference_relative_step: float = 2.0e-4

    def validate(self) -> None:
        if self.profile not in {"research", "audit"}:
            raise ValueError("profile must be research or audit")
        if self.device not in {"cuda", "cpu-bounded-reference"}:
            raise ValueError("invalid device")
        counts = (self.cases_dev, self.cases_validation, self.cases_locked, self.cases_ood)
        if any(value < 17 for value in counts):
            raise ValueError("all partitions must cover F00--F16")
        if self.contracts_per_case < 3 or self.calibration_nodes < 16:
            raise ValueError("invalid surface or quadrature size")
        if self.primary_multistarts < 2 or self.bootstrap_multistarts < 2:
            raise ValueError("O01 requires at least two independent local starts")
        if self.primary_max_iterations < 80 or self.bootstrap_max_iterations < 40:
            raise ValueError("v5 iteration budgets must materially exceed v4")
        if self.bootstrap_draws < 2 or self.materiality_threshold <= 0.0:
            raise ValueError("invalid bootstrap or materiality design")

    def engine_config(self) -> ConfirmatoryV4Config:
        return ConfirmatoryV4Config(
            profile=self.profile,
            device=self.device,
            cases_dev=self.cases_dev,
            cases_validation=self.cases_validation,
            cases_locked=self.cases_locked,
            cases_ood=self.cases_ood,
            contracts_per_case=self.contracts_per_case,
            materiality_threshold=self.materiality_threshold,
            target_fpr=self.target_fpr,
            minimum_regime_negatives=self.minimum_regime_negatives,
            truth_nodes=self.truth_nodes,
            bates_nodes=self.bates_nodes,
            calibration_nodes=self.calibration_nodes,
            population_candidates=4,
            local_iterations=1,
            bootstrap_draws=self.bootstrap_draws,
            bootstrap_candidates=4,
            bootstrap_local_iterations=1,
        )


def build_v5_seed_registry(config: ConfirmatoryV5Config) -> SeedRegistry:
    registry = SeedRegistry(
        dev=tuple(103_000_001 + index for index in range(config.cases_dev)),
        validation=tuple(104_000_001 + index for index in range(config.cases_validation)),
        locked=tuple(105_000_001 + index for index in range(config.cases_locked)),
        ood=tuple(106_000_001 + index for index in range(config.cases_ood)),
        observation_offset=50_000_000,
    )
    registry.validate()
    return registry


def _hash(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def _fault(index: int) -> FaultId:
    return FAULT_IDS[index % len(FAULT_IDS)]


def freeze_confirmatory_v5(
    path: Path = Path("artifacts/checkpoints/synthetic_confirmatory_v5.freeze.json"),
    config: ConfirmatoryV5Config | None = None,
    hypothesis_registry: Path = Path("governance/confirmatory_v5_hypothesis_registry.yaml"),
    experiment_registry: Path = Path("experiments/confirmatory_v5_registry.yaml"),
    developer_release: Path = Path("artifacts/model_releases/model_release_v1.1.json"),
    prequalification: Path = Path("artifacts/calibration/confirmatory_v5_prequalification.json"),
) -> dict[str, Any]:
    cfg = config or ConfirmatoryV5Config()
    cfg.validate()
    module_hash = _hash(Path(__file__))
    dependency_hash = _hash(Path(__file__).with_name("confirmatory_v4.py"))
    config_hash = sha256(json.dumps(asdict(cfg), sort_keys=True).encode()).hexdigest()
    if not developer_release.exists():
        raise FileNotFoundError(
            "corrected Developer Model Release v1.1 must be frozen before v5 design freeze"
        )
    if not prequalification.exists():
        raise FileNotFoundError(
            "release-aligned v5 optimizer prequalification must pass before design freeze"
        )
    prequalification_payload = json.loads(prequalification.read_text(encoding="utf-8"))
    expected_prequalification = {
        "status": "PASS",
        "module_hash": module_hash,
        "config_hash": config_hash,
        "developer_release_hash": _hash(developer_release),
    }
    if any(
        prequalification_payload.get(key) != value
        for key, value in expected_prequalification.items()
    ):
        raise ValueError(
            "v5 optimizer prequalification is stale, failed, or does not match the frozen design"
        )
    immutable_files = {
        "hypothesis_registry_hash": _hash(hypothesis_registry),
        "experiment_registry_hash": _hash(experiment_registry),
        "developer_release_hash": _hash(developer_release),
        "calibration_prequalification_hash": _hash(prequalification),
    }
    if path.exists():
        value: object = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(value, dict):
            raise ValueError("invalid v5 freeze")
        payload = cast(dict[str, Any], value)
        expected = {
            "module_hash": module_hash,
            "v4_dependency_hash": dependency_hash,
            "config_hash": config_hash,
            **immutable_files,
        }
        if any(payload.get(key) != item for key, item in expected.items()):
            raise ValueError("v5 source, configuration, release, or registry changed after freeze")
        return payload
    release = json.loads(developer_release.read_text(encoding="utf-8"))
    if (
        release.get("release_id") != "Developer Model Release v1.1"
        or release.get("objective") != "C04"
        or release.get("optimizer") != "O01"
    ):
        raise ValueError("v5 requires the corrected frozen v1.1 C04/O01 developer release")
    truth_qualification = qualify_confirmatory_engines(cfg.device)
    calibration_qualification = qualify_v4_calibration_engine(cfg.device)
    if any(item.status != "PASS" for item in truth_qualification):
        raise RuntimeError("truth-engine qualification failed")
    if calibration_qualification["status"] != "PASS":
        raise RuntimeError("calibration-pricer qualification failed")
    seeds = build_v5_seed_registry(cfg)
    schedule: list[dict[str, object]] = []
    partitions: tuple[Partition, ...] = (
        "DEV",
        "VALIDATION",
        "LOCKED_TEST",
        "OOD_LOCKED_TEST",
    )
    for partition in partitions:
        for index, seed in enumerate(seeds.for_partition(partition)):
            fault_id = _fault(index)
            schedule.append(
                {
                    "partition": partition,
                    "seed": seed,
                    "observation_seed": seed + seeds.observation_offset,
                    "calibration_seed": seed + 60_000_000,
                    "fault_id": fault_id,
                    "severity": FAULT_SEVERITIES[fault_id],
                    "truth_family": _truth_family(fault_id),
                    "implementation_label": IMPLEMENTATION_LABELS[fault_id],
                    "proxy_fault": fault_id not in {"F00", "F01", "F02", "F03"},
                }
            )
    calibration_design = {
        "developer_release": "Developer Model Release v1.1",
        "objective": "C04",
        "optimizer": "O01 true seeded multistart L-BFGS-B",
        "bounds": PARAMETER_BOUNDS,
        "primary_multistarts": cfg.primary_multistarts,
        "primary_max_iterations": cfg.primary_max_iterations,
        "bootstrap_draws": cfg.bootstrap_draws,
        "bootstrap_multistarts": cfg.bootstrap_multistarts,
        "bootstrap_max_iterations": cfg.bootstrap_max_iterations,
        "qualification": "finite optimizer success; otherwise CALIBRATION_UNQUALIFIED",
        "inputs": "observable bid/mid/ask only",
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
    seed_payload = asdict(seeds)
    payload = {
        "schema_version": "5.0",
        "status": "FROZEN_NOT_EXECUTED",
        "created_utc": datetime.now(UTC).isoformat(),
        "release_id": str(release["release_id"]),
        "module_hash": module_hash,
        "v4_dependency_hash": dependency_hash,
        "config": asdict(cfg),
        "config_hash": config_hash,
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
        "experiment_registry_path": experiment_registry.as_posix(),
        "developer_release_path": developer_release.as_posix(),
        "calibration_prequalification_path": prequalification.as_posix(),
        **immutable_files,
        "truth_engine_qualification": [asdict(item) for item in truth_qualification],
        "calibration_engine_qualification": calibration_qualification,
        "prior_v4_outcomes_used_for_selection": False,
        "locked_tuning_prohibited": True,
    }
    payload["freeze_hash"] = sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
    if _hash(Path(__file__)) != module_hash:
        raise RuntimeError("v5 source changed during freeze")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return payload


def _observable_design(surface: ObservedSurface) -> tuple[SyntheticDesign, NDArray[np.bool_]]:
    source = surface.truth.design
    observed = surface.observed & np.isfinite(surface.mid)
    if np.sum(observed) < 3:
        raise CalibrationUnqualifiedError("fewer than three observable quotes", [])
    zeros = np.zeros(int(np.sum(observed)), dtype=np.float64)
    return (
        replace(
            source,
            scenario_id=source.scenario_id[observed],
            spot=source.spot[observed],
            strike=source.strike[observed],
            maturity=source.maturity[observed],
            rate=source.rate[observed],
            dividend_yield=source.dividend_yield[observed],
            volatility=zeros,
            heston_v0=zeros,
            heston_kappa=zeros,
            heston_theta=zeros,
            heston_sigma_v=zeros,
            heston_rho=zeros,
        ),
        observed,
    )


def _c04_losses(prices: ParameterArray, target: FloatArray, spread: FloatArray) -> FloatArray:
    scaled = (prices - target[None, :]) / np.maximum(spread[None, :], 0.01)
    absolute = np.abs(scaled)
    return np.asarray(
        np.mean(np.where(absolute <= 1.5, 0.5 * scaled**2, 1.5 * (absolute - 0.75)), axis=1),
        dtype=np.float64,
    )


def calibrate_o01_observable_heston(
    surface: ObservedSurface,
    config: ConfirmatoryV5Config,
    *,
    seed: int,
    market_prices: FloatArray | None = None,
    multistarts: int | None = None,
    max_iterations: int | None = None,
) -> tuple[HestonParameters, list[dict[str, object]]]:
    """Run true O01 multistart local C04 calibration and require convergence."""

    started = time.perf_counter()
    design, observed = _observable_design(surface)
    target_all = surface.mid if market_prices is None else np.asarray(market_prices)
    target = np.asarray(target_all[observed], dtype=np.float64)
    spread = np.asarray(surface.spread[observed], dtype=np.float64)
    starts_count = multistarts or config.primary_multistarts
    iteration_budget = max_iterations or config.primary_max_iterations
    bounds = np.asarray(PARAMETER_BOUNDS, dtype=np.float64)
    rng = np.random.default_rng(seed)
    starts = [np.mean(bounds, axis=1)]
    starts.extend(
        rng.uniform(bounds[:, 0], bounds[:, 1], size=(starts_count - 1, 5)).astype(np.float64)
    )
    diagnostics: list[dict[str, object]] = []
    qualified: list[tuple[float, ParameterArray]] = []
    engine = config.engine_config()
    widths = bounds[:, 1] - bounds[:, 0]

    for start_index, start in enumerate(starts):
        evaluations = 0
        local_started = time.perf_counter()

        def value_and_gradient(raw: ParameterArray) -> tuple[float, ParameterArray]:
            nonlocal evaluations
            point = np.asarray(raw, dtype=np.float64)
            steps = config.finite_difference_relative_step * widths
            population = [point]
            denominators: list[float] = []
            for dimension in range(5):
                lower = point.copy()
                upper = point.copy()
                lower[dimension] = max(bounds[dimension, 0], point[dimension] - steps[dimension])
                upper[dimension] = min(bounds[dimension, 1], point[dimension] + steps[dimension])
                population.extend((lower, upper))
                denominators.append(upper[dimension] - lower[dimension])
            prices = _price_matrix(design, np.asarray(population), engine)
            losses = _c04_losses(prices, target, spread)
            evaluations += len(population)
            gradient = np.asarray(
                [
                    (losses[2 * index + 2] - losses[2 * index + 1]) / denominators[index]
                    for index in range(5)
                ],
                dtype=np.float64,
            )
            return float(losses[0]), gradient

        result = minimize(
            value_and_gradient,
            np.asarray(start),
            jac=True,
            method="L-BFGS-B",
            bounds=PARAMETER_BOUNDS,
            options={"maxiter": iteration_budget, "ftol": 1.0e-9, "gtol": 1.0e-5},
        )
        raw = np.asarray(result.x, dtype=np.float64)
        is_qualified = bool(
            result.success
            and raw.shape == (5,)
            and np.all(np.isfinite(raw))
            and np.isfinite(result.fun)
        )
        diagnostic: dict[str, object] = {
            "seed": seed,
            "start_index": start_index,
            "optimizer": "O01",
            "objective": "C04",
            "backend": "torch-cuda-fp64" if config.device == "cuda" else "numpy-cpu-fp64",
            "success": bool(result.success),
            "qualified": is_qualified,
            "message": str(result.message),
            "iterations": int(getattr(result, "nit", 0)),
            "evaluations": evaluations,
            "objective_value": float(result.fun),
            "gradient_infinity_norm": float(
                np.max(np.abs(np.asarray(cast(Any, result.jac), dtype=np.float64)))
            ),
            "runtime_seconds": time.perf_counter() - local_started,
            "parameters": raw.tolist(),
            "observed_quotes": int(np.sum(observed)),
        }
        diagnostics.append(diagnostic)
        if is_qualified:
            qualified.append((float(result.fun), raw))
    if not qualified:
        raise CalibrationUnqualifiedError(
            f"no converged C04/O01 start among {starts_count} attempts", diagnostics
        )
    best_loss, best = min(qualified, key=lambda item: item[0])
    diagnostics.append(
        {
            "seed": seed,
            "start_index": -1,
            "optimizer": "O01",
            "objective": "C04",
            "backend": "selection",
            "success": True,
            "qualified": True,
            "message": "best converged multistart solution",
            "iterations": 0,
            "evaluations": 0,
            "objective_value": best_loss,
            "gradient_infinity_norm": float("nan"),
            "runtime_seconds": time.perf_counter() - started,
            "parameters": best.tolist(),
            "observed_quotes": int(np.sum(observed)),
        }
    )
    params = HestonParameters(*map(float, best))
    params.validate()
    return params, diagnostics


def build_qualified_case(
    surface: ObservedSurface,
    config: ConfirmatoryV5Config,
    *,
    calibration_seed: int,
) -> tuple[ValidationCase, list[dict[str, object]], dict[str, object]]:
    params, diagnostics = calibrate_o01_observable_heston(surface, config, seed=calibration_seed)
    design = surface.truth.design
    engine = config.engine_config()
    price, delta, gamma, vega = _developer_values(design, params, engine)
    ql = QuantLibHestonParameters(params.v0, params.kappa, params.theta, params.sigma_v, params.rho)
    oracle = np.asarray(
        [
            quantlib_heston_price(
                float(design.spot[index]),
                float(design.strike[index]),
                float(design.maturity[index]),
                ql,
                float(design.rate[index]),
                float(design.dividend_yield[index]),
            ).price
            for index in range(len(design))
        ]
    )
    rng = np.random.default_rng(calibration_seed + 1_000_000)
    observed = surface.observed & np.isfinite(surface.mid)
    bootstrap = np.empty((len(design), config.bootstrap_draws), dtype=np.float64)
    for draw in range(config.bootstrap_draws):
        sampled = surface.mid.copy()
        sampled[observed] = surface.bid[observed] + rng.random(np.sum(observed)) * (
            surface.ask[observed] - surface.bid[observed]
        )
        try:
            draw_params, draw_diagnostics = calibrate_o01_observable_heston(
                surface,
                config,
                seed=calibration_seed + 10_000 + draw,
                market_prices=sampled,
                multistarts=config.bootstrap_multistarts,
                max_iterations=config.bootstrap_max_iterations,
            )
        except CalibrationUnqualifiedError as exc:
            for item in exc.diagnostics:
                item["draw"] = draw
            diagnostics.extend(exc.diagnostics)
            raise CalibrationUnqualifiedError(
                f"bootstrap draw {draw} has no converged C04/O01 fit", diagnostics
            ) from exc
        for item in draw_diagnostics:
            item["draw"] = draw
        diagnostics.extend(draw_diagnostics)
        bootstrap[:, draw] = _developer_values(design, draw_params, engine)[0]
    for item in diagnostics:
        item.setdefault("draw", -1)
    challenger_values, ssvi = _observable_challengers(surface)
    challenger_price, challenger_delta, challenger_gamma, challenger_vega = challenger_values
    clean = clean_validation_case(surface, bootstrap_draws=config.bootstrap_draws)
    case = replace(
        clean,
        developer_price=price,
        developer_delta=delta,
        developer_gamma=gamma,
        developer_vega=vega,
        numerical_prices=np.column_stack((price, oracle)),
        calibration_prices=bootstrap,
        challenger_prices=challenger_price,
        challenger_delta=challenger_delta,
        challenger_gamma=challenger_gamma,
        challenger_vega=challenger_vega,
        proxy_fault=False,
    )
    return case, diagnostics, ssvi


def _make_case(
    partition: Partition, index: int, seed: int, config: ConfirmatoryV5Config
) -> tuple[EvidenceRecord, dict[str, object], list[dict[str, object]]]:
    fault_id = _fault(index)
    family = _truth_family(fault_id)
    design = _surface_design(seed, config.contracts_per_case, ood=partition == "OOD_LOCKED_TEST")
    truth = generate_confirmatory_truth(
        design, cast(Any, family), config.engine_config().truth_config(), seed=seed
    )
    observation_seed = seed + 50_000_000
    surface = observe_truth(
        truth, config.engine_config().truth_config().observation, seed=observation_seed
    )
    case_id = f"V5-{partition}-{fault_id}-{seed}"
    try:
        case, diagnostics, ssvi = build_qualified_case(
            surface, config, calibration_seed=seed + 60_000_000
        )
    except CalibrationUnqualifiedError as exc:
        for item in exc.diagnostics:
            item.update({"case_id": case_id, "partition": partition, "fault_id": fault_id})
        raise
    case = replace(case, fault_id=fault_id)
    if fault_id not in {"F00", "F01", "F02", "F03"}:
        case = replace(
            inject_fault(case, fault_id, severity=FAULT_SEVERITIES[fault_id]), proxy_fault=True
        )
    record = make_record(
        case_id, partition, case, materiality_threshold=config.materiality_threshold
    )
    for item in diagnostics:
        item.update({"case_id": case_id, "partition": partition, "fault_id": fault_id})
    provenance = {
        "case_id": case_id,
        "partition": partition,
        "fault_id": fault_id,
        "severity": FAULT_SEVERITIES[fault_id],
        "truth_family": family,
        "implementation_label": IMPLEMENTATION_LABELS[fault_id],
        "proxy_fault": case.proxy_fault,
        "truth_seed": seed,
        "observation_seed": observation_seed,
        "calibration_seed": seed + 60_000_000,
        "calibration_inputs": "observable_quotes_only",
        "calibration_qualified": True,
        "ssvi_qualification": json.dumps(ssvi, sort_keys=True),
        "ssvi_status": ssvi["status"],
        "ssvi_static_arbitrage_valid": ssvi["static_arbitrage_valid"],
        "ssvi_fallback_to_flat_bs": ssvi["fallback_to_flat_bs"],
        "backend": truth.backend,
        "precision": truth.precision,
        "generated_utc": datetime.now(UTC).isoformat(),
    }
    return record, provenance, diagnostics


def prequalify_confirmatory_v5_calibration(
    profile: str = "research",
    device: str = "cuda",
    *,
    root: Path = Path("."),
    resume: bool = True,
) -> dict[str, Any]:
    """Qualify the exact v5 optimizer on distinct, pre-lock tuning seeds.

    These cases are never included in DEV, VALIDATION, LOCKED_TEST, or OOD
    inference.  The study exercises correctly specified Heston plus the two
    primary model-form alternatives and difficult data/combined faults before
    the design is made immutable.
    """

    config = ConfirmatoryV5Config(profile=profile, device=device)
    config.validate()
    release = root / "artifacts/model_releases/model_release_v1.1.json"
    if not release.is_file():
        raise FileNotFoundError("Developer Model Release v1.1 is required for prequalification")
    module_hash = _hash(Path(__file__))
    config_hash = sha256(json.dumps(asdict(config), sort_keys=True).encode()).hexdigest()
    release_hash = _hash(release)
    status_path = root / "artifacts/calibration/confirmatory_v5_prequalification.json"
    expected_identity = {
        "module_hash": module_hash,
        "config_hash": config_hash,
        "developer_release_hash": release_hash,
    }
    if resume and status_path.is_file():
        existing = cast(dict[str, Any], json.loads(status_path.read_text(encoding="utf-8")))
        if all(existing.get(key) == value for key, value in expected_identity.items()):
            if existing.get("status") != "PASS":
                raise RuntimeError("existing v5 optimizer prequalification did not pass")
            return existing

    tuning_cases = (
        (0, 102_000_001),
        (1, 102_000_002),
        (2, 102_000_003),
        (9, 102_000_010),
        (16, 102_000_017),
    )
    official_seeds = build_v5_seed_registry(config)
    official = set(
        official_seeds.dev + official_seeds.validation + official_seeds.locked + official_seeds.ood
    )
    if any(seed in official for _, seed in tuning_cases):
        raise RuntimeError("prequalification seed overlaps a frozen study partition")
    started = time.perf_counter()
    rows: list[dict[str, object]] = []
    diagnostics: list[dict[str, object]] = []
    for fault_index, seed in tuning_cases:
        fault_id = _fault(fault_index)
        case_started = time.perf_counter()
        try:
            record, _, case_diagnostics = _make_case("DEV", fault_index, seed, config)
            diagnostics.extend(case_diagnostics)
            rows.append(
                {
                    "seed": seed,
                    "fault_id": fault_id,
                    "truth_family": _truth_family(fault_id),
                    "status": "PASS",
                    "selected_fit_qualified": True,
                    "runtime_seconds": time.perf_counter() - case_started,
                    "normalized_true_error": record.normalized_true_error,
                }
            )
        except CalibrationUnqualifiedError as exc:
            diagnostics.extend(exc.diagnostics)
            rows.append(
                {
                    "seed": seed,
                    "fault_id": fault_id,
                    "truth_family": _truth_family(fault_id),
                    "status": "FAIL",
                    "selected_fit_qualified": False,
                    "runtime_seconds": time.perf_counter() - case_started,
                    "reason": str(exc),
                }
            )
    aggregate = root / "results/aggregated/confirmatory_v5_prequalification.csv"
    diagnostic_path = root / "results/raw/confirmatory_v5_prequalification_diagnostics.parquet"
    status_path.parent.mkdir(parents=True, exist_ok=True)
    aggregate.parent.mkdir(parents=True, exist_ok=True)
    diagnostic_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(aggregate, index=False)
    pd.DataFrame(diagnostics).to_parquet(diagnostic_path, index=False)
    passed = bool(rows) and all(row["status"] == "PASS" for row in rows)
    payload: dict[str, Any] = {
        "schema_version": "1.0",
        "status": "PASS" if passed else "FAIL",
        "study_role": "PRE_LOCK_TUNING_QUALIFICATION_NOT_INFERENTIAL",
        **expected_identity,
        "tuning_seed_region": "102m",
        "official_seed_regions": ["103m", "104m", "105m", "106m"],
        "seed_overlap": False,
        "cases_attempted": len(rows),
        "cases_passed": sum(row["status"] == "PASS" for row in rows),
        "faults_attempted": [str(row["fault_id"]) for row in rows],
        "runtime_seconds": time.perf_counter() - started,
        "summary_path": aggregate.relative_to(root).as_posix(),
        "diagnostics_path": diagnostic_path.relative_to(root).as_posix(),
    }
    status_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if not passed:
        raise RuntimeError("v5 optimizer prequalification failed; design remains unfrozen")
    return payload


def run_confirmatory_v5_study(
    profile: str = "research", device: str = "cuda", resume: bool = True
) -> dict[str, Any]:
    output = Path("results/confirmatory/synthetic_v4")
    status_path = output / "confirmatory_metrics.json"
    if resume and status_path.exists():
        return cast(dict[str, Any], json.loads(status_path.read_text(encoding="utf-8")))
    config = ConfirmatoryV5Config(profile=profile, device=device)
    frozen = freeze_confirmatory_v5(config=config)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("v5 output namespace is non-empty")
    output.mkdir(parents=True, exist_ok=True)
    seeds = build_v5_seed_registry(config)
    records: list[EvidenceRecord] = []
    provenance: list[dict[str, object]] = []
    diagnostics: list[dict[str, object]] = []

    def generate(partition: Partition) -> bool:
        for index, seed in enumerate(seeds.for_partition(partition)):
            try:
                record, source, case_diagnostics = _make_case(partition, index, seed, config)
            except CalibrationUnqualifiedError as exc:
                diagnostics.extend(exc.diagnostics)
                pd.DataFrame(diagnostics).to_parquet(
                    output / "calibration_diagnostics.parquet", index=False
                )
                failure = {
                    "status": "CALIBRATION_UNQUALIFIED",
                    "partition": partition,
                    "seed": seed,
                    "fault_id": _fault(index),
                    "reason": str(exc),
                    "freeze_hash": frozen["freeze_hash"],
                    "developer_release_id": frozen["calibration_design"]["developer_release"],
                    "developer_release_hash": frozen["developer_release_hash"],
                    "locked_evaluation_complete": False,
                }
                status_path.write_text(json.dumps(failure, indent=2, sort_keys=True) + "\n")
                return False
            source["design_freeze_hash"] = frozen["freeze_hash"]
            records.append(record)
            provenance.append(source)
            diagnostics.extend(case_diagnostics)
        return True

    if not generate("DEV") or not generate("VALIDATION"):
        return cast(dict[str, Any], json.loads(status_path.read_text(encoding="utf-8")))
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
    threshold_payload = {
        "status": "FROZEN_BEFORE_LOCKED_GENERATION",
        "frozen_utc": datetime.now(UTC).isoformat(),
        "design_freeze_hash": frozen["freeze_hash"],
        "thresholds": thresholds.to_dict(),
        "validation_diagnostics": validation_metrics,
    }
    with (output / "dve_thresholds.freeze.json").open("x", encoding="utf-8") as handle:
        json.dump(threshold_payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    if not generate("LOCKED_TEST") or not generate("OOD_LOCKED_TEST"):
        return cast(dict[str, Any], json.loads(status_path.read_text(encoding="utf-8")))
    locked = tuple(item for item in records if item.partition in {"LOCKED_TEST", "OOD_LOCKED_TEST"})
    baselines = evaluate_locked(locked, thresholds)
    ablations = evaluate_confirmatory_ablations(dev, locked, thresholds)
    result = ConfirmatoryRunResult(
        run_id=f"SYN-CONFIRMATORY-V5-{str(frozen['freeze_hash'])[:12]}",
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
        frozen["truth_engine_qualification"],
    )
    diagnostic_frame = pd.DataFrame(diagnostics)
    diagnostic_frame.to_parquet(output / "calibration_diagnostics.parquet", index=False)
    qualification = {
        "status": "PASS",
        "cases": len(records),
        "all_selected_fits_qualified": True,
        "optimizer_start_attempts": int(np.sum(diagnostic_frame["start_index"] >= 0)),
        "converged_start_attempts": int(
            np.sum((diagnostic_frame["start_index"] >= 0) & diagnostic_frame["qualified"])
        ),
    }
    (output / "calibration_qualification.json").write_text(
        json.dumps(qualification, indent=2, sort_keys=True) + "\n"
    )
    status = {
        "status": "CONFIRMATORY_LOCKED_COMPLETE",
        "run_id": result.run_id,
        "freeze_hash": result.freeze_hash,
        "developer_release_id": frozen["calibration_design"]["developer_release"],
        "developer_release_hash": frozen["developer_release_hash"],
        "locked_cases": len(locked),
        "calibration_qualification": "PASS",
        "faults_executed": list(FAULT_IDS),
        "baselines": {key: asdict(value) for key, value in baselines.items()},
        "ablations": {key: asdict(value) for key, value in ablations.items()},
    }
    status_path.write_text(json.dumps(status, indent=2, sort_keys=True) + "\n")
    return status
