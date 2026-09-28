"""Per-calibration uncertainty diagnostics for the additive v1.1 analysis.

The released parameters are never changed.  Price Jacobians are evaluated in
FP64 on CUDA.  Bid/ask quote bootstrap draws are propagated through the local
weighted calibration map; multiple-start dispersion is measured with bounded
CPU local optimizations because the released surfaces contain only 12-30 quotes.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd
import torch
from numpy.typing import NDArray

from derivguard.development.calibration import (
    PARAMETER_NAMES,
    calibrate_heston,
    parameter_vector,
)
from derivguard.development.heston_cf import HestonParameters, heston_call_price_torch
from derivguard.empirical import (
    EmpiricalConfig,
    PreparedSurface,
    _forward_lookup,
    chronological_split,
    prepare_surface,
    representative_files,
)

FloatArray = NDArray[np.float64]
BOUNDS = np.asarray(
    ((0.0025, 0.5), (0.05, 10.0), (0.0025, 0.5), (0.05, 2.5), (-0.999, 0.999)),
    dtype=np.float64,
)


def _hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _payload(path: Path) -> dict[str, Any]:
    return cast(dict[str, Any], json.loads(path.read_text(encoding="utf-8")))


def _parameters(payload: dict[str, Any]) -> HestonParameters:
    raw = cast(dict[str, Any], cast(dict[str, Any], payload["calibration"])["parameters"])
    return HestonParameters(*[float(raw[name]) for name in PARAMETER_NAMES])


def _gpu_prices(surface: PreparedSurface, matrix: FloatArray, nodes: int) -> FloatArray:
    if not torch.cuda.is_available():
        raise RuntimeError("per-calibration v1.1 diagnostics require CUDA")
    parameters = torch.as_tensor(matrix, dtype=torch.float64, device="cuda")
    output = torch.empty(
        (matrix.shape[0], surface.data.market_price.size), dtype=torch.float64, device="cuda"
    )
    groups: dict[tuple[float, float, float, float, str], list[int]] = {}
    for index in range(surface.data.market_price.size):
        key = (
            float(surface.data.spot[index]),
            float(surface.data.maturity[index]),
            float(surface.data.rate[index]),
            float(surface.data.dividend_yield[index]),
            str(surface.data.option_type[index]),
        )
        groups.setdefault(key, []).append(index)
    for (spot, maturity, rate, dividend, kind), indices in groups.items():
        idx = np.asarray(indices, dtype=np.int64)
        strikes = torch.as_tensor(surface.data.strike[idx], dtype=torch.float64, device="cuda")
        prices = heston_call_price_torch(
            spot,
            strikes,
            maturity,
            parameters,
            rate,
            dividend,
            nodes=nodes,
        )
        if kind == "put":
            parity = spot * np.exp(-dividend * maturity) - strikes * np.exp(-rate * maturity)
            prices = prices - parity[None, :]
        output[:, torch.as_tensor(idx, dtype=torch.long, device="cuda")] = prices
    return np.asarray(output.cpu(), dtype=np.float64)


def _jacobian(
    surface: PreparedSurface, parameters: HestonParameters, nodes: int
) -> tuple[FloatArray, FloatArray]:
    center = parameter_vector(parameters)
    span = BOUNDS[:, 1] - BOUNDS[:, 0]
    step = np.maximum(1.0e-4 * span, 1.0e-4 * np.maximum(np.abs(center), 1.0))
    matrix = [center]
    effective: list[tuple[float, float]] = []
    for index in range(5):
        lower = center.copy()
        upper = center.copy()
        lower[index] = max(BOUNDS[index, 0], center[index] - step[index])
        upper[index] = min(BOUNDS[index, 1], center[index] + step[index])
        matrix.extend((lower, upper))
        effective.append((lower[index], upper[index]))
    prices = _gpu_prices(surface, np.asarray(matrix), nodes)
    jacobian = np.empty((surface.data.market_price.size, 5), dtype=np.float64)
    for index, (low_value, high_value) in enumerate(effective):
        jacobian[:, index] = (prices[2 * index + 2] - prices[2 * index + 1]) / (
            high_value - low_value
        )
    return prices[0], jacobian


def _surface_iterator(root: Path) -> list[tuple[Any, PreparedSurface, dict[str, Any], int]]:
    empirical = EmpiricalConfig()
    lightweight = replace(empirical, max_expiries=2, quotes_per_expiry=6)
    files = sorted((root / "data/processed/historical_spx_sample").glob("*_options.parquet"))
    partitions = chronological_split(files)
    forwards = _forward_lookup(root)
    representative = {
        (split, path.stem.removesuffix("_options"), settlement)
        for split, split_files in partitions.items()
        for path in representative_files(split_files, empirical.representative_dates_per_split)
        for settlement in ("PM", "AM")
    }
    by_date = {path.stem.removesuffix("_options"): path for path in files}
    comparison = pd.read_parquet(root / "results/raw/real_model_comparison.parquet")
    heston = comparison[
        (comparison["model_id"] == "M10_HESTON")
        & (comparison["experiment"] == "cross_sectional")
        & comparison["converged"].fillna(False)
        & comparison["checkpoint"].notna()
    ].sort_values(["quote_date", "settlement_class"])
    result: list[tuple[Any, PreparedSurface, dict[str, Any], int]] = []
    for row in heston.itertuples(index=False):
        key = (str(row.split), str(row.quote_date), str(row.settlement_class))
        stage = empirical if key in representative else lightweight
        surface = prepare_surface(
            by_date[str(row.quote_date)],
            str(row.split),
            forwards,
            stage,
            settlement_class=str(row.settlement_class),
        )
        checkpoint = _payload(root / str(row.checkpoint))
        result.append((row, surface, checkpoint, stage.quadrature_nodes))
    return result


def run(root: Path = Path("."), *, bootstrap_draws: int = 400) -> dict[str, Any]:
    output = root / "results/v1_1/calibration_uncertainty_by_calibration.parquet"
    correlations_path = root / "results/v1_1/calibration_parameter_correlations.parquet"
    profiles_path = root / "results/v1_1/calibration_profile_widths.parquet"
    multistart_path = root / "results/v1_1/calibration_multistart_dispersion.parquet"
    manifest_path = root / "artifacts/v1_1/calibration_uncertainty_manifest.json"
    source_paths = [
        root / "results/raw/real_model_comparison.parquet",
        root / "src/derivguard/calibration_uncertainty_v1_1.py",
    ]
    identity = {path.relative_to(root).as_posix(): _hash(path) for path in source_paths}
    if manifest_path.exists() and output.exists():
        manifest = _payload(manifest_path)
        if (
            manifest.get("source_hashes") == identity
            and manifest.get("bootstrap_draws") == bootstrap_draws
        ):
            return manifest

    uncertainty: list[dict[str, object]] = []
    correlations: list[dict[str, object]] = []
    profiles: list[dict[str, object]] = []
    multistarts: list[dict[str, object]] = []
    surfaces = _surface_iterator(root)
    for surface_index, (row, surface, checkpoint, nodes) in enumerate(surfaces):
        parameters = _parameters(checkpoint)
        center = parameter_vector(parameters)
        model_price, jacobian = _jacobian(surface, parameters, nodes)
        spread = np.maximum(surface.data.ask - surface.data.bid, 0.01)
        scaled_jacobian = jacobian / spread[:, None]
        residual = (model_price - surface.data.market_price) / spread
        huber_weight = np.where(np.abs(residual) <= 1.5, 1.0, 1.5 / np.abs(residual))
        weighted = scaled_jacobian * np.sqrt(huber_weight)[:, None]
        normal = weighted.T @ weighted
        regularizer = max(float(np.trace(normal)) * 1.0e-10 / 5.0, 1.0e-12)
        inverse = np.linalg.pinv(normal + regularizer * np.eye(5), rcond=1.0e-10)
        rng = np.random.default_rng(20261101 + surface_index * 1009)
        quote_draws = rng.uniform(
            surface.data.bid, surface.data.ask, size=(bootstrap_draws, len(spread))
        )
        perturbation = (quote_draws - surface.data.market_price[None, :]) / spread[None, :]
        map_matrix = inverse @ weighted.T
        draws = center[None, :] + (map_matrix @ (perturbation * np.sqrt(huber_weight)).T).T
        draws = np.clip(draws, BOUNDS[:, 0], BOUNDS[:, 1])
        covariance = np.asarray(np.cov(draws, rowvar=False), dtype=np.float64)
        correlation = np.asarray(np.corrcoef(draws, rowvar=False), dtype=np.float64)
        for parameter_index, parameter in enumerate(PARAMETER_NAMES):
            values = draws[:, parameter_index]
            uncertainty.append(
                {
                    "quote_date": row.quote_date,
                    "split": row.split,
                    "settlement_class": row.settlement_class,
                    "parameter": parameter,
                    "released_value": center[parameter_index],
                    "bootstrap_mean": float(values.mean()),
                    "bootstrap_std": float(values.std(ddof=1)),
                    "bootstrap_q025": float(np.quantile(values, 0.025)),
                    "bootstrap_q975": float(np.quantile(values, 0.975)),
                    "draws": bootstrap_draws,
                    "method": "CUDA_FP64_LINEARIZED_BID_ASK_QUOTE_BOOTSTRAP",
                }
            )
        for left in range(5):
            for right in range(left + 1, 5):
                correlations.append(
                    {
                        "quote_date": row.quote_date,
                        "split": row.split,
                        "settlement_class": row.settlement_class,
                        "parameter_1": PARAMETER_NAMES[left],
                        "parameter_2": PARAMETER_NAMES[right],
                        "correlation": float(correlation[left, right]),
                        "covariance": float(covariance[left, right]),
                        "draws": bootstrap_draws,
                    }
                )
        base_objective = float(checkpoint["calibration"]["objective_value"])
        for parameter_index, parameter in enumerate(PARAMETER_NAMES):
            nuisance = [index for index in range(5) if index != parameter_index]
            conditional_curvature = normal[parameter_index, parameter_index]
            if nuisance:
                conditional_curvature -= (
                    normal[parameter_index, nuisance]
                    @ np.linalg.pinv(normal[np.ix_(nuisance, nuisance)], rcond=1.0e-10)
                    @ normal[nuisance, parameter_index]
                )
            target_increase = max(abs(base_objective) * 0.10, 1.0e-8)
            half_width = np.sqrt(2.0 * target_increase / max(conditional_curvature, 1.0e-12))
            low = max(BOUNDS[parameter_index, 0], center[parameter_index] - half_width)
            high = min(BOUNDS[parameter_index, 1], center[parameter_index] + half_width)
            profiles.append(
                {
                    "quote_date": row.quote_date,
                    "split": row.split,
                    "settlement_class": row.settlement_class,
                    "parameter": parameter,
                    "profile_width": float(high - low),
                    "profile_low": float(low),
                    "profile_high": float(high),
                    "relative_objective_increase": 0.10,
                    "method": "LOCAL_QUADRATIC_NUISANCE_PROFILE",
                }
            )
        span = BOUNDS[:, 1] - BOUNDS[:, 0]
        starts = [center]
        start_rng = np.random.default_rng(20261201 + surface_index * 1013)
        for _ in range(3):
            starts.append(
                np.clip(
                    center + start_rng.normal(0.0, 0.08, size=5) * span, BOUNDS[:, 0], BOUNDS[:, 1]
                )
            )
        solutions: list[FloatArray] = []
        objectives: list[float] = []
        successes = 0
        for start_index, start in enumerate(starts):
            result = calibrate_heston(
                surface.data,
                objective="C04",
                optimizer="local",
                initial=HestonParameters(*map(float, start)),
                feller_treatment="unconstrained",
                seed=20261301 + surface_index * 17 + start_index,
                max_iterations=80,
                nodes=max(32, min(nodes, 48)),
            )
            solutions.append(parameter_vector(result.parameters))
            objectives.append(result.objective_value)
            successes += int(result.success)
        solution_matrix = np.vstack(solutions)
        for parameter_index, parameter in enumerate(PARAMETER_NAMES):
            multistarts.append(
                {
                    "quote_date": row.quote_date,
                    "split": row.split,
                    "settlement_class": row.settlement_class,
                    "parameter": parameter,
                    "dispersion_std": float(solution_matrix[:, parameter_index].std(ddof=1)),
                    "range": float(np.ptp(solution_matrix[:, parameter_index])),
                    "starts": len(starts),
                    "successful_starts": successes,
                    "objective_min": float(np.min(objectives)),
                    "objective_max": float(np.max(objectives)),
                    "method": "BOUNDED_LOCAL_MULTISTART_DIAGNOSTIC",
                }
            )
    pd.DataFrame(uncertainty).to_parquet(output, index=False)
    pd.DataFrame(correlations).to_parquet(correlations_path, index=False)
    pd.DataFrame(profiles).to_parquet(profiles_path, index=False)
    pd.DataFrame(multistarts).to_parquet(multistart_path, index=False)
    manifest = {
        "status": "COMPLETE",
        "created_utc": datetime.now(UTC).isoformat(),
        "source_hashes": identity,
        "bootstrap_draws": bootstrap_draws,
        "calibrations": len(surfaces),
        "parameters_per_calibration": 5,
        "cuda": torch.cuda.get_device_name(0),
        "methods": {
            "bootstrap": "CUDA_FP64_LINEARIZED_BID_ASK_QUOTE_BOOTSTRAP",
            "profiles": "LOCAL_QUADRATIC_NUISANCE_PROFILE",
            "multistart": "BOUNDED_LOCAL_MULTISTART_DIAGNOSTIC",
        },
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest
