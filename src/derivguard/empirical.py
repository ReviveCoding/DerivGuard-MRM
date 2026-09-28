"""Bounded, reproducible real-data calibration and materiality program.

The public HistoricalData.net sample contains unsynchronised end-of-day quotes.
This module therefore performs cross-sectional and holdout experiments only; it
does not produce trading P&L or pretend the observations are an NBBO snapshot.

The complete universe is retained on disk.  Computationally expensive Heston
selection uses deterministic representative dates and stratified quote samples,
with the sampling design and all failures written to the result artifacts.
"""

from __future__ import annotations

import hashlib
import json
import time
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal, cast

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from scipy.optimize import least_squares

from derivguard.challengers.bates import BatesParameters, bates_price
from derivguard.challengers.local_vol import local_vol_mc_price
from derivguard.data.covariates import acquire_fred_covariates
from derivguard.development.calibration import (
    PARAMETER_NAMES,
    CalibrationData,
    CalibrationResult,
    calibrate_heston,
    heston_objective,
    identifiability_diagnostics,
    parameter_vector,
    price_calibration_data,
    profile_objective,
    rolling_stability,
)
from derivguard.development.heston_cf import (
    HestonParameters,
    heston_call_price_torch,
    heston_price_fixed_quad,
)
from derivguard.market.black_scholes import greeks, option_price
from derivguard.market.svi import (
    SSVIParameters,
    SSVISurface,
    SVIParameters,
    check_svi_slice,
    ssvi_total_variance,
    svi_total_variance,
)
from derivguard.materiality.portfolio import (
    InstrumentMeasures,
    OptionLeg,
    compare_materiality,
    evaluate_models,
    standardized_portfolios,
)

FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class EmpiricalConfig:
    """Controls declared before execution; defaults are the research profile."""

    seed: int = 20260927
    max_expiries: int = 3
    quotes_per_expiry: int = 10
    representative_dates_per_split: int = 3
    quadrature_nodes: int = 48
    # The earlier 75-iteration research pass produced iteration-limit exits.
    # The corrected release-aligned run uses the same budget as confirmatory
    # v5 so convergence, rather than an artificially short budget, determines
    # whether a date is qualified.
    local_iterations: int = 160
    de_iterations: int = 20
    de_population: int = 5
    multistarts: int = 5
    bootstrap_draws: int = 8
    relative_spread_limit: float = 0.25
    maximum_abs_log_moneyness: float = 0.35


@dataclass(frozen=True)
class PreparedSurface:
    quote_date: str
    split: str
    settlement_class: str
    data: CalibrationData
    forward: FloatArray
    log_moneyness: FloatArray
    expiration: NDArray[np.str_]
    source_row_id: NDArray[np.str_]


def released_optimizer_name(optimizer_id: str) -> Literal["local", "multistart", "de", "de_local"]:
    """Translate the frozen governance ID to the implemented optimizer."""

    mapping = {
        "O00": "local",
        "O01": "multistart",
        "O02": "de",
        "O03": "de_local",
    }
    if optimizer_id not in mapping:
        raise ValueError(f"optimizer {optimizer_id} is not a CPU calibration methodology")
    return cast(Literal["local", "multistart", "de", "de_local"], mapping[optimizer_id])


def _calibration_payload(result: CalibrationResult) -> dict[str, object]:
    return {
        "parameters": dict(
            zip(PARAMETER_NAMES, map(float, parameter_vector(result.parameters)), strict=True)
        ),
        "objective_value": result.objective_value,
        "objective_name": result.objective_name,
        "optimizer": result.optimizer,
        "feller_treatment": result.feller_treatment,
        "success": result.success,
        "message": result.message,
        "evaluations": result.evaluations,
        "iterations": result.iterations,
        "start_index": result.start_index,
    }


def _calibration_from_payload(payload: dict[str, Any]) -> CalibrationResult:
    parameters = payload["parameters"]
    if not isinstance(parameters, dict):
        raise ValueError("checkpoint parameters must be an object")
    return CalibrationResult(
        parameters=HestonParameters(
            *[float(parameters[name]) for name in ("v0", "kappa", "theta", "sigma_v", "rho")]
        ),
        objective_value=float(payload["objective_value"]),
        objective_name=cast(Any, payload["objective_name"]),
        optimizer=cast(Any, payload["optimizer"]),
        feller_treatment=cast(Any, payload["feller_treatment"]),
        success=bool(payload["success"]),
        message=str(payload["message"]),
        evaluations=int(payload["evaluations"]),
        iterations=int(payload["iterations"]),
        start_index=int(payload["start_index"]),
    )


def real_calibration_checkpoint_key(
    surface: PreparedSurface,
    config: EmpiricalConfig,
    *,
    objective: str,
    optimizer_id: str,
    feller: str,
    initial: HestonParameters,
    seed: int,
) -> str:
    """Content/config key for a frozen-method date-level calibration."""

    identity = {
        "quote_date": surface.quote_date,
        "split": surface.split,
        "settlement_class": surface.settlement_class,
        "source_rows": sorted(map(str, surface.source_row_id)),
        "objective": objective,
        "optimizer_id": optimizer_id,
        "feller": feller,
        "initial": parameter_vector(initial).tolist(),
        "seed": seed,
        "nodes": config.quadrature_nodes,
        "local_iterations": config.local_iterations,
        "de_iterations": config.de_iterations,
        "de_population": config.de_population,
        "multistarts": config.multistarts,
        "orchestration_code_hash": _hash_file(Path(__file__)),
        "calibration_code_hash": _hash_file(Path(__file__).parent / "development/calibration.py"),
        "heston_code_hash": _hash_file(Path(__file__).parent / "development/heston_cf.py"),
    }
    encoded = json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _calibrate_released_surface(
    surface: PreparedSurface,
    config: EmpiricalConfig,
    *,
    objective: str,
    optimizer_id: str,
    feller: str,
    initial: HestonParameters,
    seed: int,
    device: str,
    checkpoint_directory: Path,
    resume: bool,
) -> tuple[CalibrationResult, bool, Path, float]:
    """Run or resume one calibration using the exact frozen methodology."""

    key = real_calibration_checkpoint_key(
        surface,
        config,
        objective=objective,
        optimizer_id=optimizer_id,
        feller=feller,
        initial=initial,
        seed=seed,
    )
    checkpoint = checkpoint_directory / f"{key}.json"
    if resume and checkpoint.exists():
        payload = cast(dict[str, Any], json.loads(checkpoint.read_text(encoding="utf-8")))
        if payload.get("status") == "COMPLETE" and payload.get("checkpoint_key") == key:
            result_payload = payload.get("calibration")
            if isinstance(result_payload, dict):
                return (
                    _calibration_from_payload(result_payload),
                    True,
                    checkpoint,
                    float(payload.get("runtime_seconds", np.nan)),
                )
    started = _timestamp()
    started_clock = time.perf_counter()
    if optimizer_id == "O04":
        if device != "cuda":
            raise RuntimeError("frozen O04 methodology requires CUDA")
        result = _gpu_population_local(
            surface,
            config,
            device,
            objective=cast(Any, objective),
            feller=cast(Any, feller),
        )
    else:
        result = _calibrate(
            surface,
            config,
            objective=cast(Any, objective),
            optimizer=released_optimizer_name(optimizer_id),
            feller=cast(Any, feller),
            initial=initial,
            seed_offset=seed - config.seed,
        )
    checkpoint_directory.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": "1.0",
        "status": "COMPLETE",
        "checkpoint_key": key,
        "started_at": started,
        "ended_at": _timestamp(),
        "runtime_seconds": time.perf_counter() - started_clock,
        "quote_date": surface.quote_date,
        "split": surface.split,
        "settlement_class": surface.settlement_class,
        "objective": objective,
        "optimizer_id": optimizer_id,
        "feller_treatment": feller,
        "seed": seed,
        "calibration": _calibration_payload(result),
    }
    temporary = checkpoint.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    temporary.replace(checkpoint)
    return result, False, checkpoint, float(payload["runtime_seconds"])


def _timestamp() -> str:
    return datetime.now(UTC).isoformat()


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _config_hash(config: EmpiricalConfig) -> str:
    raw = json.dumps(asdict(config), sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(raw).hexdigest()


def _stable_seed(base: int, *parts: str) -> int:
    digest = hashlib.sha256("|".join(parts).encode()).digest()
    return base + int.from_bytes(digest[:4], "big") % 1_000_000


def chronological_split(files: Sequence[Path]) -> dict[str, list[Path]]:
    """Return complete-date 60/20/20 partitions without row randomisation."""

    ordered = sorted(files)
    if len(ordered) < 5:
        raise ValueError("at least five complete market dates are required")
    dev_end = int(np.floor(0.60 * len(ordered)))
    validation_end = int(np.floor(0.80 * len(ordered)))
    return {
        "DEV": list(ordered[:dev_end]),
        "VALIDATION": list(ordered[dev_end:validation_end]),
        "LOCKED_TEST": list(ordered[validation_end:]),
    }


def representative_files(files: Sequence[Path], count: int) -> list[Path]:
    """Select deterministic quantile dates, including the split endpoints."""

    if not files:
        return []
    indices = np.unique(np.rint(np.linspace(0, len(files) - 1, min(count, len(files)))).astype(int))
    return [files[int(index)] for index in indices]


def _forward_lookup(root: Path) -> pd.DataFrame:
    frame = pd.read_parquet(root / "results/raw/forward_estimator_comparison.parquet")
    selected = frame[
        (frame["subset_id"] == "D04_MATCHED_LIQUID_NARROW")
        & (frame["method"] == "F3")
        & (frame["status"] == "SUCCESS")
    ].copy()
    selected["quote_date"] = selected["quote_date"].astype(str)
    selected["expiration"] = selected["expiration"].astype(str)
    return selected.drop_duplicates(["quote_date", "expiration", "settlement_class"])


def _universe_diagnostics(files: Sequence[Path]) -> pd.DataFrame:
    """Measure liquidity, settlement, near-expiry, and timing coverage on all dates."""

    records: list[dict[str, object]] = []
    columns = [
        "quote_date",
        "expiration",
        "settlement_class",
        "relative_spread",
        "volume",
        "open_interest",
        "quote_datetime",
        "inclusion_status",
        "iv",
    ]
    for path in files:
        frame = pd.read_parquet(path, columns=columns)
        frame["dte"] = (
            pd.to_datetime(frame["expiration"]) - pd.to_datetime(frame["quote_date"])
        ).dt.days
        for settlement, group in frame.groupby("settlement_class", dropna=False):
            usable = group[group["inclusion_status"] == "INCLUDED"]
            liquid = usable[(usable["volume"] > 0) | (usable["open_interest"] > 0)]
            narrow = usable[usable["relative_spread"] <= 0.10]
            near = usable[usable["dte"].between(0, 7)]
            records.append(
                {
                    "quote_date": str(group["quote_date"].iloc[0]),
                    "settlement_class": str(settlement),
                    "raw_quotes": len(group),
                    "usable_quotes": len(usable),
                    "liquid_quotes": len(liquid),
                    "narrow_spread_quotes": len(narrow),
                    "near_expiry_0_to_7d_quotes": len(near),
                    "zero_dte_quotes": int((usable["dte"] == 0).sum()),
                    "median_relative_spread": float(usable["relative_spread"].median()),
                    "near_expiry_median_relative_spread": float(near["relative_spread"].median()),
                    "near_expiry_median_iv": float(near["iv"].median()),
                    "timestamp_coverage": float(usable["quote_datetime"].notna().mean()),
                }
            )
    return pd.DataFrame.from_records(records)


def _even_sample(frame: pd.DataFrame, count: int) -> pd.DataFrame:
    ordered = frame.sort_values("log_moneyness")
    if len(ordered) <= count:
        return ordered
    index = np.unique(np.rint(np.linspace(0, len(ordered) - 1, count)).astype(int))
    return ordered.iloc[index]


def prepare_surface(
    path: Path,
    split: str,
    forwards: pd.DataFrame,
    config: EmpiricalConfig,
    *,
    settlement_class: str = "PM",
) -> PreparedSurface:
    """Create an OTM, maturity-stratified calibration surface.

    Rates and dividend yields are implied from F3 parity regression.  Quotes
    without a successful matching forward estimate are retained in the raw
    source but are unavailable for this calibration surface.
    """

    columns = [
        "source_row_id",
        "contract",
        "quote_date",
        "expiration",
        "option_type",
        "strike",
        "bid",
        "ask",
        "midpoint",
        "relative_spread",
        "underlying_price",
        "settlement_class",
        "inclusion_status",
        "iv",
        "vega",
    ]
    frame = pd.read_parquet(path, columns=columns)
    frame = frame[
        (frame["inclusion_status"] == "INCLUDED") & (frame["settlement_class"] == settlement_class)
    ].copy()
    if frame.empty:
        raise ValueError(f"no included {settlement_class} quotes on {path.stem}")
    frame["quote_date"] = frame["quote_date"].astype(str)
    frame["expiration"] = frame["expiration"].astype(str)
    joined = frame.merge(
        forwards,
        on=["quote_date", "expiration", "settlement_class"],
        how="inner",
        suffixes=("", "_forward"),
    )
    quote_dates = pd.to_datetime(joined["quote_date"])
    expirations = pd.to_datetime(joined["expiration"])
    joined["maturity"] = (expirations - quote_dates).dt.days / 365.0
    joined["rate"] = -np.log(joined["discount_factor"]) / joined["maturity"]
    joined["dividend_yield"] = (
        joined["rate"] - np.log(joined["forward"] / joined["underlying_price"]) / joined["maturity"]
    )
    joined["log_moneyness"] = np.log(joined["strike"] / joined["forward"])
    joined = joined[
        joined["maturity"].between(7.0 / 365.0, 1.25)
        & joined["iv"].between(0.03, 1.5)
        & joined["relative_spread"].between(0.0, config.relative_spread_limit)
        & (joined["log_moneyness"].abs() <= config.maximum_abs_log_moneyness)
        & joined["rate"].between(-0.10, 0.20)
        & joined["dividend_yield"].between(-0.15, 0.25)
        & (joined["midpoint"] > 0.0)
    ]
    # OTM options avoid the largest cancellation errors in IV inversion.
    joined = joined[
        ((joined["option_type"] == "put") & (joined["strike"] <= joined["forward"]))
        | ((joined["option_type"] == "call") & (joined["strike"] >= joined["forward"]))
    ]
    expiry_sizes = joined.groupby("expiration").size()
    eligible = expiry_sizes[expiry_sizes >= 6].index
    joined = joined[joined["expiration"].isin(eligible)]
    if joined.empty:
        raise ValueError(f"no eligible calibration quotes on {path.stem}")
    expiry_table = joined.groupby("expiration", as_index=False).agg(maturity=("maturity", "first"))
    expiry_table = expiry_table.sort_values(by=["maturity"])
    chosen_index = np.unique(
        np.rint(
            np.linspace(0, len(expiry_table) - 1, min(config.max_expiries, len(expiry_table)))
        ).astype(int)
    )
    chosen = expiry_table.iloc[chosen_index]["expiration"]
    sampled = pd.concat(
        [
            _even_sample(joined[joined["expiration"] == expiry], config.quotes_per_expiry)
            for expiry in chosen
        ],
        ignore_index=True,
    )
    if len(sampled) < 12:
        raise ValueError(f"fewer than 12 stratified quotes on {path.stem}")
    market_vega = sampled["vega"].to_numpy(dtype=float)
    # Source vega can be absent; analytic market-IV vega is a documented repair.
    missing_vega = ~np.isfinite(market_vega) | (market_vega <= 0.0)
    if np.any(missing_vega):
        repaired = np.asarray(
            greeks(
                sampled["underlying_price"].to_numpy(dtype=float),
                sampled["strike"].to_numpy(dtype=float),
                sampled["maturity"].to_numpy(dtype=float),
                sampled["iv"].to_numpy(dtype=float),
                sampled["rate"].to_numpy(dtype=float),
                sampled["dividend_yield"].to_numpy(dtype=float),
                "call",
            ).vega,
            dtype=np.float64,
        )
        market_vega[missing_vega] = repaired[missing_vega]
    data = CalibrationData(
        spot=sampled["underlying_price"].to_numpy(dtype=float),
        strike=sampled["strike"].to_numpy(dtype=float),
        maturity=sampled["maturity"].to_numpy(dtype=float),
        rate=sampled["rate"].to_numpy(dtype=float),
        dividend_yield=sampled["dividend_yield"].to_numpy(dtype=float),
        option_type=sampled["option_type"].to_numpy(dtype=str),
        market_price=sampled["midpoint"].to_numpy(dtype=float),
        market_iv=sampled["iv"].to_numpy(dtype=float),
        vega=market_vega,
        bid=sampled["bid"].to_numpy(dtype=float),
        ask=sampled["ask"].to_numpy(dtype=float),
    )
    return PreparedSurface(
        str(sampled["quote_date"].iloc[0]),
        split,
        settlement_class,
        data,
        sampled["forward"].to_numpy(dtype=float),
        sampled["log_moneyness"].to_numpy(dtype=float),
        sampled["expiration"].to_numpy(dtype=str),
        sampled["source_row_id"].to_numpy(dtype=str),
    )


def _result_record(result: CalibrationResult, **extra: object) -> dict[str, object]:
    record: dict[str, object] = {
        **extra,
        "objective": result.objective_name,
        "optimizer": result.optimizer,
        "feller_treatment": result.feller_treatment,
        "objective_value": result.objective_value,
        "success": result.success,
        "evaluations": result.evaluations,
        "iterations": result.iterations,
        "message": result.message,
        "feller_ratio": result.parameters.feller_ratio,
    }
    record.update(dict(zip(PARAMETER_NAMES, parameter_vector(result.parameters), strict=True)))
    return record


def _calibrate(
    surface: PreparedSurface,
    config: EmpiricalConfig,
    *,
    objective: Literal["C00", "C01", "C02", "C03", "C04"] = "C04",
    optimizer: Literal["local", "multistart", "de", "de_local"] = "local",
    feller: Literal["unconstrained", "soft", "hard"] = "unconstrained",
    initial: HestonParameters | None = None,
    seed_offset: int = 0,
) -> CalibrationResult:
    return calibrate_heston(
        surface.data,
        objective=objective,
        optimizer=optimizer,
        initial=initial,
        feller_treatment=feller,
        multistarts=config.multistarts,
        seed=config.seed + seed_offset,
        max_iterations=(
            config.de_iterations if optimizer in {"de", "de_local"} else config.local_iterations
        ),
        de_population=config.de_population,
        nodes=config.quadrature_nodes,
    )


def _gpu_population_local(
    surface: PreparedSurface,
    config: EmpiricalConfig,
    device: str,
    *,
    objective: Literal["C00", "C01", "C02", "C03", "C04"] = "C04",
    feller: Literal["unconstrained", "soft", "hard"] = "unconstrained",
) -> CalibrationResult:
    """O04: CUDA FP64 population prescreen followed by exact local refinement."""

    if device != "cuda":
        raise RuntimeError("O04 requires device='cuda'")
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("O04 requires an available CUDA device")
    rng = np.random.default_rng(config.seed)
    bounds = np.asarray([(0.0025, 0.5), (0.05, 10.0), (0.0025, 0.5), (0.05, 2.5), (-0.999, 0.999)])
    population_size = max(64, config.de_population * 16)
    population = rng.uniform(bounds[:, 0], bounds[:, 1], size=(population_size, 5))
    tensor = torch.as_tensor(population, dtype=torch.float64, device="cuda")
    market = torch.as_tensor(surface.data.market_price, dtype=torch.float64, device="cuda")
    unique_groups: dict[tuple[float, float, float, float], list[int]] = {}
    for index in range(surface.data.market_price.size):
        key = (
            float(surface.data.spot[index]),
            float(surface.data.maturity[index]),
            float(surface.data.rate[index]),
            float(surface.data.dividend_yield[index]),
        )
        unique_groups.setdefault(key, []).append(index)
    residuals = torch.zeros(
        (population_size, surface.data.market_price.size),
        dtype=torch.float64,
        device="cuda",
    )
    for (spot, maturity, rate, dividend), indices in unique_groups.items():
        idx = np.asarray(indices, dtype=int)
        strike = torch.as_tensor(surface.data.strike[idx], dtype=torch.float64, device="cuda")
        price = heston_call_price_torch(
            spot,
            strike,
            maturity,
            tensor,
            rate,
            dividend,
            nodes=config.quadrature_nodes,
        )
        kind = surface.data.option_type[idx]
        put = torch.as_tensor(kind == "put", dtype=torch.bool, device="cuda")
        if torch.any(put):
            parity = spot * np.exp(-dividend * maturity) - strike * np.exp(-rate * maturity)
            price[:, put] -= parity[put]
        position = torch.as_tensor(idx, dtype=torch.long, device="cuda")
        residuals[:, position] = price - market[position][None, :]
    if objective in {"C03", "C04"}:
        spread = torch.as_tensor(
            np.maximum(surface.data.ask - surface.data.bid, 0.01),
            dtype=torch.float64,
            device="cuda",
        )
        scaled = residuals / spread[None, :]
        if objective == "C03":
            losses = torch.sqrt(torch.mean(scaled * scaled, dim=1))
        else:
            absolute = torch.abs(scaled)
            huber = torch.where(
                absolute <= 1.5,
                0.5 * scaled * scaled,
                1.5 * (absolute - 0.75),
            )
            losses = torch.mean(huber, dim=1)
    else:
        # IV-space inversion remains CPU-based. C01/C02 therefore use a GPU
        # price-RMSE prescreen before exact CPU objective refinement.
        losses = torch.sqrt(torch.mean(residuals * residuals, dim=1))
    best = population[int(torch.argmin(losses).item())]
    return _calibrate(
        surface,
        config,
        objective=objective,
        optimizer="local",
        feller=feller,
        initial=HestonParameters(*map(float, best)),
        seed_offset=400,
    )


def _fit_svi(surface: PreparedSurface) -> dict[float, SVIParameters]:
    fitted: dict[float, SVIParameters] = {}
    for maturity in np.unique(surface.data.maturity):
        mask = np.isclose(surface.data.maturity, maturity)
        k = surface.log_moneyness[mask]
        total = surface.data.market_iv[mask] ** 2 * maturity
        if k.size < 5:
            continue

        def residual(
            vector: FloatArray, k_slice: FloatArray = k, total_slice: FloatArray = total
        ) -> FloatArray:
            try:
                parameter = SVIParameters(*map(float, vector))
                return (
                    np.asarray(svi_total_variance(k_slice, parameter), dtype=np.float64)
                    - total_slice
                )
            except ValueError:
                return np.full_like(total_slice, 1.0e3)

        theta = max(float(np.median(total[np.argsort(np.abs(k))[: min(3, k.size)]])), 1.0e-5)
        result = least_squares(
            residual,
            np.asarray([0.5 * theta, 0.1, -0.3, 0.0, 0.1]),
            bounds=([0.0, 1.0e-6, -0.999, -1.0, 0.005], [2.0, 4.0, 0.999, 1.0, 2.0]),
            max_nfev=500,
        )
        parameter = SVIParameters(*map(float, result.x))
        parameter.validate()
        fitted[float(maturity)] = parameter
    return fitted


def _fit_ssvi(surface: PreparedSurface) -> SSVISurface:
    maturities = np.unique(surface.data.maturity)
    theta = []
    for maturity in maturities:
        mask = np.isclose(surface.data.maturity, maturity)
        order = np.argsort(np.abs(surface.log_moneyness[mask]))[: min(3, int(mask.sum()))]
        theta.append(float(np.median(surface.data.market_iv[mask][order] ** 2 * maturity)))
    theta_array = np.maximum.accumulate(np.maximum(np.asarray(theta), 1.0e-7))

    def residual(vector: FloatArray) -> FloatArray:
        parameter = SSVIParameters(*map(float, vector))
        index = np.searchsorted(maturities, surface.data.maturity)
        predicted = ssvi_total_variance(surface.log_moneyness, theta_array[index], parameter)
        observed = surface.data.market_iv**2 * surface.data.maturity
        return np.asarray(predicted, dtype=np.float64) - observed

    result = least_squares(
        residual,
        np.asarray([-0.4, 1.0, 0.5]),
        bounds=([-0.999, 0.01, 0.0], [0.999, 5.0, 1.0]),
        max_nfev=500,
    )
    return SSVISurface(maturities, theta_array, SSVIParameters(*map(float, result.x)))


def _prices_by_group(
    surface: PreparedSurface,
    pricer: Callable[[float, FloatArray, float, float, float, str], FloatArray],
) -> FloatArray:
    output = np.empty(surface.data.market_price.size, dtype=np.float64)
    groups: dict[tuple[float, float, float, float, str], list[int]] = {}
    for index in range(output.size):
        key = (
            float(surface.data.spot[index]),
            float(surface.data.maturity[index]),
            float(surface.data.rate[index]),
            float(surface.data.dividend_yield[index]),
            str(surface.data.option_type[index]),
        )
        groups.setdefault(key, []).append(index)
    for (spot, maturity, rate, dividend, kind), indices in groups.items():
        idx = np.asarray(indices, dtype=int)
        output[idx] = pricer(spot, surface.data.strike[idx], maturity, rate, dividend, kind)
    return output


def _metrics(
    surface: PreparedSurface,
    prices: FloatArray,
    *,
    model_id: str,
    experiment: str,
    subset: FloatArray | NDArray[np.bool_] | None = None,
) -> dict[str, object]:
    mask = np.ones(prices.size, dtype=bool) if subset is None else np.asarray(subset, dtype=bool)
    error = prices[mask] - surface.data.market_price[mask]
    spread = np.maximum(surface.data.ask[mask] - surface.data.bid[mask], 0.01)
    return {
        "quote_date": surface.quote_date,
        "split": surface.split,
        "settlement_class": surface.settlement_class,
        "experiment": experiment,
        "model_id": model_id,
        "observations": int(mask.sum()),
        "price_rmse": float(np.sqrt(np.mean(error**2))),
        "price_mae": float(np.mean(np.abs(error))),
        "spread_normalized_rmse": float(np.sqrt(np.mean((error / spread) ** 2))),
        "inside_bid_ask_rate": float(
            np.mean(
                (prices[mask] >= surface.data.bid[mask]) & (prices[mask] <= surface.data.ask[mask])
            )
        ),
    }


def _finite_difference_evaluator(
    pricer: Callable[[float, float, float, str], float], spot: float
) -> Callable[[OptionLeg], InstrumentMeasures]:
    spot_bump = max(0.001 * spot, 0.25)
    vol_bump = 0.01

    def evaluate(leg: OptionLeg) -> InstrumentMeasures:
        base = pricer(spot, leg.strike, leg.maturity, leg.option_type)
        up = pricer(spot + spot_bump, leg.strike, leg.maturity, leg.option_type)
        down = pricer(spot - spot_bump, leg.strike, leg.maturity, leg.option_type)
        # Vega is a model initial-variance/volatility sensitivity proxy.  For
        # arbitrary pricers without an explicit volatility state it is NaN;
        # model-specific closures below replace this calculation.
        return InstrumentMeasures(
            base,
            (up - down) / (2.0 * spot_bump),
            (up - 2.0 * base + down) / spot_bump**2,
            vol_bump * 0.0,
        )

    return evaluate


def _portfolio_rows(
    surface: PreparedSurface,
    heston: HestonParameters,
    ssvi: SSVISurface,
    config: EmpiricalConfig,
) -> list[dict[str, object]]:
    spot = float(np.median(surface.data.spot))
    maturities = np.unique(surface.data.maturity)
    if maturities.size < 2:
        return []
    front, back = float(maturities[0]), float(maturities[-1])
    portfolios = standardized_portfolios(spot, front, back)
    rate = float(np.median(surface.data.rate))
    dividend = float(np.median(surface.data.dividend_yield))

    def heston_scalar(s: float, k: float, t: float, kind: str) -> float:
        return float(
            heston_price_fixed_quad(
                s,
                k,
                t,
                heston,
                rate,
                dividend,
                cast(Literal["call", "put"], kind),
                nodes=config.quadrature_nodes,
            )
        )

    bates = BatesParameters(
        heston.v0,
        heston.kappa,
        heston.theta,
        heston.sigma_v,
        heston.rho,
        0.25,
        -0.05,
        0.12,
    )

    def bates_scalar(s: float, k: float, t: float, kind: str) -> float:
        return float(
            bates_price(
                s,
                k,
                t,
                bates,
                rate,
                dividend,
                cast(Literal["call", "put"], kind),
                nodes=config.quadrature_nodes,
            )
        )

    def ssvi_scalar(s: float, k: float, t: float, kind: str) -> float:
        forward = s * np.exp((rate - dividend) * t)
        volatility = float(ssvi.implied_volatility(np.log(k / forward), t))
        return float(
            option_price(s, k, t, volatility, rate, dividend, cast(Literal["call", "put"], kind))
        )

    # Finite differences give a common, implementation-independent Greek
    # comparison. Vega perturbs the model's volatility state explicitly.
    def with_vega(
        scalar: Callable[[float, float, float, str], float],
        volatility_scalar: Callable[[float, float, float, str, float], float],
    ) -> Callable[[OptionLeg], InstrumentMeasures]:
        base_evaluator = _finite_difference_evaluator(scalar, spot)

        def evaluate(leg: OptionLeg) -> InstrumentMeasures:
            measure = base_evaluator(leg)
            bump = 0.005
            up = volatility_scalar(spot, leg.strike, leg.maturity, leg.option_type, bump)
            down = volatility_scalar(spot, leg.strike, leg.maturity, leg.option_type, -bump)
            return InstrumentMeasures(
                measure.value,
                measure.delta,
                measure.gamma,
                (up - down) / (2.0 * bump),
            )

        return evaluate

    def heston_vol(s: float, k: float, t: float, kind: str, bump: float) -> float:
        bumped = HestonParameters(
            max((np.sqrt(heston.v0) + bump) ** 2, 1.0e-8),
            heston.kappa,
            heston.theta,
            heston.sigma_v,
            heston.rho,
        )
        return float(
            heston_price_fixed_quad(
                s,
                k,
                t,
                bumped,
                rate,
                dividend,
                cast(Literal["call", "put"], kind),
                nodes=config.quadrature_nodes,
            )
        )

    def bates_vol(s: float, k: float, t: float, kind: str, bump: float) -> float:
        bumped = BatesParameters(
            max((np.sqrt(bates.v0) + bump) ** 2, 1.0e-8),
            bates.kappa,
            bates.theta,
            bates.sigma_v,
            bates.rho,
            bates.jump_intensity,
            bates.mean_log_jump,
            bates.jump_volatility,
        )
        return float(
            bates_price(
                s,
                k,
                t,
                bumped,
                rate,
                dividend,
                cast(Literal["call", "put"], kind),
                nodes=config.quadrature_nodes,
            )
        )

    def ssvi_vol(s: float, k: float, t: float, kind: str, bump: float) -> float:
        forward = s * np.exp((rate - dividend) * t)
        base_vol = float(ssvi.implied_volatility(np.log(k / forward), t))
        return float(
            option_price(
                s,
                k,
                t,
                max(base_vol + bump, 1.0e-6),
                rate,
                dividend,
                cast(Literal["call", "put"], kind),
            )
        )

    measures = evaluate_models(
        portfolios,
        {
            "M10_HESTON": with_vega(heston_scalar, heston_vol),
            "M03_SSVI": with_vega(ssvi_scalar, ssvi_vol),
            "M21_BATES_SENSITIVITY": with_vega(bates_scalar, bates_vol),
        },
    )
    rows: list[dict[str, object]] = []
    by_portfolio = {portfolio.portfolio_id: portfolio for portfolio in portfolios}
    developer = {m.portfolio_id: m for m in measures if m.model_id == "M10_HESTON"}
    for measure in measures:
        comparison = compare_materiality(developer[measure.portfolio_id], measure)
        rows.append(
            {
                "quote_date": surface.quote_date,
                "split": surface.split,
                "portfolio_id": measure.portfolio_id,
                "description": by_portfolio[measure.portfolio_id].description,
                "normalization": by_portfolio[measure.portfolio_id].normalization,
                "model_id": measure.model_id,
                "value": measure.value,
                "delta": measure.delta,
                "gamma": measure.gamma,
                "vega": measure.vega,
                **asdict(comparison),
            }
        )
    return rows


def run_empirical_program(
    profile: str = "research",
    *,
    device: str = "cuda",
    resume: bool = True,
    root: Path | None = None,
    config: EmpiricalConfig | None = None,
    cpu_workers: int = 2,
) -> dict[str, Any]:
    """Execute feasible calibration, cross-sectional, and materiality studies."""

    workspace = Path.cwd() if root is None else root
    result_path = workspace / "artifacts/calibration/empirical_methodology.json"
    required_outputs = (
        workspace / "reports/tables/T04_CALIBRATION_OBJECTIVE_COMPARISON.csv",
        workspace / "reports/tables/T05_OPTIMIZER_IDENTIFIABILITY.csv",
        workspace / "reports/tables/T06_PRICING_MODEL_COMPARISON.csv",
        workspace / "reports/tables/T11_PORTFOLIO_MATERIALITY.csv",
    )
    requested_config = config or EmpiricalConfig()
    if resume and result_path.exists() and all(path.exists() for path in required_outputs):
        existing = cast(dict[str, Any], json.loads(result_path.read_text(encoding="utf-8")))
        prior_code_hashes = existing.get("code_hashes", {})
        current_code_hash = _hash_file(Path(__file__))
        current_calibration_hash = _hash_file(Path(__file__).parent / "development/calibration.py")
        current_heston_hash = _hash_file(Path(__file__).parent / "development/heston_cf.py")
        if (
            existing.get("profile") == profile
            and existing.get("config_hash") == _config_hash(requested_config)
            and isinstance(prior_code_hashes, dict)
            and prior_code_hashes.get("empirical.py") == current_code_hash
            and prior_code_hashes.get("calibration.py") == current_calibration_hash
            and prior_code_hashes.get("heston_cf.py") == current_heston_hash
        ):
            return existing
    if profile not in {"research", "audit", "smoke"}:
        raise ValueError("profile must be research, audit, or smoke")
    if not 1 <= cpu_workers <= 4:
        raise ValueError("cpu_workers must lie between one and four")
    if config is None and profile == "smoke":
        cfg = EmpiricalConfig(
            max_expiries=2,
            quotes_per_expiry=6,
            representative_dates_per_split=1,
            quadrature_nodes=32,
            local_iterations=8,
            de_iterations=2,
            multistarts=2,
            bootstrap_draws=2,
        )
    else:
        cfg = requested_config
    started = _timestamp()
    started_clock = time.perf_counter()
    try:
        covariates, covariate_manifest = acquire_fred_covariates(workspace)
    except (OSError, ValueError, TimeoutError) as exc:
        covariate_path = workspace / "data/processed/public_covariates.parquet"
        if covariate_path.exists():
            covariates = pd.read_parquet(covariate_path)
            covariate_manifest = {"status": "CACHED_AFTER_ACQUISITION_FAILURE", "error": str(exc)}
        else:
            covariates = pd.DataFrame(columns=["quote_date"])
            covariate_manifest = {"status": "UNAVAILABLE", "error": str(exc)}
    if not covariates.empty:
        covariates["quote_date"] = pd.to_datetime(covariates["quote_date"])
    data_dir = workspace / "data/processed/historical_spx_sample"
    files = sorted(data_dir.glob("*_options.parquet"))
    partitions = chronological_split(files)
    forwards = _forward_lookup(workspace)
    universe_diagnostics = _universe_diagnostics(files)
    availability: list[dict[str, object]] = []
    surfaces: dict[str, list[PreparedSurface]] = {name: [] for name in partitions}
    for split, split_files in partitions.items():
        for path in representative_files(split_files, cfg.representative_dates_per_split):
            for settlement in ("PM", "AM"):
                try:
                    surfaces[split].append(
                        prepare_surface(path, split, forwards, cfg, settlement_class=settlement)
                    )
                except ValueError as exc:
                    availability.append(
                        {
                            "result_id": f"surface-{split}-{path.stem}-{settlement}",
                            "status": "UNAVAILABLE",
                            "reason": str(exc),
                        }
                    )

    # Research and audit profiles evaluate every complete PM market date with
    # a smaller per-date surface. Method selection remains confined to the
    # representative DEV/VALIDATION surfaces above. This makes rolling
    # stability and lightweight cross-sectional summaries cover all 127 dates
    # without expanding the global-optimizer study by two orders of magnitude.
    comparison_surfaces = {name: list(values) for name, values in surfaces.items()}
    lightweight_config = replace(
        cfg,
        max_expiries=min(cfg.max_expiries, 2),
        quotes_per_expiry=min(cfg.quotes_per_expiry, 6),
    )
    if profile in {"research", "audit"}:
        for split, split_files in partitions.items():
            existing_dates = {
                surface.quote_date
                for surface in comparison_surfaces[split]
                if surface.settlement_class == "PM"
            }
            for path in split_files:
                quote_date = path.stem.removesuffix("_options")
                if quote_date in existing_dates:
                    continue
                try:
                    comparison_surfaces[split].append(
                        prepare_surface(
                            path,
                            split,
                            forwards,
                            lightweight_config,
                            settlement_class="PM",
                        )
                    )
                except ValueError as exc:
                    availability.append(
                        {
                            "result_id": f"all-date-PM-{split}-{quote_date}",
                            "status": "UNAVAILABLE",
                            "reason": str(exc),
                        }
                    )

    curve_sensitivity_rows: list[dict[str, object]] = []
    if not covariates.empty:
        covariates_by_date = covariates.set_index("quote_date")
        for split_surfaces in surfaces.values():
            for surface in split_surfaces:
                date = pd.Timestamp(surface.quote_date)
                if date not in covariates_by_date.index:
                    continue
                public_row = covariates_by_date.loc[date]
                if isinstance(public_row, pd.DataFrame):
                    public_row = public_row.iloc[0]
                external_prices = np.full_like(surface.data.market_price, np.nan)
                reference_series: list[str] = []
                for index, maturity in enumerate(surface.data.maturity):
                    series = (
                        "DGS1MO"
                        if maturity <= 1.5 / 12.0
                        else "DGS3MO"
                        if maturity <= 4.5 / 12.0
                        else "DGS6MO"
                        if maturity <= 0.75
                        else "DGS1"
                    )
                    reference_series.append(series)
                    raw_external_yield = public_row.get(series)
                    external_yield = (
                        float(raw_external_yield)
                        if raw_external_yield is not None and pd.notna(raw_external_yield)
                        else np.nan
                    )
                    if not np.isfinite(external_yield):
                        continue
                    external_prices[index] = option_price(
                        surface.data.spot[index],
                        surface.data.strike[index],
                        maturity,
                        surface.data.market_iv[index],
                        float(external_yield) / 100.0,
                        surface.data.dividend_yield[index],
                        cast(Literal["call", "put"], str(surface.data.option_type[index])),
                    )
                implied_prices = np.asarray(
                    [
                        option_price(
                            surface.data.spot[i],
                            surface.data.strike[i],
                            surface.data.maturity[i],
                            surface.data.market_iv[i],
                            surface.data.rate[i],
                            surface.data.dividend_yield[i],
                            cast(Literal["call", "put"], str(surface.data.option_type[i])),
                        )
                        for i in range(surface.data.market_price.size)
                    ],
                    dtype=np.float64,
                )
                valid = np.isfinite(external_prices)
                if np.any(valid):
                    shift = external_prices[valid] - implied_prices[valid]
                    curve_sensitivity_rows.append(
                        {
                            "quote_date": surface.quote_date,
                            "split": surface.split,
                            "settlement_class": surface.settlement_class,
                            "observations": int(valid.sum()),
                            "external_series": ",".join(sorted(set(reference_series))),
                            "mean_absolute_price_shift": float(np.mean(np.abs(shift))),
                            "maximum_absolute_price_shift": float(np.max(np.abs(shift))),
                            "rmse_price_shift": float(np.sqrt(np.mean(shift**2))),
                            "interpretation": (
                                "Treasury par-yield sensitivity; not a derivatives discount curve"
                            ),
                        }
                    )
    dev_pm = [surface for surface in surfaces["DEV"] if surface.settlement_class == "PM"]
    if not dev_pm:
        raise RuntimeError("no DEV PM surfaces could be prepared")
    selection_surface = dev_pm[len(dev_pm) // 2]

    objective_rows: list[dict[str, object]] = []
    for objective_index, objective in enumerate(("C00", "C01", "C02", "C03", "C04")):
        for date_index, objective_surface in enumerate(dev_pm):
            start = time.perf_counter()
            result = _calibrate(
                objective_surface,
                cfg,
                objective=cast(Any, objective),
                optimizer="local",
                seed_offset=10 * objective_index + date_index,
            )
            record = _result_record(
                result,
                quote_date=objective_surface.quote_date,
                split="DEV",
                study="calibration_objective",
                runtime_seconds=time.perf_counter() - start,
                observations=objective_surface.data.market_price.size,
            )
            # Compare every native objective on common price and robust scales.
            record["evaluation_C00"] = heston_objective(
                parameter_vector(result.parameters).tolist(),
                objective_surface.data,
                "C00",
                nodes=cfg.quadrature_nodes,
            )
            record["evaluation_C04"] = heston_objective(
                parameter_vector(result.parameters).tolist(),
                objective_surface.data,
                "C04",
                nodes=cfg.quadrature_nodes,
            )
            objective_rows.append(record)
    objective_selection_frame = pd.DataFrame(objective_rows)
    finite_selection = objective_selection_frame[
        np.isfinite(objective_selection_frame["evaluation_C04"])
    ]
    objective_summary = finite_selection.groupby("objective", as_index=False).agg(
        common_c04_mean=("evaluation_C04", "mean"),
        converged_runs=("success", "sum"),
        attempted_runs=("success", "size"),
        successful_dates=("quote_date", "nunique"),
    )
    objective_summary["convergence_rate"] = (
        objective_summary["converged_runs"] / objective_summary["attempted_runs"]
    )
    # Convergence is a hard methodological consideration, not merely a runtime
    # diagnostic.  Select on DEV by highest convergence rate first, then the
    # common robust C04 evaluation metric.  This prevents a numerically failed
    # optimization from winning solely because its last iterate reports a low
    # loss.
    selected_objective = str(
        objective_summary.sort_values(
            ["convergence_rate", "common_c04_mean", "objective"],
            ascending=[False, True, True],
        ).iloc[0]["objective"]
    )

    optimizer_rows: list[dict[str, object]] = []
    optimizer_results: dict[str, CalibrationResult] = {}
    optimizer_map = {
        "O00": "local",
        "O01": "multistart",
        "O02": "de",
        "O03": "de_local",
    }
    for offset, (optimizer_id, optimizer) in enumerate(optimizer_map.items(), 100):
        start = time.perf_counter()
        result = _calibrate(
            selection_surface,
            cfg,
            objective=cast(Any, selected_objective),
            optimizer=cast(Any, optimizer),
            seed_offset=offset,
        )
        optimizer_results[optimizer_id] = result
        optimizer_rows.append(
            _result_record(
                result,
                optimizer_id=optimizer_id,
                quote_date=selection_surface.quote_date,
                split="DEV",
                study="optimizer",
                runtime_seconds=time.perf_counter() - start,
                observations=selection_surface.data.market_price.size,
            )
        )
    try:
        start = time.perf_counter()
        o04 = _gpu_population_local(
            selection_surface,
            cfg,
            device,
            objective=cast(Any, selected_objective),
        )
        optimizer_results["O04"] = o04
        optimizer_rows.append(
            _result_record(
                o04,
                optimizer_id="O04",
                quote_date=selection_surface.quote_date,
                split="DEV",
                study="gpu_population_local",
                runtime_seconds=time.perf_counter() - start,
                observations=selection_surface.data.market_price.size,
            )
        )
    except (RuntimeError, ValueError) as exc:
        availability.append({"result_id": "O04", "status": "UNAVAILABLE", "reason": str(exc)})
    finite_optimizers = [
        row for row in optimizer_rows if np.isfinite(float(cast(float, row["objective_value"])))
    ]
    converged_optimizers = [row for row in finite_optimizers if bool(row["success"])]
    optimizer_candidates = converged_optimizers or finite_optimizers
    selected_optimizer_id = str(
        min(
            optimizer_candidates,
            key=lambda row: float(cast(float, row["objective_value"])),
        )["optimizer_id"]
    )
    selected_parameters = optimizer_results[selected_optimizer_id].parameters

    diagnostic_rows: list[dict[str, object]] = []
    feller_results: dict[str, CalibrationResult] = {}
    for offset, treatment in enumerate(("unconstrained", "soft", "hard"), 500):
        start = time.perf_counter()
        result = _calibrate(
            selection_surface,
            cfg,
            objective=cast(Any, selected_objective),
            optimizer="local",
            feller=cast(Any, treatment),
            initial=selected_parameters,
            seed_offset=offset,
        )
        feller_results[treatment] = result
        diagnostic_rows.append(
            _result_record(
                result,
                diagnostic_id=f"FELLER_{treatment.upper()}",
                quote_date=selection_surface.quote_date,
                split="DEV",
                study="feller",
                runtime_seconds=time.perf_counter() - start,
            )
        )
    successful_feller = {
        treatment: result for treatment, result in feller_results.items() if result.success
    }
    feller_candidates = successful_feller or feller_results
    selected_feller = min(
        feller_candidates,
        key=lambda treatment: feller_candidates[treatment].objective_value,
    )
    selected_parameters = feller_results[selected_feller].parameters

    multistart_results = [
        _calibrate(
            selection_surface,
            cfg,
            objective=cast(Any, selected_objective),
            optimizer="local",
            feller=cast(Any, selected_feller),
            initial=HestonParameters(
                float(np.random.default_rng(cfg.seed + i).uniform(0.01, 0.20)),
                float(np.random.default_rng(cfg.seed + i).uniform(0.2, 5.0)),
                float(np.random.default_rng(cfg.seed + i).uniform(0.01, 0.20)),
                float(np.random.default_rng(cfg.seed + i).uniform(0.1, 1.5)),
                float(np.random.default_rng(cfg.seed + i).uniform(-0.9, 0.2)),
            ),
            seed_offset=600 + i,
        )
        for i in range(cfg.multistarts)
    ]
    successful_multistarts = [result for result in multistart_results if result.success]
    if len(successful_multistarts) >= 2:
        ident = identifiability_diagnostics(successful_multistarts)
        for name, dispersion in zip(PARAMETER_NAMES, ident.parameter_dispersion, strict=True):
            diagnostic_rows.append(
                {
                    "diagnostic_id": f"MULTISTART_{name}",
                    "quote_date": selection_surface.quote_date,
                    "split": "DEV",
                    "study": "identifiability",
                    "parameter": name,
                    "dispersion": dispersion,
                    "successful_solutions": ident.successful_solutions,
                    "covariance_condition_number": ident.condition_number,
                }
            )
    else:
        availability.append(
            {
                "result_id": "multiple_start_identifiability",
                "status": "FAILED",
                "reason": "fewer than two successful bounded local solutions",
            }
        )

    objective_function = lambda vector: heston_objective(  # noqa: E731
        vector,
        selection_surface.data,
        cast(Any, selected_objective),
        feller_treatment=cast(Any, selected_feller),
        nodes=cfg.quadrature_nodes,
    )
    bounds = ((0.0025, 0.5), (0.05, 10.0), (0.0025, 0.5), (0.05, 2.5), (-0.999, 0.999))
    for parameter in ("v0", "rho"):
        index = PARAMETER_NAMES.index(parameter)
        center = parameter_vector(selected_parameters)[index]
        low, high = bounds[index]
        grid = np.linspace(
            max(low, center - 0.2 * (high - low)), min(high, center + 0.2 * (high - low)), 7
        )
        for point in profile_objective(
            objective_function,
            parameter,
            grid.tolist(),
            bounds,
            parameter_vector(selected_parameters).tolist(),
            max_iterations=max(15, cfg.local_iterations // 2),
        ):
            diagnostic_rows.append(
                {
                    "diagnostic_id": f"PROFILE_{parameter}",
                    "quote_date": selection_surface.quote_date,
                    "split": "DEV",
                    "study": "profile_objective",
                    "parameter": parameter,
                    "fixed_value": point.fixed_value,
                    "objective_value": point.objective_value,
                    "success": point.success,
                }
            )

    # Quote uncertainty: bounded bid/ask draws with warm-start local fits.
    rng = np.random.default_rng(cfg.seed + 700)
    bootstrap_vectors: list[FloatArray] = []
    for draw in range(cfg.bootstrap_draws):
        prices = rng.uniform(selection_surface.data.bid, selection_surface.data.ask)
        try:
            drawn = selection_surface.data.with_market_prices(np.asarray(prices, dtype=np.float64))
            drawn_surface = PreparedSurface(
                selection_surface.quote_date,
                "DEV",
                selection_surface.settlement_class,
                drawn,
                selection_surface.forward,
                selection_surface.log_moneyness,
                selection_surface.expiration,
                selection_surface.source_row_id,
            )
            result = _calibrate(
                drawn_surface,
                cfg,
                objective=cast(Any, selected_objective),
                optimizer="local",
                feller=cast(Any, selected_feller),
                initial=selected_parameters,
                seed_offset=700 + draw,
            )
            if result.success:
                bootstrap_vectors.append(parameter_vector(result.parameters))
            diagnostic_rows.append(
                _result_record(
                    result,
                    diagnostic_id=f"BOOTSTRAP_{draw:03d}",
                    quote_date=selection_surface.quote_date,
                    split="DEV",
                    study="bootstrap",
                    draw=draw,
                )
            )
        except (ValueError, RuntimeError, FloatingPointError) as exc:
            availability.append(
                {
                    "result_id": f"bootstrap-{draw}",
                    "status": "FAILED",
                    "reason": str(exc),
                }
            )

    validation_confirmations: list[dict[str, object]] = []
    validation_pm = [
        surface for surface in surfaces["VALIDATION"] if surface.settlement_class == "PM"
    ]
    for index, surface in enumerate(validation_pm):
        try:
            confirmation = (
                _gpu_population_local(
                    surface,
                    cfg,
                    device,
                    objective=cast(Any, selected_objective),
                    feller=cast(Any, selected_feller),
                )
                if selected_optimizer_id == "O04"
                else _calibrate(
                    surface,
                    cfg,
                    objective=cast(Any, selected_objective),
                    optimizer=cast(Any, optimizer_results[selected_optimizer_id].optimizer),
                    feller=cast(Any, selected_feller),
                    initial=selected_parameters,
                    seed_offset=900 + index,
                )
            )
            validation_confirmations.append(
                _result_record(
                    confirmation,
                    quote_date=surface.quote_date,
                    split="VALIDATION",
                    study="methodology_confirmation",
                    optimizer_id=selected_optimizer_id,
                )
            )
        except (ValueError, RuntimeError, FloatingPointError) as exc:
            availability.append(
                {
                    "result_id": f"methodology-confirmation-{surface.quote_date}",
                    "status": "FAILED",
                    "reason": str(exc),
                }
            )

    comparison_rows: list[dict[str, object]] = []
    portfolio_rows: list[dict[str, object]] = []
    rolling_dates: list[str] = []
    rolling_parameters: list[FloatArray] = []
    representative_keys = {
        (surface.split, surface.quote_date, surface.settlement_class)
        for values in surfaces.values()
        for surface in values
    }
    calibration_jobs: list[tuple[PreparedSurface, EmpiricalConfig]] = []
    for split in ("DEV", "VALIDATION", "LOCKED_TEST"):
        for surface in comparison_surfaces[split]:
            stage_config = (
                cfg
                if (surface.split, surface.quote_date, surface.settlement_class)
                in representative_keys
                else lightweight_config
            )
            calibration_jobs.append((surface, stage_config))
    calibration_jobs.sort(
        key=lambda job: (job[0].quote_date, job[0].settlement_class, job[0].split)
    )
    checkpoint_directory = (
        workspace / "artifacts/checkpoints/empirical_real_calibrations" / _config_hash(cfg)[:16]
    )
    real_calibrations: dict[tuple[str, str, str], CalibrationResult] = {}
    calibration_provenance: dict[tuple[str, str, str], dict[str, object]] = {}
    future_jobs: dict[Any, tuple[PreparedSurface, EmpiricalConfig, int]] = {}
    # O04 owns the single CUDA device and therefore remains serial. CPU
    # methodologies use at most the explicitly bounded worker count.
    worker_count = 1 if selected_optimizer_id == "O04" else cpu_workers
    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        for job_index, (surface, stage_config) in enumerate(calibration_jobs):
            seed = cfg.seed + 2_000 + job_index
            future = executor.submit(
                _calibrate_released_surface,
                surface,
                stage_config,
                objective=selected_objective,
                optimizer_id=selected_optimizer_id,
                feller=selected_feller,
                initial=selected_parameters,
                seed=seed,
                device=device,
                checkpoint_directory=checkpoint_directory,
                resume=resume,
            )
            future_jobs[future] = (surface, stage_config, seed)
        for future in as_completed(future_jobs):
            surface, _stage_config, seed = future_jobs[future]
            identity = (surface.split, surface.quote_date, surface.settlement_class)
            try:
                result, resumed_checkpoint, checkpoint_path, calibration_runtime = future.result()
                real_calibrations[identity] = result
                calibration_provenance[identity] = {
                    "released_optimizer_id": selected_optimizer_id,
                    "released_objective": selected_objective,
                    "released_feller_treatment": selected_feller,
                    "multistarts": _stage_config.multistarts,
                    "seed": seed,
                    "resumed_checkpoint": resumed_checkpoint,
                    "checkpoint": str(checkpoint_path.relative_to(workspace)),
                    "calibration_runtime_seconds": calibration_runtime,
                    "converged": result.success,
                    "optimizer_message": result.message,
                    "objective_value": result.objective_value,
                }
            except (ValueError, RuntimeError, FloatingPointError) as exc:
                availability.append(
                    {
                        "result_id": (
                            f"released-calibration-{surface.split}-{surface.quote_date}-"
                            f"{surface.settlement_class}"
                        ),
                        "status": "FAILED",
                        "reason": str(exc),
                        "optimizer_id": selected_optimizer_id,
                        "attempted": True,
                    }
                )
    for split in ("DEV", "VALIDATION", "LOCKED_TEST"):
        for surface in comparison_surfaces[split]:
            stage_config = (
                cfg
                if (surface.split, surface.quote_date, surface.settlement_class)
                in representative_keys
                else lightweight_config
            )
            try:
                identity = (surface.split, surface.quote_date, surface.settlement_class)
                if identity not in real_calibrations:
                    raise RuntimeError("released-method date calibration failed or is unavailable")
                calibrated = real_calibrations[identity]
                if not calibrated.success:
                    availability.append(
                        {
                            "result_id": (
                                f"released-calibration-{surface.split}-{surface.quote_date}-"
                                f"{surface.settlement_class}"
                            ),
                            "status": "FAILED",
                            "reason": (
                                "all selected-result evidence is finite, but the released "
                                f"optimizer did not converge: {calibrated.message}"
                            ),
                            "optimizer_id": selected_optimizer_id,
                            "attempted": True,
                        }
                    )
                    raise RuntimeError("released-method date calibration did not converge")
                if surface.settlement_class == "PM":
                    rolling_dates.append(surface.quote_date)
                    rolling_parameters.append(parameter_vector(calibrated.parameters))
                heston_prices = price_calibration_data(
                    surface.data, calibrated.parameters, nodes=stage_config.quadrature_nodes
                )
                comparison_rows.append(
                    {
                        **_metrics(
                            surface,
                            heston_prices,
                            model_id="M10_HESTON",
                            experiment="cross_sectional",
                        ),
                        "runtime_seconds": calibration_provenance[identity][
                            "calibration_runtime_seconds"
                        ],
                        **calibration_provenance[identity],
                    }
                )
                flat_vol = float(np.median(surface.data.market_iv))
                bs_prices = np.asarray(
                    [
                        option_price(
                            surface.data.spot[i],
                            surface.data.strike[i],
                            surface.data.maturity[i],
                            flat_vol,
                            surface.data.rate[i],
                            surface.data.dividend_yield[i],
                            cast(Literal["call", "put"], str(surface.data.option_type[i])),
                        )
                        for i in range(surface.data.market_price.size)
                    ],
                    dtype=np.float64,
                )
                comparison_rows.append(
                    _metrics(
                        surface, bs_prices, model_id="M00_BS_FLAT", experiment="cross_sectional"
                    )
                )
                market_iv_prices = np.asarray(
                    [
                        option_price(
                            surface.data.spot[i],
                            surface.data.strike[i],
                            surface.data.maturity[i],
                            surface.data.market_iv[i],
                            surface.data.rate[i],
                            surface.data.dividend_yield[i],
                            cast(Literal["call", "put"], str(surface.data.option_type[i])),
                        )
                        for i in range(surface.data.market_price.size)
                    ],
                    dtype=np.float64,
                )
                market_iv_record = _metrics(
                    surface,
                    market_iv_prices,
                    model_id="M01_MARKET_IV_REFERENCE",
                    experiment="cross_sectional_reference",
                )
                market_iv_record["scope"] = (
                    "same-quote supplied implied volatility; diagnostic reference, not a fit claim"
                )
                comparison_rows.append(market_iv_record)
                svi = _fit_svi(surface)
                svi_prices = np.full_like(surface.data.market_price, np.nan)
                valid_svi = np.zeros_like(surface.data.market_price, dtype=bool)
                for maturity, parameters in svi.items():
                    mask = np.isclose(surface.data.maturity, maturity)
                    volatility = np.sqrt(
                        np.maximum(svi_total_variance(surface.log_moneyness[mask], parameters), 0.0)
                        / maturity
                    )
                    for position, vol in zip(np.flatnonzero(mask), volatility, strict=True):
                        svi_prices[position] = option_price(
                            surface.data.spot[position],
                            surface.data.strike[position],
                            maturity,
                            vol,
                            surface.data.rate[position],
                            surface.data.dividend_yield[position],
                            cast(Literal["call", "put"], str(surface.data.option_type[position])),
                        )
                    valid_svi |= mask
                    diagnostics = check_svi_slice(parameters)
                    if not diagnostics.valid:
                        availability.append(
                            {
                                "result_id": f"SVI-{surface.quote_date}-{maturity:g}",
                                "status": "PARTIAL",
                                "reason": "; ".join(diagnostics.messages),
                            }
                        )
                comparison_rows.append(
                    _metrics(
                        surface,
                        svi_prices,
                        model_id="M02_SVI",
                        experiment="cross_sectional",
                        subset=valid_svi,
                    )
                )
                ssvi = _fit_ssvi(surface)
                ssvi_vol = np.asarray(
                    ssvi.implied_volatility(surface.log_moneyness, surface.data.maturity),
                    dtype=np.float64,
                )
                ssvi_prices = np.asarray(
                    [
                        option_price(
                            surface.data.spot[i],
                            surface.data.strike[i],
                            surface.data.maturity[i],
                            ssvi_vol[i],
                            surface.data.rate[i],
                            surface.data.dividend_yield[i],
                            cast(Literal["call", "put"], str(surface.data.option_type[i])),
                        )
                        for i in range(surface.data.market_price.size)
                    ],
                    dtype=np.float64,
                )
                comparison_rows.append(
                    _metrics(
                        surface, ssvi_prices, model_id="M03_SSVI", experiment="cross_sectional"
                    )
                )
                if (
                    surface.split,
                    surface.quote_date,
                    surface.settlement_class,
                ) in representative_keys:
                    bates_parameters = BatesParameters(
                        calibrated.parameters.v0,
                        calibrated.parameters.kappa,
                        calibrated.parameters.theta,
                        calibrated.parameters.sigma_v,
                        calibrated.parameters.rho,
                        0.25,
                        -0.05,
                        0.12,
                    )

                    def representative_bates_pricer(
                        spot: float,
                        strike: FloatArray,
                        maturity: float,
                        rate: float,
                        dividend: float,
                        kind: str,
                        parameters: BatesParameters = bates_parameters,
                        nodes: int = stage_config.quadrature_nodes,
                    ) -> FloatArray:
                        return np.asarray(
                            bates_price(
                                spot,
                                strike,
                                maturity,
                                parameters,
                                rate,
                                dividend,
                                cast(Literal["call", "put"], kind),
                                nodes=nodes,
                            ),
                            dtype=np.float64,
                        )

                    bates_prices = _prices_by_group(
                        surface,
                        representative_bates_pricer,
                    )
                    bates_record = _metrics(
                        surface,
                        bates_prices,
                        model_id="M21_BATES_SENSITIVITY",
                        experiment="cross_sectional_challenger",
                    )
                    bates_record["scope"] = (
                        "fixed predeclared jump sensitivity; Heston variance parameters "
                        "from same-date calibration"
                    )
                    comparison_rows.append(bates_record)
                    local_prices = np.full_like(surface.data.market_price, np.nan)
                    chosen_local = np.unique(
                        np.rint(
                            np.linspace(
                                0,
                                surface.data.market_price.size - 1,
                                min(6, surface.data.market_price.size),
                            )
                        ).astype(int)
                    )
                    local_valid = np.zeros(surface.data.market_price.size, dtype=bool)
                    for local_index in chosen_local:
                        try:
                            estimate = local_vol_mc_price(
                                ssvi,
                                float(surface.data.spot[local_index]),
                                float(surface.data.strike[local_index]),
                                float(surface.data.maturity[local_index]),
                                float(surface.data.rate[local_index]),
                                float(surface.data.dividend_yield[local_index]),
                                cast(
                                    Literal["call", "put"],
                                    str(surface.data.option_type[local_index]),
                                ),
                                paths=4_000,
                                steps=96,
                                seed=cfg.seed + int(local_index),
                            )
                            local_prices[local_index] = estimate.price
                            local_valid[local_index] = True
                        except (ValueError, FloatingPointError) as exc:
                            availability.append(
                                {
                                    "result_id": (
                                        f"M20-{surface.quote_date}-{surface.settlement_class}-"
                                        f"{local_index}"
                                    ),
                                    "status": "FAILED",
                                    "reason": str(exc),
                                }
                            )
                    if local_valid.sum() >= 2:
                        local_record = _metrics(
                            surface,
                            local_prices,
                            model_id="M20_LOCALVOL_MC",
                            experiment="cross_sectional_challenger",
                            subset=local_valid,
                        )
                        local_record["scope"] = (
                            "bounded six-contract Monte Carlo check on fitted SSVI Dupire surface"
                        )
                        comparison_rows.append(local_record)
                    else:
                        availability.append(
                            {
                                "result_id": (
                                    f"M20-{surface.quote_date}-{surface.settlement_class}"
                                ),
                                "status": "FAILED",
                                "reason": "fewer than two qualified LocalVol Monte Carlo prices",
                            }
                        )
                central = np.abs(surface.log_moneyness) <= 0.10
                wing = np.abs(surface.log_moneyness) > 0.10
                if central.sum() >= 10 and wing.sum() >= 4:
                    central_data = CalibrationData(
                        **{
                            field: np.asarray(getattr(surface.data, field))[central]
                            for field in (
                                "spot",
                                "strike",
                                "maturity",
                                "rate",
                                "dividend_yield",
                                "option_type",
                                "market_price",
                                "market_iv",
                                "vega",
                                "bid",
                                "ask",
                            )
                        }
                    )
                    central_surface = PreparedSurface(
                        surface.quote_date,
                        surface.split,
                        surface.settlement_class,
                        central_data,
                        surface.forward[central],
                        surface.log_moneyness[central],
                        surface.expiration[central],
                        surface.source_row_id[central],
                    )
                    wing_fit, _, _, _ = _calibrate_released_surface(
                        central_surface,
                        stage_config,
                        objective=cast(Any, selected_objective),
                        optimizer_id=selected_optimizer_id,
                        feller=selected_feller,
                        initial=selected_parameters,
                        seed=_stable_seed(
                            cfg.seed,
                            surface.quote_date,
                            surface.settlement_class,
                            "wing_holdout",
                        ),
                        device=device,
                        checkpoint_directory=checkpoint_directory / "holdouts",
                        resume=resume,
                    )
                    wing_prices = price_calibration_data(
                        surface.data,
                        wing_fit.parameters,
                        nodes=stage_config.quadrature_nodes,
                    )
                    comparison_rows.append(
                        _metrics(
                            surface,
                            wing_prices,
                            model_id="M10_HESTON",
                            experiment="wing_holdout",
                            subset=wing,
                        )
                    )
                else:
                    availability.append(
                        {
                            "result_id": f"wing-{surface.quote_date}-{surface.settlement_class}",
                            "status": "UNAVAILABLE",
                            "reason": "fewer than 10 central or four wing quotes",
                        }
                    )
                unique_maturities = np.unique(surface.data.maturity)
                if unique_maturities.size >= 3:
                    held_maturity = unique_maturities[len(unique_maturities) // 2]
                    held = np.isclose(surface.data.maturity, held_maturity)
                    fit = ~held
                    fit_data = CalibrationData(
                        **{
                            field: np.asarray(getattr(surface.data, field))[fit]
                            for field in (
                                "spot",
                                "strike",
                                "maturity",
                                "rate",
                                "dividend_yield",
                                "option_type",
                                "market_price",
                                "market_iv",
                                "vega",
                                "bid",
                                "ask",
                            )
                        }
                    )
                    fit_surface = PreparedSurface(
                        surface.quote_date,
                        surface.split,
                        surface.settlement_class,
                        fit_data,
                        surface.forward[fit],
                        surface.log_moneyness[fit],
                        surface.expiration[fit],
                        surface.source_row_id[fit],
                    )
                    maturity_fit, _, _, _ = _calibrate_released_surface(
                        fit_surface,
                        stage_config,
                        objective=cast(Any, selected_objective),
                        optimizer_id=selected_optimizer_id,
                        feller=selected_feller,
                        initial=selected_parameters,
                        seed=_stable_seed(
                            cfg.seed,
                            surface.quote_date,
                            surface.settlement_class,
                            "maturity_holdout",
                        ),
                        device=device,
                        checkpoint_directory=checkpoint_directory / "holdouts",
                        resume=resume,
                    )
                    maturity_prices = price_calibration_data(
                        surface.data,
                        maturity_fit.parameters,
                        nodes=stage_config.quadrature_nodes,
                    )
                    comparison_rows.append(
                        _metrics(
                            surface,
                            maturity_prices,
                            model_id="M10_HESTON",
                            experiment="maturity_holdout",
                            subset=held,
                        )
                    )
                if (
                    surface.settlement_class == "PM"
                    and split != "DEV"
                    and (surface.split, surface.quote_date, surface.settlement_class)
                    in representative_keys
                ):
                    portfolio_rows.extend(
                        _portfolio_rows(surface, calibrated.parameters, ssvi, stage_config)
                    )
            except (ValueError, RuntimeError, FloatingPointError) as exc:
                availability.append(
                    {
                        "result_id": f"comparison-{surface.quote_date}-{surface.settlement_class}",
                        "status": "FAILED",
                        "reason": str(exc),
                    }
                )

    checkpoint_manifest = {
        "schema_version": "1.0",
        "generated_at": _timestamp(),
        "released_objective": selected_objective,
        "released_optimizer_id": selected_optimizer_id,
        "released_feller_treatment": selected_feller,
        "initialization": (
            "frozen DEV-selected parameters plus deterministic bounded random starts"
        ),
        "cross_date_warm_start": False,
        "cpu_workers": worker_count,
        "jobs_expected": len(calibration_jobs),
        "jobs_completed": len(real_calibrations),
        "jobs_converged": sum(result.success for result in real_calibrations.values()),
        "jobs_nonconverged": sum(not result.success for result in real_calibrations.values()),
        "jobs_resumed": sum(
            bool(item["resumed_checkpoint"]) for item in calibration_provenance.values()
        ),
        "checkpoints": [item["checkpoint"] for item in calibration_provenance.values()],
    }
    checkpoint_directory.mkdir(parents=True, exist_ok=True)
    (checkpoint_directory / "manifest.json").write_text(
        json.dumps(checkpoint_manifest, indent=2), encoding="utf-8"
    )

    if len(rolling_parameters) >= 2:
        stability = rolling_stability(np.vstack(rolling_parameters))
        for index, quote_date in enumerate(rolling_dates):
            diagnostic_rows.append(
                {
                    "diagnostic_id": "ROLLING_STABILITY",
                    "quote_date": quote_date,
                    "split": next(
                        surface.split
                        for values in comparison_surfaces.values()
                        for surface in values
                        if surface.quote_date == quote_date and surface.settlement_class == "PM"
                    ),
                    "study": "rolling_stability",
                    "jump_score": stability.jump_score[index],
                    **dict(zip(PARAMETER_NAMES, stability.levels[index], strict=True)),
                }
            )

    objective_frame = pd.DataFrame(objective_rows + validation_confirmations)
    optimizer_frame = pd.DataFrame(optimizer_rows)
    diagnostic_frame = pd.DataFrame(diagnostic_rows)
    comparison_frame = pd.DataFrame(comparison_rows)
    if not comparison_frame.empty and not covariates.empty and "VIXCLS" in covariates:
        vix_frame = covariates[["quote_date", "VIXCLS"]].copy()
        vix_frame["quote_date"] = vix_frame["quote_date"].dt.strftime("%Y-%m-%d")
        comparison_frame = comparison_frame.merge(vix_frame, on="quote_date", how="left")
        comparison_frame["vix_regime"] = pd.cut(
            comparison_frame["VIXCLS"],
            bins=[-np.inf, 20.0, 30.0, np.inf],
            labels=["LOW", "MEDIUM", "HIGH"],
            right=False,
        ).astype("string")
        comparison_frame["vix_regime"] = comparison_frame["vix_regime"].fillna("UNAVAILABLE")
    portfolio_frame = pd.DataFrame(portfolio_rows)
    curve_sensitivity_frame = pd.DataFrame(curve_sensitivity_rows)
    feller_frame = diagnostic_frame[diagnostic_frame["study"] == "feller"].copy()
    bootstrap_frame = diagnostic_frame[diagnostic_frame["study"] == "bootstrap"].copy()
    profile_frame = diagnostic_frame[
        diagnostic_frame["study"].isin(["identifiability", "profile_objective"])
    ].copy()
    stability_frame = diagnostic_frame[diagnostic_frame["study"] == "rolling_stability"].copy()
    ablation_rows = [
        {
            "ablation_family": "objective",
            "variant": row["objective"],
            "dev_common_C04": row["evaluation_C04"],
            "native_objective_value": row["objective_value"],
            "success": row["success"],
        }
        for row in objective_rows
    ] + [
        {
            "ablation_family": "optimizer",
            "variant": row["optimizer_id"],
            "dev_common_C04": row["objective_value"],
            "native_objective_value": row["objective_value"],
            "success": row["success"],
        }
        for row in optimizer_rows
    ]
    method_ablation_frame = pd.DataFrame(ablation_rows)
    locked_frame = comparison_frame[comparison_frame["split"] == "LOCKED_TEST"].copy()
    inference_rows: list[dict[str, object]] = []
    cross = comparison_frame[
        (comparison_frame["settlement_class"] == "PM")
        & (comparison_frame["experiment"] == "cross_sectional")
    ]
    developer = cross[cross["model_id"] == "M10_HESTON"][["quote_date", "price_rmse"]]
    for model_id, challenger in cross[cross["model_id"] != "M10_HESTON"].groupby("model_id"):
        paired = developer.merge(
            challenger[["quote_date", "price_rmse"]],
            on="quote_date",
            suffixes=("_heston", "_challenger"),
        )
        differences = (paired["price_rmse_challenger"] - paired["price_rmse_heston"]).to_numpy(
            dtype=float
        )
        if differences.size < 2:
            continue
        rng_inference = np.random.default_rng(cfg.seed + len(inference_rows))
        bootstrap_means = np.mean(
            rng_inference.choice(differences, size=(2_000, differences.size), replace=True), axis=1
        )
        inference_rows.append(
            {
                "metric": "date_level_price_rmse_difference_challenger_minus_heston",
                "challenger_model_id": str(model_id),
                "market_dates": differences.size,
                "mean_difference": float(np.mean(differences)),
                "median_difference": float(np.median(differences)),
                "bootstrap_ci_low": float(np.quantile(bootstrap_means, 0.025)),
                "bootstrap_ci_high": float(np.quantile(bootstrap_means, 0.975)),
                "challenger_win_rate": float(np.mean(differences < 0.0)),
                "sampling_unit": "market date",
            }
        )
    inference_frame = pd.DataFrame(inference_rows)
    (workspace / "artifacts/calibration").mkdir(parents=True, exist_ok=True)
    (workspace / "results/raw").mkdir(parents=True, exist_ok=True)
    (workspace / "results/aggregated").mkdir(parents=True, exist_ok=True)
    (workspace / "results/materiality").mkdir(parents=True, exist_ok=True)
    (workspace / "reports/tables").mkdir(parents=True, exist_ok=True)
    objective_frame.to_parquet(
        workspace / "results/raw/calibration_objective_study.parquet", index=False
    )
    optimizer_frame.to_parquet(workspace / "results/raw/optimizer_study.parquet", index=False)
    diagnostic_frame.to_parquet(
        workspace / "results/raw/calibration_diagnostics.parquet", index=False
    )
    comparison_frame.to_parquet(
        workspace / "results/raw/real_model_comparison.parquet", index=False
    )
    universe_diagnostics.to_parquet(
        workspace / "results/raw/real_universe_diagnostics.parquet", index=False
    )
    curve_sensitivity_frame.to_parquet(
        workspace / "results/raw/external_curve_sensitivity.parquet", index=False
    )
    bootstrap_frame.to_parquet(
        workspace / "results/aggregated/bootstrap_calibration.parquet", index=False
    )
    profile_frame.to_parquet(
        workspace / "results/aggregated/identifiability_profiles.parquet", index=False
    )
    stability_frame.to_parquet(
        workspace / "results/aggregated/parameter_stability.parquet", index=False
    )
    locked_frame.to_parquet(workspace / "results/aggregated/real_locked_test.parquet", index=False)
    portfolio_frame.to_parquet(
        workspace / "results/materiality/portfolio_materiality.parquet", index=False
    )
    objective_frame.to_csv(
        workspace / "reports/tables/T04_CALIBRATION_OBJECTIVE_COMPARISON.csv", index=False
    )
    pd.concat([optimizer_frame, diagnostic_frame], ignore_index=True).to_csv(
        workspace / "reports/tables/T05_OPTIMIZER_IDENTIFIABILITY.csv", index=False
    )
    comparison_frame.to_csv(
        workspace / "reports/tables/T06_PRICING_MODEL_COMPARISON.csv", index=False
    )
    curve_sensitivity_frame.to_csv(
        workspace / "reports/tables/T02_EXTERNAL_CURVE_SENSITIVITY.csv", index=False
    )
    universe_diagnostics.to_csv(
        workspace / "reports/tables/REAL_UNIVERSE_DIAGNOSTICS.csv", index=False
    )
    feller_frame.to_csv(workspace / "results/aggregated/feller_study.csv", index=False)
    method_ablation_frame.to_csv(
        workspace / "results/aggregated/developer_method_ablations.csv", index=False
    )
    (workspace / "results/statistics").mkdir(parents=True, exist_ok=True)
    inference_frame.to_csv(workspace / "results/statistics/date_level_inference.csv", index=False)
    portfolio_frame.to_csv(workspace / "reports/tables/T11_PORTFOLIO_MATERIALITY.csv", index=False)
    split_manifest = {
        split: [path.stem.removesuffix("_options") for path in split_files]
        for split, split_files in partitions.items()
    }
    method_status = (
        "SELECTED_ON_DEV_CONFIRMED_ON_VALIDATION"
        if profile == "research"
        and validation_confirmations
        and all(bool(record["success"]) for record in validation_confirmations)
        else "SMOKE_ONLY_NOT_RELEASE_ELIGIBLE"
        if profile == "smoke"
        else "SELECTED_ON_DEV_VALIDATION_CONFIRMATION_INCOMPLETE"
    )
    methodology: dict[str, Any] = {
        "run_id": f"empirical-{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}",
        "profile": profile,
        "status": (
            "COMPLETE_WITH_RECORDED_UNAVAILABLE_RESULTS"
            if method_status == "SELECTED_ON_DEV_CONFIRMED_ON_VALIDATION"
            else "SMOKE_COMPLETE"
            if profile == "smoke"
            else "PARTIAL_VALIDATION_CONFIRMATION_INCOMPLETE"
        ),
        "started_at": started,
        "ended_at": _timestamp(),
        "runtime_seconds": time.perf_counter() - started_clock,
        "config": asdict(cfg),
        "config_hash": _config_hash(cfg),
        "code_hashes": {
            "empirical.py": _hash_file(Path(__file__)),
            "calibration.py": _hash_file(Path(__file__).parent / "development/calibration.py"),
            "heston_cf.py": _hash_file(Path(__file__).parent / "development/heston_cf.py"),
        },
        "dataset_files": len(files),
        "dataset_hashes": {_path.name: _hash_file(_path) for _path in files},
        "temporal_split": split_manifest,
        "representative_date_design": (
            "deterministic equally spaced date quantiles within each complete-date split"
        ),
        "quote_sampling_design": (
            "up to three maturity quantiles and ten evenly spaced OTM log-moneyness quotes "
            "per expiry; raw universe remains immutable"
        ),
        "selected_objective": selected_objective,
        "objective_selection_rule": (
            "highest DEV convergence rate, then lowest mean common C04 robust loss"
        ),
        "selected_optimizer": selected_optimizer_id,
        "selected_feller_treatment": selected_feller,
        "selected_parameters": dict(
            zip(PARAMETER_NAMES, parameter_vector(selected_parameters), strict=True)
        ),
        "real_date_calibration_execution": {
            "objective": selected_objective,
            "optimizer_id": selected_optimizer_id,
            "feller_treatment": selected_feller,
            "warm_start_across_dates": False,
            "initialization": (
                "frozen DEV-selected parameters plus deterministic bounded random starts"
            ),
            "cpu_workers": worker_count,
            "jobs_expected": len(calibration_jobs),
            "jobs_completed": len(real_calibrations),
            "jobs_converged": checkpoint_manifest["jobs_converged"],
            "jobs_nonconverged": checkpoint_manifest["jobs_nonconverged"],
            "convergence_rate": (
                cast(int, checkpoint_manifest["jobs_converged"]) / len(calibration_jobs)
                if calibration_jobs
                else 0.0
            ),
            "jobs_resumed": checkpoint_manifest["jobs_resumed"],
            "checkpoint_manifest": str(
                (checkpoint_directory / "manifest.json").relative_to(workspace)
            ),
        },
        "methodology_selection_status": method_status,
        "validation_confirmation_dates": [
            str(record["quote_date"]) for record in validation_confirmations
        ],
        "public_covariates": covariate_manifest,
        "all_date_pm_surface_count": sum(
            surface.settlement_class == "PM"
            for values in comparison_surfaces.values()
            for surface in values
        ),
        "limitations": [
            "public historical quotes have no synchronized quote timestamps",
            "results are cross-sectional EOD research, not trading P&L",
            (
                "global optimizer/objective/bootstrap work uses predeclared representative "
                "dates; lightweight PM calibration and stability cover all feasible dates"
            ),
            (
                "Bates portfolio row is a disclosed jump-sensitivity challenger, "
                "not a separately calibrated release"
            ),
        ],
        "row_counts": {
            "T04": len(objective_frame),
            "T05": len(optimizer_frame) + len(diagnostic_frame),
            "T06": len(comparison_frame),
            "T11": len(portfolio_frame),
            "external_curve_sensitivity": len(curve_sensitivity_frame),
            "universe_diagnostics": len(universe_diagnostics),
        },
        "availability_records": len(availability),
    }
    result_path.write_text(json.dumps(methodology, indent=2), encoding="utf-8")
    (workspace / "artifacts/calibration/developer_method_selection.json").write_text(
        json.dumps(
            {
                "status": methodology["methodology_selection_status"],
                "profile": profile,
                "selected_on": "DEV",
                "confirmed_on": "VALIDATION",
                "selected_objective": selected_objective,
                "objective_selection_rule": methodology["objective_selection_rule"],
                "selected_optimizer": selected_optimizer_id,
                "selected_feller_treatment": selected_feller,
                "selected_parameters": methodology["selected_parameters"],
                "config_hash": methodology["config_hash"],
                "validation_confirmation_dates": methodology["validation_confirmation_dates"],
                "locked_test_used_for_selection": False,
                "evidence": [
                    "results/raw/calibration_objective_study.parquet",
                    "results/raw/optimizer_study.parquet",
                    "results/aggregated/feller_study.csv",
                    "results/aggregated/bootstrap_calibration.parquet",
                    "results/aggregated/identifiability_profiles.parquet",
                    "results/aggregated/parameter_stability.parquet",
                ],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    (workspace / "results/aggregated/developer_method_selection.json").write_text(
        (workspace / "artifacts/calibration/developer_method_selection.json").read_text(
            encoding="utf-8"
        ),
        encoding="utf-8",
    )
    availability_path = workspace / "results/RESULT_AVAILABILITY.json"
    prior_availability: dict[str, object] = {}
    if availability_path.exists():
        try:
            prior_availability = cast(
                dict[str, object], json.loads(availability_path.read_text(encoding="utf-8"))
            )
        except json.JSONDecodeError:
            prior_availability = {}
    preserved_entries = prior_availability.get("entries", [])
    if not isinstance(preserved_entries, list):
        preserved_entries = []
    availability_path.write_text(
        json.dumps(
            {
                "schema_version": "1.1",
                "generated_at": _timestamp(),
                "scope": "empirical calibration, cross-sectional comparison, and materiality",
                "entries": preserved_entries,
                "experiment_records": availability,
                "structural_unavailable": [
                    {
                        "result_id": "R02_ONE_DAY_HEDGE_PROXY",
                        "status": "UNAVAILABLE_FOR_CONFIRMATORY_INFERENCE",
                        "reason": "historical EOD observations lack synchronized quote timestamps",
                    },
                    {
                        "result_id": "LIVE_PNL",
                        "status": "NOT_APPLICABLE",
                        "reason": "research dataset and intended use prohibit live-trading claims",
                    },
                ],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return methodology
