"""Post-selection real-data robustness supplement.

This stage is intentionally separate from :mod:`derivguard.empirical`.  It
may run only after the canonical research methodology has been selected on DEV
and confirmed on VALIDATION.  Historical observations remain unsynchronised
end-of-day quotes, so the outputs are cross-sectional diagnostics rather than
trading P&L or synchronized NBBO evidence.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal, cast

import numpy as np
import pandas as pd

from derivguard.development.calibration import CalibrationData, CalibrationResult
from derivguard.development.heston_cf import HestonParameters
from derivguard.empirical import (
    EmpiricalConfig,
    PreparedSurface,
    _calibrate,
    _fit_ssvi,
    _forward_lookup,
    _gpu_population_local,
    _metrics,
    chronological_split,
    representative_files,
)
from derivguard.market.black_scholes import greeks, option_price


def _timestamp() -> str:
    return datetime.now(UTC).isoformat()


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_selection_payload(payload: dict[str, Any]) -> None:
    """Refuse smoke, provisional, or validation-incomplete selections."""

    required = {
        "status",
        "profile",
        "selected_objective",
        "selected_optimizer",
        "selected_feller_treatment",
        "selected_parameters",
        "config_hash",
    }
    missing = sorted(required - payload.keys())
    if missing:
        raise RuntimeError(f"developer methodology artifact lacks fields: {missing}")
    if payload["profile"] != "research":
        raise RuntimeError("empirical supplement requires the research selection profile")
    if payload["status"] != "SELECTED_ON_DEV_CONFIRMED_ON_VALIDATION":
        raise RuntimeError("developer methodology is not research-eligible")
    if payload.get("locked_test_used_for_selection") is not False:
        raise RuntimeError("selection artifact does not prove locked-test isolation")
    if payload["selected_objective"] not in {"C00", "C01", "C02", "C03", "C04"}:
        raise RuntimeError("unknown selected objective")
    if payload["selected_optimizer"] not in {"O00", "O01", "O02", "O03", "O04"}:
        raise RuntimeError("unknown selected optimizer")
    if payload["selected_feller_treatment"] not in {"unconstrained", "soft", "hard"}:
        raise RuntimeError("unknown Feller treatment")
    parameters = payload["selected_parameters"]
    if not isinstance(parameters, dict):
        raise RuntimeError("selected_parameters must be an object")
    values = [parameters.get(name) for name in ("v0", "kappa", "theta", "sigma_v", "rho")]
    if any(not isinstance(value, (int, float)) for value in values):
        raise RuntimeError("selected Heston parameters are incomplete")
    numeric_values = cast(list[int | float], values)
    HestonParameters(*map(float, numeric_values)).validate()


def _selection(root: Path) -> dict[str, Any]:
    path = root / "results/aggregated/developer_method_selection.json"
    if not path.exists():
        raise RuntimeError("canonical developer methodology selection artifact is absent")
    payload = cast(dict[str, Any], json.loads(path.read_text(encoding="utf-8")))
    validate_selection_payload(payload)
    return payload


def filter_quote_subset(frame: pd.DataFrame, subset: str) -> pd.DataFrame:
    """Apply one declared quote-quality subset without hidden row deletion."""

    if subset == "ALL_USABLE":
        return frame.copy()
    if subset == "LIQUID":
        return frame[(frame["volume"] > 0) | (frame["open_interest"] > 0)].copy()
    if subset == "NARROW_SPREAD":
        return frame[frame["relative_spread"].between(0.0, 0.10)].copy()
    raise ValueError(f"unknown quote subset {subset}")


def _even_sample(frame: pd.DataFrame, count: int) -> pd.DataFrame:
    ordered = frame.sort_values("log_moneyness")
    if len(ordered) <= count:
        return ordered
    positions = np.unique(np.rint(np.linspace(0, len(ordered) - 1, count)).astype(int))
    return ordered.iloc[positions]


def _prepare_surface(
    path: Path,
    split: str,
    forwards: pd.DataFrame,
    *,
    settlement_class: str,
    quote_subset: str,
    maturity_band: Literal["CORE", "NEAR_EXPIRY"],
    max_expiries: int = 3,
    quotes_per_expiry: int = 10,
) -> PreparedSurface:
    columns = [
        "source_row_id",
        "quote_date",
        "expiration",
        "option_type",
        "strike",
        "bid",
        "ask",
        "midpoint",
        "relative_spread",
        "volume",
        "open_interest",
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
    frame = filter_quote_subset(frame, quote_subset)
    frame["quote_date"] = frame["quote_date"].astype(str)
    frame["expiration"] = frame["expiration"].astype(str)
    joined = frame.merge(
        forwards,
        on=["quote_date", "expiration", "settlement_class"],
        how="inner",
    )
    joined["dte"] = (
        pd.to_datetime(joined["expiration"]) - pd.to_datetime(joined["quote_date"])
    ).dt.days
    joined["maturity"] = joined["dte"] / 365.0
    if maturity_band == "NEAR_EXPIRY":
        # Zero-DTE has no positive continuous-time horizon at an unknown EOD
        # timestamp. It is counted as unavailable rather than assigned epsilon T.
        joined = joined[joined["dte"].between(1, 7)]
    else:
        joined = joined[joined["dte"].between(7, 456)]
    joined["rate"] = -np.log(joined["discount_factor"]) / joined["maturity"]
    joined["dividend_yield"] = (
        joined["rate"] - np.log(joined["forward"] / joined["underlying_price"]) / joined["maturity"]
    )
    joined["log_moneyness"] = np.log(joined["strike"] / joined["forward"])
    joined = joined[
        joined["iv"].between(0.03, 2.0)
        & joined["relative_spread"].between(0.0, 1.0)
        & (joined["log_moneyness"].abs() <= 0.35)
        & joined["rate"].between(-0.10, 0.20)
        & joined["dividend_yield"].between(-0.15, 0.25)
        & (joined["midpoint"] > 0.0)
    ]
    joined = joined[
        ((joined["option_type"] == "put") & (joined["strike"] <= joined["forward"]))
        | ((joined["option_type"] == "call") & (joined["strike"] >= joined["forward"]))
    ]
    eligible = joined.groupby("expiration").size()
    joined = joined[joined["expiration"].isin(eligible[eligible >= 6].index)]
    if joined.empty:
        raise ValueError("no expiry has six eligible OTM quotes with a qualified F3 forward")
    expiry_table = joined.groupby("expiration", as_index=False).agg(maturity=("maturity", "first"))
    expiry_table = expiry_table.sort_values("maturity")
    expiry_positions = np.unique(
        np.rint(np.linspace(0, len(expiry_table) - 1, min(max_expiries, len(expiry_table)))).astype(
            int
        )
    )
    expiries = expiry_table.iloc[expiry_positions]["expiration"]
    sampled = pd.concat(
        [
            _even_sample(joined[joined["expiration"] == expiry], quotes_per_expiry)
            for expiry in expiries
        ],
        ignore_index=True,
    )
    if len(sampled) < 8:
        raise ValueError("fewer than eight stratified quotes remain")
    market_vega = sampled["vega"].to_numpy(dtype=float)
    missing = ~np.isfinite(market_vega) | (market_vega <= 0.0)
    if np.any(missing):
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
        market_vega[missing] = repaired[missing]
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
        quote_date=str(sampled["quote_date"].iloc[0]),
        split=split,
        settlement_class=settlement_class,
        data=data,
        forward=sampled["forward"].to_numpy(dtype=float),
        log_moneyness=sampled["log_moneyness"].to_numpy(dtype=float),
        expiration=sampled["expiration"].to_numpy(dtype=str),
        source_row_id=sampled["source_row_id"].to_numpy(dtype=str),
    )


def _calibrate_selected(
    surface: PreparedSurface,
    selection: dict[str, Any],
    config: EmpiricalConfig,
    device: str,
    seed_offset: int,
) -> CalibrationResult:
    objective = cast(Literal["C00", "C01", "C02", "C03", "C04"], selection["selected_objective"])
    feller = cast(Literal["unconstrained", "soft", "hard"], selection["selected_feller_treatment"])
    parameters = selection["selected_parameters"]
    initial = HestonParameters(
        *[float(parameters[name]) for name in ("v0", "kappa", "theta", "sigma_v", "rho")]
    )
    optimizer_id = str(selection["selected_optimizer"])
    if optimizer_id == "O04":
        if device != "cuda":
            raise RuntimeError("selected O04 methodology requires device='cuda'")
        return _gpu_population_local(
            surface,
            config,
            device,
            objective=objective,
            feller=feller,
        )
    optimizer = cast(
        Literal["local", "multistart", "de", "de_local"],
        {"O00": "local", "O01": "multistart", "O02": "de", "O03": "de_local"}[optimizer_id],
    )
    return _calibrate(
        surface,
        config,
        objective=objective,
        optimizer=optimizer,
        feller=feller,
        initial=initial,
        seed_offset=seed_offset,
    )


def _model_rows(
    surface: PreparedSurface,
    calibration: CalibrationResult,
    *,
    experiment: str,
    quote_subset: str,
    maturity_band: str,
    config: EmpiricalConfig,
) -> list[dict[str, object]]:
    from derivguard.development.calibration import price_calibration_data

    heston_prices = price_calibration_data(
        surface.data, calibration.parameters, nodes=config.quadrature_nodes
    )
    flat_vol = float(np.median(surface.data.market_iv))
    bs_prices = np.asarray(
        [
            option_price(
                surface.data.spot[index],
                surface.data.strike[index],
                surface.data.maturity[index],
                flat_vol,
                surface.data.rate[index],
                surface.data.dividend_yield[index],
                cast(Literal["call", "put"], str(surface.data.option_type[index])),
            )
            for index in range(surface.data.market_price.size)
        ],
        dtype=np.float64,
    )
    ssvi = _fit_ssvi(surface)
    ssvi_vol = np.asarray(
        ssvi.implied_volatility(surface.log_moneyness, surface.data.maturity), dtype=np.float64
    )
    ssvi_prices = np.asarray(
        [
            option_price(
                surface.data.spot[index],
                surface.data.strike[index],
                surface.data.maturity[index],
                ssvi_vol[index],
                surface.data.rate[index],
                surface.data.dividend_yield[index],
                cast(Literal["call", "put"], str(surface.data.option_type[index])),
            )
            for index in range(surface.data.market_price.size)
        ],
        dtype=np.float64,
    )
    rows: list[dict[str, object]] = []
    for model_id, prices in (
        ("M10_HESTON", heston_prices),
        ("M00_BS_FLAT", bs_prices),
        ("M03_SSVI", ssvi_prices),
    ):
        row = _metrics(surface, prices, model_id=model_id, experiment=experiment)
        row.update(
            {
                "quote_subset": quote_subset,
                "maturity_band": maturity_band,
                "calibration_success": calibration.success,
                "calibration_objective_value": calibration.objective_value,
                "quote_timing": "UNSYNCHRONISED_HISTORICAL_EOD",
            }
        )
        rows.append(row)
    return rows


def run_empirical_supplement(
    *,
    root: Path | None = None,
    device: str = "cpu",
    resume: bool = True,
    representative_dates_per_split: int = 2,
) -> dict[str, Any]:
    """Execute bounded post-selection near-expiry, liquidity, and AM/PM studies."""

    workspace = Path.cwd() if root is None else root
    output = workspace / "artifacts/calibration/empirical_supplement.json"
    if resume and output.exists():
        existing = cast(dict[str, Any], json.loads(output.read_text(encoding="utf-8")))
        if existing.get("status") == "COMPLETE_WITH_RECORDED_UNAVAILABLE_RESULTS":
            return existing
    selection = _selection(workspace)
    if representative_dates_per_split < 1:
        raise ValueError("representative_dates_per_split must be positive")
    started_at = _timestamp()
    clock = time.perf_counter()
    files = sorted((workspace / "data/processed/historical_spx_sample").glob("*.parquet"))
    partitions = chronological_split(files)
    forwards = _forward_lookup(workspace)
    config = EmpiricalConfig(
        max_expiries=3,
        quotes_per_expiry=10,
        representative_dates_per_split=representative_dates_per_split,
        quadrature_nodes=48,
        local_iterations=75,
        de_iterations=20,
        multistarts=5,
        bootstrap_draws=2,
    )
    records: list[dict[str, object]] = []
    unavailable: list[dict[str, object]] = []
    counter = 0
    for split, split_files in partitions.items():
        for path in representative_files(split_files, representative_dates_per_split):
            zero_dte_source = pd.read_parquet(
                path,
                columns=["quote_date", "expiration", "settlement_class", "inclusion_status"],
            )
            zero_dte_count = int(
                (
                    (zero_dte_source["settlement_class"] == "PM")
                    & (zero_dte_source["inclusion_status"] == "INCLUDED")
                    & (
                        pd.to_datetime(zero_dte_source["expiration"])
                        == pd.to_datetime(zero_dte_source["quote_date"])
                    )
                ).sum()
            )
            designs = [
                ("NEAR_EXPIRY", "PM", "ALL_USABLE", "near_expiry_1_to_7d"),
                ("CORE", "PM", "ALL_USABLE", "liquidity_filter"),
                ("CORE", "PM", "LIQUID", "liquidity_filter"),
                ("CORE", "PM", "NARROW_SPREAD", "liquidity_filter"),
                ("CORE", "PM", "NARROW_SPREAD", "settlement_robustness"),
                ("CORE", "AM", "NARROW_SPREAD", "settlement_robustness"),
            ]
            for maturity_band, settlement, subset, experiment in designs:
                result_id = (
                    f"{experiment}-{split}-{path.stem}-{settlement}-{subset}-{maturity_band}"
                )
                try:
                    surface = _prepare_surface(
                        path,
                        split,
                        forwards,
                        settlement_class=settlement,
                        quote_subset=subset,
                        maturity_band=cast(Any, maturity_band),
                    )
                    calibration = _calibrate_selected(surface, selection, config, device, counter)
                    counter += 1
                    records.extend(
                        _model_rows(
                            surface,
                            calibration,
                            experiment=experiment,
                            quote_subset=subset,
                            maturity_band=maturity_band,
                            config=config,
                        )
                    )
                except (ValueError, RuntimeError, FloatingPointError) as exc:
                    unavailable.append(
                        {
                            "result_id": result_id,
                            "status": "UNAVAILABLE_OR_FAILED_AFTER_ATTEMPT",
                            "attempted": True,
                            "reason": str(exc),
                            "quote_timing": "UNSYNCHRONISED_HISTORICAL_EOD",
                        }
                    )
            if zero_dte_count:
                unavailable.append(
                    {
                        "result_id": f"zero-dte-{split}-{path.stem}",
                        "status": "UNAVAILABLE_FOR_CONTINUOUS_TIME_CALIBRATION",
                        "attempted": True,
                        "observations": zero_dte_count,
                        "reason": (
                            "0DTE rows have unknown intraday/EOD quote time; no positive "
                            "defensible time-to-expiry can be assigned without fabricating "
                            "a timestamp"
                        ),
                    }
                )
    frame = pd.DataFrame(records)
    if frame.empty:
        raise RuntimeError("supplement produced no measured model comparisons")
    raw_directory = workspace / "results/raw"
    table_directory = workspace / "reports/tables"
    aggregated = workspace / "results/aggregated"
    raw_directory.mkdir(parents=True, exist_ok=True)
    table_directory.mkdir(parents=True, exist_ok=True)
    aggregated.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(raw_directory / "empirical_supplement.parquet", index=False)
    frame.to_csv(table_directory / "EMPIRICAL_SUPPLEMENT.csv", index=False)
    frame[frame["experiment"] == "near_expiry_1_to_7d"].to_csv(
        aggregated / "near_expiry_study.csv", index=False
    )
    frame[frame["experiment"] == "liquidity_filter"].to_csv(
        aggregated / "liquidity_filter_study.csv", index=False
    )
    frame[frame["experiment"] == "settlement_robustness"].to_csv(
        aggregated / "am_pm_robustness.csv", index=False
    )
    availability_path = workspace / "results/EMPIRICAL_SUPPLEMENT_AVAILABILITY.json"
    availability_path.write_text(
        json.dumps(
            {
                "schema_version": "1.0",
                "generated_at": _timestamp(),
                "records": unavailable,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    payload: dict[str, Any] = {
        "run_id": f"empirical-supplement-{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}",
        "status": "COMPLETE_WITH_RECORDED_UNAVAILABLE_RESULTS",
        "started_at": started_at,
        "ended_at": _timestamp(),
        "runtime_seconds": time.perf_counter() - clock,
        "selected_method_config_hash": selection["config_hash"],
        "selection_status": selection["status"],
        "selected_objective": selection["selected_objective"],
        "selected_optimizer": selection["selected_optimizer"],
        "selected_feller_treatment": selection["selected_feller_treatment"],
        "representative_dates_per_split": representative_dates_per_split,
        "config": asdict(config),
        "measured_rows": len(frame),
        "unavailable_records": len(unavailable),
        "split_rows": frame.groupby("split").size().to_dict(),
        "experiment_rows": frame.groupby("experiment").size().to_dict(),
        "source_hash": _hash_file(Path(__file__)),
        "outputs": [
            "results/raw/empirical_supplement.parquet",
            "results/aggregated/near_expiry_study.csv",
            "results/aggregated/liquidity_filter_study.csv",
            "results/aggregated/am_pm_robustness.csv",
            "reports/tables/EMPIRICAL_SUPPLEMENT.csv",
            "results/EMPIRICAL_SUPPLEMENT_AVAILABILITY.json",
        ],
        "limitations": [
            "historical EOD option quotes have no synchronized quote timestamps",
            "results are cross-sectional research and are not live or realized P&L",
            "0DTE continuous-time calibration is unavailable without defensible timestamps",
            "Treasury or external-rate assumptions are not introduced in this supplement",
        ],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload
