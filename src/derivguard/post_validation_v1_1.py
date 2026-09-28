"""Post-validation analytics for the immutable DerivGuard v1 evidence base.

This module is deliberately additive.  It reads the frozen real-date calibration
checkpoints and v1 result artifacts, reconstructs their sampled option surfaces,
and writes a separate v1.1 analytics namespace.  It never recalibrates or mutates
the released developer model and it makes no synchronized P&L claim.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal, cast

import matplotlib
import numpy as np
import pandas as pd
import seaborn as sns  # type: ignore[import-untyped]
from matplotlib import pyplot as plt
from numpy.typing import NDArray
from scipy.stats import spearmanr
from sklearn.tree import DecisionTreeRegressor, export_text  # type: ignore[import-untyped]

from derivguard.challengers.bates import BatesParameters, bates_price
from derivguard.development.calibration import PARAMETER_NAMES, price_calibration_data
from derivguard.development.heston_cf import HestonParameters, heston_price_fixed_quad
from derivguard.empirical import (
    EmpiricalConfig,
    PreparedSurface,
    _fit_ssvi,
    _fit_svi,
    _forward_lookup,
    _prices_by_group,
    chronological_split,
    prepare_surface,
    representative_files,
)
from derivguard.market.black_scholes import greeks, option_price
from derivguard.market.implied_volatility import implied_volatility
from derivguard.market.svi import svi_total_variance

matplotlib.use("Agg")

FloatArray = NDArray[np.float64]

MONEYNESS_LABELS = (
    "deep_put_wing",
    "put_wing",
    "near_atm_put",
    "atm",
    "near_atm_call",
    "call_wing",
    "deep_call_wing",
)
MONEYNESS_EDGES = (-np.inf, -0.15, -0.05, -0.015, 0.015, 0.05, 0.15, np.inf)
MATURITY_LABELS = ("0DTE", "1-7D", "8-30D", "31-60D", "61-90D", "91-180D", "180D+")
MATURITY_EDGES = (-0.5, 0.5, 7.5, 30.5, 60.5, 90.5, 180.5, np.inf)
MODEL_ORDER = ("M00_BS_FLAT", "M02_SVI", "M03_SSVI", "M10_HESTON", "M21_BATES")
WINNER_MODELS = ("M00_BS_FLAT", "M02_SVI", "M03_SSVI", "M10_HESTON")
BOUNDS = np.asarray(
    ((0.0025, 0.5), (0.05, 10.0), (0.0025, 0.5), (0.05, 2.5), (-0.999, 0.999)),
    dtype=np.float64,
)


@dataclass(frozen=True)
class AnalyticsConfig:
    """Predeclared v1.1 descriptive-analysis controls."""

    seed: int = 20261101
    bootstrap_draws: int = 400
    minimum_slice_rows: int = 12
    minimum_slice_dates: int = 3
    tree_max_depth: int = 3
    tree_min_leaf: int = 20
    ewma_lambda: float = 0.20
    cusum_k_mad: float = 0.50
    cusum_h_mad: float = 5.00


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


def _hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _json(path: Path) -> dict[str, Any]:
    return cast(dict[str, Any], json.loads(path.read_text(encoding="utf-8")))


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8"
    )


def _parameters(payload: dict[str, Any]) -> HestonParameters:
    raw = cast(dict[str, Any], cast(dict[str, Any], payload["calibration"])["parameters"])
    return HestonParameters(*[float(raw[name]) for name in PARAMETER_NAMES])


def _model_iv(
    price: float,
    spot: float,
    strike: float,
    maturity: float,
    rate: float,
    dividend: float,
    option_type: str,
) -> float:
    try:
        result = implied_volatility(
            price,
            spot,
            strike,
            maturity,
            rate,
            dividend,
            cast(Literal["call", "put"], option_type),
        )
        return float(result.volatility) if result.converged else float("nan")
    except (ValueError, FloatingPointError, OverflowError):
        return float("nan")


def _bs_prices(surface: PreparedSurface, volatility: FloatArray) -> FloatArray:
    result = np.empty(volatility.size, dtype=np.float64)
    for kind in ("call", "put"):
        mask = surface.data.option_type == kind
        if np.any(mask):
            result[mask] = np.asarray(
                option_price(
                    surface.data.spot[mask],
                    surface.data.strike[mask],
                    surface.data.maturity[mask],
                    volatility[mask],
                    surface.data.rate[mask],
                    surface.data.dividend_yield[mask],
                    kind,
                ),
                dtype=np.float64,
            )
    return result


def _bs_greeks(
    surface: PreparedSurface, volatility: FloatArray
) -> tuple[FloatArray, FloatArray, FloatArray]:
    delta = np.empty(volatility.size, dtype=np.float64)
    gamma = np.empty(volatility.size, dtype=np.float64)
    vega = np.empty(volatility.size, dtype=np.float64)
    for kind in ("call", "put"):
        mask = surface.data.option_type == kind
        if not np.any(mask):
            continue
        risk = greeks(
            surface.data.spot[mask],
            surface.data.strike[mask],
            surface.data.maturity[mask],
            volatility[mask],
            surface.data.rate[mask],
            surface.data.dividend_yield[mask],
            kind,
        )
        delta[mask] = risk.delta
        gamma[mask] = risk.gamma
        vega[mask] = risk.vega
    return delta, gamma, vega


def _heston_prices(
    surface: PreparedSurface, parameters: HestonParameters, nodes: int
) -> FloatArray:
    return price_calibration_data(surface.data, parameters, nodes=nodes)


def _heston_greeks(
    surface: PreparedSurface, parameters: HestonParameters, nodes: int
) -> tuple[FloatArray, FloatArray, FloatArray]:
    def priced(spot_scale: float, params: HestonParameters = parameters) -> FloatArray:
        return _prices_by_group(
            surface,
            lambda spot, strike, maturity, rate, dividend, kind: np.asarray(
                heston_price_fixed_quad(
                    spot * spot_scale,
                    strike,
                    maturity,
                    params,
                    rate,
                    dividend,
                    cast(Literal["call", "put"], kind),
                    nodes=nodes,
                ),
                dtype=np.float64,
            ),
        )

    bump = 1.0e-3
    base = priced(1.0)
    up = priced(1.0 + bump)
    down = priced(1.0 - bump)
    spot = surface.data.spot
    delta = (up - down) / (2.0 * bump * spot)
    gamma = (up - 2.0 * base + down) / np.square(bump * spot)
    vol_bump = 5.0e-3
    root = np.sqrt(parameters.v0)
    upper = replace(parameters, v0=(root + vol_bump) ** 2)
    lower = replace(parameters, v0=max(root - vol_bump, 1.0e-6) ** 2)
    vega = (priced(1.0, upper) - priced(1.0, lower)) / (2.0 * vol_bump)
    return delta, gamma, vega


def _bates_prices(surface: PreparedSurface, parameters: BatesParameters, nodes: int) -> FloatArray:
    return _prices_by_group(
        surface,
        lambda spot, strike, maturity, rate, dividend, kind: np.asarray(
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
        ),
    )


def _bates_greeks(
    surface: PreparedSurface, parameters: BatesParameters, nodes: int
) -> tuple[FloatArray, FloatArray, FloatArray]:
    def priced(spot_scale: float, params: BatesParameters = parameters) -> FloatArray:
        return _prices_by_group(
            surface,
            lambda spot, strike, maturity, rate, dividend, kind: np.asarray(
                bates_price(
                    spot * spot_scale,
                    strike,
                    maturity,
                    params,
                    rate,
                    dividend,
                    cast(Literal["call", "put"], kind),
                    nodes=nodes,
                ),
                dtype=np.float64,
            ),
        )

    bump = 1.0e-3
    base, up, down = priced(1.0), priced(1.0 + bump), priced(1.0 - bump)
    delta = (up - down) / (2.0 * bump * surface.data.spot)
    gamma = (up - 2.0 * base + down) / np.square(bump * surface.data.spot)
    vol_bump = 5.0e-3
    root = np.sqrt(parameters.v0)
    upper = replace(parameters, v0=(root + vol_bump) ** 2)
    lower = replace(parameters, v0=max(root - vol_bump, 1.0e-6) ** 2)
    vega = (priced(1.0, upper) - priced(1.0, lower)) / (2.0 * vol_bump)
    return delta, gamma, vega


def _surface_source_metadata(root: Path, surface: PreparedSurface) -> pd.DataFrame:
    path = root / f"data/processed/historical_spx_sample/{surface.quote_date}_options.parquet"
    columns = [
        "source_row_id",
        "contract",
        "relative_spread",
        "volume",
        "open_interest",
        "data_quality_tier",
    ]
    available = pd.read_parquet(path).columns
    chosen = [column for column in columns if column in available]
    source = pd.read_parquet(path, columns=chosen)
    source["source_row_id"] = source["source_row_id"].astype(str)
    requested = pd.DataFrame({"source_row_id": surface.source_row_id.astype(str)})
    joined = requested.merge(source, on="source_row_id", how="left", validate="one_to_one")
    if "data_quality_tier" not in joined:
        joined["data_quality_tier"] = np.where(
            joined["relative_spread"] <= 0.05,
            "TIER_1_NARROW",
            np.where(joined["relative_spread"] <= 0.15, "TIER_2_USABLE", "TIER_3_WIDE"),
        )
    return joined


def _surface_rows(
    root: Path,
    surface: PreparedSurface,
    checkpoint: dict[str, Any],
    nodes: int,
    vix: float,
) -> list[dict[str, object]]:
    parameters = _parameters(checkpoint)
    heston = _heston_prices(surface, parameters, nodes)
    h_delta, h_gamma, h_vega = _heston_greeks(surface, parameters, nodes)
    flat = np.full(surface.data.market_iv.size, float(np.median(surface.data.market_iv)))
    bs = _bs_prices(surface, flat)
    bs_delta, bs_gamma, bs_vega = _bs_greeks(surface, flat)
    svi_fit = _fit_svi(surface)
    svi_vol = np.full(surface.data.market_iv.size, np.nan)
    for maturity, parameter in svi_fit.items():
        mask = np.isclose(surface.data.maturity, maturity)
        svi_vol[mask] = np.sqrt(
            np.maximum(svi_total_variance(surface.log_moneyness[mask], parameter), 0.0) / maturity
        )
    svi = _bs_prices(surface, np.where(np.isfinite(svi_vol), svi_vol, flat))
    svi_delta, svi_gamma, svi_vega = _bs_greeks(
        surface, np.where(np.isfinite(svi_vol), svi_vol, flat)
    )
    ssvi_fit = _fit_ssvi(surface)
    ssvi_vol = np.asarray(
        ssvi_fit.implied_volatility(surface.log_moneyness, surface.data.maturity), dtype=np.float64
    )
    ssvi = _bs_prices(surface, ssvi_vol)
    ssvi_delta, ssvi_gamma, ssvi_vega = _bs_greeks(surface, ssvi_vol)
    bates_parameters = BatesParameters(
        parameters.v0,
        parameters.kappa,
        parameters.theta,
        parameters.sigma_v,
        parameters.rho,
        0.25,
        -0.05,
        0.12,
    )
    bates = _bates_prices(surface, bates_parameters, nodes)
    b_delta, b_gamma, b_vega = _bates_greeks(surface, bates_parameters, nodes)
    metadata = _surface_source_metadata(root, surface)
    model_values = {
        "M00_BS_FLAT": (bs, bs_delta, bs_gamma, bs_vega),
        "M02_SVI": (svi, svi_delta, svi_gamma, svi_vega),
        "M03_SSVI": (ssvi, ssvi_delta, ssvi_gamma, ssvi_vega),
        "M10_HESTON": (heston, h_delta, h_gamma, h_vega),
        "M21_BATES": (bates, b_delta, b_gamma, b_vega),
    }
    records: list[dict[str, object]] = []
    for model_id, (prices, delta, gamma, vega) in model_values.items():
        for index in range(prices.size):
            market = float(surface.data.market_price[index])
            spread = max(float(surface.data.ask[index] - surface.data.bid[index]), 0.01)
            price = float(prices[index])
            records.append(
                {
                    "quote_date": surface.quote_date,
                    "split": surface.split,
                    "settlement_class": surface.settlement_class,
                    "source_row_id": str(surface.source_row_id[index]),
                    "contract": metadata.loc[index, "contract"] if "contract" in metadata else "",
                    "model_id": model_id,
                    "spot": float(surface.data.spot[index]),
                    "strike": float(surface.data.strike[index]),
                    "forward": float(surface.forward[index]),
                    "log_moneyness": float(surface.log_moneyness[index]),
                    "maturity": float(surface.data.maturity[index]),
                    "dte": round(365.0 * surface.data.maturity[index]),
                    "option_type": str(surface.data.option_type[index]),
                    "rate": float(surface.data.rate[index]),
                    "dividend_yield": float(surface.data.dividend_yield[index]),
                    "bid": float(surface.data.bid[index]),
                    "ask": float(surface.data.ask[index]),
                    "market_price": market,
                    "market_iv": float(surface.data.market_iv[index]),
                    "market_vega": float(surface.data.vega[index]),
                    "model_price": price,
                    "residual": price - market,
                    "absolute_error": abs(price - market),
                    "normalized_error": (price - market) / spread,
                    "inside_bid_ask": bool(
                        surface.data.bid[index] <= price <= surface.data.ask[index]
                    ),
                    "model_iv": _model_iv(
                        price,
                        float(surface.data.spot[index]),
                        float(surface.data.strike[index]),
                        float(surface.data.maturity[index]),
                        float(surface.data.rate[index]),
                        float(surface.data.dividend_yield[index]),
                        str(surface.data.option_type[index]),
                    ),
                    "delta": float(delta[index]),
                    "gamma": float(gamma[index]),
                    "vega": float(vega[index]),
                    "relative_spread": float(cast(Any, metadata.loc[index, "relative_spread"])),
                    "volume": float(cast(Any, metadata.loc[index, "volume"])),
                    "open_interest": float(cast(Any, metadata.loc[index, "open_interest"])),
                    "data_quality_tier": str(metadata.loc[index, "data_quality_tier"]),
                    "VIXCLS": vix,
                    "calibration_success": bool(checkpoint["calibration"]["success"]),
                }
            )
    return records


def reconstruct_contract_results(root: Path, config: AnalyticsConfig) -> pd.DataFrame:
    """Reconstruct option-level predictions from immutable calibration checkpoints."""

    output = root / "results/v1_1/contract_model_results.parquet"
    manifest = root / "artifacts/v1_1/reconstruction_manifest.json"
    source_paths = (
        root / "results/raw/real_model_comparison.parquet",
        root / "artifacts/model_releases/model_release_v1.1.json",
        root / "src/derivguard/post_validation_v1_1.py",
    )
    identity = {path.relative_to(root).as_posix(): _hash(path) for path in source_paths}
    if output.exists() and manifest.exists() and _json(manifest).get("source_hashes") == identity:
        return pd.read_parquet(output)
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
    comparison = pd.read_parquet(root / "results/raw/real_model_comparison.parquet")
    heston = comparison[
        (comparison["model_id"] == "M10_HESTON") & (comparison["experiment"] == "cross_sectional")
    ].copy()
    heston = heston[heston["converged"].fillna(False) & heston["checkpoint"].notna()]
    covariates = pd.read_parquet(root / "data/processed/public_covariates.parquet")
    covariates["quote_date"] = covariates["quote_date"].astype(str)
    vix_lookup = (
        covariates.drop_duplicates("quote_date").set_index("quote_date")["VIXCLS"].to_dict()
    )
    by_date = {path.stem.removesuffix("_options"): path for path in files}
    records: list[dict[str, object]] = []
    completed: list[dict[str, object]] = []
    for row in heston.sort_values(["quote_date", "settlement_class"]).itertuples(index=False):
        key = (str(row.split), str(row.quote_date), str(row.settlement_class))
        stage = empirical if key in representative else lightweight
        surface = prepare_surface(
            by_date[str(row.quote_date)],
            str(row.split),
            forwards,
            stage,
            settlement_class=str(row.settlement_class),
        )
        checkpoint_path = root / str(row.checkpoint)
        checkpoint = _json(checkpoint_path)
        records.extend(
            _surface_rows(
                root,
                surface,
                checkpoint,
                stage.quadrature_nodes,
                float(vix_lookup.get(str(row.quote_date), np.nan)),
            )
        )
        completed.append(
            {
                "quote_date": row.quote_date,
                "settlement_class": row.settlement_class,
                "split": row.split,
                "checkpoint": str(row.checkpoint),
                "checkpoint_hash": _hash(checkpoint_path),
                "source_rows": len(surface.source_row_id),
            }
        )
    frame = pd.DataFrame.from_records(records)
    output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(output, index=False)
    _write_json(
        manifest,
        {
            "status": "COMPLETE",
            "created_utc": _utc_now(),
            "source_hashes": identity,
            "developer_model_modified": False,
            "checkpoint_surfaces": completed,
            "rows": len(frame),
            "limitations": [
                "The reconstructed population is the deterministic released-calibration "
                "quote sample, not the full option universe.",
                "Bates is the fixed predeclared jump sensitivity, not an independently "
                "calibrated production challenger.",
                "SVI/SSVI Greeks hold each fitted volatility surface fixed under the "
                "analytic Black-Scholes risk calculation.",
                "Timestamp-free EOD data do not support synchronized temporal or P&L inference.",
            ],
        },
    )
    return frame


def _q(values: pd.Series, probability: float, fallback: float = 0.0) -> float:
    finite = values[np.isfinite(values)]
    return float(finite.quantile(probability)) if len(finite) else fallback


def freeze_slice_registry(
    root: Path, rows: pd.DataFrame, config: AnalyticsConfig
) -> dict[str, Any]:
    path = root / "artifacts/v1_1/slice_registry_v1_1_reviewed.freeze.json"
    dev = rows[(rows["split"] == "DEV") & (rows["model_id"] == "M10_HESTON")].copy()
    unique = dev.drop_duplicates(["quote_date", "source_row_id"])
    source_columns = [
        "quote_date",
        "source_row_id",
        "log_moneyness",
        "dte",
        "VIXCLS",
        "relative_spread",
        "volume",
        "open_interest",
    ]
    source_hash = hashlib.sha256(
        pd.util.hash_pandas_object(
            unique[source_columns].sort_values(["quote_date", "source_row_id"]), index=False
        )
        .to_numpy(dtype=np.uint64)
        .tobytes()
    ).hexdigest()
    if path.exists():
        existing = _json(path)
        if (
            existing.get("config") != asdict(config)
            or existing.get("dev_source_hash") != source_hash
        ):
            raise RuntimeError("existing slice registry does not match configuration/source data")
        return existing
    thresholds = {
        "vix": [_q(unique["VIXCLS"], 1 / 3), _q(unique["VIXCLS"], 2 / 3)],
        "relative_spread": [
            _q(unique["relative_spread"], 1 / 3),
            _q(unique["relative_spread"], 2 / 3),
        ],
        "positive_volume": [
            _q(unique.loc[unique["volume"] > 0, "volume"], 0.50),
            _q(unique.loc[unique["volume"] > 0, "volume"], 0.80),
        ],
        "positive_open_interest": [
            _q(unique.loc[unique["open_interest"] > 0, "open_interest"], 0.50),
            _q(unique.loc[unique["open_interest"] > 0, "open_interest"], 0.80),
        ],
    }
    money_counts = pd.cut(
        unique["log_moneyness"], MONEYNESS_EDGES, labels=MONEYNESS_LABELS
    ).value_counts(sort=False)
    maturity_counts = pd.cut(unique["dte"], MATURITY_EDGES, labels=MATURITY_LABELS).value_counts(
        sort=False
    )
    payload = {
        "schema_version": "1.0",
        "status": "FROZEN_FROM_DEV_BEFORE_VALIDATION_OR_LOCKED_SLICE_EVALUATION",
        "created_utc": _utc_now(),
        "dev_source_hash": source_hash,
        "config": asdict(config),
        "moneyness_edges": [
            str(item) if not np.isfinite(item) else item for item in MONEYNESS_EDGES
        ],
        "moneyness_labels": list(MONEYNESS_LABELS),
        "maturity_edges_days": [
            str(item) if not np.isfinite(item) else item for item in MATURITY_EDGES
        ],
        "maturity_labels": list(MATURITY_LABELS),
        "learned_thresholds": thresholds,
        "dev_bin_counts": {
            "moneyness": {str(key): int(value) for key, value in money_counts.items()},
            "maturity": {str(key): int(value) for key, value in maturity_counts.items()},
        },
        "sparse_rule": (
            "Retain mandatory economic labels; mark cells sparse unless they contain at least "
            f"{config.minimum_slice_rows} rows and {config.minimum_slice_dates} market dates. "
            "No post-DEV boundary movement is permitted."
        ),
        "zero_dte_rule": (
            "0DTE remains a separate domain and is never merged; pricing is UNAVAILABLE because "
            "released surfaces excluded sub-7D quotes and EOD files lack settlement-relative time."
        ),
    }
    _write_json(path, payload)
    return payload


def _three_regime(values: pd.Series, low: float, high: float, labels: Sequence[str]) -> pd.Series:
    return pd.cut(values, [-np.inf, low, high, np.inf], labels=labels, include_lowest=True).astype(
        str
    )


def apply_slices(rows: pd.DataFrame, frozen: dict[str, Any]) -> pd.DataFrame:
    frame = rows.copy()
    thresholds = frozen["learned_thresholds"]
    frame["moneyness_slice"] = pd.cut(
        frame["log_moneyness"], MONEYNESS_EDGES, labels=MONEYNESS_LABELS
    ).astype(str)
    frame["maturity_slice"] = pd.cut(frame["dte"], MATURITY_EDGES, labels=MATURITY_LABELS).astype(
        str
    )
    vix_low, vix_high = map(float, thresholds["vix"])
    frame["vix_regime"] = _three_regime(
        frame["VIXCLS"], vix_low, vix_high, ("LOW", "MEDIUM", "HIGH")
    )
    spread_low, spread_high = map(float, thresholds["relative_spread"])
    frame["liquidity_regime"] = _three_regime(
        frame["relative_spread"],
        spread_low,
        spread_high,
        ("NARROW", "MEDIUM", "WIDE"),
    )
    volume_low, volume_high = thresholds["positive_volume"]
    frame["volume_regime"] = np.where(
        frame["volume"] <= 0,
        "ZERO",
        np.where(
            frame["volume"] <= volume_low,
            "LOW",
            np.where(frame["volume"] <= volume_high, "MEDIUM", "HIGH"),
        ),
    )
    oi_low, oi_high = thresholds["positive_open_interest"]
    frame["open_interest_regime"] = np.where(
        frame["open_interest"] <= 0,
        "ZERO",
        np.where(
            frame["open_interest"] <= oi_low,
            "LOW",
            np.where(frame["open_interest"] <= oi_high, "MEDIUM", "HIGH"),
        ),
    )
    return frame


def _metrics(group: pd.DataFrame) -> dict[str, float | int]:
    residual = group["residual"].to_numpy(dtype=float)
    normalized = group["normalized_error"].to_numpy(dtype=float)
    iv_error = (group["model_iv"] - group["market_iv"]).to_numpy(dtype=float)
    finite_iv = np.isfinite(iv_error)
    result: dict[str, float | int] = {
        "count": len(group),
        "market_dates": int(group["quote_date"].nunique()),
        "rmse": float(np.sqrt(np.mean(np.square(residual)))),
        "mae": float(np.mean(np.abs(residual))),
        "mean_signed_error": float(np.mean(residual)),
        "median_signed_error": float(np.median(residual)),
        "iv_rmse": float(np.sqrt(np.mean(np.square(iv_error[finite_iv]))))
        if np.any(finite_iv)
        else np.nan,
        "spread_normalized_mae": float(np.mean(np.abs(normalized))),
        "p90_normalized_error": float(np.quantile(np.abs(normalized), 0.90)),
        "p95_normalized_error": float(np.quantile(np.abs(normalized), 0.95)),
        "within_bid_ask_rate": float(group["inside_bid_ask"].mean()),
        "calibration_failure_rate": float(1.0 - group["calibration_success"].mean()),
    }
    for greek in ("delta", "gamma", "vega"):
        column = f"{greek}_disagreement"
        result[column] = float(group[column].mean()) if column in group else np.nan
    return result


def _attach_heston_greek_disagreement(rows: pd.DataFrame) -> pd.DataFrame:
    keys = ["quote_date", "split", "settlement_class", "source_row_id"]
    heston = rows[rows["model_id"] == "M10_HESTON"][[*keys, "delta", "gamma", "vega"]].rename(
        columns={name: f"heston_{name}" for name in ("delta", "gamma", "vega")}
    )
    enriched = rows.merge(heston, on=keys, how="left", validate="many_to_one")
    for greek in ("delta", "gamma", "vega"):
        enriched[f"{greek}_disagreement"] = np.abs(enriched[greek] - enriched[f"heston_{greek}"])
    return enriched.drop(columns=[f"heston_{name}" for name in ("delta", "gamma", "vega")])


BOOTSTRAP_METRICS = (
    "rmse",
    "mae",
    "mean_signed_error",
    "median_signed_error",
    "iv_rmse",
    "spread_normalized_mae",
    "p90_normalized_error",
    "p95_normalized_error",
    "within_bid_ask_rate",
    "delta_disagreement",
    "gamma_disagreement",
    "vega_disagreement",
)


def _block_intervals(group: pd.DataFrame, draws: int, seed: int) -> dict[str, tuple[float, float]]:
    dates = np.asarray(sorted(group["quote_date"].unique()))
    if dates.size < 2:
        return {metric: (np.nan, np.nan) for metric in BOOTSTRAP_METRICS}
    blocks: dict[object, dict[str, FloatArray]] = {}
    for date in dates:
        block = group[group["quote_date"] == date]
        blocks[date] = {
            "residual": block["residual"].to_numpy(dtype=np.float64),
            "normalized_error": block["normalized_error"].to_numpy(dtype=np.float64),
            "iv_error": (block["model_iv"] - block["market_iv"]).to_numpy(dtype=np.float64),
            "inside_bid_ask": block["inside_bid_ask"].to_numpy(dtype=np.float64),
            "delta_disagreement": block["delta_disagreement"].to_numpy(dtype=np.float64),
            "gamma_disagreement": block["gamma_disagreement"].to_numpy(dtype=np.float64),
            "vega_disagreement": block["vega_disagreement"].to_numpy(dtype=np.float64),
        }
    rng = np.random.default_rng(seed)
    values = {metric: np.empty(draws, dtype=np.float64) for metric in BOOTSTRAP_METRICS}
    for draw in range(draws):
        sampled_dates = dates[rng.integers(0, len(dates), size=len(dates))]
        residual = np.concatenate([blocks[date]["residual"] for date in sampled_dates])
        normalized = np.concatenate([blocks[date]["normalized_error"] for date in sampled_dates])
        absolute_normalized = np.abs(normalized)
        iv_error = np.concatenate([blocks[date]["iv_error"] for date in sampled_dates])
        finite = np.isfinite(iv_error)
        values["rmse"][draw] = np.sqrt(np.mean(np.square(residual)))
        values["mae"][draw] = np.mean(np.abs(residual))
        values["mean_signed_error"][draw] = np.mean(residual)
        values["median_signed_error"][draw] = np.median(residual)
        values["iv_rmse"][draw] = (
            np.sqrt(np.mean(np.square(iv_error[finite]))) if np.any(finite) else np.nan
        )
        values["spread_normalized_mae"][draw] = np.mean(absolute_normalized)
        values["p90_normalized_error"][draw] = np.quantile(absolute_normalized, 0.90)
        values["p95_normalized_error"][draw] = np.quantile(absolute_normalized, 0.95)
        values["within_bid_ask_rate"][draw] = np.mean(
            np.concatenate([blocks[date]["inside_bid_ask"] for date in sampled_dates])
        )
        for metric in ("delta_disagreement", "gamma_disagreement", "vega_disagreement"):
            values[metric][draw] = np.mean(
                np.concatenate([blocks[date][metric] for date in sampled_dates])
            )
    return {
        metric: (float(np.nanquantile(value, 0.025)), float(np.nanquantile(value, 0.975)))
        for metric, value in values.items()
    }


def slice_metrics(rows: pd.DataFrame, config: AnalyticsConfig) -> pd.DataFrame:
    rows = _attach_heston_greek_disagreement(rows)
    dimensions = (
        "moneyness_slice",
        "maturity_slice",
        "vix_regime",
        "liquidity_regime",
        "volume_regime",
        "open_interest_regime",
        "option_type",
        "settlement_class",
        "data_quality_tier",
    )
    records: list[dict[str, object]] = []
    for dimension in dimensions:
        for keys, group in rows.groupby(["split", "model_id", dimension], dropna=False):
            split, model_id, value = cast(tuple[str, str, str], keys)
            result: dict[str, object] = {
                "split": split,
                "model_id": model_id,
                "slice_dimension": dimension,
                "slice_value": value,
                **_metrics(group),
            }
            result["defensible"] = bool(
                len(group) >= config.minimum_slice_rows
                and group["quote_date"].nunique() >= config.minimum_slice_dates
            )
            intervals = (
                _block_intervals(group, config.bootstrap_draws, config.seed + len(records) * 17)
                if result["defensible"]
                else {metric: (np.nan, np.nan) for metric in BOOTSTRAP_METRICS}
            )
            for metric, (low, high) in intervals.items():
                result[f"{metric}_ci_low"] = low
                result[f"{metric}_ci_high"] = high
            records.append(result)
    return pd.DataFrame(records)


def two_dimensional_metrics(rows: pd.DataFrame, config: AnalyticsConfig) -> pd.DataFrame:
    rows = _attach_heston_greek_disagreement(rows)
    pairs = (
        ("moneyness_slice", "maturity_slice"),
        ("moneyness_slice", "vix_regime"),
        ("maturity_slice", "vix_regime"),
        ("moneyness_slice", "liquidity_regime"),
        ("maturity_slice", "liquidity_regime"),
    )
    records: list[dict[str, object]] = []
    for first, second in pairs:
        for keys, group in rows.groupby(["split", "model_id", first, second], dropna=False):
            split, model, left, right = cast(tuple[str, str, str, str], keys)
            result: dict[str, object] = {
                "split": split,
                "model_id": model,
                "dimension_1": first,
                "value_1": left,
                "dimension_2": second,
                "value_2": right,
                **_metrics(group),
                "defensible": bool(
                    len(group) >= config.minimum_slice_rows
                    and group["quote_date"].nunique() >= config.minimum_slice_dates
                ),
            }
            intervals = (
                _block_intervals(group, config.bootstrap_draws, config.seed + len(records) * 29)
                if result["defensible"]
                else {metric: (np.nan, np.nan) for metric in BOOTSTRAP_METRICS}
            )
            for metric, (low, high) in intervals.items():
                result[f"{metric}_ci_low"] = low
                result[f"{metric}_ci_high"] = high
            records.append(result)
    return pd.DataFrame(records)


def _plot_heatmap(
    table: pd.DataFrame,
    value: str,
    title: str,
    path: Path,
    *,
    center: float | None = None,
) -> None:
    if table.empty:
        return
    pivot = table.pivot(index="value_1", columns="value_2", values=value)
    fig, ax = plt.subplots(figsize=(9, 5.5))
    sns.heatmap(
        pivot,
        annot=True,
        fmt=".2f",
        cmap="coolwarm" if center is not None else "viridis",
        center=center,
        ax=ax,
    )
    ax.set_title(title)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def disagreement_tables(
    rows: pd.DataFrame, frozen: dict[str, Any]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    keys = ["quote_date", "split", "settlement_class", "source_row_id"]
    wide = rows.pivot(
        index=keys, columns="model_id", values=["model_price", "delta", "gamma", "vega"]
    ).reset_index()
    wide.columns = ["__".join(filter(None, map(str, item))) for item in wide.columns]
    base = rows[rows["model_id"] == "M10_HESTON"].copy()
    records: list[dict[str, object]] = []
    for challenger in ("M02_SVI", "M03_SSVI", "M21_BATES"):
        selected = base[
            [*keys, "moneyness_slice", "maturity_slice", "vix_regime", "liquidity_regime"]
        ].copy()
        value_columns = [
            f"{measure}__{model}"
            for measure in ("model_price", "delta", "gamma", "vega")
            for model in ("M10_HESTON", challenger)
        ]
        selected = selected.merge(
            wide[[*keys, *value_columns]], on=keys, how="inner", validate="one_to_one"
        )
        for measure in ("model_price", "delta", "gamma", "vega"):
            selected[f"{measure}_disagreement"] = np.abs(
                selected[f"{measure}__M10_HESTON"] - selected[f"{measure}__{challenger}"]
            )
        selected = selected.drop(columns=value_columns)
        selected["challenger"] = challenger
        records.extend(cast(list[dict[str, object]], selected.to_dict("records")))
    disagreement = pd.DataFrame(records)
    dev = disagreement[disagreement["split"] == "DEV"]
    val_threshold = float(dev["model_price_disagreement"].quantile(0.25))
    percentile_columns = []
    for measure in ("delta_disagreement", "gamma_disagreement", "vega_disagreement"):
        reference = np.sort(dev[measure].to_numpy(dtype=float))
        percentile = np.searchsorted(
            reference, disagreement[measure].to_numpy(dtype=float), side="right"
        ) / max(len(reference), 1)
        column = f"{measure}_dev_percentile"
        disagreement[column] = percentile
        percentile_columns.append(column)
    greek_score = disagreement.loc[disagreement["split"] == "DEV", percentile_columns].max(axis=1)
    greek_threshold = float(greek_score.quantile(0.75))
    all_score = disagreement[percentile_columns].max(axis=1)
    finding = disagreement[
        (disagreement["model_price_disagreement"] <= val_threshold) & (all_score >= greek_threshold)
    ].copy()
    finding["selection_rule"] = (
        f"valuation disagreement <= DEV q25 ({val_threshold:.6g}); "
        f"max Greek percentile >= DEV q75 ({greek_threshold:.6g})"
    )
    frozen["low_value_high_greek_rule"] = {
        "valuation_q25": val_threshold,
        "greek_percentile_q75": greek_threshold,
    }
    return disagreement, finding


def model_winner_tables(
    rows: pd.DataFrame, config: AnalyticsConfig
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    qualified = rows[rows["model_id"].isin(WINNER_MODELS)].copy()
    dimensions = ["split", "moneyness_slice", "maturity_slice", "model_id"]
    summary = qualified.groupby(dimensions, as_index=False).agg(
        count=("residual", "size"),
        dates=("quote_date", "nunique"),
        error=("normalized_error", lambda x: float(np.mean(np.abs(x)))),
    )
    valid = summary[
        (summary["count"] >= config.minimum_slice_rows)
        & (summary["dates"] >= config.minimum_slice_dates)
    ]
    winners: list[dict[str, object]] = []
    for keys, group in valid.groupby(["split", "moneyness_slice", "maturity_slice"]):
        ordered = group.sort_values(["error", "model_id"])
        if len(ordered) < 2:
            continue
        selected_winner = str(ordered.iloc[0]["model_id"])
        selected_runner_up = str(ordered.iloc[1]["model_id"])
        cell = qualified[
            (qualified["split"] == keys[0])
            & (qualified["moneyness_slice"] == keys[1])
            & (qualified["maturity_slice"] == keys[2])
        ]
        dates = np.asarray(sorted(cell["quote_date"].unique()))
        rng = np.random.default_rng(config.seed + len(winners) * 43)
        bootstrap_winners: list[str] = []
        bootstrap_margins: list[float] = []
        for _ in range(config.bootstrap_draws):
            sampled_dates = dates[rng.integers(0, len(dates), size=len(dates))]
            sampled = pd.concat(
                [cell[cell["quote_date"] == date] for date in sampled_dates], ignore_index=True
            )
            errors = sampled.groupby("model_id")["normalized_error"].apply(
                lambda value: float(np.mean(np.abs(value)))
            )
            ranked = errors.sort_values()
            bootstrap_winners.append(str(ranked.index[0]))
            bootstrap_margins.append(float(errors[selected_runner_up] - errors[selected_winner]))
        winners.append(
            {
                "split": keys[0],
                "moneyness_slice": keys[1],
                "maturity_slice": keys[2],
                "winner": ordered.iloc[0]["model_id"],
                "second_best": ordered.iloc[1]["model_id"],
                "winner_error": ordered.iloc[0]["error"],
                "second_error": ordered.iloc[1]["error"],
                "winner_margin": ordered.iloc[1]["error"] - ordered.iloc[0]["error"],
                "winner_bootstrap_probability": float(
                    np.mean(np.asarray(bootstrap_winners) == selected_winner)
                ),
                "winner_margin_ci_low": float(np.quantile(bootstrap_margins, 0.025)),
                "winner_margin_ci_high": float(np.quantile(bootstrap_margins, 0.975)),
            }
        )
    winner = pd.DataFrame(winners)
    date_errors = qualified.groupby(["split", "quote_date", "model_id"], as_index=False).agg(
        error=("normalized_error", lambda x: float(np.mean(np.abs(x))))
    )
    date_errors["date_winner"] = (
        date_errors.groupby(["split", "quote_date"])["error"].transform("min")
        == date_errors["error"]
    )
    date_win = date_errors.groupby(["split", "model_id"], as_index=False).agg(
        date_level_win_rate=("date_winner", "mean"), dates=("quote_date", "nunique")
    )
    date_ci: list[tuple[float, float]] = []
    for item in date_win.itertuples(index=False):
        values = date_errors[
            (date_errors["split"] == item.split) & (date_errors["model_id"] == item.model_id)
        ]["date_winner"].to_numpy(dtype=float)
        rng = np.random.default_rng(config.seed + len(date_ci) * 47)
        draws = np.asarray(
            [
                rng.choice(values, size=len(values), replace=True).mean()
                for _ in range(config.bootstrap_draws)
            ]
        )
        date_ci.append((float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))))
    date_win["win_rate_ci_low"] = [value[0] for value in date_ci]
    date_win["win_rate_ci_high"] = [value[1] for value in date_ci]
    pairwise: list[dict[str, object]] = []
    pivot = date_errors.pivot(index=["split", "quote_date"], columns="model_id", values="error")
    for split in rows["split"].unique():
        block = pivot.loc[split]
        for left in WINNER_MODELS:
            for right in WINNER_MODELS:
                if left == right or left not in block or right not in block:
                    continue
                difference = (block[left] < block[right]).dropna()
                rng = np.random.default_rng(config.seed + len(pairwise) * 53)
                samples = np.asarray(
                    [
                        rng.choice(
                            difference.to_numpy(dtype=float), size=len(difference), replace=True
                        ).mean()
                        for _ in range(config.bootstrap_draws)
                    ]
                )
                pairwise.append(
                    {
                        "split": split,
                        "model": left,
                        "opponent": right,
                        "win_rate": float(difference.mean()),
                        "dates": int(difference.size),
                        "win_rate_ci_low": float(np.quantile(samples, 0.025)),
                        "win_rate_ci_high": float(np.quantile(samples, 0.975)),
                    }
                )
    return winner, pd.DataFrame(pairwise), date_win


def calibration_diagnostics(
    root: Path, rows: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    comparison = pd.read_parquet(root / "results/raw/real_model_comparison.parquet")
    heston = comparison[
        (comparison["model_id"] == "M10_HESTON") & (comparison["experiment"] == "cross_sectional")
    ].copy()
    records: list[dict[str, object]] = []
    for item in heston.itertuples(index=False):
        subset = rows[
            (rows["model_id"] == "M10_HESTON")
            & (rows["quote_date"] == item.quote_date)
            & (rows["settlement_class"] == item.settlement_class)
        ]
        record: dict[str, object] = {
            "quote_date": item.quote_date,
            "split": item.split,
            "settlement_class": item.settlement_class,
            "objective_value": item.objective_value,
            "convergence_status": bool(item.converged),
            "optimizer_message": item.optimizer_message,
            "price_rmse": item.price_rmse,
            "price_mae": item.price_mae,
            "inside_bid_ask_rate": item.inside_bid_ask_rate,
            "calibration_runtime_seconds": item.calibration_runtime_seconds,
            "iterations": np.nan,
            "evaluations": np.nan,
        }
        if pd.notna(item.checkpoint) and (root / str(item.checkpoint)).exists():
            checkpoint = _json(root / str(item.checkpoint))
            checkpoint_calibration = cast(dict[str, Any], checkpoint["calibration"])
            parameter = _parameters(checkpoint)
            vector = np.asarray([getattr(parameter, name) for name in PARAMETER_NAMES])
            distance = np.minimum((vector - BOUNDS[:, 0]), (BOUNDS[:, 1] - vector)) / (
                BOUNDS[:, 1] - BOUNDS[:, 0]
            )
            record.update(
                {
                    "iterations": checkpoint_calibration["iterations"],
                    "evaluations": checkpoint_calibration["evaluations"],
                    "boundary_proximity_min_fraction": float(np.min(distance)),
                    "nearest_boundary_parameter": PARAMETER_NAMES[int(np.argmin(distance))],
                    "feller_ratio": parameter.feller_ratio,
                    "feller_satisfied": parameter.feller_ratio >= 1.0,
                    **{name: getattr(parameter, name) for name in PARAMETER_NAMES},
                }
            )
        if not subset.empty:
            iv_error = (subset["model_iv"] - subset["market_iv"]).to_numpy(dtype=float)
            valid = np.isfinite(iv_error)
            weights = np.maximum(subset["market_vega"].to_numpy(dtype=float)[valid], 1.0e-12)
            record.update(
                {
                    "iv_rmse": float(np.sqrt(np.mean(np.square(iv_error[valid]))))
                    if np.any(valid)
                    else np.nan,
                    "vega_weighted_iv_error": float(
                        np.sqrt(np.average(np.square(iv_error[valid]), weights=weights))
                    )
                    if np.any(valid)
                    else np.nan,
                    "spread_normalized_mae": float(np.mean(np.abs(subset["normalized_error"]))),
                    "signed_residual_bias": float(np.mean(subset["residual"])),
                    "moneyness_support": float(
                        subset["log_moneyness"].max() - subset["log_moneyness"].min()
                    ),
                    "maturity_support_days": int(subset["dte"].max() - subset["dte"].min()),
                    "median_relative_spread": float(subset["relative_spread"].median()),
                    "VIXCLS": float(subset["VIXCLS"].median()),
                }
            )
        records.append(record)
    calibration = pd.DataFrame(records)
    bootstrap = pd.read_parquet(root / "results/aggregated/bootstrap_calibration.parquet")
    parameters = bootstrap[list(PARAMETER_NAMES)].dropna()
    uncertainty = pd.DataFrame(
        [
            {
                "parameter": name,
                "bootstrap_draws": len(parameters),
                "mean": parameters[name].mean(),
                "std": parameters[name].std(ddof=1),
                "q025": parameters[name].quantile(0.025),
                "q975": parameters[name].quantile(0.975),
            }
            for name in PARAMETER_NAMES
        ]
    )
    correlation = (
        parameters.corr()
        .rename_axis("parameter")
        .reset_index()
        .melt("parameter", var_name="parameter_2", value_name="correlation")
    )
    return calibration, uncertainty, correlation


def calibration_slice_analysis(
    root: Path, calibration: pd.DataFrame, frozen: dict[str, Any]
) -> pd.DataFrame:
    """Summarize calibration quality by predeclared or DEV-frozen support regimes."""

    frame = calibration.copy()
    dev = frame[frame["split"] == "DEV"]
    support_thresholds = {
        "moneyness_support": [
            float(dev["moneyness_support"].quantile(1 / 3)),
            float(dev["moneyness_support"].quantile(2 / 3)),
        ],
        "maturity_support_days": [
            float(dev["maturity_support_days"].quantile(1 / 3)),
            float(dev["maturity_support_days"].quantile(2 / 3)),
        ],
    }
    registry = {
        "status": "FROZEN_FROM_DEV",
        "support_thresholds": support_thresholds,
        "vix_thresholds": frozen["learned_thresholds"]["vix"],
        "liquidity_thresholds": frozen["learned_thresholds"]["relative_spread"],
    }
    registry_path = root / "artifacts/v1_1/calibration_slice_registry.freeze.json"
    if registry_path.exists():
        if _json(registry_path) != registry:
            raise RuntimeError("calibration slice registry mismatch; refusing to overwrite")
    else:
        _write_json(registry_path, registry)
    vix_low, vix_high = map(float, registry["vix_thresholds"])
    frame["vix_regime"] = _three_regime(
        frame["VIXCLS"], vix_low, vix_high, ("LOW", "MEDIUM", "HIGH")
    )
    liquidity_low, liquidity_high = map(float, registry["liquidity_thresholds"])
    frame["liquidity_regime"] = _three_regime(
        frame["median_relative_spread"],
        liquidity_low,
        liquidity_high,
        ("NARROW", "MEDIUM", "WIDE"),
    )
    money_low, money_high = support_thresholds["moneyness_support"]
    frame["moneyness_support_regime"] = _three_regime(
        frame["moneyness_support"],
        money_low,
        money_high,
        ("NARROW", "MEDIUM", "BROAD"),
    )
    maturity_low, maturity_high = support_thresholds["maturity_support_days"]
    frame["maturity_support_regime"] = _three_regime(
        frame["maturity_support_days"],
        maturity_low,
        maturity_high,
        ("SHORT", "MEDIUM", "BROAD"),
    )
    records = []
    for dimension in (
        "quote_date",
        "vix_regime",
        "liquidity_regime",
        "moneyness_support_regime",
        "maturity_support_regime",
    ):
        for keys, group in frame.groupby(["split", dimension], dropna=False):
            records.append(
                {
                    "split": keys[0],
                    "dimension": dimension,
                    "value": keys[1],
                    "calibrations": len(group),
                    "market_dates": group["quote_date"].nunique(),
                    "median_objective": float(group["objective_value"].median()),
                    "median_price_rmse": float(group["price_rmse"].median()),
                    "median_iv_rmse": float(group["iv_rmse"].median()),
                    "median_vega_weighted_iv_error": float(
                        group["vega_weighted_iv_error"].median()
                    ),
                    "median_spread_normalized_mae": float(group["spread_normalized_mae"].median()),
                    "median_inside_bid_ask_rate": float(group["inside_bid_ask_rate"].median()),
                    "mean_signed_residual_bias": float(group["signed_residual_bias"].mean()),
                    "feller_rate": float(group["feller_satisfied"].mean()),
                }
            )
    return pd.DataFrame(records)


def parameter_stability(
    root: Path, calibration: pd.DataFrame, rows: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    frame = calibration[
        (calibration["settlement_class"] == "PM") & calibration["convergence_status"]
    ].copy()
    frame["quote_date"] = pd.to_datetime(frame["quote_date"])
    frame = frame.sort_values("quote_date")
    date_disagreement = rows.pivot_table(
        index="quote_date", columns="model_id", values="model_price", aggfunc="mean"
    )
    date_disagreement["challenger_disagreement"] = (
        date_disagreement[["M02_SVI", "M03_SSVI", "M21_BATES"]]
        .sub(date_disagreement["M10_HESTON"], axis=0)
        .abs()
        .mean(axis=1)
    )
    frame = frame.merge(
        date_disagreement[["challenger_disagreement"]],
        left_on=frame["quote_date"].astype(str),
        right_index=True,
        how="left",
    ).drop(columns="key_0")
    for name in PARAMETER_NAMES:
        frame[f"{name}_change"] = frame[name].diff()
        median = frame[name].rolling(21, min_periods=5).median()
        mad = (frame[name] - median).abs().rolling(21, min_periods=5).median()
        frame[f"{name}_rolling_median"] = median
        frame[f"{name}_rolling_mad"] = mad
        frame[f"{name}_robust_z"] = (frame[name] - median) / (1.4826 * mad.replace(0.0, np.nan))
    robust_columns = [f"{name}_robust_z" for name in PARAMETER_NAMES]
    frame["parameter_jump_score_v1_1"] = frame[robust_columns].abs().max(axis=1)
    correlation_rows: list[dict[str, object]] = []
    targets = ["VIXCLS", "median_relative_spread", "price_rmse", "challenger_disagreement"]
    for name in PARAMETER_NAMES:
        for target in targets:
            valid = frame[[name, target]].dropna()
            value, pvalue = (
                spearmanr(valid[name], valid[target]) if len(valid) >= 3 else (np.nan, np.nan)
            )
            correlation_rows.append(
                {
                    "parameter": name,
                    "target": target,
                    "spearman_correlation": value,
                    "p_value_exploratory": pvalue,
                    "dates": len(valid),
                }
            )
    similar: list[dict[str, object]] = []
    values = frame[list(PARAMETER_NAMES)].to_numpy(dtype=float)
    fit = frame["price_rmse"].to_numpy(dtype=float)
    for left in range(len(frame)):
        for right in range(left + 1, len(frame)):
            fit_gap = abs(fit[left] - fit[right]) / max(fit[left], fit[right], 1.0e-12)
            scaled_distance = float(
                np.linalg.norm((values[left] - values[right]) / (BOUNDS[:, 1] - BOUNDS[:, 0]))
            )
            if fit_gap <= 0.05 and scaled_distance >= 0.20:
                similar.append(
                    {
                        "date_1": frame.iloc[left]["quote_date"],
                        "date_2": frame.iloc[right]["quote_date"],
                        "relative_fit_gap": fit_gap,
                        "scaled_parameter_distance": scaled_distance,
                    }
                )
    return (
        frame,
        pd.DataFrame(correlation_rows),
        pd.DataFrame(similar).sort_values("scaled_parameter_distance", ascending=False)
        if similar
        else pd.DataFrame(
            columns=["date_1", "date_2", "relative_fit_gap", "scaled_parameter_distance"]
        ),
    )


def fit_segmentation_trees(rows: pd.DataFrame, config: AnalyticsConfig) -> tuple[pd.DataFrame, str]:
    heston = rows[rows["model_id"] == "M10_HESTON"].copy()
    svi = rows[rows["model_id"] == "M02_SVI"][
        ["quote_date", "source_row_id", "model_price", "delta", "gamma", "vega"]
    ].rename(
        columns={
            "model_price": "svi_price",
            "delta": "svi_delta",
            "gamma": "svi_gamma",
            "vega": "svi_vega",
        }
    )
    challengers = (
        rows[rows["model_id"].isin(["M02_SVI", "M03_SSVI", "M21_BATES"])]
        .groupby(["quote_date", "source_row_id"], as_index=False)
        .agg(challenger_mean=("model_price", "mean"), challenger_dispersion=("model_price", "std"))
    )
    frame = heston.merge(challengers, on=["quote_date", "source_row_id"], how="left")
    frame = frame.merge(svi, on=["quote_date", "source_row_id"], how="left")
    frame["heston_svi_difference"] = np.abs(frame["model_price"] - frame["svi_price"])
    greek_components = [
        np.abs(frame[measure] - frame[f"svi_{measure}"])
        / np.maximum(np.abs(frame[measure]), 1.0e-8)
        for measure in ("delta", "gamma", "vega")
    ]
    frame["greek_disagreement"] = np.column_stack(greek_components).max(axis=1)
    frame["settlement_code"] = (frame["settlement_class"] == "AM").astype(float)
    frame["option_code"] = (frame["option_type"] == "call").astype(float)
    features = [
        "log_moneyness",
        "dte",
        "VIXCLS",
        "relative_spread",
        "volume",
        "open_interest",
        "settlement_code",
        "option_code",
    ]
    targets = [
        "absolute_error",
        "heston_svi_difference",
        "challenger_dispersion",
        "greek_disagreement",
    ]
    output: list[pd.DataFrame] = []
    descriptions: list[str] = []
    for target in targets:
        dev = frame[frame["split"] == "DEV"].dropna(subset=[*features, target])
        tree = DecisionTreeRegressor(
            max_depth=config.tree_max_depth,
            min_samples_leaf=config.tree_min_leaf,
            random_state=config.seed,
        ).fit(dev[features], dev[target])
        descriptions.append(
            f"## {target}\n\n```text\n{export_text(tree, feature_names=features)}\n```"
        )
        for split in ("DEV", "VALIDATION", "LOCKED_TEST"):
            subset = frame[frame["split"] == split].dropna(subset=[*features, target]).copy()
            subset["leaf_id"] = tree.apply(subset[features])
            summary = subset.groupby("leaf_id", as_index=False).agg(
                count=(target, "size"),
                dates=("quote_date", "nunique"),
                target_mean=(target, "mean"),
                target_median=(target, "median"),
            )
            summary["split"] = split
            summary["target"] = target
            output.append(summary)
    return pd.concat(output, ignore_index=True), "\n\n".join(descriptions)


def portfolio_slices(root: Path, frozen: dict[str, Any]) -> pd.DataFrame:
    frame = pd.read_parquet(root / "results/materiality/portfolio_materiality.parquet")
    covariates = pd.read_parquet(root / "data/processed/public_covariates.parquet")[
        ["quote_date", "VIXCLS"]
    ]
    covariates["quote_date"] = covariates["quote_date"].astype(str)
    frame = frame.merge(covariates.drop_duplicates("quote_date"), on="quote_date", how="left")
    vix_low, vix_high = map(float, frozen["learned_thresholds"]["vix"])
    frame["vix_regime"] = _three_regime(
        frame["VIXCLS"], vix_low, vix_high, ("LOW", "MEDIUM", "HIGH")
    )
    maturity = {
        "P01": "FRONT_MATURITY",
        "P02": "FRONT_MATURITY",
        "P03": "FRONT_MATURITY",
        "P04": "TERM_STRUCTURE",
        "P05": "FRONT_MATURITY",
        "P06": "MIXED_FRONT_BACK",
    }
    moneyness = {
        "P01": "ATM_DOMINANT",
        "P02": "WING_OR_SKEW",
        "P03": "WING_STRUCTURE",
        "P04": "ATM_DOMINANT",
        "P05": "WING_OR_SKEW",
        "P06": "MIXED_ATM_WING",
    }
    frame["maturity_exposure"] = frame["portfolio_id"].map(maturity)
    frame["moneyness_exposure"] = frame["portfolio_id"].map(moneyness)
    # Portfolio artifacts do not retain leg quote spreads.  Keep the dimension
    # explicit and unavailable.
    frame["liquidity_regime"] = "UNAVAILABLE_LEG_LEVEL_SPREAD"
    return frame


def monitoring_table(
    calibration: pd.DataFrame,
    stability: pd.DataFrame,
    rows: pd.DataFrame,
    portfolio: pd.DataFrame,
    config: AnalyticsConfig,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    heston = (
        rows[rows["model_id"] == "M10_HESTON"]
        .groupby(["quote_date", "split"], as_index=False)
        .agg(spread_normalized_error=("normalized_error", lambda x: float(np.mean(np.abs(x)))))
    )
    disagreement = rows.pivot_table(
        index=["quote_date", "split"], columns="model_id", values="model_price", aggfunc="mean"
    ).reset_index()
    disagreement["E_form"] = (
        disagreement[["M02_SVI", "M03_SSVI", "M21_BATES"]]
        .sub(disagreement["M10_HESTON"], axis=0)
        .abs()
        .mean(axis=1)
    )
    risk = rows.pivot_table(
        index=["quote_date", "split", "source_row_id"],
        columns="model_id",
        values=["delta", "gamma", "vega"],
    )
    greek_parts: list[pd.Series] = []
    for measure in ("delta", "gamma", "vega"):
        developer = risk[(measure, "M10_HESTON")]
        challenger_columns = [
            (measure, model)
            for model in ("M02_SVI", "M03_SSVI", "M21_BATES")
            if (measure, model) in risk
        ]
        scale = np.maximum(np.abs(developer), 1.0e-8)
        greek_parts.append(
            risk[challenger_columns].sub(developer, axis=0).abs().div(scale, axis=0).mean(axis=1)
        )
    greek = pd.concat(greek_parts, axis=1).median(axis=1).rename("E_greek").reset_index()
    greek_grouped = cast(
        pd.DataFrame,
        greek.groupby(["quote_date", "split"], as_index=False)["E_greek"].median(),
    )
    portfolio_monitor = portfolio.groupby(["quote_date", "split"], as_index=False).agg(
        portfolio_materiality=("valuation_materiality", "max")
    )
    frame = calibration[calibration["settlement_class"] == "PM"][
        ["quote_date", "split", "objective_value"]
    ].rename(columns={"objective_value": "calibration_loss"})
    frame = frame.merge(heston, on=["quote_date", "split"], how="left").merge(
        disagreement[["quote_date", "split", "E_form"]], on=["quote_date", "split"], how="left"
    )
    frame = frame.merge(greek_grouped, on=["quote_date", "split"], how="left")
    frame = frame.merge(portfolio_monitor, on=["quote_date", "split"], how="left")
    jumps = stability.copy()
    jumps["quote_date"] = jumps["quote_date"].astype(str)
    frame = frame.merge(
        jumps[["quote_date", "parameter_jump_score_v1_1"]], on="quote_date", how="left"
    )
    measures = [
        "calibration_loss",
        "spread_normalized_error",
        "parameter_jump_score_v1_1",
        "E_form",
        "E_greek",
        "portfolio_materiality",
    ]
    registry: dict[str, Any] = {}
    frame = frame.sort_values("quote_date")
    for measure in measures:
        dev = frame.loc[frame["split"] == "DEV", measure].dropna()
        if dev.empty:
            registry[measure] = {
                "status": "UNAVAILABLE_NO_DEV_REFERENCE_DISTRIBUTION",
                "dev_median": None,
                "dev_mad": None,
                "amber": None,
                "red": None,
            }
            for suffix in ("rolling_median", "rolling_mad", "ewma", "cusum"):
                frame[f"{measure}_{suffix}"] = np.nan
            frame[f"{measure}_status"] = "UNAVAILABLE"
            continue
        center = float(dev.median())
        mad = float((dev - center).abs().median())
        scale = max(1.4826 * mad, 1.0e-12)
        amber, red = center + 3.0 * scale, center + 5.0 * scale
        cusum_amber = 0.5 * config.cusum_h_mad * scale
        cusum_red = config.cusum_h_mad * scale
        registry[measure] = {
            "dev_median": center,
            "dev_mad": mad,
            "amber": amber,
            "red": red,
            "cusum_amber": cusum_amber,
            "cusum_red": cusum_red,
        }
        frame[f"{measure}_rolling_median"] = frame[measure].rolling(21, min_periods=5).median()
        frame[f"{measure}_rolling_mad"] = (
            (frame[measure] - frame[f"{measure}_rolling_median"])
            .abs()
            .rolling(21, min_periods=5)
            .median()
        )
        frame[f"{measure}_ewma"] = frame[measure].ewm(alpha=config.ewma_lambda, adjust=False).mean()
        centered = frame[measure].fillna(center) - center - config.cusum_k_mad * scale
        cusum: list[float] = []
        state = 0.0
        for value in centered:
            state = max(0.0, state + float(value))
            cusum.append(state)
        frame[f"{measure}_cusum"] = cusum
        raw_status = np.where(
            frame[measure] >= red, "RED", np.where(frame[measure] >= amber, "AMBER", "GREEN")
        )
        frame[f"{measure}_status"] = np.where(
            frame[f"{measure}_cusum"] >= cusum_red,
            "RED",
            np.where(frame[f"{measure}_cusum"] >= cusum_amber, "AMBER", raw_status),
        )
    return frame, registry


def _report(path: Path, title: str, sections: Iterable[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("# " + title + "\n\n" + "\n\n".join(sections).strip() + "\n", encoding="utf-8")


def _table_preview(frame: pd.DataFrame, columns: Sequence[str], rows: int = 12) -> str:
    selected = [column for column in columns if column in frame]
    if not selected or frame.empty:
        return "No defensible rows available."
    preview = frame[selected].head(rows).copy()

    def clean(value: object) -> str:
        if value is None or value is pd.NA or value is pd.NaT:
            return "NA"
        if isinstance(value, float):
            if np.isnan(value):
                return "NA"
            return f"{value:.6g}"
        return str(value).replace("|", "\\|").replace("\n", " ")

    header = "| " + " | ".join(selected) + " |"
    separator = "| " + " | ".join("---" for _ in selected) + " |"
    body = [
        "| " + " | ".join(clean(value) for value in row) + " |"
        for row in preview.itertuples(index=False, name=None)
    ]
    return "\n".join([header, separator, *body])


def generate_reports(
    root: Path,
    rows: pd.DataFrame,
    slices: pd.DataFrame,
    two_dim: pd.DataFrame,
    calibration: pd.DataFrame,
    winners: pd.DataFrame,
    pairwise: pd.DataFrame,
    stability: pd.DataFrame,
    correlations: pd.DataFrame,
    similar: pd.DataFrame,
    disagreement: pd.DataFrame,
    low_value_high_greek: pd.DataFrame,
    portfolio: pd.DataFrame,
    monitoring: pd.DataFrame,
    trees: pd.DataFrame,
    tree_text: str,
) -> None:
    report = root / "reports/v1_1"
    limitation = (
        "All real-data findings are cross-sectional on deterministic released-calibration samples. "
        "Historical quotes lack quote timestamps, so temporal OOS, hedge outcomes, "
        "and P&L remain unavailable."
    )
    _report(
        report / "CALIBRATION_DIAGNOSTICS.md",
        "Calibration Diagnostics",
        [
            limitation,
            _table_preview(
                calibration,
                [
                    "quote_date",
                    "split",
                    "objective_value",
                    "price_rmse",
                    "iv_rmse",
                    "spread_normalized_mae",
                    "inside_bid_ask_rate",
                    "feller_satisfied",
                    "boundary_proximity_min_fraction",
                ],
            ),
        ],
    )
    _report(
        report / "SLICING_ANALYSIS.md",
        "Fixed Economic Slicing Analysis",
        [
            limitation,
            "0DTE is retained as a separate stress domain but is unavailable for "
            "released-surface pricing because the preserved samples excluded sub-7D "
            "contracts and the EOD files do not contain settlement-relative quote time.",
            _table_preview(
                slices[slices["defensible"]],
                [
                    "split",
                    "model_id",
                    "slice_dimension",
                    "slice_value",
                    "count",
                    "rmse",
                    "mae",
                    "spread_normalized_mae",
                    "within_bid_ask_rate",
                ],
            ),
        ],
    )
    _report(
        report / "RESIDUAL_ANALYSIS.md",
        "Residual Analysis",
        [
            limitation,
            "Residual is model price minus observed midpoint; positive values indicate "
            "overpricing.",
            _table_preview(
                slices[(slices["slice_dimension"] == "moneyness_slice") & slices["defensible"]],
                [
                    "split",
                    "model_id",
                    "slice_value",
                    "mean_signed_error",
                    "median_signed_error",
                    "mae",
                ],
            ),
        ],
    )
    _report(
        report / "MODEL_WINNER_MAP.md",
        "Model Winner Map",
        [
            "Winners use predefined spread-normalized MAE and are reported only for "
            "sufficiently populated cells, with market-date bootstrap uncertainty. "
            "The unqualified fixed Bates sensitivity is excluded from winner selection; "
            "LocalVol is unavailable without contract-level qualification. No global-best "
            "claim is made.",
            _table_preview(winners, list(winners.columns)),
            "## Pairwise date wins",
            _table_preview(pairwise, list(pairwise.columns)),
        ],
    )
    _report(
        report / "PARAMETER_STABILITY.md",
        "Heston Parameter Stability",
        [
            limitation,
            _table_preview(
                stability.sort_values("parameter_jump_score_v1_1", ascending=False),
                [
                    "quote_date",
                    "split",
                    "parameter_jump_score_v1_1",
                    *PARAMETER_NAMES,
                    "price_rmse",
                    "challenger_disagreement",
                ],
            ),
            "## Exploratory correlations",
            _table_preview(correlations, list(correlations.columns)),
            "## Similar fit, different parameters",
            _table_preview(similar, list(similar.columns)),
        ],
    )
    _report(
        report / "CHALLENGER_ATTRIBUTION.md",
        "Challenger Disagreement Attribution",
        [
            limitation,
            "SVI and SSVI are qualified contract-level challengers. Bates is retained only "
            "as the inherited fixed jump sensitivity and cannot support winner claims; "
            "LocalVol attribution is unavailable without contract-level qualification.",
            _table_preview(
                disagreement.sort_values("model_price_disagreement", ascending=False),
                [
                    "quote_date",
                    "split",
                    "challenger",
                    "moneyness_slice",
                    "maturity_slice",
                    "model_price_disagreement",
                    "delta_disagreement",
                    "gamma_disagreement",
                    "vega_disagreement",
                ],
            ),
        ],
    )
    _report(
        report / "GREEK_RISK_ANALYSIS.md",
        "Greek Risk Analysis",
        [
            "SVI/SSVI Greeks are frozen-surface analytic risks; Heston and the unqualified "
            "fixed Bates sensitivity use common finite differences. LocalVol Greek slicing "
            "is unavailable without contract-level qualification.",
            "## Low valuation disagreement / high Greek disagreement",
            _table_preview(
                low_value_high_greek.sort_values("vega_disagreement", ascending=False),
                list(low_value_high_greek.columns),
            ),
        ],
    )
    _report(
        report / "PORTFOLIO_MATERIALITY_SLICES.md",
        "Portfolio Materiality Slices",
        [
            limitation,
            "Portfolio leg-level liquidity is unavailable in the inherited aggregate "
            "artifact and is not imputed.",
            _table_preview(
                portfolio.sort_values("valuation_materiality", ascending=False),
                [
                    "quote_date",
                    "split",
                    "portfolio_id",
                    "description",
                    "model_id",
                    "vix_regime",
                    "maturity_exposure",
                    "moneyness_exposure",
                    "valuation_materiality",
                    "delta_materiality",
                    "gamma_materiality",
                    "vega_materiality",
                ],
            ),
        ],
    )
    _report(
        report / "MONITORING_ANALYTICS.md",
        "Monitoring Analytics",
        [
            "GREEN/AMBER/RED are transparent research statuses derived from DEV medians "
            "and MADs; they are not institutional JPMorgan rules.",
            _table_preview(
                monitoring.tail(20),
                [
                    "quote_date",
                    "split",
                    "calibration_loss",
                    "calibration_loss_status",
                    "spread_normalized_error",
                    "spread_normalized_error_status",
                    "parameter_jump_score_v1_1",
                    "parameter_jump_score_v1_1_status",
                    "E_form",
                    "E_form_status",
                    "E_greek",
                    "E_greek_status",
                    "portfolio_materiality",
                    "portfolio_materiality_status",
                ],
            ),
        ],
    )
    _report(
        report / "POST_VALIDATION_ANALYTICS.md",
        "Post-Validation Analytics",
        [
            "This is an additive v1.1 analysis extension. The v1 locked evidence was "
            "not rewritten, retuned, deleted, or reinterpreted.",
            limitation,
            "## Data-driven DEV segmentation",
            tree_text,
            "## Leaf transfer summaries",
            _table_preview(trees, list(trees.columns)),
        ],
    )


def run_post_validation_analytics(
    root: Path = Path("."), config: AnalyticsConfig | None = None
) -> dict[str, Any]:
    cfg = config or AnalyticsConfig()
    result_dir = root / "results/v1_1"
    figure_dir = root / "reports/v1_1/figures"
    result_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)
    reconstructed = reconstruct_contract_results(root, cfg)
    frozen = freeze_slice_registry(root, reconstructed, cfg)
    rows = apply_slices(reconstructed, frozen)
    rows.to_parquet(result_dir / "contract_model_results_sliced.parquet", index=False)
    slices = slice_metrics(rows, cfg)
    slices.to_parquet(result_dir / "core_slice_metrics.parquet", index=False)
    two_dim = two_dimensional_metrics(rows, cfg)
    two_dim.to_parquet(result_dir / "two_dimensional_slice_metrics.parquet", index=False)
    residual = rows[
        [
            "quote_date",
            "split",
            "model_id",
            "source_row_id",
            "log_moneyness",
            "dte",
            "VIXCLS",
            "relative_spread",
            "volume",
            "settlement_class",
            "residual",
            "absolute_error",
            "normalized_error",
        ]
    ]
    residual.to_parquet(result_dir / "residual_records.parquet", index=False)
    winners, pairwise, date_wins = model_winner_tables(rows, cfg)
    winners.to_parquet(result_dir / "model_winner_map.parquet", index=False)
    pairwise.to_csv(result_dir / "pairwise_win_matrix.csv", index=False)
    date_wins.to_csv(result_dir / "date_level_win_rates.csv", index=False)
    calibration, uncertainty, parameter_correlation = calibration_diagnostics(root, rows)
    calibration.to_parquet(result_dir / "calibration_deep_dive.parquet", index=False)
    uncertainty.to_csv(result_dir / "bootstrap_parameter_uncertainty.csv", index=False)
    parameter_correlation.to_csv(result_dir / "bootstrap_parameter_correlations.csv", index=False)
    calibration_slices = calibration_slice_analysis(root, calibration, frozen)
    calibration_slices.to_parquet(result_dir / "calibration_diagnostic_slices.parquet", index=False)
    stability, correlations, similar = parameter_stability(root, calibration, rows)
    stability.to_parquet(result_dir / "parameter_stability_v1_1.parquet", index=False)
    correlations.to_csv(result_dir / "parameter_correlations.csv", index=False)
    similar.to_csv(result_dir / "similar_fit_different_parameters.csv", index=False)
    disagreement, low_value_high_greek = disagreement_tables(rows, frozen)
    disagreement.to_parquet(result_dir / "challenger_disagreement.parquet", index=False)
    low_value_high_greek.to_parquet(result_dir / "low_value_high_greek_cases.parquet", index=False)
    portfolio = portfolio_slices(root, frozen)
    portfolio.to_parquet(result_dir / "portfolio_materiality_slices.parquet", index=False)
    monitoring, monitoring_registry = monitoring_table(calibration, stability, rows, portfolio, cfg)
    monitoring.to_parquet(result_dir / "monitoring_status.parquet", index=False)
    monitoring_path = root / "artifacts/v1_1/monitoring_thresholds_v1_1_reviewed.freeze.json"
    if monitoring_path.exists():
        if _json(monitoring_path) != monitoring_registry:
            raise RuntimeError("monitoring threshold freeze mismatch; refusing to overwrite")
    else:
        _write_json(monitoring_path, monitoring_registry)
    trees, tree_text = fit_segmentation_trees(rows, cfg)
    trees.to_parquet(result_dir / "dev_tree_segment_transfer.parquet", index=False)
    (result_dir / "dev_tree_rules.txt").write_text(tree_text + "\n", encoding="utf-8")
    # Required heatmaps use measured validation+locked cells only.
    plot = two_dim[
        (two_dim["dimension_1"] == "moneyness_slice")
        & (two_dim["dimension_2"] == "maturity_slice")
        & (two_dim["split"].isin(["VALIDATION", "LOCKED_TEST"]))
        & two_dim["defensible"]
    ]
    for model in ("M10_HESTON", "M02_SVI", "M03_SSVI"):
        selected = (
            plot[plot["model_id"] == model]
            .groupby(["value_1", "value_2"], as_index=False)
            .agg(rmse=("rmse", "mean"))
        )
        _plot_heatmap(
            selected,
            "rmse",
            f"{model} RMSE by moneyness and maturity",
            figure_dir / f"{model.lower()}_error_heatmap.png",
        )
    comparison = plot.pivot_table(
        index=["value_1", "value_2"], columns="model_id", values="rmse", aggfunc="mean"
    ).reset_index()
    for challenger in ("M02_SVI", "M03_SSVI"):
        if challenger in comparison and "M10_HESTON" in comparison:
            comparison[f"diff_{challenger}"] = comparison["M10_HESTON"] - comparison[challenger]
            _plot_heatmap(
                comparison,
                f"diff_{challenger}",
                f"Heston minus {challenger} RMSE",
                figure_dir / f"heston_vs_{challenger.lower()}_heatmap.png",
                center=0.0,
            )
    generate_reports(
        root,
        rows,
        slices,
        two_dim,
        calibration,
        winners,
        pairwise,
        stability,
        correlations,
        similar,
        disagreement,
        low_value_high_greek,
        portfolio,
        monitoring,
        trees,
        tree_text,
    )
    manifest = {
        "status": "REAL_POST_VALIDATION_ANALYTICS_COMPLETE",
        "created_utc": _utc_now(),
        "config": asdict(cfg),
        "contract_model_rows": len(rows),
        "unique_market_dates": int(rows["quote_date"].nunique()),
        "models": sorted(rows["model_id"].unique()),
        "v1_artifacts_modified": False,
        "temporal_oos": "UNAVAILABLE",
        "hedge_outcomes": "UNAVAILABLE",
        "zero_dte_pricing": "UNAVAILABLE_SEPARATELY_IDENTIFIED",
    }
    _write_json(root / "artifacts/v1_1/post_validation_manifest.json", manifest)
    return manifest
