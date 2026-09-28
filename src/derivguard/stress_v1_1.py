"""Controlled v1.1 repricing stress matrix with CUDA FP64 Heston evaluation."""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal, cast

import numpy as np
import pandas as pd
import torch

from derivguard.development.heston_cf import (
    HestonParameters,
    heston_call_price_torch,
    heston_price_fixed_quad,
)
from derivguard.market.black_scholes import greeks, option_price

KEYS = ["quote_date", "split", "settlement_class", "source_row_id"]
CORE_MODELS = ("M00_BS_FLAT", "M02_SVI", "M03_SSVI", "M10_HESTON")


def _hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _bs_reprice(
    frame: pd.DataFrame, spot: np.ndarray, rate: np.ndarray, vol: np.ndarray
) -> pd.DataFrame:
    price = np.empty(len(frame), dtype=np.float64)
    delta = np.empty(len(frame), dtype=np.float64)
    gamma = np.empty(len(frame), dtype=np.float64)
    vega = np.empty(len(frame), dtype=np.float64)
    for kind in ("call", "put"):
        mask = frame["option_type"].to_numpy() == kind
        if not np.any(mask):
            continue
        price[mask] = np.asarray(
            option_price(
                spot[mask],
                frame.loc[mask, "strike"].to_numpy(float),
                frame.loc[mask, "maturity"].to_numpy(float),
                vol[mask],
                rate[mask],
                frame.loc[mask, "dividend_yield"].to_numpy(float),
                kind,
            )
        )
        risk = greeks(
            spot[mask],
            frame.loc[mask, "strike"].to_numpy(float),
            frame.loc[mask, "maturity"].to_numpy(float),
            vol[mask],
            rate[mask],
            frame.loc[mask, "dividend_yield"].to_numpy(float),
            kind,
        )
        delta[mask], gamma[mask], vega[mask] = risk.delta, risk.gamma, risk.vega
    return pd.DataFrame(
        {
            "stressed_price": price,
            "stressed_delta": delta,
            "stressed_gamma": gamma,
            "stressed_vega": vega,
        }
    )


def _gpu_heston_group(
    spot: float,
    strikes: np.ndarray,
    maturity: float,
    rate: float,
    dividend: float,
    kind: str,
    parameters: HestonParameters,
    *,
    nodes: int = 64,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    matrix = np.asarray(
        [[parameters.v0, parameters.kappa, parameters.theta, parameters.sigma_v, parameters.rho]],
        dtype=np.float64,
    )

    def priced(current_spot: float, current: np.ndarray) -> np.ndarray:
        tensor = torch.as_tensor(current, dtype=torch.float64, device="cuda")
        strike_tensor = torch.as_tensor(strikes, dtype=torch.float64, device="cuda")
        values = heston_call_price_torch(
            current_spot,
            strike_tensor,
            maturity,
            tensor,
            rate,
            dividend,
            nodes=nodes,
        )
        if kind == "put":
            parity = current_spot * np.exp(-dividend * maturity) - strike_tensor * np.exp(
                -rate * maturity
            )
            values = values - parity[None, :]
        return np.asarray(values[0].cpu(), dtype=np.float64)

    base = priced(spot, matrix)
    spot_bump = 1.0e-3
    up = priced(spot * (1.0 + spot_bump), matrix)
    down = priced(spot * (1.0 - spot_bump), matrix)
    delta = (up - down) / (2.0 * spot_bump * spot)
    gamma = (up - 2.0 * base + down) / (spot_bump * spot) ** 2
    root = np.sqrt(parameters.v0)
    vol_bump = 5.0e-3
    upper = matrix.copy()
    lower = matrix.copy()
    upper[0, 0] = (root + vol_bump) ** 2
    lower[0, 0] = max(root - vol_bump, 1.0e-6) ** 2
    vega = (priced(spot, upper) - priced(spot, lower)) / (2.0 * vol_bump)
    return base, delta, gamma, vega


def _heston_reprice(frame: pd.DataFrame, scenario: dict[str, float]) -> pd.DataFrame:
    output = pd.DataFrame(
        index=frame.index,
        columns=["stressed_price", "stressed_delta", "stressed_gamma", "stressed_vega"],
        dtype=float,
    )
    group_columns = [
        "quote_date",
        "settlement_class",
        "spot",
        "maturity",
        "rate",
        "dividend_yield",
        "option_type",
    ]
    for _, group in frame.groupby(group_columns, sort=False):
        first = group.iloc[0]
        spot = float(first["spot"]) * scenario.get("spot_factor", 1.0)
        rate = float(first["rate"]) + scenario.get("rate_shift", 0.0)
        root_v0 = np.sqrt(float(first["v0"])) * scenario.get("vol_factor", 1.0)
        parameters = HestonParameters(
            max(root_v0 * root_v0, 1.0e-8),
            float(first["kappa"]),
            float(first["theta"]),
            float(first["sigma_v"]) * scenario.get("sigma_factor", 1.0),
            float(np.clip(float(first["rho"]) + scenario.get("rho_shift", 0.0), -0.999, 0.999)),
        )
        values = _gpu_heston_group(
            spot,
            group["strike"].to_numpy(float),
            float(first["maturity"]),
            rate,
            float(first["dividend_yield"]),
            str(first["option_type"]),
            parameters,
        )
        for column, value in zip(output.columns, values, strict=True):
            output.loc[group.index, column] = value
    return output


def _scenarios() -> dict[str, dict[str, float]]:
    result: dict[str, dict[str, float]] = {}
    for shock in (-0.10, -0.05, 0.05, 0.10):
        result[f"SPOT_{shock:+.0%}"] = {"spot_factor": 1.0 + shock}
    for shock in (-0.50, -0.20, 0.20, 0.50):
        result[f"VOL_{shock:+.0%}"] = {"vol_factor": 1.0 + shock}
    for shift in (-0.02, -0.01, 0.01, 0.02):
        result[f"RATE_{shift * 10000:+.0f}BP"] = {"rate_shift": shift}
    for shift in (-0.20, 0.20):
        result[f"RHO_{shift:+.2f}"] = {"rho_shift": shift}
    for shock in (-0.50, -0.20, 0.20, 0.50):
        result[f"VOL_OF_VOL_{shock:+.0%}"] = {"sigma_factor": 1.0 + shock}
    result["SPREAD_X2"] = {"spread_factor": 2.0}
    result["SPREAD_X4"] = {"spread_factor": 4.0}
    return result


def run(root: Path = Path(".")) -> dict[str, Any]:
    source = root / "results/v1_1/contract_model_results_sliced.parquet"
    calibration_path = root / "results/v1_1/calibration_deep_dive.parquet"
    module_path = root / "src/derivguard/stress_v1_1.py"
    manifest_path = root / "artifacts/v1_1/stress_repricing_manifest.json"
    output_path = root / "results/v1_1/stress_sensitivity_matrix.parquet"
    contract_path = root / "results/v1_1/stress_contract_results.parquet"
    identity = {
        path.relative_to(root).as_posix(): _hash(path)
        for path in (source, calibration_path, module_path)
    }
    if manifest_path.exists() and output_path.exists() and contract_path.exists():
        manifest = cast(dict[str, Any], json.loads(manifest_path.read_text(encoding="utf-8")))
        if manifest.get("source_hashes") == identity:
            return manifest
    if not torch.cuda.is_available():
        raise RuntimeError("controlled Heston stress repricing requires CUDA")
    rows = pd.read_parquet(source)
    rows = rows[rows["model_id"].isin(CORE_MODELS)].copy()
    calibration = pd.read_parquet(calibration_path)[
        ["quote_date", "settlement_class", "v0", "kappa", "theta", "sigma_v", "rho"]
    ].drop_duplicates(["quote_date", "settlement_class"])
    rows = rows.merge(
        calibration, on=["quote_date", "settlement_class"], how="left", validate="many_to_one"
    )
    stressed_records: list[pd.DataFrame] = []
    for scenario_name, scenario in _scenarios().items():
        for model, group in rows.groupby("model_id", sort=False):
            group = group.copy()
            if "spread_factor" in scenario:
                stressed = group[["model_price", "delta", "gamma", "vega"]].rename(
                    columns={
                        "model_price": "stressed_price",
                        "delta": "stressed_delta",
                        "gamma": "stressed_gamma",
                        "vega": "stressed_vega",
                    }
                )
            elif model == "M10_HESTON":
                stressed = _heston_reprice(group, scenario)
            else:
                spot = group["spot"].to_numpy(float) * scenario.get("spot_factor", 1.0)
                rate = group["rate"].to_numpy(float) + scenario.get("rate_shift", 0.0)
                vol = group["model_iv"].to_numpy(float) * scenario.get("vol_factor", 1.0)
                stressed = _bs_reprice(group, spot, rate, np.maximum(vol, 1.0e-6))
            result = group[
                [*KEYS, "model_id", "model_price", "delta", "gamma", "vega", "bid", "ask"]
            ].copy()
            for column in stressed.columns:
                result[column] = stressed[column].to_numpy(float)
            result["scenario"] = scenario_name
            result["spread_factor"] = scenario.get("spread_factor", 1.0)
            result["model_scenario_applicability"] = np.where(
                (model != "M10_HESTON") & ("rho_shift" in scenario or "sigma_factor" in scenario),
                "UNCHANGED_REFERENCE_MODEL_NOT_APPLICABLE",
                "APPLIED",
            )
            stressed_records.append(result)
    contract = pd.concat(stressed_records, ignore_index=True)
    heston = contract[contract["model_id"] == "M10_HESTON"][
        [*KEYS, "scenario", "stressed_price", "stressed_delta", "stressed_gamma", "stressed_vega"]
    ].rename(
        columns={
            "stressed_price": "heston_stressed_price",
            "stressed_delta": "heston_stressed_delta",
            "stressed_gamma": "heston_stressed_gamma",
            "stressed_vega": "heston_stressed_vega",
        }
    )
    contract = contract.merge(heston, on=[*KEYS, "scenario"], how="left", validate="many_to_one")
    for measure in ("price", "delta", "gamma", "vega"):
        contract[f"{measure}_disagreement"] = np.abs(
            contract[f"stressed_{measure}"] - contract[f"heston_stressed_{measure}"]
        )
    contract["value_change"] = contract["stressed_price"] - contract["model_price"]
    contract["spread_normalized_effect"] = np.abs(contract["value_change"]) / (
        np.maximum(contract["ask"] - contract["bid"], 0.01) * contract["spread_factor"]
    )
    aggregate = contract.groupby(["split", "model_id", "scenario"], as_index=False).agg(
        count=("source_row_id", "size"),
        mean_value_change=("value_change", "mean"),
        mean_absolute_value_change=("value_change", lambda value: float(np.abs(value).mean())),
        p95_absolute_value_change=(
            "value_change",
            lambda value: float(np.quantile(np.abs(value), 0.95)),
        ),
        mean_spread_normalized_effect=("spread_normalized_effect", "mean"),
        mean_abs_delta=("stressed_delta", lambda value: float(np.abs(value).mean())),
        mean_abs_gamma=("stressed_gamma", lambda value: float(np.abs(value).mean())),
        mean_abs_vega=("stressed_vega", lambda value: float(np.abs(value).mean())),
        mean_value_disagreement=("price_disagreement", "mean"),
        mean_delta_disagreement=("delta_disagreement", "mean"),
        mean_gamma_disagreement=("gamma_disagreement", "mean"),
        mean_vega_disagreement=("vega_disagreement", "mean"),
    )
    aggregate["method"] = "ACTUAL_REPRICING_CUDA_FP64_HESTON_EXACT_BS_SURFACE"
    aggregate["status"] = "MEASURED_CONTROLLED_REPRICING"
    # Independent CPU price checks on deterministic Heston scenario rows.
    checks = contract[
        (contract["model_id"] == "M10_HESTON") & ~contract["scenario"].str.startswith("SPREAD")
    ].head(12)
    errors = []
    for item in checks.itertuples(index=False):
        base = rows[
            (rows["model_id"] == "M10_HESTON")
            & (rows["quote_date"] == item.quote_date)
            & (rows["settlement_class"] == item.settlement_class)
            & (rows["source_row_id"] == item.source_row_id)
        ].iloc[0]
        spec = _scenarios()[str(item.scenario)]
        parameters = HestonParameters(
            max((np.sqrt(base.v0) * spec.get("vol_factor", 1.0)) ** 2, 1.0e-8),
            base.kappa,
            base.theta,
            base.sigma_v * spec.get("sigma_factor", 1.0),
            float(np.clip(base.rho + spec.get("rho_shift", 0.0), -0.999, 0.999)),
        )
        cpu = heston_price_fixed_quad(
            base.spot * spec.get("spot_factor", 1.0),
            base.strike,
            base.maturity,
            parameters,
            base.rate + spec.get("rate_shift", 0.0),
            base.dividend_yield,
            cast(Literal["call", "put"], base.option_type),
            nodes=64,
        )
        errors.append(abs(float(cpu) - float(cast(Any, item.stressed_price))))
    max_error = max(errors, default=float("nan"))
    if not np.isfinite(max_error) or max_error > 1.0e-6:
        raise RuntimeError(f"CUDA stress repricing failed CPU spot check: {max_error}")
    contract.to_parquet(contract_path, index=False)
    aggregate.to_parquet(output_path, index=False)
    manifest = {
        "status": "COMPLETE",
        "created_utc": datetime.now(UTC).isoformat(),
        "source_hashes": identity,
        "cuda_device": torch.cuda.get_device_name(0),
        "scenarios": list(_scenarios()),
        "models": list(CORE_MODELS),
        "contract_scenario_rows": len(contract),
        "aggregate_rows": len(aggregate),
        "cpu_gpu_max_absolute_error": max_error,
        "bates_status": "EXCLUDED_UNQUALIFIED_FIXED_SENSITIVITY",
        "localvol_status": "UNAVAILABLE_NO_CONTRACT_LEVEL_QUALIFICATION",
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest
