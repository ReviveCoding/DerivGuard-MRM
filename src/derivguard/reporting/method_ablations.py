"""Derive developer-method ablations from executed experiment artifacts."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def build_developer_method_ablation_summary(root: Path) -> dict[str, Any]:
    raw = pd.read_csv(root / "results/aggregated/developer_method_ablations.csv")
    bootstrap = pd.read_parquet(root / "results/aggregated/bootstrap_calibration.parquet")
    feller = pd.read_csv(root / "results/aggregated/feller_study.csv")
    forward = pd.read_parquet(root / "results/raw/forward_estimator_comparison.parquet")
    objective = raw.loc[raw["ablation_family"] == "objective"].copy()
    optimizer = raw.loc[raw["ablation_family"] == "optimizer"].copy()

    def median(frame: pd.DataFrame, variant: str, column: str) -> float:
        values = pd.to_numeric(frame.loc[frame["variant"] == variant, column], errors="coerce")
        if not values.notna().any():
            raise ValueError(f"missing measured {variant}/{column} ablation evidence")
        return float(values.median())

    selected_c04 = median(objective, "C04", "dev_common_C04")
    selected_o01 = median(optimizer, "O01", "dev_common_C04")
    parameter_columns = ["v0", "kappa", "theta", "sigma_v", "rho"]
    successful_bootstrap = bootstrap.loc[bootstrap["success"].astype(bool), parameter_columns]
    if len(successful_bootstrap) < 2:
        raise ValueError("at least two successful bootstrap draws are required")
    mean_parameter_std = float(
        np.mean(successful_bootstrap.astype(float).std(axis=0, ddof=1).to_numpy())
    )
    feller_objectives = feller.groupby("feller_treatment")["objective_value"].median().astype(float)
    forward_success = forward.loc[
        (forward["status"] == "SUCCESS") & forward["method"].isin(["F0", "F1", "F2", "F3"])
    ].copy()
    forward_rmse = forward_success.groupby("method")["residual_rmse"].median().astype(float)
    if not {"unconstrained", "soft", "hard"}.issubset(feller_objectives.index):
        raise ValueError("Feller ablation evidence is incomplete")
    if not {"F0", "F1", "F2", "F3"}.issubset(forward_rmse.index):
        raise ValueError("forward-estimator ablation evidence is incomplete")

    rows = [
        {
            "ablation_id": "DA01",
            "description": "no global optimization",
            "metric": "DEV common C04 loss",
            "selected_variant": "O01",
            "ablated_variant": "O00",
            "selected_value": selected_o01,
            "ablated_value": median(optimizer, "O00", "dev_common_C04"),
            "evidence": "results/aggregated/developer_method_ablations.csv",
        },
        {
            "ablation_id": "DA02",
            "description": "no robust loss",
            "metric": "DEV common C04 loss",
            "selected_variant": "C04",
            "ablated_variant": "C00",
            "selected_value": selected_c04,
            "ablated_value": median(objective, "C00", "dev_common_C04"),
            "evidence": "results/aggregated/developer_method_ablations.csv",
        },
        {
            "ablation_id": "DA03",
            "description": "no spread weighting",
            "metric": "DEV common C04 loss",
            "selected_variant": "C04",
            "ablated_variant": "C01",
            "selected_value": selected_c04,
            "ablated_value": median(objective, "C01", "dev_common_C04"),
            "evidence": "results/aggregated/developer_method_ablations.csv",
        },
        {
            "ablation_id": "DA04",
            "description": "no bootstrap",
            "metric": "mean parameter standard deviation",
            "selected_variant": f"bootstrap ({len(successful_bootstrap)} successful draws)",
            "ablated_variant": "point estimate only",
            "selected_value": mean_parameter_std,
            "ablated_value": 0.0,
            "evidence": "results/aggregated/bootstrap_calibration.parquet",
        },
        {
            "ablation_id": "DA05",
            "description": "Feller variants",
            "metric": "median calibration objective",
            "selected_variant": "unconstrained",
            "ablated_variant": "best of soft/hard",
            "selected_value": float(feller_objectives["unconstrained"]),
            "ablated_value": float(min(feller_objectives["soft"], feller_objectives["hard"])),
            "evidence": "results/aggregated/feller_study.csv",
        },
        {
            "ablation_id": "DA06",
            "description": "forward estimator variants",
            "metric": "median parity residual RMSE",
            "selected_variant": "F3 spread-weighted robust",
            "ablated_variant": "F0 ordinary least squares",
            "selected_value": float(forward_rmse["F3"]),
            "ablated_value": float(forward_rmse["F0"]),
            "evidence": "results/raw/forward_estimator_comparison.parquet",
        },
    ]
    frame = pd.DataFrame(rows)
    frame["difference_ablated_minus_selected"] = frame["ablated_value"] - frame["selected_value"]
    output = root / "results/aggregated/developer_method_ablation_summary.csv"
    frame.to_csv(output, index=False)
    payload = {
        "status": "MEASURED",
        "generated_at": datetime.now(UTC).isoformat(),
        "ablation_ids": list(frame["ablation_id"]),
        "rows": len(frame),
        "artifact": str(output.relative_to(root)),
    }
    (root / "results/aggregated/developer_method_ablation_summary.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return payload
