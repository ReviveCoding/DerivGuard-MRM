"""Measured GPU Monte Carlo variance-reduction comparison (experiment N06)."""

from __future__ import annotations

import json
import time
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

import pandas as pd

from derivguard.validation.heston_mc import ValidationHestonParameters, heston_mc_torch


def run_mc_variance_reduction(root: Path, device: str = "cuda") -> dict[str, Any]:
    """Compare no reduction, antithetic sampling, and antithetic+control variate."""

    if device != "cuda":
        raise RuntimeError("research N06 must execute on the qualified CUDA backend")
    parameters = ValidationHestonParameters(0.04, 2.0, 0.04, 0.8, -0.7)
    designs = (
        ("none", False, False),
        ("antithetic", True, False),
        ("antithetic_control", True, True),
    )
    rows: list[dict[str, object]] = []
    for seed in range(20261101, 20261106):
        seed_rows: list[dict[str, object]] = []
        for method, antithetic, control in designs:
            started = time.perf_counter()
            estimate = heston_mc_torch(
                100.0,
                100.0,
                1.0,
                parameters,
                0.03,
                0.01,
                paths=150_000,
                steps=128,
                scheme="qe_m",
                seed=seed,
                antithetic=antithetic,
                control_variate=control,
                device="cuda",
                chunk_paths=150_000,
            )
            row: dict[str, object] = {
                **asdict(estimate),
                "experiment_id": "N06",
                "method": method,
                "runtime_seconds": time.perf_counter() - started,
            }
            seed_rows.append(row)
        base_variance = float(cast(float, seed_rows[0]["standard_error"])) ** 2
        for row in seed_rows:
            variance = float(cast(float, row["standard_error"])) ** 2
            row["variance_reduction_ratio"] = base_variance / variance if variance > 0.0 else None
            rows.append(row)
    frame = pd.DataFrame(rows)
    output = root / "results/aggregated/mc_variance_reduction.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output, index=False)
    summary = {
        "status": "MEASURED",
        "experiment_id": "N06",
        "generated_at": datetime.now(UTC).isoformat(),
        "backend": "torch-cuda-fp64",
        "scheme": "qe_m",
        "paths_per_run": 150_000,
        "steps": 128,
        "seeds": 5,
        "methods": list(frame["method"].drop_duplicates()),
        "rows": len(frame),
        "median_standard_error": {
            str(method): float(group["standard_error"].median())
            for method, group in frame.groupby("method")
        },
        "median_variance_reduction_ratio": {
            str(method): float(group["variance_reduction_ratio"].median())
            for method, group in frame.groupby("method")
        },
        "artifact": str(output.relative_to(root)),
    }
    (root / "results/aggregated/mc_variance_reduction.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return summary
