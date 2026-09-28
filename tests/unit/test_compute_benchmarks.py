from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from derivguard.compute.benchmarks import (
    _bootstrap_losses_numpy,
    _population_design,
    _population_prices_numpy,
    run_final_compute_benchmarks,
)


def test_population_and_bootstrap_benchmark_shapes_are_explicit() -> None:
    population, strikes, maturities, spread, draws = _population_design(6, 8, 4, 17)
    prices = _population_prices_numpy(population, strikes, maturities, nodes=40)
    losses, best = _bootstrap_losses_numpy(prices, draws, spread)
    assert prices.shape == (6, 8)
    assert losses.shape == (4, 6)
    assert best.shape == (4,)
    assert np.all(np.isfinite(prices))
    assert np.all(np.isfinite(losses))
    assert np.all((best >= 0) & (best < 6))


def test_cpu_smoke_runner_records_gpu_as_not_run() -> None:
    output = Path("build/compute-benchmark-test-output")
    output.mkdir(parents=True, exist_ok=True)
    result = run_final_compute_benchmarks("smoke", "cpu", output_dir=output)
    table = pd.read_csv(output / "final_compute_benchmarks.csv")
    performance = pd.read_csv(output / "T12_CPU_GPU_PERFORMANCE.csv")
    manifest = json.loads((output / "final_compute_benchmarks.json").read_text(encoding="utf-8"))
    assert result["all_rows_measured_or_explicitly_unavailable"]
    assert manifest["device"] == "cpu"
    assert {
        "synthetic_heston_truth_and_greeks",
        "heston_population_evaluation",
        "bootstrap_population_calibration",
        "heston_pde_mcs_representative_grid",
    } <= set(table["workload"])
    cupy = table[table["backend"] == "cupyx-sparse-cuda-fp64"].iloc[0]
    assert cupy["status"] == "NOT_RUN_DEVICE_CPU"
    assert len(performance) == 4
    assert {"workload", "cpu_seconds", "gpu_seconds", "speedup", "gpu_status"} <= set(
        performance.columns
    )
