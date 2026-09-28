"""Measured final CPU/GPU compute benchmarks.

The benchmark runner is intentionally independent of report generation.  It
records measured rows for synthetic Heston truth generation, batched Heston
population evaluation, coarse bootstrap population calibration, the CPU ADI
validator, and an actually attempted CuPy sparse MCS implementation.  CuPy
incompatibility or poor performance is retained as evidence; it is never
silently replaced by a claimed GPU result.
"""

from __future__ import annotations

import hashlib
import json
import time
import tracemalloc
from collections.abc import Callable
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml  # type: ignore[import-untyped]
from numpy.typing import NDArray
from scipy.sparse import eye

from derivguard.development.heston_cf import (
    HestonParameters,
    heston_call_price_torch,
    heston_price_adaptive,
    heston_price_fixed_quad,
)
from derivguard.synthetic.confirmatory import (
    ConfirmatoryConfig,
    _surface_design,
    generate_confirmatory_truth,
)
from derivguard.validation.heston_pde import (
    PDEGrid,
    PDEHestonParameters,
    _centered_nonuniform_grid,
    _set_adi_boundary_matrix,
    _split_operators,
    heston_pde_price,
)

FloatArray = NDArray[np.float64]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _measure_cpu(function: Callable[[], Any]) -> tuple[Any, float, int]:
    tracemalloc.start()
    before = time.perf_counter()
    result = function()
    runtime = time.perf_counter() - before
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return result, runtime, peak


def _population_prices_numpy(
    parameter_matrix: FloatArray,
    strikes: FloatArray,
    maturities: FloatArray,
    *,
    nodes: int,
) -> FloatArray:
    """CPU FP64 population prices, grouped by maturity but not by model code."""

    output = np.empty((len(parameter_matrix), len(strikes)), dtype=np.float64)
    for row, raw in enumerate(parameter_matrix):
        params = HestonParameters(*map(float, raw))
        for maturity in np.unique(maturities):
            mask = np.isclose(maturities, maturity)
            output[row, mask] = np.asarray(
                heston_price_fixed_quad(
                    100.0,
                    strikes[mask],
                    float(maturity),
                    params,
                    0.03,
                    0.01,
                    nodes=nodes,
                ),
                dtype=np.float64,
            )
    return output


def _bootstrap_losses_numpy(
    prices: FloatArray, market_draws: FloatArray, spread: FloatArray
) -> tuple[FloatArray, NDArray[np.int64]]:
    scaled = (prices[None, :, :] - market_draws[:, None, :]) / spread[None, None, :]
    losses = np.mean(
        np.where(np.abs(scaled) <= 1.5, 0.5 * scaled**2, 1.5 * (np.abs(scaled) - 0.75)), axis=2
    )
    best = np.asarray(np.argmin(losses, axis=1), dtype=np.int64)
    return np.asarray(losses, dtype=np.float64), best


def _population_design(
    candidates: int, contracts: int, draws: int, seed: int
) -> tuple[FloatArray, FloatArray, FloatArray, FloatArray, FloatArray]:
    rng = np.random.default_rng(seed)
    bounds = np.asarray(((0.015, 0.10), (0.5, 4.0), (0.015, 0.10), (0.20, 0.90), (-0.90, -0.15)))
    population = rng.uniform(bounds[:, 0], bounds[:, 1], size=(candidates, 5))
    # Include the generating parameters, avoiding a benchmark whose minimum is
    # entirely an optimizer-design accident.
    population[0] = np.asarray((0.04, 2.0, 0.04, 0.5, -0.7))
    strikes = np.linspace(75.0, 125.0, contracts, dtype=np.float64)
    maturities = np.where(np.arange(contracts) % 2 == 0, 0.5, 1.0).astype(np.float64)
    true = _population_prices_numpy(population[:1], strikes, maturities, nodes=64)[0]
    spread = np.maximum(0.08, 0.015 * np.maximum(true, 1.0))
    market_draws = rng.uniform(true - 0.5 * spread, true + 0.5 * spread, size=(draws, contracts))
    return population, strikes, maturities, spread, np.asarray(market_draws, dtype=np.float64)


def _population_prices_torch(
    parameter_matrix: FloatArray,
    strikes: FloatArray,
    maturities: FloatArray,
    *,
    nodes: int,
) -> Any:
    import torch

    parameters = torch.as_tensor(parameter_matrix, dtype=torch.float64, device="cuda")
    output = torch.empty((len(parameter_matrix), len(strikes)), dtype=torch.float64, device="cuda")
    for maturity in np.unique(maturities):
        indices = np.flatnonzero(np.isclose(maturities, maturity))
        strike_tensor = torch.as_tensor(strikes[indices], dtype=torch.float64, device="cuda")
        values = heston_call_price_torch(
            100.0,
            strike_tensor,
            float(maturity),
            parameters,
            0.03,
            0.01,
            nodes=nodes,
        )
        output[:, torch.as_tensor(indices, dtype=torch.long, device="cuda")] = values
    return output


def _benchmark_cupy_mcs(*, spot_nodes: int, variance_nodes: int, time_steps: int) -> dict[str, Any]:
    """Actually execute a CuPy sparse MCS solve or return the concrete blocker."""

    started = time.perf_counter()
    try:
        import cupy as cp  # type: ignore[import-untyped]
        import cupyx.scipy.sparse as cpsparse  # type: ignore[import-untyped]
        import cupyx.scipy.sparse.linalg as cplinalg  # type: ignore[import-untyped]

        params = PDEHestonParameters(0.04, 2.0, 0.04, 0.5, -0.7)
        spot, strike, maturity = 100.0, 100.0, 1.0
        rate, dividend = 0.03, 0.01
        smax, vmax = 400.0, 0.5
        spots = _centered_nonuniform_grid(
            0.0, spot, smax, spot_nodes, 0.5, max(0.04 * spot, 1.0e-4)
        )
        variances = _centered_nonuniform_grid(
            0.0,
            params.v0,
            vmax,
            variance_nodes,
            0.3,
            max(0.25 * max(params.v0, params.theta), 1.0e-4),
        )
        a0_cpu, a1_cpu, a2_cpu = _split_operators(spots, variances, params, rate, dividend)
        ns, nv = len(spots), len(variances)
        dt, theta = maturity / time_steps, 1.0 / 3.0
        identity = eye(ns * nv, format="csc", dtype=np.float64)
        factors_cpu = []
        for operator in (a1_cpu, a2_cpu):
            matrix = (identity - theta * dt * operator).tolil()
            _set_adi_boundary_matrix(matrix, ns, nv)
            factors_cpu.append(matrix.tocsr())

        pool = cp.get_default_memory_pool()
        pool.free_all_blocks()
        a0 = cpsparse.csr_matrix(a0_cpu)
        a1 = cpsparse.csr_matrix(a1_cpu)
        a2 = cpsparse.csr_matrix(a2_cpu)
        total = a0 + a1 + a2
        factors = tuple(cpsparse.csr_matrix(matrix) for matrix in factors_cpu)
        terminal = np.maximum(spots - strike, 0.0)
        values = cp.asarray(np.tile(terminal, (nv, 1)).reshape(-1), dtype=cp.float64)

        def enforce(vector: Any, tau: float, *, rhs: bool = False) -> Any:
            shaped = vector.reshape(nv, ns).copy()
            shaped[:, 0] = 0.0
            shaped[:, -1] = max(smax * np.exp(-dividend * tau) - strike * np.exp(-rate * tau), 0.0)
            if rhs:
                shaped[-1, 1:-1] = 0.0
            else:
                shaped[-1, 1:-1] = shaped[-2, 1:-1]
            return shaped.reshape(-1)

        cp.cuda.Stream.null.synchronize()
        solve_started = time.perf_counter()
        for step in range(time_steps):
            tau = (step + 1) * dt
            previous = values
            y0 = enforce(previous + dt * (total @ previous), tau)
            y = y0
            for operator, matrix in zip((a1, a2), factors, strict=True):
                rhs = enforce(y - theta * dt * (operator @ previous), tau, rhs=True)
                y = enforce(cplinalg.spsolve(matrix, rhs), tau)
            corrected = y0 + theta * dt * (a0 @ (y - previous))
            corrected += (0.5 - theta) * dt * (total @ (y - previous))
            corrected = enforce(corrected, tau)
            for operator, matrix in zip((a1, a2), factors, strict=True):
                rhs = enforce(corrected - theta * dt * (operator @ previous), tau, rhs=True)
                corrected = enforce(cplinalg.spsolve(matrix, rhs), tau)
            values = corrected
        cp.cuda.Stream.null.synchronize()
        runtime = time.perf_counter() - solve_started
        spot_index = int(np.flatnonzero(np.isclose(spots, spot))[0])
        variance_index = int(np.flatnonzero(np.isclose(variances, params.v0))[0])
        price = float(values.reshape(nv, ns)[variance_index, spot_index].get())
        peak = int(pool.total_bytes())
        return {
            "status": "MEASURED",
            "price": price,
            "runtime_seconds": runtime,
            "peak_memory_bytes": peak,
            "peak_memory_kind": "cupy_memory_pool_allocated_after_run",
            "setup_seconds": solve_started - started,
            "notes": "cupyx sparse MCS uses spsolve per directional stage; no reusable GPU LU",
        }
    except Exception as exc:  # actual incompatibility is scientific evidence
        return {
            "status": "ATTEMPTED_INCOMPATIBLE",
            "price": None,
            "runtime_seconds": time.perf_counter() - started,
            "peak_memory_bytes": None,
            "peak_memory_kind": None,
            "error_type": type(exc).__name__,
            "error": str(exc),
            "notes": "CuPy sparse MCS attempt failed; CPU result remains the selected backend",
        }


def _performance_table(frame: pd.DataFrame) -> pd.DataFrame:
    """Pivot long-form evidence into the canonical T12 comparison schema."""

    rows: list[dict[str, Any]] = []
    for workload, group in frame.groupby("workload", sort=False):
        cpu = group[group["backend"].str.contains("cpu", case=False, na=False)]
        gpu = group[group["backend"].str.contains("cuda|cupyx", case=False, regex=True, na=False)]
        cpu_row = cpu.iloc[0] if not cpu.empty else None
        gpu_row = gpu.iloc[0] if not gpu.empty else None
        selected = group[group["selected_backend"].fillna(False).astype(bool)]
        rows.append(
            {
                "workload": workload,
                "cpu_backend": None if cpu_row is None else cpu_row.get("backend"),
                "cpu_seconds": None if cpu_row is None else cpu_row.get("runtime_seconds"),
                "cpu_peak_memory_bytes": None
                if cpu_row is None
                else cpu_row.get("peak_memory_bytes"),
                "gpu_backend": None if gpu_row is None else gpu_row.get("backend"),
                "gpu_seconds": None if gpu_row is None else gpu_row.get("runtime_seconds"),
                "gpu_peak_memory_bytes": None
                if gpu_row is None
                else gpu_row.get("peak_memory_bytes"),
                "speedup": None if gpu_row is None else gpu_row.get("speedup_vs_cpu"),
                "absolute_error": None if gpu_row is None else gpu_row.get("absolute_error"),
                "relative_error": None if gpu_row is None else gpu_row.get("relative_error"),
                "accuracy_pass": None
                if gpu_row is None
                else gpu_row.get("passes_cpu_gpu_accuracy"),
                "gpu_status": None if gpu_row is None else gpu_row.get("status"),
                "selected_backend": None if selected.empty else selected.iloc[0].get("backend"),
                "availability_note": None if gpu_row is None else gpu_row.get("notes"),
            }
        )
    return pd.DataFrame(rows)


def run_final_compute_benchmarks(
    profile: str = "research",
    device: str = "cuda",
    *,
    output_dir: Path | None = None,
) -> dict[str, Any]:
    """Run bounded final workload benchmarks and persist measured evidence."""

    if profile not in {"smoke", "audit", "research"}:
        raise ValueError("profile must be smoke, audit, or research")
    if device not in {"cpu", "cuda"}:
        raise ValueError("device must be cpu or cuda")
    root = Path.cwd()
    destination = output_dir or root / "results/aggregated"
    destination.mkdir(parents=True, exist_ok=True)
    tolerance_path = root / "configs/numerical_tolerances.yaml"
    tolerance_document = yaml.safe_load(tolerance_path.read_text(encoding="utf-8"))
    gpu_tolerance = tolerance_document["tolerances"]["cpu_gpu"]
    gpu_absolute = float(gpu_tolerance["float64_absolute"])
    gpu_relative = float(gpu_tolerance["float64_relative"])
    started = datetime.now(UTC).isoformat()
    rows: list[dict[str, Any]] = []
    if profile == "research":
        contracts, candidates, draws, nodes = 48, 128, 64, 96
        pde_shape = (81, 41, 100)
    elif profile == "audit":
        contracts, candidates, draws, nodes = 24, 64, 32, 64
        pde_shape = (61, 31, 60)
    else:
        contracts, candidates, draws, nodes = 8, 12, 8, 48
        pde_shape = (41, 21, 30)

    design = _surface_design(20261101, contracts, ood=False)
    cpu_config = ConfirmatoryConfig(
        contracts_per_case=contracts,
        max_cpu_contracts=max(contracts, 256),
        heston_nodes=nodes,
        device="cpu-bounded-reference",
    )
    cpu_truth, cpu_truth_seconds, cpu_truth_peak = _measure_cpu(
        lambda: generate_confirmatory_truth(design, "HESTON", cpu_config, seed=20261102)
    )
    rows.append(
        {
            "workload": "synthetic_heston_truth_and_greeks",
            "backend": "numpy-cpu-fp64",
            "runtime_seconds": cpu_truth_seconds,
            "throughput_per_second": contracts / cpu_truth_seconds,
            "absolute_error": 0.0,
            "relative_error": 0.0,
            "peak_memory_bytes": cpu_truth_peak,
            "peak_memory_kind": "python_tracemalloc",
            "status": "MEASURED",
            "selected_backend": device == "cpu",
        }
    )
    cpu_truth_row_index = len(rows) - 1

    population, strikes, maturities, spread, market_draws = _population_design(
        candidates, contracts, draws, 20261103
    )
    cpu_prices, cpu_population_seconds, cpu_population_peak = _measure_cpu(
        lambda: _population_prices_numpy(population, strikes, maturities, nodes=nodes)
    )
    one_market = market_draws[0]
    population_loss = np.mean(((cpu_prices - one_market[None, :]) / spread[None, :]) ** 2, axis=1)
    rows.append(
        {
            "workload": "heston_population_evaluation",
            "backend": "numpy-cpu-fp64",
            "runtime_seconds": cpu_population_seconds,
            "throughput_per_second": candidates * contracts / cpu_population_seconds,
            "absolute_error": 0.0,
            "relative_error": 0.0,
            "peak_memory_bytes": cpu_population_peak,
            "peak_memory_kind": "python_tracemalloc",
            "status": "MEASURED",
            "best_objective": float(np.min(population_loss)),
            "selected_backend": device == "cpu",
        }
    )
    cpu_population_row_index = len(rows) - 1
    (cpu_losses, cpu_best), cpu_bootstrap_seconds, cpu_bootstrap_peak = _measure_cpu(
        lambda: _bootstrap_losses_numpy(cpu_prices, market_draws, spread)
    )
    rows.append(
        {
            "workload": "bootstrap_population_calibration",
            "backend": "numpy-cpu-fp64",
            "runtime_seconds": cpu_population_seconds + cpu_bootstrap_seconds,
            "throughput_per_second": draws
            * candidates
            * contracts
            / (cpu_population_seconds + cpu_bootstrap_seconds),
            "absolute_error": 0.0,
            "relative_error": 0.0,
            "peak_memory_bytes": max(cpu_population_peak, cpu_bootstrap_peak),
            "peak_memory_kind": "python_tracemalloc",
            "status": "MEASURED",
            "successful_draws": len(cpu_best),
            "selected_backend": device == "cpu",
        }
    )
    cpu_bootstrap_row_index = len(rows) - 1

    pde_params = PDEHestonParameters(0.04, 2.0, 0.04, 0.5, -0.7)
    pde_grid = PDEGrid(*pde_shape, variance_max=0.5, nonuniform=True, rannacher_steps=0)
    cpu_pde, cpu_pde_seconds, cpu_pde_peak = _measure_cpu(
        lambda: heston_pde_price(
            100.0,
            100.0,
            1.0,
            pde_params,
            0.03,
            0.01,
            scheme="modified_craig_sneyd",
            grid=pde_grid,
        )
    )
    cf_reference = float(
        heston_price_adaptive(
            100.0, 100.0, 1.0, HestonParameters(0.04, 2.0, 0.04, 0.5, -0.7), 0.03, 0.01
        )
    )
    rows.append(
        {
            "workload": "heston_pde_mcs_representative_grid",
            "backend": "scipy-sparse-cpu-fp64",
            "runtime_seconds": cpu_pde_seconds,
            "throughput_per_second": pde_shape[0] * pde_shape[1] * pde_shape[2] / cpu_pde_seconds,
            "absolute_error": abs(cpu_pde.price - cf_reference),
            "relative_error": abs(cpu_pde.price - cf_reference) / abs(cf_reference),
            "peak_memory_bytes": cpu_pde_peak,
            "peak_memory_kind": "python_tracemalloc",
            "status": "MEASURED",
            "price": cpu_pde.price,
            "selected_backend": True,
        }
    )
    cpu_pde_row_index = len(rows) - 1

    if device == "cuda":
        import torch

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA benchmark requested but Torch reports CUDA unavailable")
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        gpu_config = replace(cpu_config, device="cuda")
        _ = generate_confirmatory_truth(
            _surface_design(20261100, min(4, contracts), ood=False),
            "HESTON",
            replace(gpu_config, contracts_per_case=min(4, contracts)),
            seed=20261102,
        )
        torch.cuda.synchronize()
        before = time.perf_counter()
        gpu_truth = generate_confirmatory_truth(design, "HESTON", gpu_config, seed=20261102)
        torch.cuda.synchronize()
        gpu_truth_seconds = time.perf_counter() - before
        truth_error = max(
            float(np.max(np.abs(getattr(gpu_truth, name) - getattr(cpu_truth, name))))
            for name in ("true_price", "true_delta", "true_gamma", "true_vega")
        )
        truth_scale = max(float(np.max(np.abs(cpu_truth.true_price))), 1.0e-12)
        truth_relative = truth_error / truth_scale
        truth_passes = truth_error <= max(gpu_absolute, gpu_relative * truth_scale)
        truth_faster = gpu_truth_seconds < cpu_truth_seconds
        rows.append(
            {
                "workload": "synthetic_heston_truth_and_greeks",
                "backend": "torch-cuda-fp64",
                "runtime_seconds": gpu_truth_seconds,
                "throughput_per_second": contracts / gpu_truth_seconds,
                "speedup_vs_cpu": cpu_truth_seconds / gpu_truth_seconds,
                "absolute_error": truth_error,
                "relative_error": truth_relative,
                "peak_memory_bytes": int(torch.cuda.max_memory_allocated()),
                "peak_memory_kind": "torch_cuda_max_memory_allocated",
                "status": "MEASURED",
                "passes_cpu_gpu_accuracy": truth_passes,
                "faster_than_cpu": truth_faster,
                "selected_backend": bool(truth_passes and truth_faster),
            }
        )
        rows[cpu_truth_row_index]["selected_backend"] = not bool(truth_passes and truth_faster)

        torch.cuda.reset_peak_memory_stats()
        _ = _population_prices_torch(population[:2], strikes, maturities, nodes=nodes)
        torch.cuda.synchronize()
        before = time.perf_counter()
        gpu_price_tensor = _population_prices_torch(population, strikes, maturities, nodes=nodes)
        torch.cuda.synchronize()
        gpu_population_seconds = time.perf_counter() - before
        gpu_prices = np.asarray(gpu_price_tensor.detach().cpu().numpy(), dtype=np.float64)
        price_error = float(np.max(np.abs(gpu_prices - cpu_prices)))
        price_scale = max(float(np.max(np.abs(cpu_prices))), 1.0e-12)
        price_relative = price_error / price_scale
        population_passes = price_error <= max(gpu_absolute, gpu_relative * price_scale)
        population_faster = gpu_population_seconds < cpu_population_seconds
        rows.append(
            {
                "workload": "heston_population_evaluation",
                "backend": "torch-cuda-fp64",
                "runtime_seconds": gpu_population_seconds,
                "throughput_per_second": candidates * contracts / gpu_population_seconds,
                "speedup_vs_cpu": cpu_population_seconds / gpu_population_seconds,
                "absolute_error": price_error,
                "relative_error": price_relative,
                "peak_memory_bytes": int(torch.cuda.max_memory_allocated()),
                "peak_memory_kind": "torch_cuda_max_memory_allocated",
                "status": "MEASURED",
                "passes_cpu_gpu_accuracy": population_passes,
                "faster_than_cpu": population_faster,
                "selected_backend": bool(population_passes and population_faster),
            }
        )
        rows[cpu_population_row_index]["selected_backend"] = not bool(
            population_passes and population_faster
        )

        torch.cuda.reset_peak_memory_stats()
        draw_tensor = torch.as_tensor(market_draws, dtype=torch.float64, device="cuda")
        spread_tensor = torch.as_tensor(spread, dtype=torch.float64, device="cuda")
        before = time.perf_counter()
        # Reprice inside the timed batch: this is valuation plus coarse
        # population calibration, not merely post-processing cached values.
        bootstrap_prices = _population_prices_torch(population, strikes, maturities, nodes=nodes)
        scaled = (bootstrap_prices[None, :, :] - draw_tensor[:, None, :]) / spread_tensor[
            None, None, :
        ]
        absolute = torch.abs(scaled)
        bootstrap_losses = torch.mean(
            torch.where(absolute <= 1.5, 0.5 * scaled * scaled, 1.5 * (absolute - 0.75)),
            dim=2,
        )
        gpu_best = torch.argmin(bootstrap_losses, dim=1)
        torch.cuda.synchronize()
        gpu_bootstrap_seconds = time.perf_counter() - before
        gpu_loss_values = np.asarray(bootstrap_losses.detach().cpu().numpy(), dtype=np.float64)
        bootstrap_error = float(np.max(np.abs(gpu_loss_values - cpu_losses)))
        selection_match = float(np.mean(np.asarray(gpu_best.cpu()) == cpu_best))
        bootstrap_scale = max(float(np.max(np.abs(cpu_losses))), 1.0e-12)
        bootstrap_relative = bootstrap_error / bootstrap_scale
        bootstrap_passes = bootstrap_error <= max(gpu_absolute, gpu_relative * bootstrap_scale)
        bootstrap_faster = gpu_bootstrap_seconds < (cpu_population_seconds + cpu_bootstrap_seconds)
        rows.append(
            {
                "workload": "bootstrap_population_calibration",
                "backend": "torch-cuda-fp64",
                "runtime_seconds": gpu_bootstrap_seconds,
                "throughput_per_second": draws * candidates * contracts / gpu_bootstrap_seconds,
                "speedup_vs_cpu": (cpu_population_seconds + cpu_bootstrap_seconds)
                / gpu_bootstrap_seconds,
                "absolute_error": bootstrap_error,
                "relative_error": bootstrap_relative,
                "peak_memory_bytes": int(torch.cuda.max_memory_allocated()),
                "peak_memory_kind": "torch_cuda_max_memory_allocated",
                "status": "MEASURED",
                "selection_match_fraction": selection_match,
                "passes_cpu_gpu_accuracy": bootstrap_passes,
                "faster_than_cpu": bootstrap_faster,
                "selected_backend": bool(bootstrap_passes and bootstrap_faster),
            }
        )
        rows[cpu_bootstrap_row_index]["selected_backend"] = not bool(
            bootstrap_passes and bootstrap_faster
        )

        cupy_result = _benchmark_cupy_mcs(
            spot_nodes=pde_shape[0],
            variance_nodes=pde_shape[1],
            time_steps=pde_shape[2],
        )
        cupy_price = cupy_result.get("price")
        cupy_runtime = float(cupy_result["runtime_seconds"])
        cupy_error = abs(float(cupy_price) - cpu_pde.price) if cupy_price is not None else None
        cupy_passes_accuracy = cupy_error is not None and cupy_error <= max(
            gpu_absolute, gpu_relative * abs(cpu_pde.price)
        )
        cupy_faster = cupy_runtime < cpu_pde_seconds
        rows.append(
            {
                "workload": "heston_pde_mcs_representative_grid",
                "backend": "cupyx-sparse-cuda-fp64",
                **cupy_result,
                "speedup_vs_cpu": cpu_pde_seconds / cupy_runtime,
                "absolute_error": cupy_error,
                "relative_error": (
                    cupy_error / abs(cpu_pde.price) if cupy_error is not None else None
                ),
                "passes_cpu_gpu_accuracy": cupy_passes_accuracy,
                "faster_than_cpu": cupy_faster,
                "selected_backend": bool(cupy_passes_accuracy and cupy_faster),
            }
        )
        # CPU remains selected unless the measured GPU implementation is both
        # accurate and faster; update the CPU row without changing PDE prices.
        rows[cpu_pde_row_index]["selected_backend"] = not bool(cupy_passes_accuracy and cupy_faster)
    else:
        rows.append(
            {
                "workload": "heston_pde_mcs_representative_grid",
                "backend": "cupyx-sparse-cuda-fp64",
                "status": "NOT_RUN_DEVICE_CPU",
                "selected_backend": False,
                "notes": "GPU execution was not requested; no performance claim made",
            }
        )

    frame = pd.DataFrame(rows)
    csv_path = destination / "final_compute_benchmarks.csv"
    frame.to_csv(csv_path, index=False)
    performance = _performance_table(frame)
    performance_path = destination / "T12_CPU_GPU_PERFORMANCE.csv"
    performance.to_csv(performance_path, index=False)
    if output_dir is None:
        report_table_dir = root / "reports/tables"
        report_table_dir.mkdir(parents=True, exist_ok=True)
        performance.to_csv(report_table_dir / "T12_CPU_GPU_PERFORMANCE.csv", index=False)
    ended = datetime.now(UTC).isoformat()
    payload: dict[str, Any] = {
        "schema_version": "1.0",
        "profile": profile,
        "device": device,
        "start_timestamp": started,
        "end_timestamp": ended,
        "rows": len(frame),
        "result": str(csv_path),
        "result_sha256": _sha256(csv_path),
        "canonical_t12": str(performance_path),
        "canonical_t12_sha256": _sha256(performance_path),
        "tolerance_registry": str(tolerance_path),
        "tolerance_registry_sha256": _sha256(tolerance_path),
        "all_rows_measured_or_explicitly_unavailable": bool(
            frame["status"].isin({"MEASURED", "ATTEMPTED_INCOMPATIBLE", "NOT_RUN_DEVICE_CPU"}).all()
        ),
        "selection_rule": "GPU only when measured accuracy passes and runtime improves",
    }
    json_path = destination / "final_compute_benchmarks.json"
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    payload["manifest"] = str(json_path)
    return payload
