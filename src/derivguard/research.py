# ruff: noqa: E501
"""Executable, provenance-aware research and reporting workflows.

The module deliberately separates observed measurements from capability
status.  Unsupported or unexecuted studies are recorded as such; no result is
back-filled to satisfy an acceptance gate.
"""

from __future__ import annotations

import hashlib
import json
import math
import platform
import subprocess
import sys
import time
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal, cast

import numpy as np
import pandas as pd

from derivguard.development.heston_cf import (
    HestonParameters,
    heston_call_price_torch,
    heston_price_fixed_quad,
)
from derivguard.synthetic.faults import (
    FAULT_DESCRIPTIONS,
    clean_validation_case,
    inject_fault,
)
from derivguard.synthetic.generation import SyntheticDesign, SyntheticTruth, stratified_design
from derivguard.synthetic.observation import observe_truth
from derivguard.validation.dve import (
    EvidenceRecord,
    baseline_scores,
    evaluate_locked,
    fit_dev_thresholds,
    make_record,
    run_component_ablations,
)
from derivguard.validation.heston_mc import ValidationHestonParameters, heston_mc_torch
from derivguard.validation.heston_pde import PDEGrid, PDEHestonParameters, heston_pde_price
from derivguard.validation.quantlib_oracle import QuantLibHestonParameters, quantlib_heston_price


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


def repository_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8"
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _git_commit(root: Path) -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def capture_environment() -> dict[str, Any]:
    """Record the qualified local environment and a real CUDA smoke result."""

    import cupy  # type: ignore[import-untyped]
    import numpy
    import QuantLib  # type: ignore[import-untyped]
    import scipy
    import torch

    root = repository_root()
    started = time.perf_counter()
    tensor = torch.arange(1_000_000, dtype=torch.float64, device="cuda")
    tensor_sum = float(torch.sum(tensor * tensor).item())
    torch.cuda.synchronize()
    torch_seconds = time.perf_counter() - started
    cupy_tensor = cupy.arange(1024, dtype=cupy.float64)
    cupy_sum = float(cupy.sum(cupy_tensor).get())
    props = torch.cuda.get_device_properties(0)
    import shutil

    free_bytes = shutil.disk_usage(root).free
    payload: dict[str, Any] = {
        "captured_at": utc_now(),
        "python": sys.version,
        "sys_executable": sys.executable,
        "architecture": platform.architecture(),
        "platform": platform.platform(),
        "numpy": numpy.__version__,
        "scipy": scipy.__version__,
        "torch": torch.__version__,
        "torch_cuda_runtime": torch.version.cuda,
        "torch_cuda_available": torch.cuda.is_available(),
        "cuda_device_count": torch.cuda.device_count(),
        "gpu_name": torch.cuda.get_device_name(0),
        "gpu_total_memory_bytes": props.total_memory,
        "quantlib": QuantLib.__version__,
        "cupy": cupy.__version__,
        "cupy_runtime": cupy.cuda.runtime.runtimeGetVersion(),
        "torch_fp64_smoke_sum": tensor_sum,
        "torch_fp64_smoke_seconds": torch_seconds,
        "cupy_fp64_smoke_sum": cupy_sum,
        "disk_free_bytes": free_bytes,
        "disk_reserve_bytes": 20 * 1024**3,
        "git_commit": _git_commit(root),
    }
    hardware = {
        "captured_at": payload["captured_at"],
        "cpu": platform.processor(),
        "machine": platform.machine(),
        "gpu_name": payload["gpu_name"],
        "gpu_total_memory_bytes": payload["gpu_total_memory_bytes"],
        "cuda_device_count": payload["cuda_device_count"],
        "disk_free_bytes": free_bytes,
        "disk_reserve_satisfied": bool(free_bytes and free_bytes >= 20 * 1024**3),
    }
    _write_json(root / "artifacts/system/environment.json", payload)
    _write_json(root / "artifacts/system/hardware_profile.json", hardware)
    return payload


def _relative_error(actual: np.ndarray, reference: np.ndarray) -> float:
    return float(np.max(np.abs(actual - reference) / np.maximum(np.abs(reference), 1.0e-12)))


def run_numerical_benchmarks(profile: str = "research") -> dict[str, Any]:
    """Run measured CPU/GPU CF, GPU Monte Carlo, PDE, and oracle checks."""

    import torch

    started_at = utc_now()
    root = repository_root()
    result_dir = root / "results/aggregated"
    result_dir.mkdir(parents=True, exist_ok=True)
    params = HestonParameters(0.04, 2.0, 0.04, 0.5, -0.7)
    qparams = PDEHestonParameters(0.04, 2.0, 0.04, 0.5, -0.7)
    vparams = ValidationHestonParameters(0.04, 2.0, 0.04, 0.5, -0.7)
    rows: list[dict[str, Any]] = []

    for label, parameter_count, strike_count in (("small", 1, 128), ("medium", 32, 256)):
        strikes_np = np.linspace(60.0, 140.0, strike_count)
        parameter_np = np.repeat(
            np.asarray([[params.v0, params.kappa, params.theta, params.sigma_v, params.rho]]),
            parameter_count,
            axis=0,
        )
        start = time.perf_counter()
        cpu = np.vstack(
            [
                np.asarray(heston_price_fixed_quad(100.0, strikes_np, 1.0, params, 0.03, 0.01))
                for _ in range(parameter_count)
            ]
        )
        cpu_seconds = time.perf_counter() - start
        strikes_gpu = torch.as_tensor(strikes_np, dtype=torch.float64, device="cuda")
        parameters_gpu = torch.as_tensor(parameter_np, dtype=torch.float64, device="cuda")
        _ = heston_call_price_torch(100.0, strikes_gpu, 1.0, parameters_gpu, 0.03, 0.01)
        torch.cuda.synchronize()
        start = time.perf_counter()
        gpu_tensor = heston_call_price_torch(100.0, strikes_gpu, 1.0, parameters_gpu, 0.03, 0.01)
        torch.cuda.synchronize()
        gpu_seconds = time.perf_counter() - start
        gpu = gpu_tensor.detach().cpu().numpy()
        rows.append(
            {
                "workload": f"heston_cf_{label}",
                "cpu_seconds": cpu_seconds,
                "gpu_seconds": gpu_seconds,
                "speedup": cpu_seconds / gpu_seconds,
                "max_absolute_error": float(np.max(np.abs(cpu - gpu))),
                "max_relative_error": _relative_error(gpu, cpu),
                "backend": "torch-cuda-fp64",
                "shape": f"{parameter_count}x{strike_count}",
            }
        )

    oracle = quantlib_heston_price(
        100.0,
        100.0,
        1.0,
        params=QuantLibHestonParameters(0.04, 2.0, 0.04, 0.5, -0.7),
        rate=0.03,
        dividend_yield=0.01,
    )
    cf_price = float(heston_price_fixed_quad(100.0, 100.0, 1.0, params, 0.03, 0.01))
    mc_paths = 250_000 if profile == "research" else 25_000
    step_grid = (32, 64, 128, 256) if profile == "research" else (16, 32)
    mc_convergence: list[dict[str, Any]] = []
    mc = None
    mc_seconds = 0.0
    for scheme in ("full_truncation", "qe"):
        for mc_steps in step_grid:
            start = time.perf_counter()
            estimate = heston_mc_torch(
                100.0,
                100.0,
                1.0,
                vparams,
                0.03,
                0.01,
                paths=mc_paths,
                steps=mc_steps,
                scheme=scheme,
                seed=20260927,
                chunk_paths=min(mc_paths, 131_072),
            )
            elapsed = time.perf_counter() - start
            mc_convergence.append(
                {
                    **asdict(estimate),
                    "runtime_seconds": elapsed,
                    "absolute_error_vs_cf": abs(estimate.price - cf_price),
                    "reference_inside_95pct_interval": (
                        estimate.confidence_low <= cf_price <= estimate.confidence_high
                    ),
                }
            )
            if scheme == "qe" and mc_steps == step_grid[-1]:
                mc = estimate
                mc_seconds = elapsed
    assert mc is not None
    mc_seed_robustness: list[dict[str, Any]] = []
    if profile == "research":
        for seed in range(20260927, 20260932):
            if seed == 20260927:
                estimate = mc
                elapsed = mc_seconds
            else:
                start = time.perf_counter()
                estimate = heston_mc_torch(
                    100.0,
                    100.0,
                    1.0,
                    vparams,
                    0.03,
                    0.01,
                    paths=mc_paths,
                    steps=step_grid[-1],
                    scheme="qe",
                    seed=seed,
                    chunk_paths=min(mc_paths, 131_072),
                )
                elapsed = time.perf_counter() - start
            mc_seed_robustness.append(
                {
                    **asdict(estimate),
                    "runtime_seconds": elapsed,
                    "absolute_error_vs_cf": abs(estimate.price - cf_price),
                    "reference_inside_95pct_interval": (
                        estimate.confidence_low <= cf_price <= estimate.confidence_high
                    ),
                }
            )
    rows.append(
        {
            "workload": "heston_mc_qe",
            "cpu_seconds": None,
            "gpu_seconds": mc_seconds,
            "speedup": None,
            "max_absolute_error": abs(mc.price - cf_price),
            "max_relative_error": abs(mc.price - cf_price) / abs(cf_price),
            "backend": mc.backend,
            "shape": f"{mc_paths} paths x {mc_steps} steps",
            "standard_error": mc.standard_error,
            "confidence_low": mc.confidence_low,
            "confidence_high": mc.confidence_high,
            "oom_retries": mc.oom_retries,
        }
    )

    pde_rows: list[dict[str, Any]] = []
    pde_schemes: tuple[Literal["implicit_euler", "crank_nicolson"], ...] = (
        "implicit_euler",
        "crank_nicolson",
    )
    pde_grids = (
        PDEGrid(81, 41, 100, 4.0, 1.0),
        PDEGrid(121, 61, 150, 4.0, 1.0),
        PDEGrid(161, 81, 200, 4.0, 1.0),
    )
    for scheme in pde_schemes:
        for grid_index, grid in enumerate(pde_grids):
            start = time.perf_counter()
            pde = heston_pde_price(100.0, 100.0, 1.0, qparams, 0.03, 0.01, grid=grid, scheme=scheme)
            elapsed = time.perf_counter() - start
            pde_rows.append(
                {
                    "scheme": scheme,
                    "grid_index": grid_index,
                    "price": pde.price,
                    "runtime_seconds": elapsed,
                    "absolute_error_vs_cf": abs(pde.price - cf_price),
                    "grid": asdict(grid),
                }
            )

    payload = {
        "run_id": f"numerical-{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}",
        "started_at": started_at,
        "captured_at": utc_now(),
        "status": "PARTIAL_PDE_TOLERANCE_FAIL",
        "experiment_ids": ["N02", "N03", "N04", "N05", "N06", "N08", "N09"],
        "config_hash": _sha256(root / "configs/numerical_tolerances.yaml"),
        "hardware_profile_hash": _sha256(root / "artifacts/system/hardware_profile.json"),
        "profile": profile,
        "device": torch.cuda.get_device_name(0),
        "precision": "float64",
        "cf_price": cf_price,
        "quantlib_price": oracle.price,
        "cf_quantlib_absolute_error": abs(cf_price - oracle.price),
        "mc": asdict(mc),
        "mc_reference_inside_95pct_interval": (mc.confidence_low <= cf_price <= mc.confidence_high),
        "mc_convergence": mc_convergence,
        "mc_seed_robustness": mc_seed_robustness,
        "pde": pde_rows,
        "benchmarks": rows,
        "pde_gate_tolerance": 5.0e-4,
        "pde_gate_pass": all(
            row["absolute_error_vs_cf"] <= 5.0e-4
            for row in pde_rows
            if row["grid_index"] == len(pde_grids) - 1
        ),
    }
    _write_json(result_dir / "numerical_benchmarks.json", payload)
    pd.DataFrame(mc_convergence).to_csv(result_dir / "mc_convergence.csv", index=False)
    pd.DataFrame(mc_seed_robustness).to_csv(result_dir / "mc_seed_robustness.csv", index=False)
    pd.DataFrame(rows).to_csv(result_dir / "T12_CPU_GPU_PERFORMANCE.csv", index=False)
    pd.DataFrame(
        [
            {
                "engine": "Developer CF",
                "price": cf_price,
                "absolute_error_vs_cf": 0.0,
            },
            {
                "engine": "QuantLib analytic Heston",
                "price": oracle.price,
                "absolute_error_vs_cf": abs(oracle.price - cf_price),
            },
            {
                "engine": "Independent GPU MC QE",
                "price": mc.price,
                "absolute_error_vs_cf": abs(mc.price - cf_price),
            },
            *[
                {
                    "engine": f"Independent PDE {row['scheme']}",
                    "price": row["price"],
                    "absolute_error_vs_cf": row["absolute_error_vs_cf"],
                }
                for row in pde_rows
            ],
        ]
    ).to_csv(result_dir / "T03_NUMERICAL_ENGINE_VERIFICATION.csv", index=False)
    return payload


def _slice_design(design: SyntheticDesign, start: int, end: int) -> SyntheticDesign:
    values = {name: getattr(design, name)[start:end] for name in design.__dataclass_fields__}
    return SyntheticDesign(**values)


def _gpu_bs_truth(design: SyntheticDesign) -> SyntheticTruth:
    import torch

    device = torch.device("cuda")
    dtype = torch.float64
    s = torch.as_tensor(design.spot, dtype=dtype, device=device)
    k = torch.as_tensor(design.strike, dtype=dtype, device=device)
    t = torch.as_tensor(design.maturity, dtype=dtype, device=device)
    r = torch.as_tensor(design.rate, dtype=dtype, device=device)
    q = torch.as_tensor(design.dividend_yield, dtype=dtype, device=device)
    vol = torch.as_tensor(design.volatility, dtype=dtype, device=device)
    sqrt_t = torch.sqrt(t)
    d1 = (torch.log(s / k) + (r - q + 0.5 * vol * vol) * t) / (vol * sqrt_t)
    d2 = d1 - vol * sqrt_t
    cdf1 = 0.5 * (1.0 + torch.erf(d1 / math.sqrt(2.0)))
    cdf2 = 0.5 * (1.0 + torch.erf(d2 / math.sqrt(2.0)))
    pdf1 = torch.exp(-0.5 * d1 * d1) / math.sqrt(2.0 * math.pi)
    price = s * torch.exp(-q * t) * cdf1 - k * torch.exp(-r * t) * cdf2
    delta = torch.exp(-q * t) * cdf1
    gamma = torch.exp(-q * t) * pdf1 / (s * vol * sqrt_t)
    vega = s * torch.exp(-q * t) * pdf1 * sqrt_t
    torch.cuda.synchronize()
    return SyntheticTruth(
        design=design,
        model_id="SYN-BS",
        true_price=price.cpu().numpy(),
        true_delta=delta.cpu().numpy(),
        true_gamma=gamma.cpu().numpy(),
        true_vega=vega.cpu().numpy(),
        backend="torch-cuda-fp64",
        precision="float64",
    )


def _record_to_row(record: EvidenceRecord, scores: dict[str, float]) -> dict[str, Any]:
    return {
        "case_id": record.case_id,
        "partition": record.partition,
        "regime": record.regime,
        "fault_id": record.fault_id,
        "proxy_fault": record.proxy_fault,
        "material_error": record.material_error,
        "normalized_true_error": record.normalized_true_error,
        "fit_error": record.fit_error,
        "oos_error": record.oos_error,
        **record.evidence.as_dict(),
        **{f"score_{key}": value for key, value in scores.items()},
    }


def run_synthetic_dve(profile: str = "research") -> dict[str, Any]:
    """Execute a locked synthetic fault study with GPU-generated BS truth."""

    root = repository_root()
    started_at = utc_now()
    options_per_case = 24
    counts = {
        "research": {"DEV": 30, "VALIDATION": 10, "LOCKED_TEST": 20},
        "audit": {"DEV": 10, "VALIDATION": 4, "LOCKED_TEST": 6},
        "smoke": {"DEV": 3, "VALIDATION": 2, "LOCKED_TEST": 3},
    }[profile]
    faults = tuple(FAULT_DESCRIPTIONS)
    case_total = sum(counts.values()) * len(faults)
    design = stratified_design(case_total * options_per_case, seed=20260927)
    started = time.perf_counter()
    truth_all = _gpu_bs_truth(design)
    gpu_truth_seconds = time.perf_counter() - started
    records: list[EvidenceRecord] = []
    cursor = 0
    partition_offsets = {"DEV": 0, "VALIDATION": 100_000, "LOCKED_TEST": 200_000}
    for partition, repetitions in counts.items():
        for fault_index, fault in enumerate(faults):
            for replicate in range(repetitions):
                start, end = cursor * options_per_case, (cursor + 1) * options_per_case
                surface_design = _slice_design(design, start, end)
                truth = SyntheticTruth(
                    surface_design,
                    "SYN-BS",
                    truth_all.true_price[start:end],
                    truth_all.true_delta[start:end],
                    truth_all.true_gamma[start:end],
                    truth_all.true_vega[start:end],
                    truth_all.backend,
                    truth_all.precision,
                )
                surface = observe_truth(
                    truth,
                    seed=20260927 + partition_offsets[partition] + fault_index * 1000 + replicate,
                )
                case = clean_validation_case(surface)
                severity = 0.20 + 3.30 * ((replicate + 0.5) / repetitions)
                fault_case = inject_fault(case, fault, severity=severity)
                record = make_record(
                    f"{partition}-{fault}-{replicate:03d}",
                    partition,  # type: ignore[arg-type]
                    fault_case,
                    materiality_threshold=1.0,
                )
                records.append(record)
                cursor += 1

    dev = tuple(row for row in records if row.partition == "DEV")
    locked = tuple(row for row in records if row.partition == "LOCKED_TEST")
    frozen = fit_dev_thresholds(dev, target_fpr=0.05)
    metrics = evaluate_locked(locked, frozen)
    ablations = run_component_ablations(dev, locked)
    raw_rows = [
        _record_to_row(row, baseline_scores(row, frozen.evidence_scales)) for row in records
    ]
    raw = pd.DataFrame(raw_rows)
    raw_dir = root / "results/raw"
    agg_dir = root / "results/aggregated"
    stats_dir = root / "results/statistics"
    failure_dir = root / "results/failures"
    for directory in (raw_dir, agg_dir, stats_dir, failure_dir, root / "artifacts/checkpoints"):
        directory.mkdir(parents=True, exist_ok=True)
    raw.to_parquet(raw_dir / "synthetic_dve_records.parquet", index=False)
    _write_json(root / "artifacts/checkpoints/dve_thresholds_v1.json", frozen.to_dict())

    metric_rows = [asdict(value) for value in metrics.values()]
    pd.DataFrame(metric_rows).to_csv(agg_dir / "T08_DVE_BASELINES.csv", index=False)
    _write_json(agg_dir / "synthetic_detection_metrics.json", metric_rows)
    ablation_rows = [
        {
            "removed_dimension": name,
            "recall_change": value.recall_change,
            "auprc_change": value.auprc_change,
            "false_alarm_change": value.false_alarm_change,
            "miss_rate_change": value.miss_rate_change,
            "full_recall": value.full.recall_at_dev_5pct_fpr,
            "ablated_recall": value.ablated.recall_at_dev_5pct_fpr,
        }
        for name, value in ablations.items()
    ]
    full_metrics = metrics["V06"]
    non_regime_metrics = metrics["V05"]
    ablation_rows.append(
        {
            "removed_dimension": "regime_conditioning",
            "recall_change": (
                non_regime_metrics.recall_at_dev_5pct_fpr - full_metrics.recall_at_dev_5pct_fpr
            ),
            "auprc_change": non_regime_metrics.auprc - full_metrics.auprc,
            "false_alarm_change": (
                non_regime_metrics.false_alarm_rate - full_metrics.false_alarm_rate
            ),
            "miss_rate_change": (
                non_regime_metrics.material_risk_miss_rate - full_metrics.material_risk_miss_rate
            ),
            "full_recall": full_metrics.recall_at_dev_5pct_fpr,
            "ablated_recall": non_regime_metrics.recall_at_dev_5pct_fpr,
            "interpretation": "A09 compares V05 global DVE with V06 regime-conditioned DVE",
        }
    )

    locked_frame = raw.loc[raw["partition"] == "LOCKED_TEST"].copy()
    locked_frame["threshold_V06"] = [
        frozen.threshold("V06", regime) for regime in locked_frame["regime"]
    ]
    locked_frame["predicted_V06"] = locked_frame["score_V06"] > locked_frame["threshold_V06"]
    from sklearn.metrics import average_precision_score  # type: ignore[import-untyped]

    fault_labels = locked_frame["fault_id"] != "F00"
    fault_recall = float(locked_frame.loc[fault_labels, "predicted_V06"].mean())
    fault_fpr = float(locked_frame.loc[~fault_labels, "predicted_V06"].mean())
    fault_auprc = float(average_precision_score(fault_labels, locked_frame["score_V06"]))
    ablation_rows.append(
        {
            "removed_dimension": "materiality_prioritization",
            "recall_change": fault_recall - full_metrics.recall_at_dev_5pct_fpr,
            "auprc_change": fault_auprc - full_metrics.auprc,
            "false_alarm_change": fault_fpr - full_metrics.false_alarm_rate,
            "miss_rate_change": (1.0 - fault_recall) - full_metrics.material_risk_miss_rate,
            "full_recall": full_metrics.recall_at_dev_5pct_fpr,
            "ablated_recall": fault_recall,
            "interpretation": (
                "A11 treats every non-F00 mechanism as positive regardless of materiality"
            ),
        }
    )
    pd.DataFrame(ablation_rows).to_csv(agg_dir / "T09_DVE_ABLATION.csv", index=False)
    fault_summary_rows: list[dict[str, Any]] = []
    for fault_id, group in locked_frame.groupby("fault_id", observed=True):
        material = group["material_error"].astype(bool)
        fault_summary_rows.append(
            {
                "fault_id": fault_id,
                "observations": len(group),
                "material_cases": int(material.sum()),
                "recall": (
                    float(group.loc[material, "predicted_V06"].mean()) if material.any() else np.nan
                ),
                "false_positive_rate": (
                    float(group.loc[~material, "predicted_V06"].mean())
                    if (~material).any()
                    else np.nan
                ),
                "median_normalized_true_error": float(group["normalized_true_error"].median()),
                "proxy_fault_fraction": float(group["proxy_fault"].mean()),
            }
        )
    fault_summary = pd.DataFrame(fault_summary_rows)
    fault_summary.to_csv(agg_dir / "T07_SYNTHETIC_FAULT_DETECTION.csv", index=False)
    false_positive = locked_frame.loc[
        locked_frame["predicted_V06"] & ~locked_frame["material_error"]
    ]
    false_negative = locked_frame.loc[
        ~locked_frame["predicted_V06"] & locked_frame["material_error"]
    ]
    false_positive.to_parquet(failure_dir / "false_positive_cases.parquet", index=False)
    false_negative.to_parquet(failure_dir / "false_negative_cases.parquet", index=False)
    worst_oos = locked_frame.nlargest(50, "normalized_true_error").copy()
    worst_oos["selection_basis"] = (
        "synthetic normalized truth error; not a real temporal OOS outcome"
    )
    worst_oos.to_parquet(failure_dir / "worst_oos_cases.parquet", index=False)
    locked_frame.nlargest(50, "E_form").to_parquet(
        failure_dir / "largest_challenger_disagreement.parquet", index=False
    )
    locked_frame.nlargest(50, "E_param").to_parquet(
        failure_dir / "largest_parameter_jump.parquet", index=False
    )
    largest_hedge = locked_frame.nlargest(50, "E_greek").copy()
    largest_hedge["selection_basis"] = (
        "synthetic Greek-disagreement evidence; not realized hedge P&L"
    )
    largest_hedge.to_parquet(failure_dir / "largest_hedge_error.parquet", index=False)
    payload = {
        "run_id": f"synthetic-{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}",
        "started_at": started_at,
        "captured_at": utc_now(),
        "status": "DESCRIPTIVE_PARTIAL_NOT_PREREGISTERED",
        "experiment_ids": [f"S{index:02d}" for index in range(1, 11)]
        + [f"A{index:02d}" for index in range(1, 12)],
        "profile": profile,
        "seed": 20260927,
        "config_hash": _sha256(root / "configs/numerical_tolerances.yaml"),
        "hardware_profile_hash": _sha256(root / "artifacts/system/hardware_profile.json"),
        "truth_model": "SYN-BS",
        "truth_backend": "torch-cuda-fp64",
        "gpu_truth_seconds": gpu_truth_seconds,
        "case_count": case_total,
        "option_observation_count": case_total * options_per_case,
        "partition_case_counts": {
            partition: counts[partition] * len(faults) for partition in counts
        },
        "materiality_threshold_spread_units": 1.0,
        "threshold_config_hash": frozen.config_hash,
        "metrics": {key: asdict(value) for key, value in metrics.items()},
        "proxy_limitations": [
            "F01 and F02 are deterministic proxy faults because qualified Bates/LocalVol truth was unavailable at execution time.",
            "Synthetic observation parameters are explicit design assumptions, not fitted claims about the public sample.",
            "Numerical, calibration, challenger, and Greek evidence are deterministic synthetic perturbations initialized from latent truth; they form a fault harness, not independently priced market evidence.",
            "E_outcome is zero in this study because no genuinely prior outcome series exists.",
            "Regimes use only synthetic maturity, moneyness, and liquidity buckets; VIX conditioning is unavailable.",
        ],
    }
    _write_json(stats_dir / "synthetic_locked_test.json", payload)
    return payload


def freeze_developer_release() -> dict[str, Any]:
    """Freeze v1.0 only after measured DEV selection and validation confirmation."""

    root = repository_root()
    release_path = root / "artifacts/model_releases/model_release_v1.0.json"
    if release_path.exists():
        return cast(dict[str, Any], json.loads(release_path.read_text(encoding="utf-8")))
    selection_path = root / "results/aggregated/developer_method_selection.json"
    if not selection_path.exists():
        raise RuntimeError(
            "Developer v1.0 cannot be frozen before the measured DEV methodology-selection stage"
        )
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    if selection.get("status") != "SELECTED_ON_DEV_CONFIRMED_ON_VALIDATION":
        raise RuntimeError(
            "developer methodology has not passed DEV selection and VALIDATION confirmation"
        )
    source = root / "src/derivguard/development/heston_cf.py"
    calibration_source = root / "src/derivguard/development/calibration.py"
    selection_source = root / "src/derivguard/empirical.py"
    tolerance = root / "configs/numerical_tolerances.yaml"
    import yaml  # type: ignore[import-untyped]

    tolerance_payload = yaml.safe_load(tolerance.read_text(encoding="utf-8"))
    if tolerance_payload.get("registry_status") != "FROZEN_FOR_DEVELOPER_RELEASE_V1_0":
        raise RuntimeError("numerical tolerance registry must be frozen before release v1.0")
    release = {
        "release_id": "Developer Model Release v1.0",
        "model_id": "M10",
        "released_at": utc_now(),
        "status": "FROZEN_BEFORE_CALIBRATED_CONFIRMATORY_LOCKED_TEST",
        "intended_use": "European SPXW vanilla-option research valuation, Greeks, and model-risk research",
        "prohibited_use": [
            "live trading",
            "brokerage execution",
            "regulatory capital",
            "American or exotic options",
            "production institutional valuation",
        ],
        "methodology": "Little-Heston-Trap-style stable characteristic function with FP64 Gauss-Legendre integration",
        "parameter_schema": ["v0", "kappa", "theta", "sigma_v", "rho"],
        "objective": selection["selected_objective"],
        "optimizer": selection["selected_optimizer"],
        "constraints": selection["selected_feller_treatment"],
        "numerical_tolerances_file": str(tolerance.relative_to(root)),
        "source_sha256": _sha256(source),
        "source_hashes": {
            "development/heston_cf.py": _sha256(source),
            "development/calibration.py": _sha256(calibration_source),
            "empirical.py": _sha256(selection_source),
        },
        "config_sha256": _sha256(tolerance),
        "selection_evidence": str(selection_path.relative_to(root)),
        "selection_evidence_hash": _sha256(selection_path),
        "known_limitations": [
            "Public historical quotes are standing EOD observations without quote timestamps.",
            "The model remains restricted to European vanilla research use.",
        ],
    }
    _write_json(release_path, release)
    return release


def freeze_developer_release_v1_1() -> dict[str, Any]:
    """Freeze the release-conformance remediation before confirmatory v5.

    Version 1.0 remains immutable historical evidence.  Version 1.1 records
    the same DEV-selected C04/O01/unconstrained method together with proof
    that the real-date execution actually used that method without cross-date
    warm starts.
    """

    root = repository_root()
    release_path = root / "artifacts/model_releases/model_release_v1.1.json"
    if release_path.exists():
        return cast(dict[str, Any], json.loads(release_path.read_text(encoding="utf-8")))
    prior_path = root / "artifacts/model_releases/model_release_v1.0.json"
    methodology_path = root / "artifacts/calibration/empirical_methodology.json"
    selection_path = root / "results/aggregated/developer_method_selection.json"
    if not prior_path.exists() or not methodology_path.exists() or not selection_path.exists():
        raise RuntimeError("v1.0 and corrected empirical execution evidence are required")
    methodology = json.loads(methodology_path.read_text(encoding="utf-8"))
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    execution = methodology.get("real_date_calibration_execution", {})
    release_method = (
        selection.get("selected_objective"),
        selection.get("selected_optimizer"),
        selection.get("selected_feller_treatment"),
    )
    executed_method = (
        execution.get("objective"),
        execution.get("optimizer_id"),
        execution.get("feller_treatment"),
    )
    execution_complete = (
        methodology.get("methodology_selection_status") == "SELECTED_ON_DEV_CONFIRMED_ON_VALIDATION"
        and release_method == ("C04", "O01", "unconstrained")
        and executed_method == release_method
        and execution.get("warm_start_across_dates") is False
        and int(execution.get("jobs_completed", -1)) == int(execution.get("jobs_expected", -2))
        and int(execution.get("jobs_completed", 0)) > 0
        and int(execution.get("jobs_converged", 0)) > 0
    )
    if not execution_complete:
        raise RuntimeError("corrected C04/O01 real-date execution is incomplete or inconsistent")
    source = root / "src/derivguard/development/heston_cf.py"
    calibration_source = root / "src/derivguard/development/calibration.py"
    empirical_source = root / "src/derivguard/empirical.py"
    tolerance = root / "configs/numerical_tolerances.yaml"
    release = {
        "release_id": "Developer Model Release v1.1",
        "supersedes_for_new_validation": "Developer Model Release v1.0",
        "model_id": "M10",
        "released_at": utc_now(),
        "status": "FROZEN_BEFORE_CONFIRMATORY_V5_LOCKED_TEST",
        "intended_use": (
            "European SPXW vanilla-option research valuation, Greeks, and model-risk research"
        ),
        "prohibited_use": [
            "live trading",
            "brokerage execution",
            "regulatory capital",
            "American or exotic options",
            "production institutional valuation",
        ],
        "methodology": (
            "Little-Heston-Trap-style stable characteristic function with FP64 "
            "Gauss-Legendre integration"
        ),
        "parameter_schema": ["v0", "kappa", "theta", "sigma_v", "rho"],
        "objective": "C04",
        "optimizer": "O01",
        "constraints": "unconstrained",
        "real_date_execution_conformant": True,
        "real_date_jobs_expected": int(execution["jobs_expected"]),
        "real_date_jobs_converged": int(execution["jobs_converged"]),
        "real_date_convergence_rate": float(execution["convergence_rate"]),
        "cross_date_warm_start": False,
        "source_sha256": _sha256(source),
        "source_hashes": {
            "development/heston_cf.py": _sha256(source),
            "development/calibration.py": _sha256(calibration_source),
            "empirical.py": _sha256(empirical_source),
        },
        "config_sha256": _sha256(tolerance),
        "numerical_tolerances_file": str(tolerance.relative_to(root)),
        "selection_evidence": str(selection_path.relative_to(root)),
        "selection_evidence_hash": _sha256(selection_path),
        "execution_evidence": str(methodology_path.relative_to(root)),
        "execution_evidence_hash": _sha256(methodology_path),
        "prior_release_hash": _sha256(prior_path),
        "known_limitations": [
            "Public historical quotes are standing EOD observations without quote timestamps.",
            "Version 1.0 validation evidence is method-variant and retained as historical evidence.",
            "Non-converged real-date calibrations are retained as failures and excluded from model comparisons.",
            "The model remains restricted to European vanilla research use.",
        ],
    }
    _write_json(release_path, release)
    return release


def run_data_eda() -> dict[str, Any]:
    """Aggregate the acquired public sample without treating quotes as synchronized NBBO."""

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import polars as pl

    root = repository_root()
    processing_path = root / "artifacts/data/processing_summary.json"
    processing = json.loads(processing_path.read_text(encoding="utf-8"))
    summaries = sorted((root / "data/quality_flagged/historical_spx_sample").glob("*.summary.json"))
    daily: list[dict[str, Any]] = []
    for path in summaries:
        payload = json.loads(path.read_text(encoding="utf-8"))
        quality = payload["quality_summary"]
        daily.append(
            {
                "date": path.name[:10],
                "rows_total": quality.get("ROWS_TOTAL", 0),
                "rows_included": quality.get("ROWS_INCLUDED", 0),
                "rows_excluded": quality.get("ROWS_EXCLUDED", 0),
                "monotonicity_flags": quality.get("STRIKE_MONOTONICITY_VIOLATION", 0),
                "convexity_flags": quality.get("BUTTERFLY_CONVEXITY_VIOLATION", 0),
                "extreme_spread_flags": quality.get("EXTREME_RELATIVE_SPREAD", 0),
            }
        )
    daily_frame = pd.DataFrame(daily)
    table_dir = root / "reports/tables"
    figure_dir = root / "reports/figures"
    table_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)
    daily_frame.to_csv(table_dir / "data_coverage_daily.csv", index=False)

    pattern = str(root / "data/processed/historical_spx_sample/*.parquet")
    scan = pl.scan_parquet(pattern).filter(pl.col("underlying").is_in(["SPX", "SPXW"]))
    coverage = (
        scan.group_by(["underlying", "settlement_class"])
        .agg(
            pl.len().alias("rows"),
            pl.col("quote_date").n_unique().alias("dates"),
            pl.col("expiration").n_unique().alias("expirations"),
            pl.col("strike").n_unique().alias("strikes"),
            pl.col("relative_spread").median().alias("median_relative_spread"),
            pl.col("volume").median().alias("median_volume"),
            pl.col("open_interest").median().alias("median_open_interest"),
        )
        .collect(engine="streaming")
        .to_pandas()
    )
    coverage.to_csv(table_dir / "T01_DATA_COVERAGE_QUALITY.csv", index=False)
    distribution = (
        scan.select(
            [
                "underlying",
                "quote_date",
                "expiration",
                "strike",
                "underlying_price",
                "relative_spread",
                "iv",
                "volume",
                "open_interest",
            ]
        )
        .filter(pl.col("relative_spread").is_finite() & pl.col("iv").is_finite())
        .collect(engine="streaming")
    )
    sample_size = min(200_000, distribution.height)
    sample = distribution.sample(n=sample_size, seed=20260927).to_pandas()
    sample["moneyness"] = sample["strike"] / sample["underlying_price"]
    sample["dte"] = (
        pd.to_datetime(sample["expiration"]) - pd.to_datetime(sample["quote_date"])
    ).dt.days

    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.plot(pd.to_datetime(daily_frame["date"]), daily_frame["rows_included"], label="included")
    ax.plot(pd.to_datetime(daily_frame["date"]), daily_frame["rows_excluded"], label="excluded")
    ax.set(title="Historical public SPX/SPXW sample coverage", ylabel="rows", xlabel="quote date")
    ax.legend()
    fig.tight_layout()
    fig.savefig(figure_dir / "data_coverage.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    clipped = sample["relative_spread"].clip(upper=float(sample["relative_spread"].quantile(0.99)))
    ax.hist(clipped, bins=80)
    ax.set(
        title="Relative spread distribution (99th-percentile clipped)",
        xlabel="(ask-bid)/mid",
        ylabel="sample count",
    )
    fig.tight_layout()
    fig.savefig(figure_dir / "spread_distribution.png", dpi=160)
    plt.close(fig)

    surface = sample.loc[
        sample["moneyness"].between(0.65, 1.35) & sample["dte"].between(1, 730)
    ].sample(n=min(30_000, len(sample)), random_state=20260927)
    fig, ax = plt.subplots(figsize=(8, 5))
    points = ax.scatter(surface["moneyness"], surface["iv"], c=surface["dte"], s=3, alpha=0.25)
    fig.colorbar(points, ax=ax, label="DTE")
    ax.set(
        title="Observed implied volatility by moneyness",
        xlabel="K / underlying close proxy",
        ylabel="vendor IV",
    )
    fig.tight_layout()
    fig.savefig(figure_dir / "raw_iv_surface_projection.png", dpi=160)
    plt.close(fig)

    result = {
        "captured_at": utc_now(),
        "daily_surfaces": len(daily),
        "processing_summary": processing,
        "coverage": coverage.to_dict(orient="records"),
        "eda_sample_rows": sample_size,
        "limitations": [
            "All 4,294,301 source rows lack quote_time; historical observations are not treated as synchronized NBBO.",
            "Vendor IV and underlying_close are descriptive fields, not independently reconstructed intraday state.",
            "Static-arbitrage flags can reflect asynchrony and quote quality and are classified as data evidence.",
        ],
    }
    _write_json(root / "results/aggregated/data_eda.json", result)
    return result


def generate_reports() -> dict[str, Any]:
    """Generate current reports solely from measured machine-readable artifacts."""

    from derivguard.reporting.final_reports import generate_mandatory_reports

    return generate_mandatory_reports(repository_root())


def run_full(
    profile: str = "research", *, device: str = "cuda", resume: bool = False
) -> dict[str, Any]:
    """Invoke the dependency-ordered scientific pipeline."""

    from derivguard.pipeline import run_full_pipeline

    return run_full_pipeline(profile, device=device, resume=resume)
