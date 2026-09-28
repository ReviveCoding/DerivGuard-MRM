"""Bounded FP32-versus-FP64 Heston characteristic-function sensitivity study.

This module deliberately does not alter the frozen developer engine.  It
implements a benchmark-only Torch transcription with explicit precision so
that N10 can measure the loss from complex64/float32 against the canonical
NumPy FP64 implementation.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from derivguard.development.heston_cf import HestonParameters, heston_price_fixed_quad


@dataclass(frozen=True)
class PrecisionResult:
    case: str
    precision: str
    backend: str
    contracts: int
    runtime_seconds: float
    maximum_absolute_error: float
    maximum_relative_error: float
    mean_absolute_error: float
    finite_fraction: float
    status: str


def _torch_prices(
    strikes: np.ndarray,
    maturity: float,
    params: HestonParameters,
    *,
    device: str,
    precision: str,
    nodes: int = 128,
) -> np.ndarray:
    import torch

    real_dtype = torch.float64 if precision == "float64" else torch.float32
    complex_dtype = torch.complex128 if precision == "float64" else torch.complex64
    parameter = torch.tensor(
        [[params.v0, params.kappa, params.theta, params.sigma_v, params.rho]],
        dtype=real_dtype,
        device=device,
    )
    strike = torch.as_tensor(strikes, dtype=real_dtype, device=device)
    roots, raw_weights = np.polynomial.legendre.leggauss(nodes)
    u = torch.as_tensor(100.0 * (roots + 1.0), dtype=real_dtype, device=device)
    weights = torch.as_tensor(100.0 * raw_weights, dtype=real_dtype, device=device)
    uc = u.to(complex_dtype)[None, :]
    v0, kappa, theta, sigma, rho = (parameter[:, index : index + 1] for index in range(5))

    def cf(z: Any) -> Any:
        iu = 1j * z
        beta = kappa - rho * sigma * iu
        d = torch.sqrt(beta * beta + sigma * sigma * (z * z + iu))
        d = torch.where(torch.real(d) < 0.0, -d, d)
        g = (beta - d) / (beta + d)
        exp_dt = torch.exp(-d * maturity)
        log_term = torch.log1p(-g * exp_dt) - torch.log1p(-g)
        c = iu * (np.log(100.0) + 0.015 * maturity) + (kappa * theta / (sigma * sigma)) * (
            (beta - d) * maturity - 2.0 * log_term
        )
        dc = (beta - d) / (sigma * sigma) * (1.0 - exp_dt) / (1.0 - g * exp_dt)
        return torch.exp(c + dc * v0)

    phase = torch.exp(-1j * torch.log(strike)[:, None].to(complex_dtype) * uc)
    denominator = 1j * uc
    phi_minus_i = torch.full(
        (1, 1), 100.0 * np.exp(0.015 * maturity), dtype=complex_dtype, device=device
    )
    phi1, phi2 = cf(uc - 1j), cf(uc)
    i1 = torch.sum(
        weights[None, None, :]
        * torch.real(
            phase[None, :, :]
            * phi1[:, None, :]
            / (denominator[:, None, :] * phi_minus_i[:, :, None])
        ),
        dim=-1,
    )
    i2 = torch.sum(
        weights[None, None, :]
        * torch.real(phase[None, :, :] * phi2[:, None, :] / denominator[:, None, :]),
        dim=-1,
    )
    calls = 100.0 * (0.5 + i1 / np.pi) - strike[None, :] * np.exp(-0.015 * maturity) * (
        0.5 + i2 / np.pi
    )
    return np.asarray(calls[0].detach().cpu().numpy(), dtype=np.float64)


def run_precision_sensitivity(root: Path, device: str = "cuda") -> dict[str, Any]:
    """Execute and persist N10 without changing the frozen pricing release."""

    import torch

    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("N10 CUDA execution requested but CUDA is unavailable")
    torch_device = "cuda" if device == "cuda" else "cpu"
    cases = (
        ("base", 1.0, HestonParameters(0.04, 1.5, 0.04, 0.5, -0.7)),
        ("short_high_vov", 0.05, HestonParameters(0.09, 0.7, 0.06, 1.2, -0.9)),
        ("long_feller_violated", 3.0, HestonParameters(0.04, 0.4, 0.05, 0.9, -0.6)),
        ("positive_rho", 1.5, HestonParameters(0.06, 1.0, 0.05, 0.8, 0.75)),
    )
    strikes = np.linspace(55.0, 155.0, 64, dtype=np.float64)
    rows: list[PrecisionResult] = []
    for case, maturity, params in cases:
        reference = np.asarray(
            heston_price_fixed_quad(100.0, strikes, maturity, params, rate=0.015, nodes=128),
            dtype=np.float64,
        )
        for precision in ("float64", "float32"):
            if torch_device == "cuda":
                torch.cuda.synchronize()
            started = time.perf_counter()
            measured = _torch_prices(
                strikes, maturity, params, device=torch_device, precision=precision
            )
            if torch_device == "cuda":
                torch.cuda.synchronize()
            runtime = time.perf_counter() - started
            absolute = np.abs(measured - reference)
            relative = absolute / np.maximum(np.abs(reference), 1.0e-8)
            finite = float(np.mean(np.isfinite(measured)))
            rows.append(
                PrecisionResult(
                    case,
                    precision,
                    f"torch-{torch_device}",
                    len(strikes),
                    runtime,
                    float(np.nanmax(absolute)),
                    float(np.nanmax(relative)),
                    float(np.nanmean(absolute)),
                    finite,
                    "MEASURED" if finite == 1.0 else "NUMERICAL_FAILURE",
                )
            )
    output = root / "results/aggregated/precision_sensitivity.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(asdict(row) for row in rows)
    frame.to_csv(output, index=False)
    payload = {
        "status": "MEASURED",
        "experiment_id": "N10",
        "generated_at": datetime.now(UTC).isoformat(),
        "device": torch_device,
        "rows": len(frame),
        "cases": len(cases),
        "precisions": ["float64", "float32"],
        "maximum_fp64_absolute_error": float(
            frame.loc[frame["precision"] == "float64", "maximum_absolute_error"].max()
        ),
        "maximum_fp32_absolute_error": float(
            frame.loc[frame["precision"] == "float32", "maximum_absolute_error"].max()
        ),
        "scientific_rule": "FP64 remains mandatory; FP32 is diagnostic only",
        "artifact": str(output.relative_to(root)),
    }
    manifest = root / "results/aggregated/precision_sensitivity.json"
    manifest.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload
