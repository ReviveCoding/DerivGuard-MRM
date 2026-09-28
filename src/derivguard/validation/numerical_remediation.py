"""Executable MC/PDE numerical-remediation study.

The entry point in this module is deliberately separate from report
generation.  It performs the calculations, records every observation, and
derives qualification status from the registered tolerances.  Research-scale
Monte Carlo uses one coordinated Torch/CUDA executor; PDE ADI remains CPU
because its sparse directional solves have not demonstrated GPU advantage.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import asdict
from datetime import UTC, datetime
from functools import partial
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml  # type: ignore[import-untyped]

from derivguard.development.heston_cf import HestonParameters, heston_price_adaptive
from derivguard.validation.heston_mc import (
    ValidationHestonParameters,
    heston_mc_numpy,
    heston_mc_torch,
)
from derivguard.validation.heston_pde import (
    PDEGrid,
    PDEHestonParameters,
    PDEScheme,
    heston_pde_price,
)
from derivguard.validation.quantlib_oracle import (
    QuantLibHestonParameters,
    quantlib_heston_price,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_write(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _mc_estimate(
    *,
    device: str,
    paths: int,
    steps: int,
    scheme: str,
    seed: int,
    option_type: str = "call",
    control_variate: bool = True,
    spot: float = 100.0,
    strike: float = 100.0,
    maturity: float = 1.0,
    raw_params: tuple[float, float, float, float, float] = (0.04, 2.0, 0.04, 0.5, -0.7),
    rate: float = 0.03,
    dividend_yield: float = 0.01,
) -> Any:
    params = ValidationHestonParameters(*raw_params)
    kwargs: dict[str, Any] = {
        "paths": paths,
        "steps": steps,
        "scheme": scheme,
        "seed": seed,
        "antithetic": True,
        "control_variate": control_variate,
        "option_type": option_type,
    }
    if device.startswith("cuda"):
        return heston_mc_torch(
            spot,
            strike,
            maturity,
            params,
            rate,
            dividend_yield,
            device=device,
            chunk_paths=min(paths, 262_144),
            **kwargs,
        )
    return heston_mc_numpy(spot, strike, maturity, params, rate, dividend_yield, **kwargs)


def _record_pde_run(
    scheme: PDEScheme,
    study: str,
    level: int,
    spot_nodes: int,
    variance_nodes: int,
    time_steps: int,
    spot_max_multiple: float,
    run_variance_max: float,
    *,
    rows: list[dict[str, Any]],
    case_name: str,
    spot: float,
    strike: float,
    maturity: float,
    params: PDEHestonParameters,
    cf_reference: float,
    quantlib_price: float,
    quantlib_effective_maturity: float,
    mc_comparator: Any,
) -> None:
    """Execute one measured ADI grid and append its full evidence row."""

    before = time.perf_counter()
    result = heston_pde_price(
        spot,
        strike,
        maturity,
        params,
        0.03,
        0.01,
        scheme=scheme,
        grid=PDEGrid(
            spot_nodes=spot_nodes,
            variance_nodes=variance_nodes,
            time_steps=time_steps,
            spot_max_multiple=spot_max_multiple,
            variance_max=run_variance_max,
            nonuniform=True,
            rannacher_steps=2,
        ),
    )
    rows.append(
        {
            **asdict(result),
            "case": case_name,
            "study": study,
            "level": level,
            "grid_type": "nonuniform_sinh",
            "spot_max_multiple": spot_max_multiple,
            "cf_reference": cf_reference,
            "quantlib_reference": quantlib_price,
            "quantlib_effective_maturity": quantlib_effective_maturity,
            "mc_qem_price": mc_comparator.price,
            "mc_qem_standard_error": mc_comparator.standard_error,
            "mc_qem_confidence_low": mc_comparator.confidence_low,
            "mc_qem_confidence_high": mc_comparator.confidence_high,
            "pde_inside_mc_qem_95pct_ci": (
                mc_comparator.confidence_low <= result.price <= mc_comparator.confidence_high
            ),
            "absolute_error_cf": abs(result.price - cf_reference),
            "absolute_error_quantlib": abs(result.price - quantlib_price),
            "runtime_seconds": time.perf_counter() - before,
        }
    )


def run_numerical_remediation(profile: str = "research", device: str = "cuda") -> dict[str, Any]:
    """Run and record the QE/QE-M and nonuniform ADI qualification program."""

    if profile not in {"smoke", "audit", "research"}:
        raise ValueError("profile must be smoke, audit, or research")
    root = Path.cwd()
    tolerance_path = root / "configs/numerical_tolerances.yaml"
    tolerance_document = yaml.safe_load(tolerance_path.read_text(encoding="utf-8"))
    tolerances = tolerance_document["tolerances"]
    started = datetime.now(UTC).isoformat()
    started_clock = time.perf_counter()

    mc_steps: tuple[int, ...]
    mc_seeds: range
    path_levels: tuple[int, ...]
    pde_levels: tuple[tuple[int, int, int], ...]
    pde_cases: tuple[
        tuple[str, float, float, float, tuple[float, float, float, float, float], float], ...
    ]

    if profile == "research":
        mc_paths, mc_steps, mc_seeds = 150_000, (32, 64, 128, 256), range(5)
        path_levels = (50_000, 100_000, 250_000)
        pde_levels = ((81, 41, 120), (161, 81, 240), (301, 121, 400))
        pde_cases = (
            ("base", 100.0, 100.0, 1.0, (0.04, 2.0, 0.04, 0.5, -0.7), 0.5),
            ("feller_violated", 100.0, 110.0, 1.5, (0.02, 0.8, 0.03, 0.8, -0.5), 1.5),
            ("positive_rho", 100.0, 110.0, 2.0, (0.09, 0.5, 0.06, 1.0, 0.9), 1.5),
        )
    elif profile == "audit":
        mc_paths, mc_steps, mc_seeds = 50_000, (32, 128, 256), range(3)
        path_levels = (20_000, 50_000, 100_000)
        pde_levels = ((61, 31, 80), (101, 51, 140), (161, 81, 240))
        pde_cases = (("base", 100.0, 100.0, 1.0, (0.04, 2.0, 0.04, 0.5, -0.7), 0.5),)
    else:
        mc_paths, mc_steps, mc_seeds = 20_000, (32, 128), range(2)
        path_levels = (10_000, 20_000)
        pde_levels = ((51, 25, 60), (81, 41, 100))
        pde_cases = (("base", 100.0, 100.0, 1.0, (0.04, 2.0, 0.04, 0.5, -0.7), 0.5),)

    cf_params = HestonParameters(0.04, 2.0, 0.04, 0.5, -0.7)
    mc_reference = float(heston_price_adaptive(100.0, 100.0, 1.0, cf_params, 0.03, 0.01))
    mc_rows: list[dict[str, Any]] = []
    for scheme in ("full_truncation", "qe", "qe_m"):
        for steps in mc_steps:
            for seed_offset in mc_seeds:
                seed = 20260927 + seed_offset
                before = time.perf_counter()
                estimate = _mc_estimate(
                    device=device,
                    paths=mc_paths,
                    steps=steps,
                    scheme=scheme,
                    seed=seed,
                )
                mc_rows.append(
                    {
                        **asdict(estimate),
                        "study": "time_step_seed",
                        "reference_price": mc_reference,
                        "bias": estimate.price - mc_reference,
                        "absolute_bias": abs(estimate.price - mc_reference),
                        "reference_in_95pct_ci": (
                            estimate.confidence_low <= mc_reference <= estimate.confidence_high
                        ),
                        "runtime_seconds": time.perf_counter() - before,
                    }
                )

    for paths in path_levels:
        before = time.perf_counter()
        estimate = _mc_estimate(
            device=device,
            paths=paths,
            steps=max(mc_steps),
            scheme="qe_m",
            seed=20261007,
        )
        mc_rows.append(
            {
                **asdict(estimate),
                "study": "path_count",
                "reference_price": mc_reference,
                "bias": estimate.price - mc_reference,
                "absolute_bias": abs(estimate.price - mc_reference),
                "reference_in_95pct_ci": (
                    estimate.confidence_low <= mc_reference <= estimate.confidence_high
                ),
                "runtime_seconds": time.perf_counter() - before,
            }
        )

    # Call-put parity with common random numbers isolates stock-martingale
    # behavior without using the discounted-stock control variate.
    parity_target = 100.0 * np.exp(-0.01) - 100.0 * np.exp(-0.03)
    for scheme in ("qe", "qe_m"):
        pair: dict[str, Any] = {}
        for option_type in ("call", "put"):
            pair[option_type] = _mc_estimate(
                device=device,
                paths=mc_paths,
                steps=min(mc_steps),
                scheme=scheme,
                seed=20261017,
                option_type=option_type,
                control_variate=False,
            )
        mc_rows.append(
            {
                "study": "martingale_parity",
                "scheme": scheme,
                "backend": pair["call"].backend,
                "paths": mc_paths,
                "steps": min(mc_steps),
                "seed": 20261017,
                "price": pair["call"].price - pair["put"].price,
                "standard_error": np.sqrt(
                    pair["call"].standard_error ** 2 + pair["put"].standard_error ** 2
                ),
                "reference_price": parity_target,
                "bias": pair["call"].price - pair["put"].price - parity_target,
                "absolute_bias": abs(pair["call"].price - pair["put"].price - parity_target),
                "reference_in_95pct_ci": None,
                "runtime_seconds": None,
            }
        )

    mc_frame = pd.DataFrame(mc_rows)
    aggregated = root / "results/aggregated"
    aggregated.mkdir(parents=True, exist_ok=True)
    mc_path = aggregated / "mc_qe_qem_audit.csv"
    mc_frame.to_csv(mc_path, index=False)

    pde_rows: list[dict[str, Any]] = []
    for case_name, spot, strike, maturity, raw_params, variance_max in pde_cases:
        params = PDEHestonParameters(*raw_params)
        developer_params = HestonParameters(*raw_params)
        ql_params = QuantLibHestonParameters(*raw_params)
        cf_reference = float(
            heston_price_adaptive(spot, strike, maturity, developer_params, 0.03, 0.01)
        )
        ql_result = quantlib_heston_price(spot, strike, maturity, ql_params, 0.03, 0.01)
        mc_comparator = _mc_estimate(
            device=device,
            paths=max(path_levels),
            steps=max(mc_steps),
            scheme="qe_m",
            seed=20261027,
            spot=spot,
            strike=strike,
            maturity=maturity,
            raw_params=raw_params,
        )
        adi_schemes: tuple[PDEScheme, ...] = (
            "modified_craig_sneyd",
            "hundsdorfer_verwer",
        )

        append_pde_result = partial(
            _record_pde_run,
            rows=pde_rows,
            case_name=case_name,
            spot=spot,
            strike=strike,
            maturity=maturity,
            params=params,
            cf_reference=cf_reference,
            quantlib_price=ql_result.price,
            quantlib_effective_maturity=ql_result.effective_maturity,
            mc_comparator=mc_comparator,
        )

        for scheme in adi_schemes:
            for level, (spot_nodes, variance_nodes, time_steps) in enumerate(pde_levels, start=1):
                append_pde_result(
                    scheme,
                    "combined_grid_convergence",
                    level,
                    spot_nodes,
                    variance_nodes,
                    time_steps,
                    4.0,
                    variance_max,
                )
            if profile == "research" and case_name == "base":
                # Isolate each resolution axis and both truncation domains;
                # combined-grid refinement alone cannot identify the source
                # of the observed discretization error.
                for level, spot_nodes in enumerate((81, 161, 301), start=1):
                    append_pde_result(
                        scheme, "spot_grid", level, spot_nodes, 81, 240, 4.0, variance_max
                    )
                for level, variance_nodes in enumerate((41, 81, 121), start=1):
                    append_pde_result(
                        scheme,
                        "variance_grid",
                        level,
                        161,
                        variance_nodes,
                        240,
                        4.0,
                        variance_max,
                    )
                for level, time_steps in enumerate((80, 160, 320), start=1):
                    append_pde_result(
                        scheme, "time_grid", level, 161, 81, time_steps, 4.0, variance_max
                    )
                for level, (spot_multiple, vmax_multiple) in enumerate(
                    ((3.0, 0.75), (4.0, 1.0), (5.0, 1.5)), start=1
                ):
                    append_pde_result(
                        scheme,
                        "domain_boundary",
                        level,
                        161,
                        81,
                        240,
                        spot_multiple,
                        variance_max * vmax_multiple,
                    )
    pde_frame = pd.DataFrame(pde_rows)
    pde_frame["successive_grid_change"] = (
        pde_frame.groupby(["case", "scheme", "study"], sort=False)["price"].diff().abs()
    )
    pde_path = aggregated / "pde_adi_convergence.csv"
    pde_frame.to_csv(pde_path, index=False)

    mc_tolerance = tolerances["monte_carlo"]
    mc_qualification = {
        "minimum_seed_requirement_met": len(tuple(mc_seeds))
        >= int(mc_tolerance["minimum_independent_seeds"]),
        "minimum_time_levels_met": len(mc_steps) >= int(mc_tolerance["minimum_convergence_levels"]),
        "all_finest_qem_seed_biases_within_3se": bool(
            (
                mc_frame.loc[
                    (mc_frame["study"] == "time_step_seed")
                    & (mc_frame["scheme"] == "qe_m")
                    & (mc_frame["steps"] == max(mc_steps)),
                    "absolute_bias",
                ]
                <= float(mc_tolerance["reference_bias_standard_error_multiple"])
                * mc_frame.loc[
                    (mc_frame["study"] == "time_step_seed")
                    & (mc_frame["scheme"] == "qe_m")
                    & (mc_frame["steps"] == max(mc_steps)),
                    "standard_error",
                ]
            ).all()
        ),
    }
    pde_tolerance = tolerances["pde"]
    convergence = pde_frame[pde_frame["study"] == "combined_grid_convergence"]
    finest = convergence[convergence["level"] == convergence["level"].max()]
    pde_qualification = {
        "minimum_grid_levels_met": len(pde_levels) >= int(pde_tolerance["minimum_grid_levels"]),
        "all_finest_reference_errors_pass": bool(
            (
                finest["absolute_error_cf"]
                <= np.maximum(
                    float(pde_tolerance["reference_price_absolute"]),
                    float(pde_tolerance["reference_price_relative"]) * finest["cf_reference"].abs(),
                )
            ).all()
        ),
        "all_finest_successive_changes_pass": bool(
            (
                finest["successive_grid_change"]
                <= np.maximum(
                    float(pde_tolerance["successive_grid_absolute"]),
                    float(pde_tolerance["successive_grid_relative"]) * finest["price"].abs(),
                )
            ).all()
        ),
        "nonnegative_within_registry_slack": bool(
            (pde_frame["minimum_grid_value"] >= -float(pde_tolerance["negative_price_slack"])).all()
        ),
    }
    ended = datetime.now(UTC).isoformat()
    payload: dict[str, Any] = {
        "schema_version": "1.0",
        "run_type": "NUMERICAL_REMEDIATION",
        "profile": profile,
        "device": device,
        "start_timestamp": started,
        "end_timestamp": ended,
        "runtime_seconds": time.perf_counter() - started_clock,
        "tolerance_registry": str(tolerance_path.relative_to(root)),
        "tolerance_registry_sha256": _sha256(tolerance_path),
        "mc_result": str(mc_path.relative_to(root)),
        "mc_result_sha256": _sha256(mc_path),
        "pde_result": str(pde_path.relative_to(root)),
        "pde_result_sha256": _sha256(pde_path),
        "mc_qualification": mc_qualification,
        "pde_qualification": pde_qualification,
        "mc_status": "PASS" if all(mc_qualification.values()) else "PARTIAL",
        "pde_status": "PASS" if all(pde_qualification.values()) else "PARTIAL",
        "notes": [
            "No tolerance was changed by this run.",
            "PDE qualification requires both reference accuracy and successive-grid convergence.",
            "Monte Carlo bias and sampling standard error are recorded separately for every seed.",
        ],
    }
    qualification_path = aggregated / "numerical_remediation_qualification.json"
    _json_write(qualification_path, payload)
    payload["qualification_result"] = str(qualification_path.relative_to(root))
    return payload
