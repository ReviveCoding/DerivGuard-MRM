"""Put-call-parity forward and discount-factor estimators F0--F3."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import numpy.typing as npt
import pandas as pd

ForwardMethod = Literal["F0", "F1", "F2", "F3"]


@dataclass(frozen=True)
class ForwardEstimate:
    method: ForwardMethod
    forward: float
    discount_factor: float
    intercept: float
    slope: float
    residual_rmse: float
    weighted_rmse: float
    observations: int
    iterations: int
    converged: bool


def matched_put_call_pairs(frame: pd.DataFrame) -> pd.DataFrame:
    """Build one matched call/put midpoint pair per strike.

    If duplicate quotes exist, the narrowest nonnegative-spread quote is used
    deterministically. This routine intentionally does not synchronize EOD
    observations; downstream reports must retain the source timing label.
    """
    required = {"strike", "option_type", "bid", "ask"}
    if missing := required - set(frame.columns):
        raise ValueError(f"missing put-call pair fields: {sorted(missing)}")
    data = frame.copy()
    data["midpoint"] = (
        pd.to_numeric(data["bid"], errors="coerce") + pd.to_numeric(data["ask"], errors="coerce")
    ) / 2.0
    data["spread"] = pd.to_numeric(data["ask"], errors="coerce") - pd.to_numeric(
        data["bid"], errors="coerce"
    )
    data["option_type"] = data["option_type"].astype("string").str.lower()
    data = data.dropna(subset=["strike", "midpoint", "spread"])
    data = data[data["spread"] >= 0].sort_values(["strike", "option_type", "spread"])
    data = data.drop_duplicates(["strike", "option_type"], keep="first")
    prices = data.pivot(index="strike", columns="option_type", values="midpoint")
    spreads = data.pivot(index="strike", columns="option_type", values="spread")
    if not {"call", "put"}.issubset(prices.columns):
        return pd.DataFrame(columns=["strike", "call", "put", "call_spread", "put_spread"])
    result = pd.DataFrame(
        {
            "call": prices["call"],
            "put": prices["put"],
            "call_spread": spreads["call"],
            "put_spread": spreads["put"],
        }
    ).dropna()
    return result.reset_index()


def compare_forward_estimators(frame: pd.DataFrame) -> list[ForwardEstimate]:
    pairs = matched_put_call_pairs(frame)
    if len(pairs) < 3:
        return []
    results: list[ForwardEstimate] = []
    for method in ("F0", "F1", "F2", "F3"):
        call_spreads = pairs["call_spread"].to_numpy() if method in {"F1", "F3"} else None
        put_spreads = pairs["put_spread"].to_numpy() if method in {"F1", "F3"} else None
        results.append(
            estimate_forward_discount(
                pairs["strike"].to_numpy(),
                pairs["call"].to_numpy(),
                pairs["put"].to_numpy(),
                method=method,
                call_spreads=call_spreads,
                put_spreads=put_spreads,
            )
        )
    return results


def estimate_grouped_forwards(
    frame: pd.DataFrame,
    *,
    group_columns: tuple[str, ...] = ("quote_date", "expiration", "settlement_class"),
) -> pd.DataFrame:
    """Run F0--F3 by complete market slice and retain failures explicitly."""
    missing = set(group_columns) - set(frame.columns)
    if missing:
        raise ValueError(f"missing forward grouping columns: {sorted(missing)}")
    records: list[dict[str, object]] = []
    for keys, group in frame.groupby(list(group_columns), dropna=False):
        key_tuple = keys if isinstance(keys, tuple) else (keys,)
        identity = dict(zip(group_columns, key_tuple, strict=True))
        pairs = matched_put_call_pairs(group)
        if len(pairs) < 3:
            records.append(
                {
                    **identity,
                    "method": None,
                    "status": "UNAVAILABLE",
                    "reason": "fewer than three matched put-call strikes",
                    "matched_pairs": len(pairs),
                }
            )
            continue
        for method in ("F0", "F1", "F2", "F3"):
            call_spreads = pairs["call_spread"].to_numpy() if method in {"F1", "F3"} else None
            put_spreads = pairs["put_spread"].to_numpy() if method in {"F1", "F3"} else None
            try:
                estimate = estimate_forward_discount(
                    pairs["strike"].to_numpy(),
                    pairs["call"].to_numpy(),
                    pairs["put"].to_numpy(),
                    method=method,
                    call_spreads=call_spreads,
                    put_spreads=put_spreads,
                )
                records.append(
                    {
                        **identity,
                        **estimate.__dict__,
                        "status": "SUCCESS" if estimate.converged else "FAILED_DIAGNOSTIC",
                        "reason": None if estimate.converged else "Huber IRLS did not converge",
                        "matched_pairs": len(pairs),
                    }
                )
            except ValueError as exc:
                records.append(
                    {
                        **identity,
                        "method": method,
                        "status": "FAILED_DIAGNOSTIC",
                        "reason": str(exc),
                        "matched_pairs": len(pairs),
                    }
                )
    return pd.DataFrame.from_records(records)


def _as_vector(value: npt.ArrayLike, name: str) -> npt.NDArray[np.float64]:
    array = np.asarray(value, dtype=np.float64)
    if array.ndim != 1 or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be a finite one-dimensional array")
    return array


def _weighted_fit(
    strikes: npt.NDArray[np.float64],
    differences: npt.NDArray[np.float64],
    weights: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    design = np.column_stack((np.ones_like(strikes), strikes))
    root_w = np.sqrt(weights)
    beta, _, rank, _ = np.linalg.lstsq(design * root_w[:, None], differences * root_w, rcond=None)
    if rank != 2:
        raise ValueError("put-call pairs do not identify intercept and slope")
    return np.asarray(beta, dtype=np.float64)


def estimate_forward_discount(
    strikes: npt.ArrayLike,
    calls: npt.ArrayLike,
    puts: npt.ArrayLike,
    *,
    method: ForwardMethod = "F3",
    call_spreads: npt.ArrayLike | None = None,
    put_spreads: npt.ArrayLike | None = None,
    spread_floor: float = 0.01,
    huber_delta: float = 1.345,
    max_iterations: int = 100,
    tolerance: float = 1.0e-10,
) -> ForwardEstimate:
    """Estimate ``C-P = D*F - D*K`` using F0/F1/F2/F3.

    F0 is OLS, F1 spread-weighted least squares, F2 Huber IRLS, and F3
    spread-weighted Huber IRLS. The spread weight is inverse variance using
    the root-sum-square of call and put quoted spreads.
    """
    k = _as_vector(strikes, "strikes")
    c = _as_vector(calls, "calls")
    p = _as_vector(puts, "puts")
    if not (len(k) == len(c) == len(p)) or len(k) < 3:
        raise ValueError("at least three equally sized call/put pairs are required")
    if np.any(k <= 0) or np.any(c < 0) or np.any(p < 0):
        raise ValueError("strikes must be positive and option prices nonnegative")
    if method not in {"F0", "F1", "F2", "F3"}:
        raise ValueError(f"unknown forward estimator: {method}")
    spread_weighted = method in {"F1", "F3"}
    robust = method in {"F2", "F3"}
    base_weights = np.ones_like(k)
    if spread_weighted:
        if call_spreads is None or put_spreads is None:
            raise ValueError(f"{method} requires call_spreads and put_spreads")
        cs = _as_vector(call_spreads, "call_spreads")
        ps = _as_vector(put_spreads, "put_spreads")
        if len(cs) != len(k) or len(ps) != len(k) or np.any(cs < 0) or np.any(ps < 0):
            raise ValueError("spreads must be nonnegative and match pair count")
        scale = np.maximum(np.hypot(cs, ps), spread_floor)
        base_weights = 1.0 / np.square(scale)
        base_weights /= np.mean(base_weights)
    differences = c - p
    weights = base_weights.copy()
    beta = _weighted_fit(k, differences, weights)
    converged = not robust
    iterations = 1
    if robust:
        for iteration in range(1, max_iterations + 1):
            iterations = iteration
            residual = differences - (beta[0] + beta[1] * k)
            median = np.median(residual)
            mad = np.median(np.abs(residual - median))
            robust_scale = max(1.4826 * mad, np.finfo(np.float64).eps)
            standardized = np.abs(residual) / robust_scale
            huber_weights = np.ones_like(standardized)
            outside = standardized > huber_delta
            huber_weights[outside] = huber_delta / standardized[outside]
            new_beta = _weighted_fit(k, differences, base_weights * huber_weights)
            if np.linalg.norm(new_beta - beta) <= tolerance * (1.0 + np.linalg.norm(beta)):
                beta = new_beta
                weights = base_weights * huber_weights
                converged = True
                break
            beta = new_beta
            weights = base_weights * huber_weights
    intercept, slope = float(beta[0]), float(beta[1])
    discount = -slope
    if not (0.0 < discount <= 1.5):
        raise ValueError(f"parity regression produced implausible discount factor {discount}")
    forward = intercept / discount
    if forward <= 0:
        raise ValueError(f"parity regression produced nonpositive forward {forward}")
    residual = differences - (intercept + slope * k)
    return ForwardEstimate(
        method=method,
        forward=forward,
        discount_factor=discount,
        intercept=intercept,
        slope=slope,
        residual_rmse=float(np.sqrt(np.mean(np.square(residual)))),
        weighted_rmse=float(np.sqrt(np.average(np.square(residual), weights=weights))),
        observations=len(k),
        iterations=iterations,
        converged=converged,
    )
