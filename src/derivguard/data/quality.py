"""Auditable option-quote quality checks and static-arbitrage diagnostics."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class DataQualityConfig:
    critical_columns: tuple[str, ...] = (
        "quote_date",
        "expiration",
        "strike",
        "option_type",
        "bid",
        "ask",
    )
    price_bound_slack: float = 1.0e-8
    monotonicity_slack: float = 1.0e-8
    convexity_slack: float = 1.0e-8
    extreme_relative_spread: float = 1.0
    minimum_strikes_per_slice: int = 5
    minimum_maturities_per_date: int = 2


def _add(flags: list[set[str]], indices: np.ndarray | list[int], code: str) -> None:
    for index in indices:
        flags[int(index)].add(code)


def audit_option_quotes(
    frame: pd.DataFrame, config: DataQualityConfig | None = None
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Return all rows with flags; never silently drop an observation."""
    cfg = config or DataQualityConfig()
    data = frame.copy().reset_index(drop=True)
    missing_columns = set(cfg.critical_columns) - set(data.columns)
    if missing_columns:
        raise ValueError(f"missing critical schema columns: {sorted(missing_columns)}")
    if "source_row_id" not in data:
        raise ValueError("source_row_id is required for auditable exclusions")
    flags: list[set[str]] = [set() for _ in range(len(data))]

    missing = data[list(cfg.critical_columns)].isna().any(axis=1).to_numpy()
    _add(flags, np.flatnonzero(missing), "MISSING_CRITICAL")
    option_type = data["option_type"].astype("string").str.lower()
    _add(flags, np.flatnonzero(~option_type.isin(["call", "put"])), "MALFORMED_OPTION_TYPE")
    strike = pd.to_numeric(data["strike"], errors="coerce")
    bid = pd.to_numeric(data["bid"], errors="coerce")
    ask = pd.to_numeric(data["ask"], errors="coerce")
    _add(flags, np.flatnonzero(strike.isna() | (strike <= 0)), "INVALID_STRIKE")
    _add(flags, np.flatnonzero(bid.notna() & (bid < 0)), "NEGATIVE_BID")
    _add(flags, np.flatnonzero(ask.notna() & bid.notna() & (ask < bid)), "ASK_BELOW_BID")
    _add(flags, np.flatnonzero((bid == 0) & (ask == 0)), "UNUSABLE_ZERO_QUOTE")

    quote_dates = pd.to_datetime(data["quote_date"], errors="coerce")
    expirations = pd.to_datetime(data["expiration"], errors="coerce")
    _add(flags, np.flatnonzero(expirations < quote_dates), "EXPIRATION_BEFORE_VALUATION")
    if "settlement_class" in data:
        ambiguous = data["settlement_class"].astype("string") == "AMBIGUOUS"
        _add(flags, np.flatnonzero(ambiguous), "SETTLEMENT_AMBIGUOUS")

    duplicate_keys = [
        name
        for name in (
            "source_file",
            "quote_datetime",
            "quote_date",
            "contract",
            "expiration",
            "strike",
            "option_type",
        )
        if name in data
    ]
    if duplicate_keys:
        duplicates = data.duplicated(duplicate_keys, keep=False).to_numpy()
        _add(flags, np.flatnonzero(duplicates), "DUPLICATE_OBSERVATION")

    midpoint = (bid + ask) / 2.0
    spread = ask - bid
    relative_spread = spread / midpoint.where(midpoint > 0)
    data["midpoint"] = midpoint
    data["spread"] = spread
    data["relative_spread"] = relative_spread
    _add(
        flags,
        np.flatnonzero(relative_spread > cfg.extreme_relative_spread),
        "EXTREME_RELATIVE_SPREAD",
    )

    if {"forward", "discount_factor"}.issubset(data.columns):
        forward = pd.to_numeric(data["forward"], errors="coerce")
        discount = pd.to_numeric(data["discount_factor"], errors="coerce")
        call = option_type == "call"
        lower = discount * np.where(
            call, np.maximum(forward - strike, 0), np.maximum(strike - forward, 0)
        )
        upper = discount * np.where(call, forward, strike)
        _add(
            flags,
            np.flatnonzero(midpoint < lower - cfg.price_bound_slack),
            "INTRINSIC_BOUND_VIOLATION",
        )
        _add(
            flags, np.flatnonzero(midpoint > upper + cfg.price_bound_slack), "UPPER_BOUND_VIOLATION"
        )

    group_keys = [name for name in ("quote_date", "expiration", "option_type") if name in data]
    if len(group_keys) == 3:
        for _, group in data.assign(_k=strike, _m=midpoint).groupby(group_keys, dropna=False):
            valid = group.dropna(subset=["_k", "_m"]).sort_values("_k")
            indices = valid.index.to_numpy()
            prices = valid["_m"].to_numpy(float)
            strikes = valid["_k"].to_numpy(float)
            if len(valid) < cfg.minimum_strikes_per_slice:
                _add(flags, indices, "SPARSE_STRIKE_SLICE")
            if len(valid) < 2:
                continue
            kind = str(valid["option_type"].iloc[0]).lower()
            diffs = np.diff(prices)
            bad_edges = (
                np.flatnonzero(diffs > cfg.monotonicity_slack)
                if kind == "call"
                else np.flatnonzero(diffs < -cfg.monotonicity_slack)
            )
            for edge in bad_edges:
                _add(flags, indices[[edge, edge + 1]], "STRIKE_MONOTONICITY_VIOLATION")
            if len(valid) >= 3 and np.all(np.diff(strikes) > 0):
                slopes = np.diff(prices) / np.diff(strikes)
                bad_convex = np.flatnonzero(np.diff(slopes) < -cfg.convexity_slack)
                for edge in bad_convex:
                    _add(
                        flags, indices[[edge, edge + 1, edge + 2]], "BUTTERFLY_CONVEXITY_VIOLATION"
                    )

    maturity_keys = [name for name in ("quote_date",) if name in data]
    if maturity_keys:
        expiry_counts = data.groupby(maturity_keys, dropna=False)["expiration"].transform("nunique")
        _add(
            flags,
            np.flatnonzero(expiry_counts < cfg.minimum_maturities_per_date),
            "SPARSE_MATURITY_DATE",
        )

    pair_keys = [name for name in ("quote_date", "expiration", "strike") if name in data]
    if len(pair_keys) == 3 and {"forward", "discount_factor"}.issubset(data.columns):
        usable = data.assign(_type=option_type, _mid=midpoint).dropna(subset=["_mid"])
        for _, group in usable.groupby(pair_keys, dropna=False):
            calls = group[group["_type"] == "call"]
            puts = group[group["_type"] == "put"]
            if calls.empty or puts.empty:
                continue
            call_row, put_row = calls.iloc[0], puts.iloc[0]
            expected = float(call_row["discount_factor"]) * (
                float(call_row["forward"]) - float(call_row["strike"])
            )
            residual = float(call_row["_mid"]) - float(put_row["_mid"]) - expected
            combined_spread = max(float(call_row["spread"]) + float(put_row["spread"]), 1.0e-12)
            if abs(residual) > 2.0 * combined_spread:
                positions = data.index.get_indexer(pd.Index([call_row.name, put_row.name]))
                _add(flags, positions, "PARITY_RESIDUAL_WARNING")

    exclusionary = {
        "MISSING_CRITICAL",
        "MALFORMED_OPTION_TYPE",
        "INVALID_STRIKE",
        "NEGATIVE_BID",
        "ASK_BELOW_BID",
        "UNUSABLE_ZERO_QUOTE",
        "EXPIRATION_BEFORE_VALUATION",
        "SETTLEMENT_AMBIGUOUS",
        "DUPLICATE_OBSERVATION",
        "INTRINSIC_BOUND_VIOLATION",
        "UPPER_BOUND_VIOLATION",
    }
    data["quality_flags"] = ["|".join(sorted(row_flags)) for row_flags in flags]
    data["exclusion_codes"] = ["|".join(sorted(row_flags & exclusionary)) for row_flags in flags]
    data["inclusion_status"] = np.where(data["exclusion_codes"] == "", "INCLUDED", "EXCLUDED")
    data["exclusion_reason"] = data["exclusion_codes"].str.replace("|", "; ", regex=False)
    data["repair_flag"] = False
    counts = Counter(flag for row_flags in flags for flag in row_flags)
    counts["ROWS_TOTAL"] = len(data)
    counts["ROWS_INCLUDED"] = int((data["inclusion_status"] == "INCLUDED").sum())
    counts["ROWS_EXCLUDED"] = int((data["inclusion_status"] == "EXCLUDED").sum())
    return data, dict(sorted(counts.items()))
