"""Explicit RAW -> NORMALIZED -> QUALITY_FLAGGED -> FILTERED -> MODEL_READY pipeline."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from .quality import DataQualityConfig, audit_option_quotes
from .settlement import classify_settlement

CANONICAL_ALIASES: dict[str, tuple[str, ...]] = {
    "contract": ("contract", "option_symbol", "option symbol"),
    "underlying": ("underlying", "underlying_symbol", "underlying symbol"),
    "quote_date": ("quote_date", "quote date", "trade_date"),
    "quote_datetime": ("quote_datetime", "quote datetime", "quote_time", "timestamp"),
    "expiration": ("expiration", "expiration_date", "expiry"),
    "strike": ("strike", "strike_price"),
    "option_type": ("option_type", "type", "call_put", "put_call"),
    "bid": ("bid", "bid_price"),
    "ask": ("ask", "ask_price"),
    "bid_size": ("bid_size", "bid size"),
    "ask_size": ("ask_size", "ask size"),
    "volume": ("volume", "trade_volume", "trade volume"),
    "open_interest": ("open_interest", "open interest"),
    "underlying_price": ("underlying_close", "active_underlying_price", "underlying price"),
    "settlement_time": ("settlement_time", "settlement time"),
}


@dataclass(frozen=True)
class PipelineResult:
    raw: pd.DataFrame
    normalized: pd.DataFrame
    quality_flagged: pd.DataFrame
    filtered: pd.DataFrame
    model_ready: pd.DataFrame
    quality_summary: dict[str, int]


def _canonical_name(name: str) -> str:
    return name.strip().lower().replace("-", "_")


def normalize_quotes(
    raw: pd.DataFrame,
    *,
    source_id: str,
    source_file: str,
    aliases: Mapping[str, tuple[str, ...]] = CANONICAL_ALIASES,
) -> pd.DataFrame:
    """Normalize names/types and assign deterministic source-row identities."""
    data = raw.copy()
    standardized = {_canonical_name(str(column)): str(column) for column in data.columns}
    rename: dict[str, str] = {}
    for canonical, candidates in aliases.items():
        for candidate in candidates:
            existing = standardized.get(_canonical_name(candidate))
            if existing is not None:
                rename[existing] = canonical
                break
    data = data.rename(columns=rename)
    data.insert(0, "source_row_number", range(1, len(data) + 1))
    identifiers = [
        hashlib.sha256(f"{source_id}|{source_file}|{number}".encode()).hexdigest()
        for number in data["source_row_number"]
    ]
    data.insert(0, "source_row_id", identifiers)
    data.insert(1, "source_id", source_id)
    data.insert(2, "source_file", source_file)
    for column in (
        "strike",
        "bid",
        "ask",
        "bid_size",
        "ask_size",
        "volume",
        "open_interest",
        "underlying_price",
    ):
        if column in data:
            data[column] = pd.to_numeric(data[column], errors="coerce")
    for column in ("quote_date", "expiration"):
        if column in data:
            data[column] = pd.to_datetime(data[column], errors="coerce").dt.date
    if "option_type" in data:
        replacements = {"c": "call", "call": "call", "p": "put", "put": "put"}
        data["option_type"] = (
            data["option_type"].astype("string").str.strip().str.lower().map(replacements)
        )
    root = data["underlying"] if "underlying" in data else pd.Series([None] * len(data))
    settlement = (
        data["settlement_time"] if "settlement_time" in data else pd.Series([None] * len(data))
    )
    data["settlement_class"] = [
        classify_settlement(r, s) for r, s in zip(root, settlement, strict=True)
    ]
    return data


def run_audit_pipeline(
    raw: pd.DataFrame,
    *,
    source_id: str,
    source_file: str,
    quality_config: DataQualityConfig | None = None,
) -> PipelineResult:
    raw_stage = raw.copy()
    normalized = normalize_quotes(raw_stage, source_id=source_id, source_file=source_file)
    flagged, summary = audit_option_quotes(normalized, quality_config)
    # Excluded rows remain durable in QUALITY_FLAGGED with their source identity.
    filtered = flagged.loc[flagged["inclusion_status"] == "INCLUDED"].copy()
    model_ready = filtered.copy()
    return PipelineResult(raw_stage, normalized, flagged, filtered, model_ready, summary)


def write_pipeline_artifacts(
    result: PipelineResult, output_root: Path, stem: str
) -> dict[str, Path]:
    """Persist each explicit stage; quality-flagged retains every exclusion."""
    paths: dict[str, Path] = {}
    stages = {
        "raw": result.raw,
        "normalized": result.normalized,
        "quality_flagged": result.quality_flagged,
        "filtered": result.filtered,
        "model_ready": result.model_ready,
    }
    for stage, frame in stages.items():
        directory = output_root / stage
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f"{stem}.parquet"
        frame.to_parquet(path, index=False)
        paths[stage] = path
    return paths
