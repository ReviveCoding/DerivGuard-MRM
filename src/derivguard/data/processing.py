"""Restartable processing jobs for acquired public option artifacts."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd

from .adapters import iter_public_spx_sample, read_cboe_marking_long
from .pipeline import run_audit_pipeline


def _write_parquet_atomic(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    frame.to_parquet(temporary, index=False)
    temporary.replace(path)


def _stage_paths(data_root: Path, dataset: str, stem: str) -> dict[str, Path]:
    return {
        stage: data_root / stage / dataset / f"{stem}.parquet"
        for stage in ("normalized", "quality_flagged", "filtered", "processed")
    }


def process_historical_spx_sample(
    archive: Path,
    data_root: Path,
    *,
    resume: bool = True,
) -> dict[str, Any]:
    """Process all 127 SPX/SPXW dates in bounded daily chunks."""
    quality_totals: Counter[str] = Counter()
    rows_by_stage: Counter[str] = Counter()
    files_processed = 0
    files_resumed = 0
    for member, raw in iter_public_spx_sample(archive):
        stem = Path(member).stem
        paths = _stage_paths(data_root, "historical_spx_sample", stem)
        if resume and all(path.exists() for path in paths.values()):
            files_resumed += 1
            # Durable per-date summaries allow an exact resume without rereading Parquet.
            sidecar = paths["quality_flagged"].with_suffix(".summary.json")
            if not sidecar.exists():
                raise FileNotFoundError(f"resume sidecar missing: {sidecar}")
            summary = json.loads(sidecar.read_text(encoding="utf-8"))
            quality_totals.update(summary["quality_summary"])
            rows_by_stage.update(summary["rows_by_stage"])
            continue
        result = run_audit_pipeline(
            raw,
            source_id="historicaldata_spx_sample",
            source_file=member,
        )
        _write_parquet_atomic(result.normalized, paths["normalized"])
        _write_parquet_atomic(result.quality_flagged, paths["quality_flagged"])
        _write_parquet_atomic(result.filtered, paths["filtered"])
        _write_parquet_atomic(result.model_ready, paths["processed"])
        stage_counts = {
            "normalized": len(result.normalized),
            "quality_flagged": len(result.quality_flagged),
            "filtered": len(result.filtered),
            "processed": len(result.model_ready),
        }
        sidecar = paths["quality_flagged"].with_suffix(".summary.json")
        sidecar.write_text(
            json.dumps(
                {"quality_summary": result.quality_summary, "rows_by_stage": stage_counts},
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        quality_totals.update(result.quality_summary)
        rows_by_stage.update(stage_counts)
        files_processed += 1
    return {
        "dataset": "historical_spx_sample",
        "daily_files_processed": files_processed,
        "daily_files_resumed": files_resumed,
        "rows_by_stage": dict(sorted(rows_by_stage.items())),
        "quality_summary": dict(sorted(quality_totals.items())),
    }


def process_cboe_marking(path: Path, data_root: Path) -> dict[str, Any]:
    raw = read_cboe_marking_long(path)
    result = run_audit_pipeline(
        raw,
        source_id="cboe_marking_1500ct",
        source_file=path.name,
    )
    paths = _stage_paths(data_root, "cboe_marking_1500ct", path.stem)
    _write_parquet_atomic(result.normalized, paths["normalized"])
    _write_parquet_atomic(result.quality_flagged, paths["quality_flagged"])
    _write_parquet_atomic(result.filtered, paths["filtered"])
    _write_parquet_atomic(result.model_ready, paths["processed"])
    return {
        "dataset": "cboe_marking_1500ct",
        "rows_by_stage": {
            "normalized": len(result.normalized),
            "quality_flagged": len(result.quality_flagged),
            "filtered": len(result.filtered),
            "processed": len(result.model_ready),
        },
        "quality_summary": result.quality_summary,
        "classification": "PROSPECTIVE_LOCKED_TEST",
    }
