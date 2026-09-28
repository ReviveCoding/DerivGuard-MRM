"""Acquire no-key Federal Reserve Economic Data research covariates.

The fredgraph CSV endpoint is an official public FRED distribution mechanism.
Treasury constant-maturity series are par-yield references, not derivatives
discount curves; consumers must label comparisons as sensitivities.
"""

from __future__ import annotations

import hashlib
import json
import urllib.request
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

FRED_SERIES = ("SOFR", "DGS1MO", "DGS3MO", "DGS6MO", "DGS1", "VIXCLS", "SP500")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def acquire_fred_covariates(
    root: Path,
    *,
    start: str = "2022-06-01",
    end: str = "2023-01-31",
    refresh: bool = False,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Download, archive, hash, and merge the declared public FRED series."""

    raw_directory = root / "data/raw/fred_public_covariates"
    raw_directory.mkdir(parents=True, exist_ok=True)
    frames: list[pd.DataFrame] = []
    artifacts: list[dict[str, object]] = []
    for series in FRED_SERIES:
        url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series}&cosd={start}&coed={end}"
        path = raw_directory / f"{series}_{start}_{end}.csv"
        if refresh or not path.exists():
            with urllib.request.urlopen(url, timeout=60) as response:
                payload = response.read()
            path.write_bytes(payload)
        frame = pd.read_csv(path, na_values=".")
        date_column = "DATE" if "DATE" in frame.columns else "observation_date"
        frame = frame.rename(columns={date_column: "quote_date"})
        frame["quote_date"] = pd.to_datetime(frame["quote_date"])
        frame[series] = pd.to_numeric(frame[series], errors="coerce")
        frames.append(frame[["quote_date", series]])
        artifacts.append(
            {
                "series": series,
                "provider": "Federal Reserve Bank of St. Louis (FRED)",
                "source_url": url,
                "retrieval_timestamp": datetime.now(UTC).isoformat(),
                "sha256": _sha256(path),
                "file_size": path.stat().st_size,
                "row_count": len(frame),
                "minimum_date": frame["quote_date"].min().date().isoformat(),
                "maximum_date": frame["quote_date"].max().date().isoformat(),
                "schema_version": "fredgraph-csv-v1",
                "license_source_note": "official public FRED series; cite originating agency",
                "redistribution_note": "retain series attribution and FRED terms",
                "known_issues": (
                    "DGS tenors are Treasury constant-maturity par yields, not exact "
                    "derivatives discount factors"
                    if series.startswith("DGS")
                    else None
                ),
            }
        )
    combined = frames[0]
    for frame in frames[1:]:
        combined = combined.merge(frame, on="quote_date", how="outer")
    combined = combined.sort_values("quote_date")
    processed = root / "data/processed/public_covariates.parquet"
    processed.parent.mkdir(parents=True, exist_ok=True)
    combined.to_parquet(processed, index=False)
    manifest: dict[str, Any] = {
        "source_id": "fred_public_covariates",
        "retrieval_completed_at": datetime.now(UTC).isoformat(),
        "query": {"start": start, "end": end, "series": list(FRED_SERIES)},
        "artifacts": artifacts,
        "processed_path": str(processed.relative_to(root)),
        "processed_sha256": _sha256(processed),
        "limitations": [
            "daily series have different publication calendars and missing observations",
            "Treasury par yields are sensitivity references rather than an option discount curve",
            "SP500 is a daily close and is not synchronized to historical option quotes",
        ],
    }
    (raw_directory / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return combined, manifest
