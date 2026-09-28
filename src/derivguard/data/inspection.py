"""Independent schema and content inspection for acquired public artifacts."""

from __future__ import annotations

import csv
import hashlib
import io
import json
import zipfile
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd

from .adapters import iter_datashop_nested_csv, read_cboe_marking_long
from .lineage import sha256_file


def _stream_digest_and_lines(stream: Any) -> tuple[str, int]:
    digest = hashlib.sha256()
    lines = 0
    for chunk in iter(lambda: stream.read(1024 * 1024), b""):
        digest.update(chunk)
        lines += chunk.count(b"\n")
    return digest.hexdigest(), lines


def inspect_historical_sample(path: Path) -> dict[str, Any]:
    """Verify every packaged daily checksum/row count and independently summarize scope."""
    with zipfile.ZipFile(path) as archive:
        vendor_manifest = json.loads(archive.read("day_by_date/manifest.json"))
        files = vendor_manifest["files"]
        failures: list[str] = []
        observed_columns: list[str] | None = None
        underlying_counts: Counter[str] = Counter()
        missing_quote_time = 0
        inspected_rows = 0
        for record in files:
            member = f"day_by_date/{record['name']}"
            with archive.open(member) as stream:
                digest, line_count = _stream_digest_and_lines(stream)
            if digest != record["sha256"]:
                failures.append(f"{member}: SHA-256 mismatch")
            if line_count - 1 != record["rows"]:
                failures.append(f"{member}: row-count mismatch")
            with archive.open(member) as stream:
                subset = pd.read_csv(stream, usecols=["underlying", "quote_time"], low_memory=False)
            underlying_counts.update(subset["underlying"].astype(str))
            missing_quote_time += int(subset["quote_time"].isna().sum())
            inspected_rows += len(subset)
            if observed_columns is None:
                with archive.open(member) as stream:
                    header = next(csv.reader(io.TextIOWrapper(stream, encoding="utf-8-sig")))
                observed_columns = header
    return {
        "artifact_sha256": sha256_file(path),
        "artifact_bytes": path.stat().st_size,
        "vendor_product": vendor_manifest.get("product"),
        "vendor_rows_total": vendor_manifest.get("rows_total"),
        "independently_inspected_rows": inspected_rows,
        "trading_days_present": len(files),
        "minimum_date": vendor_manifest.get("first_day"),
        "maximum_date": vendor_manifest.get("last_day"),
        "columns": observed_columns,
        "column_count": len(observed_columns or []),
        "underlying_counts": dict(sorted(underlying_counts.items())),
        "quote_time_missing_rows": missing_quote_time,
        "verification_failures": failures,
        "verification_status": "PASS" if not failures else "FAIL",
    }


def inspect_cboe_marking(path: Path) -> dict[str, Any]:
    frame = read_cboe_marking_long(path)
    dates = pd.Series(frame["quote_date"]).dropna().astype(str)
    return {
        "artifact_sha256": sha256_file(path),
        "artifact_bytes": path.stat().st_size,
        "rows_wide": len(frame) // 2,
        "rows_long": len(frame),
        "minimum_date": dates.min() if not dates.empty else None,
        "maximum_date": dates.max() if not dates.empty else None,
        "root_counts_long": {
            str(key): int(value)
            for key, value in frame["underlying"].value_counts().sort_index().items()
        },
        "observation_class": "PROSPECTIVE_LOCKED_TEST",
        "nbbo_status": "Cboe indicative mark and actual Cboe BBO fields; not an OPRA NBBO update",
    }


def inspect_datashop_sample(path: Path) -> dict[str, Any]:
    members: list[dict[str, Any]] = []
    for member, frame in iter_datashop_nested_csv(path):
        date_columns = [name for name in frame.columns if "date" in str(name).lower()]
        members.append(
            {
                "member": member,
                "rows": len(frame),
                "columns": [str(column) for column in frame.columns],
                "date_columns": date_columns,
            }
        )
    return {
        "artifact_sha256": sha256_file(path),
        "artifact_bytes": path.stat().st_size,
        "nested_csv_count": len(members),
        "rows_total": sum(int(item["rows"]) for item in members),
        "members": members,
        "scope_note": "Official integration sample; no full-history entitlement inferred.",
    }
