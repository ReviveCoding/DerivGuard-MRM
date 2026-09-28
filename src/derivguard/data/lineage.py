"""Immutable raw-data storage and machine-readable lineage manifests."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, BinaryIO


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    """Return the SHA-256 digest of *path* without loading it into memory."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


@dataclass(frozen=True)
class RawArtifactManifest:
    source_id: str
    provider: str
    retrieval_timestamp_utc: str
    source_url: str
    query_parameters: dict[str, str]
    sha256: str
    file_size: int
    relative_path: str
    row_count: int | None = None
    minimum_date: str | None = None
    maximum_date: str | None = None
    schema_version: str = "uninspected"
    license_source_note: str = "Review upstream terms before redistribution."
    redistribution_note: str = "Raw redistribution not authorized by this manifest."
    known_issues: tuple[str, ...] = field(default_factory=tuple)
    observation_class: str = "UNCLASSIFIED"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def write_json_exclusive(path: Path, payload: dict[str, Any]) -> None:
    """Create a JSON file and refuse to replace an existing artifact."""
    path.parent.mkdir(parents=True, exist_ok=True)
    data = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode()
    try:
        with path.open("xb") as handle:
            handle.write(data)
    except FileExistsError:
        existing = path.read_bytes()
        if existing != data:
            raise FileExistsError(
                f"immutable artifact already exists with different bytes: {path}"
            ) from None


def archive_stream_immutable(stream: BinaryIO, destination: Path) -> tuple[str, int]:
    """Write a stream once, fsync it, and return its digest and byte count."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256()
    size = 0
    created = False
    try:
        with destination.open("xb") as output:
            created = True
            while chunk := stream.read(1024 * 1024):
                output.write(chunk)
                digest.update(chunk)
                size += len(chunk)
            output.flush()
            os.fsync(output.fileno())
    except BaseException:
        if created:
            destination.unlink(missing_ok=True)
        raise
    return digest.hexdigest(), size


def archive_local_file_immutable(source: Path, destination: Path) -> tuple[str, int]:
    """Archive a local source without modifying it or overwriting prior raw bytes."""
    with source.open("rb") as stream:
        return archive_stream_immutable(stream, destination)


def available_disk_bytes(path: Path) -> int:
    return shutil.disk_usage(path.resolve()).free
