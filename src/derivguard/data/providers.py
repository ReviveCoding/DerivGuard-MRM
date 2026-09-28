"""Provider abstractions with immutable acquisition and explicit skip/block states."""

from __future__ import annotations

import os
import urllib.error
import urllib.request
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from .lineage import (
    RawArtifactManifest,
    archive_stream_immutable,
    available_disk_bytes,
    utc_now,
    write_json_exclusive,
)

AcquisitionStatus = Literal["ACQUIRED", "CACHED", "SKIPPED", "BLOCKED"]


@dataclass(frozen=True)
class AcquisitionResult:
    provider_id: str
    status: AcquisitionStatus
    artifact_path: Path | None
    manifest_path: Path | None
    reason: str | None = None


class DataProvider(ABC):
    provider_id: str
    provider_name: str

    @abstractmethod
    def acquire(self, raw_root: Path, *, reserve_bytes: int = 20 * 1024**3) -> AcquisitionResult:
        """Acquire or explicitly skip/block one provider artifact."""


class PublicFileProvider(DataProvider):
    source_url: str
    filename: str
    estimated_bytes: int
    observation_class: str
    license_note: str
    known_issues: tuple[str, ...] = ()

    def acquire(self, raw_root: Path, *, reserve_bytes: int = 20 * 1024**3) -> AcquisitionResult:
        provider_dir = raw_root / self.provider_id
        artifact = provider_dir / self.filename
        manifest_path = artifact.with_suffix(artifact.suffix + ".manifest.json")
        if artifact.exists() and manifest_path.exists():
            return AcquisitionResult(self.provider_id, "CACHED", artifact, manifest_path)
        if artifact.exists():
            return AcquisitionResult(
                self.provider_id,
                "BLOCKED",
                artifact,
                None,
                "raw bytes exist without a completed manifest; refusing overwrite",
            )
        free = available_disk_bytes(raw_root)
        if free - self.estimated_bytes < reserve_bytes:
            return AcquisitionResult(
                self.provider_id,
                "BLOCKED",
                None,
                None,
                f"disk reserve would be breached: free={free}, estimate={self.estimated_bytes}",
            )
        request = urllib.request.Request(
            self.source_url,
            headers={"User-Agent": "DerivGuard-MRM/0.1 research data collector"},
        )
        try:
            with urllib.request.urlopen(request, timeout=120) as response:
                digest, size = archive_stream_immutable(response, artifact)
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            return AcquisitionResult(self.provider_id, "BLOCKED", None, None, str(exc))
        manifest = RawArtifactManifest(
            source_id=self.provider_id,
            provider=self.provider_name,
            retrieval_timestamp_utc=utc_now(),
            source_url=self.source_url,
            query_parameters={},
            sha256=digest,
            file_size=size,
            relative_path=artifact.as_posix(),
            license_source_note=self.license_note,
            known_issues=self.known_issues,
            observation_class=self.observation_class,
        )
        write_json_exclusive(manifest_path, manifest.to_dict())
        return AcquisitionResult(self.provider_id, "ACQUIRED", artifact, manifest_path)


class PublicSPXSampleProvider(PublicFileProvider):
    provider_id = "historicaldata_spx_sample"
    provider_name = "HistoricalData.net"
    source_url = "https://historicaldata.net/file/options_sample_2022H2.zip"
    filename = "options_sample_2022H2.zip"
    estimated_bytes = 260 * 1024**2
    observation_class = "HISTORICAL_STANDING_EOD_QUOTE"
    license_note = "HistoricalData.net terms apply; derived research only, no raw redistribution."
    known_issues = (
        "Historical quote_time is expected to be blank; do not call the surface synchronized NBBO.",
        "SPX and SPXW settlement classes must remain separate.",
    )


class CboeMarkingProvider(PublicFileProvider):
    provider_id = "cboe_marking_1500ct"
    provider_name = "Cboe Global Markets"
    source_url = "https://cdn.cboe.com/resources/marking_prices/eod_marking_prices_list.csv"
    filename = "eod_marking_prices_list.csv"
    estimated_bytes = 100 * 1024**2
    observation_class = "PROSPECTIVE_LOCKED_TEST"
    license_note = "Cboe data is provided as-is; Cboe terms and redistribution restrictions apply."
    known_issues = (
        "Indicative marks are not actual OPRA NBBO updates.",
        "Rotating publication is prospective and does not establish historical availability.",
    )


class CboeDataShopSampleProvider(PublicFileProvider):
    provider_id = "cboe_datashop_quote_interval_sample"
    provider_name = "Cboe DataShop"
    source_url = "https://datashop.cboe.com/download/sample/215"
    filename = "option_quote_intervals_sample.zip"
    estimated_bytes = 2 * 1024**3
    observation_class = "INTEGRATION_SAMPLE"
    license_note = "Cboe DataShop terms apply; sample only, no full-history entitlement inferred."
    known_issues = (
        "Sample scope must be inspected; it does not establish access to full history.",
        "Quote-size methodology changes effective 2026-06-22.",
    )


class MassiveProvider(DataProvider):
    provider_id = "massive_optional"
    provider_name = "Massive"

    def acquire(self, raw_root: Path, *, reserve_bytes: int = 20 * 1024**3) -> AcquisitionResult:
        del raw_root, reserve_bytes
        if not os.environ.get("MASSIVE_API_KEY"):
            return AcquisitionResult(
                self.provider_id,
                "SKIPPED",
                None,
                None,
                "MASSIVE_API_KEY is absent; optional authenticated source skipped.",
            )
        return AcquisitionResult(
            self.provider_id,
            "BLOCKED",
            None,
            None,
            "Key detected but endpoint/query entitlement was not configured; no request made.",
        )
