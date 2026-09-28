"""Read-only verification of the published v1.1 DVE freeze and evidence hashes."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, cast

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "artifacts/v1_1/v1_1_evidence_integrity_manifest.json"
TEXT_EVIDENCE_SUFFIXES = frozenset({".csv", ".json", ".md", ".txt", ".yaml", ".yml"})


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    return cast(dict[str, Any], json.loads(path.read_text(encoding="utf-8")))


def evidence_hash_matches(path: Path, expected: str) -> tuple[bool, bool]:
    """Match original bytes or the Windows-origin CRLF representation of text."""

    data = path.read_bytes()
    if hashlib.sha256(data).hexdigest() == expected:
        return True, False
    if path.suffix.lower() not in TEXT_EVIDENCE_SUFFIXES:
        return False, False
    crlf = data.replace(b"\r\n", b"\n").replace(b"\n", b"\r\n")
    return hashlib.sha256(crlf).hexdigest() == expected, True


def main() -> None:
    manifest = read_json(MANIFEST)
    if manifest.get("status") != "V1_1_EVIDENCE_INTEGRITY_VERIFIED":
        raise RuntimeError("published integrity manifest does not record verified status")

    for check in manifest["design_checks"]:
        module = ROOT / check["module"]
        if sha256(module) != check["module_hash"]:
            raise RuntimeError(f"frozen module hash mismatch: {module.relative_to(ROOT)}")

    aggregation = read_json(ROOT / "artifacts/v1_1/dve_v1_1b_aggregation.freeze.json")
    design = read_json(ROOT / "artifacts/v1_1/dve_v1_1b_design.freeze.json")
    if aggregation["design_freeze_hash"] != design["freeze_hash"]:
        raise RuntimeError("aggregation freeze does not reference the v1.1b design freeze")

    line_ending_normalized = 0
    for relative, expected in manifest["evidence_hashes"].items():
        path = ROOT / relative
        if not path.is_file():
            raise RuntimeError(f"published evidence file is missing: {relative}")
        matched, normalized = evidence_hash_matches(path, expected)
        if not matched:
            raise RuntimeError(f"published evidence hash mismatch: {relative}")
        line_ending_normalized += int(normalized)

    print(
        "V1_1_PUBLICATION_INTEGRITY_VERIFIED "
        f"files={len(manifest['evidence_hashes'])} designs={len(manifest['design_checks'])} "
        f"line_ending_normalized={line_ending_normalized}"
    )


if __name__ == "__main__":
    main()
