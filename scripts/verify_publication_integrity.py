"""Read-only verification of the published v1.1 DVE freeze and evidence hashes."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, cast

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "artifacts/v1_1/v1_1_evidence_integrity_manifest.json"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    return cast(dict[str, Any], json.loads(path.read_text(encoding="utf-8")))


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

    for relative, expected in manifest["evidence_hashes"].items():
        path = ROOT / relative
        if not path.is_file():
            raise RuntimeError(f"published evidence file is missing: {relative}")
        if sha256(path) != expected:
            raise RuntimeError(f"published evidence hash mismatch: {relative}")

    print(
        "V1_1_PUBLICATION_INTEGRITY_VERIFIED "
        f"files={len(manifest['evidence_hashes'])} designs={len(manifest['design_checks'])}"
    )


if __name__ == "__main__":
    main()
