"""Closure-time integrity checks for additive v1.1 evidence."""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast


def _hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _json(path: Path) -> dict[str, Any]:
    return cast(dict[str, Any], json.loads(path.read_text(encoding="utf-8")))


def create_integrity_manifest(root: Path = Path(".")) -> dict[str, Any]:
    design_pairs = (
        (
            root / "artifacts/v1_1/dve_v1_1_design.freeze.json",
            root / "src/derivguard/synthetic/dve_v1_1.py",
        ),
        (
            root / "artifacts/v1_1/dve_v1_1b_design.freeze.json",
            root / "src/derivguard/synthetic/dve_v1_1b.py",
        ),
    )
    design_checks = []
    for freeze_path, module_path in design_pairs:
        freeze = _json(freeze_path)
        current = _hash(module_path)
        matched = current == freeze["module_hash"]
        if not matched:
            raise RuntimeError(f"frozen DVE module hash mismatch: {module_path}")
        design_checks.append(
            {
                "freeze": freeze_path.relative_to(root).as_posix(),
                "module": module_path.relative_to(root).as_posix(),
                "module_hash": current,
                "matched": matched,
            }
        )
    aggregation = _json(root / "artifacts/v1_1/dve_v1_1b_aggregation.freeze.json")
    design = _json(root / "artifacts/v1_1/dve_v1_1b_design.freeze.json")
    if aggregation["design_freeze_hash"] != design["freeze_hash"]:
        raise RuntimeError("aggregation freeze does not reference the v1.1b design freeze")
    evidence_paths = sorted(
        path
        for base in (
            root / "results/v1_1/dve_v1_1b",
            root / "results/v1_1/dve_failure",
        )
        for path in base.rglob("*")
        if path.is_file()
    )
    evidence_hashes = {path.relative_to(root).as_posix(): _hash(path) for path in evidence_paths}
    payload = {
        "status": "V1_1_EVIDENCE_INTEGRITY_VERIFIED",
        "created_utc": datetime.now(UTC).isoformat(),
        "design_checks": design_checks,
        "aggregation_design_hash_matched": True,
        "evidence_hashes": evidence_hashes,
        "note": (
            "This closure manifest prevents future silent cache substitution. The completed "
            "fresh run was generated before this closure and did not reuse v1 locked observations."
        ),
    }
    output = root / "artifacts/v1_1/v1_1_evidence_integrity_manifest.json"
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload
