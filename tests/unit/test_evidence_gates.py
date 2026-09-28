from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from derivguard.governance.gates import (
    PHASE_REQUIREMENTS,
    ArtifactRequirement,
    _active_confirmatory,
    _inspect_artifact,
    _read_engine_qualification,
    source_tree_sha256,
    verify_final_evidence,
)
from derivguard.reporting.final_reports import _synthetic_evidence


def test_status_only_csv_is_not_scientific_evidence(tmp_path: Path) -> None:
    path = tmp_path / "placeholder.csv"
    pd.DataFrame([{"execution_status": "PARTIAL", "measured_value": float("nan")}]).to_csv(
        path, index=False
    )
    result = _inspect_artifact(
        tmp_path, ArtifactRequirement("placeholder.csv", "csv", minimum_rows=1)
    )
    assert not result.passed
    assert "placeholder" in result.detail


def test_measured_csv_passes_row_and_column_checks(tmp_path: Path) -> None:
    pd.DataFrame([{"model": "M00", "rmse": 0.2}]).to_csv(tmp_path / "measured.csv", index=False)
    result = _inspect_artifact(
        tmp_path,
        ArtifactRequirement(
            "measured.csv", "csv", minimum_rows=1, required_columns=("model", "rmse")
        ),
    )
    assert result.passed


def test_json_requires_semantic_keys(tmp_path: Path) -> None:
    (tmp_path / "evidence.json").write_text(json.dumps({"status": "ok"}), encoding="utf-8")
    result = _inspect_artifact(
        tmp_path,
        ArtifactRequirement("evidence.json", "json", required_json_keys=("metrics",)),
    )
    assert not result.passed
    assert "metrics" in result.detail


def test_truth_qualification_accepts_record_list_schema(tmp_path: Path) -> None:
    path = tmp_path / "truth_qualification.json"
    path.write_text(
        json.dumps([{"engine": "BATES_CF", "status": "PASS", "measured_error": 1.0e-12}]),
        encoding="utf-8",
    )
    rows = _read_engine_qualification(path)
    assert rows == [{"engine": "BATES_CF", "status": "PASS", "measured_error": 1.0e-12}]


def test_truth_qualification_rejects_non_record_payload(tmp_path: Path) -> None:
    path = tmp_path / "truth_qualification.json"
    path.write_text(json.dumps(["PASS"]), encoding="utf-8")
    try:
        _read_engine_qualification(path)
    except ValueError as exc:
        assert "engine-record list" in str(exc)
    else:
        raise AssertionError("invalid truth-qualification schema was accepted")


def test_primary_confirmatory_requires_qualified_v6_calibration(tmp_path: Path) -> None:
    v6 = tmp_path / "results/confirmatory/synthetic_v5"
    preserved_v5 = tmp_path / "results/confirmatory/synthetic_v4"
    oracle = tmp_path / "results/confirmatory/synthetic_v2"
    v6.mkdir(parents=True)
    preserved_v5.mkdir(parents=True)
    oracle.mkdir(parents=True)
    payload = {"status": "CONFIRMATORY_LOCKED_COMPLETE", "baselines": {}}
    (preserved_v5 / "confirmatory_metrics.json").write_text(
        json.dumps({"status": "CALIBRATION_UNQUALIFIED", "baselines": {}}), encoding="utf-8"
    )
    (oracle / "confirmatory_metrics.json").write_text(json.dumps(payload), encoding="utf-8")

    primary, fallback_label, fallback_dir = _synthetic_evidence(tmp_path)
    assert primary is None
    assert fallback_label == "confirmatory execution unavailable"
    assert fallback_dir is None

    pd.DataFrame([{"case_id": "V4-DEV-F00-1"}]).to_parquet(
        v6 / "calibration_diagnostics.parquet", index=False
    )
    (v6 / "confirmatory_metrics.json").write_text(json.dumps(payload), encoding="utf-8")
    (v6 / "calibration_qualification.json").write_text(
        json.dumps({"status": "FAIL", "all_selected_fits_qualified": False}), encoding="utf-8"
    )
    assert _synthetic_evidence(tmp_path)[0] is None

    (v6 / "calibration_qualification.json").write_text(
        json.dumps({"status": "PASS", "all_selected_fits_qualified": True}), encoding="utf-8"
    )
    _, corrected_label, corrected_dir = _synthetic_evidence(tmp_path)
    assert corrected_label.startswith("confirmatory v6 release-aligned")
    assert corrected_dir == v6


def test_v6_is_the_only_pass_eligible_confirmatory_phase_contract(tmp_path: Path) -> None:
    preserved_v5 = tmp_path / "results/confirmatory/synthetic_v4"
    preserved_v5.mkdir(parents=True)
    (preserved_v5 / "confirmatory_metrics.json").write_text("{}", encoding="utf-8")
    active_result, active_freeze, active_source = _active_confirmatory(tmp_path)
    assert active_result == preserved_v5
    assert active_freeze.name == "synthetic_confirmatory_v5.freeze.json"
    assert active_source.name == "confirmatory_v5.py"

    v6 = tmp_path / "results/confirmatory/synthetic_v5"
    v6.mkdir(parents=True)
    (v6 / "confirmatory_metrics.json").write_text("{}", encoding="utf-8")
    active_result, active_freeze, active_source = _active_confirmatory(tmp_path)
    assert active_result == v6
    assert active_freeze.name == "synthetic_confirmatory_v6.freeze.json"
    assert active_source.name == "confirmatory_v6.py"

    confirmatory_paths = {
        artifact.path
        for phase in PHASE_REQUIREMENTS
        if phase.phase_id in {"25", "26", "27", "28", "33", "35", "36", "38", "39"}
        for artifact in phase.artifacts
    }
    assert any("synthetic_v5" in path for path in confirmatory_paths)
    assert not any("synthetic_v4" in path for path in confirmatory_paths)


def test_final_review_requires_current_source_hash_and_zero_high_findings(
    tmp_path: Path,
) -> None:
    source = tmp_path / "src/derivguard/example.py"
    source.parent.mkdir(parents=True)
    source.write_text("VALUE = 1\n", encoding="utf-8")
    source_hash = source_tree_sha256(tmp_path)
    tests = tmp_path / "artifacts/tests/final_test_summary.json"
    review = tmp_path / "artifacts/review/final_review.json"
    tests.parent.mkdir(parents=True)
    review.parent.mkdir(parents=True)
    tests.write_text(
        json.dumps({"status": "PASS", "source_tree_sha256": source_hash}), encoding="utf-8"
    )
    required_scope = [
        "developer_validator_independence",
        "heston_cf",
        "mc_qe_qem",
        "pde_adi",
        "calibration_leakage",
        "locked_test_discipline",
        "dve",
        "synthetic_faults",
        "statistics",
        "phase_state",
        "gpu_correctness",
        "report_claims",
    ]
    review.write_text(
        json.dumps(
            {
                "status": "PASS",
                "reviewed_source_sha256": source_hash,
                "unresolved_severity_counts": {"Critical": 0, "High": 1},
                "review_scope": required_scope,
            }
        ),
        encoding="utf-8",
    )
    assert verify_final_evidence(tmp_path)["status"] == "FAIL"

    value = json.loads(review.read_text(encoding="utf-8"))
    value["unresolved_severity_counts"]["High"] = 0
    review.write_text(json.dumps(value), encoding="utf-8")
    assert verify_final_evidence(tmp_path)["status"] == "PASS"

    source.write_text("VALUE = 2\n", encoding="utf-8")
    assert verify_final_evidence(tmp_path)["status"] == "FAIL"
