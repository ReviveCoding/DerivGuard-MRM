from __future__ import annotations

import json
from pathlib import Path

import pytest

from derivguard.governance.run_state import RunStateStore, sha256_file, sha256_manifest


def _initial_state(path: Path) -> None:
    path.write_text(
        json.dumps(
            {
                "run_id": "test-run",
                "overall_status": "PENDING_EXECUTION",
                "current_phase": "00",
                "resume_phase": "00",
                "updated_at": None,
                "phases": {
                    "00": {
                        "status": "PENDING",
                        "started_at": None,
                        "ended_at": None,
                        "input_artifacts": [],
                        "config_hash": None,
                        "code_hash": None,
                        "output_artifacts": [],
                        "test_status": "NOT_RUN",
                        "errors": [],
                        "resume_point": None,
                    },
                    "01": {"status": "PENDING"},
                },
            }
        ),
        encoding="utf-8",
    )


def test_phase_lifecycle_is_durable(tmp_path: Path) -> None:
    state_path = tmp_path / "run_state.json"
    ledger_path = tmp_path / "ledger.jsonl"
    _initial_state(state_path)
    store = RunStateStore(state_path, ledger_path)

    store.start_phase(
        "00",
        input_artifacts=["AGENTS.md"],
        config_hash="config-hash",
        code_hash="code-hash",
        resume_point="reconcile",
    )
    finished = store.finish_phase(
        "00",
        status="COMPLETED",
        output_artifacts=["evidence.json"],
        test_status="PASS",
    )

    assert finished["phases"]["00"]["status"] == "COMPLETED"
    assert finished["resume_phase"] == "01"
    events = [json.loads(line) for line in ledger_path.read_text(encoding="utf-8").splitlines()]
    assert [event["event_type"] for event in events] == ["PHASE_STARTED", "PHASE_FINISHED"]


def test_completion_requires_evidence(tmp_path: Path) -> None:
    state_path = tmp_path / "run_state.json"
    ledger_path = tmp_path / "ledger.jsonl"
    _initial_state(state_path)
    store = RunStateStore(state_path, ledger_path)
    store.start_phase(
        "00",
        input_artifacts=[],
        config_hash="config-hash",
        code_hash="code-hash",
        resume_point="work",
    )

    with pytest.raises(ValueError, match="evidence output"):
        store.finish_phase("00", status="COMPLETED", output_artifacts=[], test_status="PASS")


def test_content_hashes_are_deterministic(tmp_path: Path) -> None:
    first = tmp_path / "a.txt"
    second = tmp_path / "b.txt"
    first.write_text("alpha", encoding="utf-8")
    second.write_text("beta", encoding="utf-8")

    assert sha256_file(first) == sha256_file(first)
    assert sha256_manifest([second, first], root=tmp_path) == sha256_manifest(
        [first, second], root=tmp_path
    )
