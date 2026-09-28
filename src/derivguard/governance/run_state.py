"""Durable phase state and append-only execution-ledger support.

The module records provenance; it never infers scientific success from file
presence. Callers must supply the evidence, hashes, and test status used to
justify a transition.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
import threading
import uuid
from collections.abc import Iterable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, TextIO

PHASE_STATUSES = frozenset(
    {"PENDING", "RUNNING", "COMPLETED", "PARTIAL", "FAILED", "UNAVAILABLE", "BLOCKED"}
)
TEST_STATUSES = frozenset({"NOT_RUN", "PASS", "FAIL", "PARTIAL", "NOT_APPLICABLE"})


def timestamp() -> str:
    """Return an unambiguous ISO-8601 UTC timestamp."""

    return datetime.now(UTC).isoformat()


def sha256_file(path: str | Path, *, chunk_size: int = 1024 * 1024) -> str:
    """Hash a file without loading it into memory."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_manifest(paths: Iterable[str | Path], *, root: str | Path | None = None) -> str:
    """Hash sorted relative names and content hashes for a deterministic manifest."""

    root_path = Path(root).resolve() if root is not None else None
    entries: list[tuple[str, str]] = []
    for raw_path in paths:
        path = Path(raw_path).resolve()
        name = path.relative_to(root_path).as_posix() if root_path is not None else path.as_posix()
        entries.append((name, sha256_file(path)))
    canonical = json.dumps(sorted(entries), separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


@contextmanager
def _advisory_lock(lock_path: Path) -> Iterator[None]:
    """Use a one-byte Windows lock, with a process-local fallback elsewhere."""

    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+b") as lock_file:
        lock_file.seek(0, os.SEEK_END)
        if lock_file.tell() == 0:
            lock_file.write(b"0")
            lock_file.flush()
        lock_file.seek(0)
        if os.name == "nt":
            import msvcrt

            msvcrt.locking(lock_file.fileno(), msvcrt.LK_LOCK, 1)
            try:
                yield
            finally:
                lock_file.seek(0)
                msvcrt.locking(lock_file.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            # The supported execution platform is Windows. The lock still
            # serializes threads during bounded cross-platform unit tests.
            with _PROCESS_LOCK:
                yield


_PROCESS_LOCK = threading.Lock()


def _write_json_atomic(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
            json.dump(value, stream, indent=2, sort_keys=False, ensure_ascii=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_name, path)
    except BaseException:
        Path(temporary_name).unlink(missing_ok=True)
        raise


class RunStateStore:
    """Update run state and ledger while preserving explicit evidence fields."""

    def __init__(self, state_path: str | Path, ledger_path: str | Path) -> None:
        self.state_path = Path(state_path)
        self.ledger_path = Path(ledger_path)
        self.lock_path = self.state_path.with_suffix(self.state_path.suffix + ".lock")

    def load(self) -> dict[str, Any]:
        with self.state_path.open(encoding="utf-8") as stream:
            value = json.load(stream)
        if not isinstance(value, dict):
            raise ValueError("Run state root must be a JSON object")
        return value

    def start_phase(
        self,
        phase_id: str,
        *,
        input_artifacts: Sequence[str],
        config_hash: str,
        code_hash: str,
        resume_point: str,
    ) -> dict[str, Any]:
        """Move a pending/restartable phase to RUNNING with provenance."""

        if not config_hash or not code_hash:
            raise ValueError("Starting a phase requires non-empty config and code/content hashes")
        now = timestamp()
        with _advisory_lock(self.lock_path):
            state = self.load()
            phase = self._phase(state, phase_id)
            if phase["status"] not in {"PENDING", "PARTIAL", "FAILED", "BLOCKED"}:
                raise ValueError(f"Phase {phase_id} cannot start from {phase['status']}")
            phase.update(
                {
                    "status": "RUNNING",
                    "started_at": now,
                    "ended_at": None,
                    "input_artifacts": list(input_artifacts),
                    "config_hash": config_hash,
                    "code_hash": code_hash,
                    "test_status": "NOT_RUN",
                    "errors": [],
                    "resume_point": resume_point,
                }
            )
            state["current_phase"] = phase_id
            state["resume_phase"] = phase_id
            state["overall_status"] = "RUNNING"
            state["updated_at"] = now
            _write_json_atomic(self.state_path, state)
            self._append_event_locked(
                state,
                phase_id=phase_id,
                event_type="PHASE_STARTED",
                status="RUNNING",
                message=resume_point,
            )
        return state

    def finish_phase(
        self,
        phase_id: str,
        *,
        status: str,
        output_artifacts: Sequence[str],
        test_status: str,
        errors: Sequence[str] = (),
        resume_point: str | None = None,
    ) -> dict[str, Any]:
        """Finish or checkpoint a running phase without manufacturing success."""

        if status not in PHASE_STATUSES - {"PENDING", "RUNNING"}:
            raise ValueError(f"Unsupported terminal/checkpoint status: {status}")
        if test_status not in TEST_STATUSES:
            raise ValueError(f"Unsupported test status: {test_status}")
        if status == "COMPLETED" and test_status not in {"PASS", "NOT_APPLICABLE"}:
            raise ValueError("COMPLETED requires PASS or justified NOT_APPLICABLE test status")
        if status == "COMPLETED" and not output_artifacts:
            raise ValueError("COMPLETED requires at least one evidence output artifact")
        if status == "FAILED" and not errors:
            raise ValueError("FAILED requires at least one actual error")
        now = timestamp()
        with _advisory_lock(self.lock_path):
            state = self.load()
            phase = self._phase(state, phase_id)
            if phase["status"] != "RUNNING":
                raise ValueError(f"Phase {phase_id} must be RUNNING, not {phase['status']}")
            if status == "COMPLETED" and (
                not phase.get("config_hash") or not phase.get("code_hash")
            ):
                raise ValueError("COMPLETED requires the hashes recorded at phase start")
            phase.update(
                {
                    "status": status,
                    "ended_at": now,
                    "output_artifacts": list(output_artifacts),
                    "test_status": test_status,
                    "errors": list(errors),
                    "resume_point": resume_point,
                }
            )
            state["updated_at"] = now
            state["resume_phase"] = (
                phase_id if status != "COMPLETED" else self._next_phase(state, phase_id)
            )
            state["overall_status"] = "RUNNING" if status == "COMPLETED" else status
            _write_json_atomic(self.state_path, state)
            self._append_event_locked(
                state,
                phase_id=phase_id,
                event_type="PHASE_FINISHED" if status == "COMPLETED" else "PHASE_CHECKPOINTED",
                status=status,
                message=resume_point,
            )
        return state

    @staticmethod
    def _phase(state: Mapping[str, Any], phase_id: str) -> dict[str, Any]:
        phases = state.get("phases")
        if not isinstance(phases, dict) or phase_id not in phases:
            raise KeyError(f"Unknown phase: {phase_id}")
        phase = phases[phase_id]
        if not isinstance(phase, dict):
            raise ValueError(f"Phase {phase_id} record must be an object")
        return phase

    @staticmethod
    def _next_phase(state: Mapping[str, Any], phase_id: str) -> str | None:
        phases = state.get("phases")
        if not isinstance(phases, dict):
            return None
        phase_ids = list(phases)
        try:
            index = phase_ids.index(phase_id)
        except ValueError:
            return None
        return phase_ids[index + 1] if index + 1 < len(phase_ids) else None

    def _append_event_locked(
        self,
        state: Mapping[str, Any],
        *,
        phase_id: str,
        event_type: str,
        status: str,
        message: str | None,
    ) -> None:
        phase = self._phase(state, phase_id)
        event = {
            "schema_version": "1.0",
            "event_id": str(uuid.uuid4()),
            "timestamp": timestamp(),
            "run_id": state.get("run_id"),
            "phase_id": phase_id,
            "event_type": event_type,
            "status": status,
            "inputs": phase.get("input_artifacts", []),
            "outputs": phase.get("output_artifacts", []),
            "config_hash": phase.get("config_hash"),
            "code_hash": phase.get("code_hash"),
            "test_status": phase.get("test_status"),
            "errors": phase.get("errors", []),
            "message": message,
        }
        self.ledger_path.parent.mkdir(parents=True, exist_ok=True)
        with self.ledger_path.open("a", encoding="utf-8", newline="\n") as stream:
            self._write_ledger_line(stream, event)

    @staticmethod
    def _write_ledger_line(stream: TextIO, event: Mapping[str, Any]) -> None:
        stream.write(json.dumps(event, separators=(",", ":"), ensure_ascii=False))
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
