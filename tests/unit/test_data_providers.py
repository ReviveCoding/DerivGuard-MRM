from derivguard.data.providers import MassiveProvider


def test_massive_skips_without_key(monkeypatch, tmp_path) -> None:
    monkeypatch.delenv("MASSIVE_API_KEY", raising=False)
    result = MassiveProvider().acquire(tmp_path)
    assert result.status == "SKIPPED"
    assert "absent" in (result.reason or "")


def test_massive_does_not_guess_entitlement_when_key_exists(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("MASSIVE_API_KEY", "not-used")
    result = MassiveProvider().acquire(tmp_path)
    assert result.status == "BLOCKED"
    assert result.artifact_path is None
