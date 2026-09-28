import io
import json

import pytest

from derivguard.data.lineage import archive_stream_immutable, sha256_file, write_json_exclusive


def test_archive_is_immutable_and_hashed(tmp_path) -> None:
    destination = tmp_path / "raw" / "sample.csv"
    digest, size = archive_stream_immutable(io.BytesIO(b"a,b\n1,2\n"), destination)
    assert size == 8
    assert digest == sha256_file(destination)
    with pytest.raises(FileExistsError):
        archive_stream_immutable(io.BytesIO(b"different"), destination)
    assert destination.read_bytes() == b"a,b\n1,2\n"


def test_manifest_write_is_idempotent_but_not_replaceable(tmp_path) -> None:
    path = tmp_path / "manifest.json"
    write_json_exclusive(path, {"sha256": "abc", "rows": 2})
    write_json_exclusive(path, {"sha256": "abc", "rows": 2})
    assert json.loads(path.read_text()) == {"rows": 2, "sha256": "abc"}
    with pytest.raises(FileExistsError):
        write_json_exclusive(path, {"sha256": "changed"})
