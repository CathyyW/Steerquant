from pathlib import Path

import pytest

from fastwam_steerquant.inspect import expected_sha256, sha256_file


def test_manifest_lookup_and_checksum(tmp_path: Path):
    checkpoint = tmp_path / "deployment.pt"
    checkpoint.write_bytes(b"packed-checkpoint")
    checksum = sha256_file(checkpoint)
    (tmp_path / "SHA256SUMS").write_text(f"{checksum}  deployment.pt\n", encoding="utf-8")
    assert expected_sha256(checkpoint) == checksum


def test_manifest_rejects_duplicate_entry(tmp_path: Path):
    checkpoint = tmp_path / "deployment.pt"
    checkpoint.write_bytes(b"x")
    checksum = sha256_file(checkpoint)
    (tmp_path / "SHA256SUMS").write_text(
        f"{checksum}  deployment.pt\n{checksum}  deployment.pt\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="exactly one"):
        expected_sha256(checkpoint)
