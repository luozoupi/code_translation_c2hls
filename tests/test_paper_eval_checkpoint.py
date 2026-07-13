from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import pytest


REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "scripts" / "create_paper_eval_checkpoint.py"
SPEC = importlib.util.spec_from_file_location("paper_checkpoint", SCRIPT)
assert SPEC and SPEC.loader
checkpoint = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(checkpoint)


def test_required_ignored_controller_is_hashed_without_scanning_env(tmp_path: Path) -> None:
    controller = tmp_path / "c2hls_temp.py"
    controller.write_text("controller = True\n", encoding="utf-8")
    (tmp_path / ".env").write_text("SECRET=do-not-record\n", encoding="utf-8")

    records = checkpoint.required_ignored_records(tmp_path)

    assert [row["path"] for row in records] == ["c2hls_temp.py"]
    assert records[0]["sha256"] == hashlib.sha256(controller.read_bytes()).hexdigest()


def test_checkpoint_directory_is_immutable(tmp_path: Path) -> None:
    destination = tmp_path / "checkpoint"
    checkpoint.create_checkpoint_dir(destination)
    (destination / "manifest.json").write_text("{}\n", encoding="utf-8")

    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        checkpoint.create_checkpoint_dir(destination)


def test_checkpoint_preserves_untracked_ignored_and_external_inputs(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.email", "checkpoint@example.invalid"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.name", "Checkpoint Test"], cwd=repo, check=True)
    (repo / ".gitignore").write_text("c2hls_temp.py\n", encoding="utf-8")
    (repo / "tracked.py").write_text("value = 1\n", encoding="utf-8")
    subprocess.run(["git", "add", ".gitignore", "tracked.py"], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-qm", "base"], cwd=repo, check=True)
    (repo / "tracked.py").write_text("value = 2\n", encoding="utf-8")
    (repo / "new_config.json").write_text('{"enabled": true}\n', encoding="utf-8")
    (repo / "c2hls_temp.py").write_text("controller = True\n", encoding="utf-8")
    external = tmp_path / "external"
    external.mkdir()
    (external / "testbench.cpp").write_text("int main() { return 0; }\n", encoding="utf-8")

    completed = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--repo",
            str(repo),
            "--name",
            "final",
            "--purpose",
            "post-hardening test",
            "--external-input-root",
            f"hlsfactory={external}",
        ],
        text=True,
        capture_output=True,
    )
    assert completed.returncode == 0, completed.stderr + completed.stdout
    output = repo / "paper_eval" / "checkpoints" / "final"
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["schema_version"] == 3
    assert (output / "base-head.tar").is_file()
    assert (output / "tracked.patch").is_file()
    assert (output / "payload" / "untracked" / "new_config.json").is_file()
    assert (output / "payload" / "ignored" / "c2hls_temp.py").is_file()
    assert (output / "payload" / "external" / "hlsfactory" / "testbench.cpp").is_file()
    assert checkpoint.verify_checkpoint(output)["valid"]

    (output / "payload" / "untracked" / "new_config.json").write_text(
        "tampered\n", encoding="utf-8"
    )
    report = checkpoint.verify_checkpoint(output)
    assert not report["valid"]
    assert any("hash_mismatch" in item for item in report["failures"])
