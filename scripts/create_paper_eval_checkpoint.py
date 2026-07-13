#!/usr/bin/env python3
"""Create a non-destructive manifest for an HPCA paper-evaluation checkpoint.

The checkpoint records the dirty Git state, a binary patch for tracked changes,
content hashes for source/configuration inputs, and content hashes for compact
experiment artifacts.  Large, reproducible Vitis work trees are inventoried by
path/size/mtime instead of being copied into the checkpoint.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
from typing import Iterable


INPUT_SUFFIXES = {
    ".c",
    ".cc",
    ".cfg",
    ".cmake",
    ".cpp",
    ".csv",
    ".h",
    ".hpp",
    ".ini",
    ".json",
    ".jsonl",
    ".md",
    ".mk",
    ".py",
    ".sh",
    ".tcl",
    ".toml",
    ".txt",
    ".yaml",
    ".yml",
}

# These trees contain reproducible compiler work products or model-training
# payloads measured in gigabytes.  Their path/size/mtime inventory is hashed,
# while compact result summaries and manifests elsewhere under artifacts/ are
# content-hashed individually.
DEFAULT_INVENTORY_ONLY = (
    "artifacts/tmp_vitis",
    "artifacts/vllm_cosim_compare",
    "artifacts/vllm_vitis_smoke",
    "artifacts/rl_corpus",
)

# Git ignores generated data and local environment files aggressively in this
# repository.  This controller module is nevertheless an implementation input
# and must never disappear from a reproducibility checkpoint merely because it
# is listed in .gitignore.  Keep this allow-list narrow so secrets such as .env
# are not inventoried accidentally.
REQUIRED_IGNORED_INPUTS = ("c2hls_temp.py",)


def run(repo: Path, *args: str) -> bytes:
    return subprocess.check_output(args, cwd=repo)


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path, chunk_size: int = 4 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _stable_stat(path: Path) -> tuple[int, int, int, int]:
    stat = path.stat()
    return (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns)


def file_record(repo: Path, path: Path) -> dict[str, object]:
    before = _stable_stat(path)
    digest = sha256_file(path)
    after = _stable_stat(path)
    if before != after:
        raise RuntimeError(f"input mutated while checkpoint was hashing it: {path}")
    stat = path.stat()
    return {
        "path": path.relative_to(repo).as_posix(),
        "bytes": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "sha256": digest,
    }


def external_file_record(root: Path, path: Path) -> dict[str, object]:
    before = _stable_stat(path)
    digest = sha256_file(path)
    after = _stable_stat(path)
    if before != after:
        raise RuntimeError(f"external input mutated while checkpoint was hashing it: {path}")
    stat = path.stat()
    return {
        "path": path.relative_to(root).as_posix(),
        "bytes": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "sha256": digest,
    }


def preserve_file(source: Path, destination: Path, expected_sha256: str) -> None:
    """Copy one input and prove both source and payload retained its identity."""

    before = _stable_stat(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    after = _stable_stat(source)
    if before != after:
        raise RuntimeError(f"input mutated while checkpoint was copying it: {source}")
    if sha256_file(destination) != expected_sha256:
        raise RuntimeError(f"checkpoint payload hash mismatch after copy: {destination}")


def parse_status_z(raw: bytes) -> list[tuple[str, str]]:
    fields = raw.split(b"\0")
    records: list[tuple[str, str]] = []
    index = 0
    while index < len(fields) and fields[index]:
        field = fields[index].decode("utf-8", errors="surrogateescape")
        status, path = field[:2], field[3:]
        records.append((status, path))
        index += 1
        if "R" in status or "C" in status:
            # With -z, Git emits the source name as the following field.
            index += 1
    return records


def under_any(path: str, roots: Iterable[str]) -> bool:
    return any(path == root or path.startswith(root + "/") for root in roots)


def inventory_tree(repo: Path, relative_root: str) -> dict[str, object]:
    root = repo / relative_root
    digest = hashlib.sha256()
    count = 0
    total_bytes = 0
    latest_mtime_ns = 0
    if root.exists():
        for path in sorted(item for item in root.rglob("*") if item.is_file()):
            stat = path.stat()
            relative = path.relative_to(repo).as_posix()
            row = f"{relative}\0{stat.st_size}\0{stat.st_mtime_ns}\n".encode()
            digest.update(row)
            count += 1
            total_bytes += stat.st_size
            latest_mtime_ns = max(latest_mtime_ns, stat.st_mtime_ns)
    return {
        "path": relative_root,
        "files": count,
        "bytes": total_bytes,
        "latest_mtime_ns": latest_mtime_ns,
        "inventory_sha256": digest.hexdigest(),
        "content_hashed": False,
    }


def required_ignored_records(repo: Path) -> list[dict[str, object]]:
    """Hash only explicitly approved ignored implementation inputs."""
    return [
        file_record(repo, repo / relative)
        for relative in REQUIRED_IGNORED_INPUTS
        if (repo / relative).is_file()
    ]


def create_checkpoint_dir(path: Path) -> None:
    """Create a new checkpoint directory without permitting replacement."""
    if path.exists() and any(path.iterdir()):
        raise FileExistsError(f"refusing to overwrite immutable checkpoint: {path}")
    path.mkdir(parents=True, exist_ok=True)


def parse_external_root(value: str) -> tuple[str, Path]:
    if "=" in value:
        label, raw_path = value.split("=", 1)
    else:
        raw_path = value
        label = Path(value).name
    label = label.strip()
    source = Path(raw_path).expanduser().resolve()
    if not label or not all(ch.isalnum() or ch in "._-" for ch in label):
        raise ValueError(f"invalid external-input label: {label!r}")
    if not source.is_dir():
        raise FileNotFoundError(f"external input root is not a directory: {source}")
    return label, source


def preserve_external_tree(
    source: Path, destination: Path, *, label: str
) -> dict[str, object]:
    files: list[dict[str, object]] = []
    for path in sorted(item for item in source.rglob("*") if item.is_file()):
        record = external_file_record(source, path)
        payload_path = Path("payload") / "external" / label / str(record["path"])
        preserve_file(path, destination / payload_path, str(record["sha256"]))
        record["payload_path"] = payload_path.as_posix()
        files.append(record)
    tree_digest = hashlib.sha256()
    for record in files:
        tree_digest.update(
            (
                f"{record['path']}\0{record['bytes']}\0{record['sha256']}\n"
            ).encode("utf-8")
        )
    return {
        "label": label,
        "source": str(source),
        "file_count": len(files),
        "bytes": sum(int(row["bytes"]) for row in files),
        "content_manifest_sha256": tree_digest.hexdigest(),
        "files": files,
    }


def verify_checkpoint(checkpoint_dir: Path) -> dict[str, object]:
    """Verify every self-contained payload in an immutable checkpoint."""

    checkpoint_dir = checkpoint_dir.resolve()
    manifest_path = checkpoint_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    failures: list[str] = []

    def verify_one(relative: str, expected: str) -> None:
        path = checkpoint_dir / relative
        if not path.is_file():
            failures.append(f"missing:{relative}")
        elif sha256_file(path) != expected:
            failures.append(f"hash_mismatch:{relative}")

    git = manifest.get("git") or {}
    verify_one("tracked.patch", str(git.get("tracked_patch_sha256") or ""))
    verify_one("base-head.tar", str(git.get("base_archive_sha256") or ""))
    verifier = manifest.get("verifier") or {}
    verify_one(str(verifier.get("path") or ""), str(verifier.get("sha256") or ""))
    for collection in (
        "untracked_source_and_config_inputs",
        "required_ignored_source_and_config_inputs",
        "artifact_content_hashes",
    ):
        for record in manifest.get(collection) or []:
            verify_one(str(record.get("payload_path") or ""), str(record.get("sha256") or ""))
    for tree in manifest.get("external_input_snapshots") or []:
        for record in tree.get("files") or []:
            verify_one(str(record.get("payload_path") or ""), str(record.get("sha256") or ""))
    return {
        "valid": not failures,
        "failure_count": len(failures),
        "failures": failures,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--name", default="2026-07-13_initial")
    parser.add_argument(
        "--purpose",
        default="HPCA 2027 paper-evaluation checkpoint",
        help="Human-readable reason recorded in the manifest.",
    )
    parser.add_argument(
        "--external-input-root",
        action="append",
        default=[],
        metavar="[LABEL=]PATH",
        help="Content-pin a benchmark/input tree outside the Git worktree.",
    )
    parser.add_argument(
        "--verify",
        type=Path,
        help="Verify an existing checkpoint and do not create a new one.",
    )
    args = parser.parse_args()

    if args.verify:
        report = verify_checkpoint(args.verify)
        print(json.dumps(report, indent=2, sort_keys=True))
        return 0 if report["valid"] else 1

    repo = args.repo.resolve()
    output_dir = repo / "paper_eval" / "checkpoints" / args.name
    create_checkpoint_dir(output_dir)

    head = run(repo, "git", "rev-parse", "HEAD").decode().strip()
    base_archive = output_dir / "base-head.tar"
    subprocess.check_call(
        ["git", "archive", "--format=tar", "--output", str(base_archive), head],
        cwd=repo,
    )

    status_raw = run(
        repo,
        "git",
        "status",
        "--porcelain=v1",
        "-z",
        "--untracked-files=all",
    )
    status = [
        row
        for row in parse_status_z(status_raw)
        if not row[1].startswith("paper_eval/checkpoints/")
    ]
    status_identity = json.dumps(status, sort_keys=True, separators=(",", ":")).encode()
    tracked_patch = run(repo, "git", "diff", "--binary", "--no-ext-diff", "HEAD", "--", ".")
    (output_dir / "tracked.patch").write_bytes(tracked_patch)

    tracked_changes: list[dict[str, object]] = []
    untracked_inputs: list[dict[str, object]] = []
    for code, relative in status:
        path = repo / relative
        if code == "??":
            if (
                path.is_file()
                and not relative.startswith("artifacts/")
                and not relative.startswith("paper_eval/checkpoints/")
                and (path.suffix.lower() in INPUT_SUFFIXES or path.name in {"Makefile", "Dockerfile"})
            ):
                record = file_record(repo, path)
                payload_path = Path("payload") / "untracked" / relative
                preserve_file(path, output_dir / payload_path, str(record["sha256"]))
                record["payload_path"] = payload_path.as_posix()
                untracked_inputs.append(record)
            continue

        record: dict[str, object] = {"path": relative, "status": code}
        if path.is_file():
            record.update(file_record(repo, path))
        else:
            record["workspace_state"] = "deleted_or_non_file"
        try:
            head_payload = run(repo, "git", "show", f"HEAD:{relative}")
            record["head_blob_sha256"] = sha256_bytes(head_payload)
        except subprocess.CalledProcessError:
            record["head_blob_sha256"] = None
        tracked_changes.append(record)

    compact_artifacts: list[dict[str, object]] = []
    artifact_root = repo / "artifacts"
    if artifact_root.exists():
        for path in sorted(item for item in artifact_root.rglob("*") if item.is_file()):
            relative = path.relative_to(repo).as_posix()
            if under_any(relative, DEFAULT_INVENTORY_ONLY):
                continue
            record = file_record(repo, path)
            payload_path = Path("payload") / relative
            preserve_file(path, output_dir / payload_path, str(record["sha256"]))
            record["payload_path"] = payload_path.as_posix()
            compact_artifacts.append(record)

    inventory_only = [inventory_tree(repo, root) for root in DEFAULT_INVENTORY_ONLY]
    ignored_inputs = required_ignored_records(repo)
    for record in ignored_inputs:
        payload_path = Path("payload") / "ignored" / str(record["path"])
        preserve_file(repo / str(record["path"]), output_dir / payload_path, str(record["sha256"]))
        record["payload_path"] = payload_path.as_posix()

    external_snapshots: list[dict[str, object]] = []
    seen_labels: set[str] = set()
    for raw_root in args.external_input_root:
        label, source = parse_external_root(raw_root)
        if label in seen_labels:
            raise ValueError(f"duplicate external-input label: {label}")
        seen_labels.add(label)
        external_snapshots.append(
            preserve_external_tree(source, output_dir, label=label)
        )

    # The verifier is copied into the checkpoint, so its integrity can be
    # checked even if the working tree subsequently changes.
    verifier_path = output_dir / "verify_checkpoint.py"
    shutil.copy2(Path(__file__).resolve(), verifier_path)
    verifier_sha256 = sha256_file(verifier_path)

    # Recheck mutable sources after all hashing/copying.  A stable Git status
    # alone is insufficient because file contents can change without changing
    # the porcelain path list.
    for record in tracked_changes + untracked_inputs + compact_artifacts + ignored_inputs:
        relative = str(record.get("path") or "")
        source = repo / relative
        if source.is_file() and record.get("sha256") != sha256_file(source):
            raise RuntimeError(f"input changed before checkpoint finalization: {source}")
    for tree in external_snapshots:
        source_root = Path(str(tree["source"]))
        for record in tree["files"]:
            source = source_root / str(record["path"])
            if not source.is_file() or record.get("sha256") != sha256_file(source):
                raise RuntimeError(
                    f"external input changed before checkpoint finalization: {source}"
                )
    final_status_raw = run(
        repo, "git", "status", "--porcelain=v1", "-z", "--untracked-files=all"
    )
    final_status = [
        row
        for row in parse_status_z(final_status_raw)
        if not row[1].startswith("paper_eval/checkpoints/")
    ]
    final_patch = run(repo, "git", "diff", "--binary", "--no-ext-diff", "HEAD", "--", ".")
    if final_status != status or final_patch != tracked_patch:
        raise RuntimeError("working tree changed while checkpoint was being created")
    if run(repo, "git", "rev-parse", "HEAD").decode().strip() != head:
        raise RuntimeError("Git HEAD changed while checkpoint was being created")

    manifest = {
        "schema_version": 3,
        "purpose": args.purpose,
        "created_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "repository": str(repo),
        "git": {
            "head": head,
            "branch": run(repo, "git", "branch", "--show-current").decode().strip(),
            "status_sha256": sha256_bytes(status_identity),
            "tracked_patch_sha256": sha256_bytes(tracked_patch),
            "base_archive_sha256": sha256_file(base_archive),
        },
        "tracked_changes": tracked_changes,
        "untracked_source_and_config_inputs": untracked_inputs,
        "required_ignored_source_and_config_inputs": ignored_inputs,
        "artifact_content_hashes": compact_artifacts,
        "inventory_only_generated_trees": inventory_only,
        "external_input_snapshots": external_snapshots,
        "verifier": {
            "path": "verify_checkpoint.py",
            "sha256": verifier_sha256,
            "usage": "python verify_checkpoint.py --verify CHECKPOINT_DIR",
        },
        "policy": {
            "input_suffixes": sorted(INPUT_SUFFIXES),
            "inventory_only_roots": list(DEFAULT_INVENTORY_ONLY),
            "required_ignored_inputs": list(REQUIRED_IGNORED_INPUTS),
            "note": (
                "No working-tree file is deleted, staged, or committed. The base HEAD archive, "
                "tracked binary patch, untracked inputs, approved ignored inputs, compact "
                "artifacts, and explicit external benchmark roots are preserved as payloads. "
                "Large reproducible generated trees receive deterministic inventory hashes."
            ),
        },
    }

    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    readme = f"""# HPCA paper-evaluation checkpoint: {args.name}

Purpose: {args.purpose}

This is a self-contained, non-destructive record of the implementation and
benchmark-input state. `base-head.tar` preserves Git HEAD and `tracked.patch`
preserves every tracked working-tree change relative to
`{manifest['git']['head']}`. Untracked inputs, approved ignored implementation
inputs, compact artifacts, and explicitly named external benchmark trees are
copied under `payload/` and bound by SHA-256 in `manifest.json`.

The four multi-gigabyte generated trees named under
`inventory_only_generated_trees` are not copied or content-hashed; their full
path/size/mtime inventories are hashed and their byte/file counts are recorded.
This exception is explicit so the checkpoint does not silently absorb or mutate
unrelated Vitis work products or model-training payloads.

Verify this checkpoint with:

```sh
python verify_checkpoint.py --verify {output_dir}
```
"""
    (output_dir / "README.md").write_text(readme, encoding="utf-8")
    verification = verify_checkpoint(output_dir)
    if not verification["valid"]:
        raise RuntimeError(f"checkpoint self-verification failed: {verification}")
    print(output_dir)
    print(f"tracked changes: {len(tracked_changes)}")
    print(f"untracked inputs: {len(untracked_inputs)}")
    print(f"required ignored inputs: {len(ignored_inputs)}")
    print(f"content-hashed artifacts: {len(compact_artifacts)}")
    print(f"external input trees: {len(external_snapshots)}")
    print("self verification: passed")
    print(f"patch sha256: {manifest['git']['tracked_patch_sha256']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
