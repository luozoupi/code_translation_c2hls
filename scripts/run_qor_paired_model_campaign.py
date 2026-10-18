#!/usr/bin/env python3
"""Run fresh model trajectories and audit their exact pre/post-QoR pairs."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
from collections import Counter
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
import re
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import threading
import time
import urllib.request

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from evaluation_repro import skill_snapshot_manifest  # noqa: E402


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    tmp.replace(path)


def cells_for(config: dict) -> list[dict]:
    cells = []
    for bench in config["benchmarks"]:
        for strategy in config["strategies"]:
            cell_id = f"{bench['name']}_{strategy}_{config['skill_mode']}"
            stamp = f"{config['campaign_id']}_{cell_id}"
            out = Path(config["storage_root"]) / "results" / f"agentic_no_streamcluster_{stamp}"
            out /= f"{bench['name']}_{model_label(config)}_{config['skill_mode']}"
            cells.append({
                "cell_id": cell_id, "benchmark": bench["name"],
                "family": bench["family"], "strategy": strategy,
                "skill_mode": config["skill_mode"], "stamp": stamp,
                "result_path": str(out / f"{bench['name']}_multistep_results.json"),
                "parent_path": str(out / f"{bench['name']}_pre_qor.json"),
                "schema_jsonl": str(REPO / "artifacts" / f"agentic_no_streamcluster_{stamp}.jsonl"),
                "status": "pending",
            })
    smoke = config["smoke_cell_ids"]
    if not set(smoke).issubset({cell["cell_id"] for cell in cells}):
        raise ValueError("smoke cells are outside the registered matrix")
    return sorted(cells, key=lambda cell: (
        smoke.index(cell["cell_id"]) if cell["cell_id"] in smoke else len(smoke),
    ))


def model_label(config: dict) -> str:
    return re.sub(r"[^a-zA-Z0-9_]+", "_", config["model"]).strip("_")


def endpoint_check(config: dict) -> dict:
    key = Path(config["key_file"]).read_text().strip()
    if not key or "\n" in key:
        raise ValueError("API key file must contain one nonempty key")
    request = urllib.request.Request(
        config["endpoint"].rstrip("/") + "/models",
        headers={"Authorization": "Bearer " + key},
    )
    with urllib.request.urlopen(request, timeout=25) as response:
        payload = json.load(response)
    models = [item.get("id") for item in payload.get("data", [])]
    if config["model"] not in models:
        raise ValueError("requested model is absent from the configured endpoint")
    return {"checked_at": now(), "model": config["model"], "available": True}


def prepare(config_path: Path, config: dict, *, refresh_empty_failure: bool = False) -> dict:
    root = Path(config["report_root"])
    if (root / "campaign_state.json").exists():
        previous = json.loads((root / "campaign_state.json").read_text())
        if not refresh_empty_failure or previous["status"] != "smoke_failed":
            raise ValueError("campaign already has state; use run to resume it")
        for cell in previous["cells"]:
            directory = Path(cell["result_path"]).parent
            if directory.is_dir() and any(directory.rglob("*.json")):
                raise ValueError("cannot refresh a campaign that has model or evaluation artifacts")
        archive = root / "prelaunch_attempts" / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
        archive.mkdir(parents=True, exist_ok=False)
        for name in ("manifest.json", "campaign_state.json"):
            shutil.copy2(root / name, archive / name)
    if config["workers"] not in (1, 2):
        raise ValueError("this campaign supports at most two workers")
    registry_path = Path(config["shape_registry"])
    registry = json.loads(registry_path.read_text())
    files = [config_path, registry_path, Path(config["skill_library"])]
    for filename in config.get("required_regression_outputs", []):
        path = Path(filename)
        smoke = json.loads(path.read_text())
        design = smoke.get("design_sweep") or {}
        candidates = design.get("candidates") or []
        if (smoke.get("model_calls") != 0 or not candidates
                or not design.get("parent_integrity_passed")
                or not all(candidate.get("feasible") is True
                           and (candidate.get("csim") or {}).get("passed") is True
                           for candidate in candidates)):
            raise ValueError(f"required AXI regression smoke failed: {path}")
        source = Path(smoke["source_result"]["path"])
        if sha256(source) != smoke["source_result"]["sha256"]:
            raise ValueError(f"regression source changed: {source}")
        files.extend([path, source])
    for bench in config["benchmarks"]:
        directory = Path(config["benchmarks_root"]) / bench["name"]
        meta = json.loads((directory / "metadata.json").read_text())
        testbench = directory / meta["testbench_file"]
        if sha256(testbench) != registry["benchmarks"][bench["name"]]["testbench_sha256"]:
            raise ValueError(f"shape-contract mismatch: {bench['name']}")
        if meta.get("status") == "disabled":
            raise ValueError(f"disabled benchmark: {bench['name']}")
        files.extend(path for path in directory.rglob("*") if path.is_file())
    files.extend(REPO.glob("*.py"))
    files.extend((REPO / "scripts").glob("*qor*.py"))
    root.mkdir(parents=True, exist_ok=True)
    audit_dir = root / "reference_audit"
    with (root / "preflight.log").open("w") as log:
        subprocess.run([
            sys.executable, str(REPO / "scripts/audit_reference_isolation_campaign.py"),
            "--benchmarks-dir", config["benchmarks_root"],
            "--skill-library", config["skill_library"],
            "--benchmarks", ",".join(item["name"] for item in config["benchmarks"]),
            "--output-dir", str(audit_dir),
        ], cwd=REPO, stdout=log, stderr=subprocess.STDOUT, check=True)
    audit = json.loads((audit_dir / "audit.json").read_text())
    if audit["status"] != "passed":
        raise ValueError("reference-isolation preflight failed")
    skill_sha = skill_snapshot_manifest(REPO, environ={
        "C2HLS_SKILL_LIBRARY_PATH": config["skill_library"],
        "C2HLS_SKILL_LIBRARY_FROZEN": "1",
    })["sha256"]
    manifest = {
        "schema_version": "c2hls.qor-paired-campaign.v1",
        "prepared_at": now(), "config": config,
        "config_sha256": sha256(config_path),
        "skill_snapshot_sha256": skill_sha,
        "endpoint": endpoint_check(config),
        "input_files": [
            {"path": str(path.resolve()), "sha256": sha256(path)}
            for path in sorted(set(files))
        ],
        "cells": cells_for(config),
        "comparison": "same fresh LLM trajectory before and after additive QoR exploration",
        "statistical_unit": "benchmark; strategies are clustered within benchmark",
        "replicates": 1,
        "limitations": [
            "single fresh trajectory per benchmark and workflow",
            "seed 42 is registered; DeepSeek omits unsupported seed from API requests",
            "extra synthesis cost is charged to QoR; this is not equal-tool-budget superiority",
            "in-domain PolyBench subset; no external generalization claim",
            "automatic source-local controls; no GEMM/2MM template transplantation",
            "CSynth estimates only; COSIM and physical implementation are excluded",
        ],
    }
    atomic_json(root / "manifest.json", manifest)
    atomic_json(root / "campaign_state.json", {
        "schema_version": "c2hls.qor-paired-state.v1",
        "status": "prepared", "updated_at": now(),
        "config_sha256": manifest["config_sha256"], "cells": manifest["cells"],
    })
    return manifest


def verify_inputs(manifest: dict) -> None:
    for entry in manifest["input_files"]:
        if sha256(Path(entry["path"])) != entry["sha256"]:
            raise RuntimeError(f"frozen campaign input changed: {entry['path']}")


def cell_environment(config: dict, manifest: dict, cell: dict) -> dict[str, str]:
    env = {
        key: value for key, value in os.environ.items()
        if not key.startswith(("C2HLS_", "OPENAI_", "ANTHROPIC_"))
    }
    env.update(config["environment"])
    scratch = Path(config["scratch_root"]) / cell["cell_id"]
    env.update({
        "OPENAI_API_KEY": Path(config["key_file"]).read_text().strip(),
        "OPENAI_BASE_URL": config["endpoint"],
        "C2HLS_MODEL": config["model"],
        "C2HLS_MODEL_LABEL": model_label(config),
        "C2HLS_MODEL_REVISION": config["model_revision"],
        "C2HLS_SWEEP_MODELS": config["model"],
        "C2HLS_SWEEP_BENCHES": cell["benchmark"],
        "C2HLS_SWEEP_BENCHMARKS_DIR": config["benchmarks_root"],
        "C2HLS_HLSFACTORY_SHAPE_REGISTRY": config["shape_registry"],
        "C2HLS_SKILL_LIBRARY_PATH": config["skill_library"],
        "C2HLS_SKILL_SNAPSHOT_SHA256": manifest["skill_snapshot_sha256"],
        "C2HLS_SWEEP_STRATEGY": cell["strategy"],
        "C2HLS_DYNAMIC_ROUTING": "1" if cell["strategy"] == "dynamic" else "0",
        "C2HLS_SWEEP_SKILL_MODES": cell["skill_mode"],
        "C2HLS_SWEEP_STAMP": cell["stamp"],
        "C2HLS_SWEEP_RESULTS_ROOT": str(Path(config["storage_root"]) / "results"),
        "C2HLS_TMP_ROOT": str(scratch / "work"),
        "C2HLS_SWEEP_TMP_ROOT": str(scratch / "work"),
        "C2HLS_VITIS_USER_HOME": str(scratch / "user_home"),
        "XILINX_LOCAL_USER_DATA": str(scratch / "user_home"),
        "TMPDIR": str(scratch / "work"),
        "PYTHONUNBUFFERED": "1",
    })
    for role in ("TRANSLATOR", "SYNTHESIS", "QUALITY_REPAIR", "FEEDBACK"):
        env[f"C2HLS_{role}_MODEL"] = config["model"]
    return env


def write_report(config_path: Path) -> None:
    subprocess.run([
        sys.executable, str(REPO / "scripts/analyze_qor_paired_model_campaign.py"),
        "--config", str(config_path),
    ], check=True, cwd=REPO, stdout=subprocess.DEVNULL)


def run_campaign(config_path: Path, config: dict) -> None:
    root = Path(config["report_root"])
    manifest = json.loads((root / "manifest.json").read_text())
    verify_inputs(manifest)
    endpoint_check(config)
    state_path = root / "campaign_state.json"
    state = json.loads(state_path.read_text())
    lock_file = (root / "campaign.lock").open("w")
    fcntl.flock(lock_file, fcntl.LOCK_EX | fcntl.LOCK_NB)
    mutex = threading.Lock()
    stopping = threading.Event()
    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, lambda _signal, _frame: stopping.set())
    state.update(status="running", pid=os.getpid(), updated_at=now())
    log_dir = Path(config["storage_root"]) / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    def update(cell: dict | None = None, **changes) -> None:
        with mutex:
            (cell if cell is not None else state).update(changes)
            state["updated_at"] = now()
            state["counts"] = dict(Counter(item["status"] for item in state["cells"]))
            atomic_json(state_path, state)

    def run_cell(cell: dict) -> None:
        if stopping.is_set():
            update(cell, status="interrupted")
            return
        verify_inputs(manifest)
        if shutil.disk_usage(config["scratch_root"]).free < 25 * 1024 ** 3:
            raise RuntimeError("local scratch has less than 25 GiB free")
        env = cell_environment(config, manifest, cell)
        if cell["status"] in {"failed", "timeout", "invalid_artifact"}:
            env["C2HLS_SWEEP_RESUME"] = "0"
        for key in ("C2HLS_TMP_ROOT", "C2HLS_VITIS_USER_HOME"):
            Path(env[key]).mkdir(parents=True, exist_ok=True)
        log_path = log_dir / f"{cell['cell_id']}.log"
        update(cell, status="running", started_at=now(), log_path=str(log_path))
        command = [
            "bash", "-c",
            'source "$C2HLS_VITIS_SETTINGS" && exec "$1" "$2"',
            "c2hls-qor-worker", sys.executable, str(REPO / "run_agentic_sweep.py"),
        ]
        started = time.monotonic()
        with log_path.open("a") as log:
            process = subprocess.Popen(
                command, cwd=REPO, env=env, stdout=log,
                stderr=subprocess.STDOUT, start_new_session=True,
            )
            update(cell, pid=process.pid)
            timeout = config.get("cell_timeout_seconds", 43200)
            while process.poll() is None:
                if stopping.is_set() or time.monotonic() - started > timeout:
                    os.killpg(process.pid, signal.SIGTERM)
                    try:
                        process.wait(timeout=20)
                    except subprocess.TimeoutExpired:
                        os.killpg(process.pid, signal.SIGKILL)
                        process.wait()
                    update(cell, status="interrupted" if stopping.is_set() else "timeout", finished_at=now())
                    return
                time.sleep(5)
        update(cell, returncode=process.returncode, elapsed_seconds=time.monotonic() - started)
        from analyze_qor_paired_model_campaign import analyze_cell
        row = analyze_cell(cell, config)
        status = "complete" if process.returncode == 0 and row["pair_valid"] else "failed"
        update(cell, status=status, finished_at=now(), measurement=row)
        print(f"{now()} {cell['cell_id']} {status} speedup={row.get('speedup')}", flush=True)
        if config.get("archive_scratch"):
            scratch = Path(config["scratch_root"]) / cell["cell_id"]
            archive = Path(config["storage_root"]) / "vitis_archives" / f"{cell['cell_id']}.tar"
            archive.parent.mkdir(parents=True, exist_ok=True)
            partial = archive.with_suffix(".tar.partial")
            try:
                subprocess.run(["tar", "-cf", str(partial), "-C", str(scratch), "."],
                               check=True, timeout=900, stderr=subprocess.DEVNULL)
                subprocess.run(["tar", "-tf", str(partial)], check=True, timeout=300,
                               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                partial.replace(archive)
                update(cell, scratch_archive={"path": str(archive), "sha256": sha256(archive),
                                              "original_root": str(scratch)})
                shutil.rmtree(scratch)
            except (OSError, subprocess.SubprocessError) as error:
                update(cell, archive_error=type(error).__name__)

    Path(config["scratch_root"]).mkdir(parents=True, exist_ok=True)
    from analyze_qor_paired_model_campaign import analyze_cell
    for cell in state["cells"]:
        if cell["status"] == "complete" and not analyze_cell(cell, config)["pair_valid"]:
            cell["status"] = "invalid_artifact"
    pending = [cell for cell in state["cells"] if cell["status"] != "complete"]
    smoke_ids = set(config["smoke_cell_ids"])
    groups = [
        [cell for cell in pending if cell["cell_id"] in smoke_ids],
        [cell for cell in pending if cell["cell_id"] not in smoke_ids],
    ]
    update()
    with ThreadPoolExecutor(max_workers=config["workers"]) as pool:
        for index, group in enumerate(groups):
            if stopping.is_set():
                break
            futures = {pool.submit(run_cell, cell): cell for cell in group}
            while futures:
                done, _ = wait(futures, timeout=30, return_when=FIRST_COMPLETED)
                for future in done:
                    cell = futures.pop(future)
                    try:
                        future.result()
                    except Exception as error:
                        update(cell, status="failed", error=str(error), finished_at=now())
                update()
                write_report(config_path)
            if index == 0 and any(
                cell["status"] != "complete" for cell in state["cells"]
                if cell["cell_id"] in smoke_ids
            ):
                update(status="smoke_failed")
                write_report(config_path)
                return
    update(status="interrupted" if stopping.is_set() else
           "complete" if all(cell["status"] == "complete" for cell in state["cells"])
           else "complete_with_failures", finished_at=now())
    write_report(config_path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "refresh-empty-preflight", "run", "report", "endpoint"))
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    config_path = args.config.resolve()
    config = json.loads(config_path.read_text())
    if args.action in {"prepare", "refresh-empty-preflight"}:
        result = prepare(config_path, config, refresh_empty_failure=args.action == "refresh-empty-preflight")
        print(json.dumps({"prepared": len(result["cells"]), "report_root": config["report_root"]}))
        write_report(config_path)
    elif args.action == "endpoint":
        print(json.dumps(endpoint_check(config)))
    elif args.action == "report":
        write_report(config_path)
    else:
        run_campaign(config_path, config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
