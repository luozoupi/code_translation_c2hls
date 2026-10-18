#!/usr/bin/env python3
"""Smoke-gated, restartable reviewer A/B with frozen Phase-B parents."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
from collections import Counter
import fcntl
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import threading
import time

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from evaluation_repro import skill_snapshot_manifest
from phase_b_manifest import load_phase_b_seed
from freeze_qor_ab_seeds import freeze
from run_qor_paired_model_campaign import (atomic_json, cell_environment as base_environment,
                                           endpoint_check, model_label, now, sha256, verify_inputs)
from analyze_qor_review_ab import analyze_cell


def cells_for(config):
    cells = []
    for repeat in range(1, config["replicates"] + 1):
        for index, bench in enumerate(config["benchmarks"]):
            for mode_index, strategy in enumerate(config["strategies"]):
                pair_id = f"{bench['name']}_{strategy}_r{repeat}"
                order = ["review_off", "review_on"]
                if (index + mode_index + repeat) % 2:
                    order.reverse()
                for arm in order:
                    cell_id = f"{pair_id}_{arm}"
                    stamp = config["campaign_id"] + "_" + cell_id
                    directory = Path(config["storage_root"]) / "results" / ("agentic_no_streamcluster_" + stamp)
                    directory /= f"{bench['name']}_{model_label(config)}_{config['skill_mode']}"
                    cells.append({"cell_id": cell_id, "pair_id": pair_id, "benchmark": bench["name"],
                                  "strategy": strategy, "arm": arm, "repeat": repeat, "stamp": stamp,
                                  "skill_mode": config["skill_mode"], "status": "pending",
                                  "result_path": str(directory / (bench["name"] + "_multistep_results.json")),
                                  "schema_jsonl": str(REPO / "artifacts" / ("agentic_no_streamcluster_" + stamp + ".jsonl"))})
    smoke_ids = config["smoke_cell_ids"]
    if not set(smoke_ids).issubset(c["cell_id"] for c in cells):
        raise ValueError("smoke cells are outside the matrix")
    return sorted(cells, key=lambda c: smoke_ids.index(c["cell_id"]) if c["cell_id"] in smoke_ids else len(smoke_ids))


def cell_environment(config, manifest, cell):
    env = base_environment(config, manifest, cell)
    env.update({"C2HLS_QOR_PROFILE_REVIEW": "1" if cell["arm"] == "review_on" else "0",
                "C2HLS_PHASE_B_SEED_MANIFEST": config["phase_b_manifest"], "C2HLS_SAMPLE_ID": cell["cell_id"],
                "C2HLS_SWEEP_QOR_DESIGN_SWEEP": "0", "C2HLS_QOR_DESIGN_SWEEP": "0",
                "C2HLS_SWEEP_ATTEMPTS_PER_CANDIDATE": "2" if cell["strategy"] == "flash" else "1",
                "C2HLS_SWEEP_EXHAUSTIVE_CANDIDATE_ATTEMPTS": "1",
                "C2HLS_LLM_CANDIDATE_BUDGET": "5", "C2HLS_SYNTHESIS_EVAL_BUDGET": "5"})
    # The paper fingerprint requires a seed field; the DeepSeek client omits it.
    env["C2HLS_LLM_SEED"] = "42"
    return env


def prepare(config_path, config):
    import c2hls
    root = Path(config["report_root"])
    if (root / "campaign_state.json").exists():
        raise ValueError("campaign already prepared; run it to resume")
    if config["workers"] not in (1, 2) or config["replicates"] != 3:
        raise ValueError("registration requires three samples and at most two workers")
    root.mkdir(parents=True, exist_ok=True)
    seed_path = Path(config["phase_b_manifest"])
    seeds = json.loads(seed_path.read_text()) if seed_path.exists() else freeze(config)
    files = [config_path, seed_path, Path(config["skill_library"]), Path(config["shape_registry"])]
    registry = json.loads(Path(config["shape_registry"]).read_text())
    for bench in config["benchmarks"]:
        name = bench["name"]
        directory = Path(config["benchmarks_root"]) / name
        inputs = c2hls._load_benchmark_inputs(str(directory))
        seed = load_phase_b_seed(seed_path, benchmark=name, input_c=inputs["c_code"], header_code=inputs["header_code"],
                                 expected_part="xcu280-fsvh2892-2L-e", expected_clock_ns=3.33, expected_vitis_version="2023.2")
        files.append(Path(seed["provenance"]["code_path"]))
        for key in ("result_path", "history_path"):
            p = Path(seed["provenance"]["source_artifacts"][key])
            if sha256(p) != seed["provenance"]["source_artifacts"][key.replace("_path", "_sha256")]:
                raise ValueError("historical seed provenance changed")
        metadata = json.loads((directory / "metadata.json").read_text())
        if sha256(directory / metadata["testbench_file"]) != registry["benchmarks"][name]["testbench_sha256"]:
            raise ValueError("shape contract mismatch: " + name)
        files.extend(p for p in directory.rglob("*") if p.is_file())
    audit_dir = root / "reference_audit"
    with (root / "preflight.log").open("w") as log:
        subprocess.run([sys.executable, str(REPO / "scripts/audit_reference_isolation_campaign.py"),
                        "--benchmarks-dir", config["benchmarks_root"], "--skill-library", config["skill_library"],
                        "--benchmarks", ",".join(b["name"] for b in config["benchmarks"]),
                        "--output-dir", str(audit_dir)], check=True, cwd=REPO, stdout=log, stderr=subprocess.STDOUT, timeout=180)
    if json.loads((audit_dir / "audit.json").read_text())["status"] != "passed":
        raise ValueError("reference isolation audit failed")
    files.extend(REPO.glob("*.py"))
    files.extend((REPO / "scripts").glob("*qor*.py"))
    manifest = {"schema_version": "c2hls.qor-review-ab-manifest.v1", "prepared_at": now(), "config": config,
                "config_sha256": sha256(config_path), "endpoint": endpoint_check(config),
                "skill_snapshot_sha256": skill_snapshot_manifest(REPO, environ={"C2HLS_SKILL_LIBRARY_PATH": config["skill_library"], "C2HLS_SKILL_LIBRARY_FROZEN": "1"})["sha256"],
                "input_files": [{"path":str(p.resolve()), "sha256":sha256(p)} for p in sorted(set(files))],
                "cells": cells_for(config)}
    atomic_json(root / "manifest.json", manifest)
    atomic_json(root / "campaign_state.json", {"schema_version": "c2hls.qor-review-ab-state.v1", "status": "prepared",
                                                "updated_at": now(), "config_sha256":manifest["config_sha256"], "cells":manifest["cells"]})
    report(config_path)


def report(config_path):
    subprocess.run([sys.executable, str(REPO / "scripts/analyze_qor_review_ab.py"), "--config", str(config_path)],
                   cwd=REPO, check=True, timeout=120, stdout=subprocess.DEVNULL)


def refresh_empty_preflight(config_path, config):
    root = Path(config["report_root"])
    state = json.loads((root / "campaign_state.json").read_text())
    if state["status"] != "smoke_failed":
        raise ValueError("only an empty failed smoke preflight may be refreshed")
    for cell in state["cells"]:
        if any(Path(cell["result_path"]).parent.glob("*.json")):
            raise ValueError("cannot refresh a campaign with evaluation/model artifacts")
        log_path = Path(cell.get("log_path", root / "absent.log"))
        if log_path.is_file() and 'HTTP Request: POST' in log_path.read_text():
            raise ValueError("cannot refresh after a model API request")
    archive = root / "prelaunch_attempts" / now().replace(":", "-")
    archive.mkdir(parents=True, exist_ok=False)
    for name in ("manifest.json", "campaign_state.json", "launcher.log"):
        if (root / name).exists():
            (root / name).replace(archive / name)
    prepare(config_path, config)


def archive_cell(config, cell):
    scratch = Path(config["scratch_root"]) / cell["cell_id"]
    archive = Path(config["storage_root"]) / "vitis_archives" / (cell["cell_id"] + ".tar")
    archive.parent.mkdir(parents=True, exist_ok=True)
    partial = archive.with_suffix(".tar.partial")
    subprocess.run(["tar", "-cf", str(partial), "-C", str(scratch), "."], check=True, timeout=600)
    subprocess.run(["tar", "-tf", str(partial)], check=True, timeout=180, stdout=subprocess.DEVNULL)
    digest = subprocess.check_output(["sha256sum", str(partial)], text=True, timeout=180).split()[0]
    partial.replace(archive)
    receipt = {"path": str(archive), "sha256": digest, "original_root": str(scratch)}
    atomic_json(archive.with_suffix(".receipt.json"), receipt)
    shutil.rmtree(scratch)
    return receipt


def run(config_path, config):
    root = Path(config["report_root"])
    manifest = json.loads((root / "manifest.json").read_text())
    verify_inputs(manifest)
    endpoint_check(config)
    lock = (root / "campaign.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    state_path = root / "campaign_state.json"
    state = json.loads(state_path.read_text())
    seeds = json.loads(Path(config["phase_b_manifest"]).read_text())
    state.setdefault("restarts", []).append({"at": now(), "previous_status":state["status"], "previous_pid":state.get("pid")})
    state.update(status="running", pid=os.getpid())
    mutex, stopping = threading.Lock(), threading.Event()
    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, lambda *_: stopping.set())
    Path(config["scratch_root"]).mkdir(parents=True, exist_ok=True)
    logs = Path(config["storage_root"]) / "logs"
    logs.mkdir(parents=True, exist_ok=True)

    def update(cell=None, **changes):
        with mutex:
            (state if cell is None else cell).update(changes)
            state["updated_at"] = now()
            state["counts"] = dict(Counter(c["status"] for c in state["cells"]))
            atomic_json(state_path, state)

    def managed_process(command, log, *, env=None, timeout=7200):
        process = subprocess.Popen(command, cwd=REPO, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        start = time.monotonic()
        while process.poll() is None:
            if stopping.is_set() or time.monotonic() - start > timeout:
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=20)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait(timeout=20)
                raise TimeoutError("interrupted" if stopping.is_set() else "process wall-time limit")
            time.sleep(2)
        return process.returncode

    def worker(cell):
        if stopping.is_set():
            return
        verify_inputs(manifest)
        if shutil.disk_usage(config["scratch_root"]).free < 25 * 1024**3:
            stopping.set()
            raise RuntimeError("less than 25 GiB scratch free; queue stopped")
        env = cell_environment(config, manifest, cell)
        for key in ("C2HLS_TMP_ROOT", "C2HLS_VITIS_USER_HOME"):
            Path(env[key]).mkdir(parents=True, exist_ok=True)
        log_path = logs / (cell["cell_id"] + ".log")
        update(cell, status="running", started_at=now(), log_path=str(log_path))
        with log_path.open("a") as log:
            command = ["bash", "-c", 'source "$C2HLS_VITIS_SETTINGS" && exec "$1" "$2"',
                       "c2hls-review-ab", sys.executable, str(REPO / "run_agentic_sweep.py")]
            code = managed_process(command, log, env=env, timeout=config["cell_timeout_seconds"])
            row = analyze_cell(cell, config, seeds)
            if code or not row["audit_ok"]:
                update(cell, status="failed", returncode=code, measurement=row, finished_at=now())
                print(now(), cell["cell_id"], "failed", row["issues"], flush=True)
                return
            update(cell, status="complete", measurement=row, finished_at=now())
            print(now(), cell["cell_id"], "complete", row["cycles"], flush=True)
            try:
                code = managed_process([sys.executable, __file__, "archive", "--config", str(config_path),
                                        "--cell-id", cell["cell_id"]], log, timeout=1100)
                receipt = Path(config["storage_root"]) / "vitis_archives" / (cell["cell_id"] + ".receipt.json")
                if code:
                    raise RuntimeError("archive process exit " + str(code))
                update(cell, scratch_archive=json.loads(receipt.read_text()))
            except (OSError, RuntimeError, TimeoutError) as error:
                update(cell, archive_error=str(error))

    # Re-adopt complete results after an interruption, but never rerun an audited cell.
    for cell in state["cells"]:
        if cell["status"] == "pending":
            continue
        row = analyze_cell(cell, config, seeds)
        if row["audit_ok"]:
            cell.update(status="complete", measurement=row)
        elif cell["status"] == "complete":
            raise RuntimeError("previous completed cell no longer validates: " + cell["cell_id"])
    smoke = set(config["smoke_cell_ids"])
    groups = [[c for c in state["cells"] if c["cell_id"] in smoke and c["status"] != "complete"],
              [c for c in state["cells"] if c["cell_id"] not in smoke and c["status"] != "complete"]]
    update()
    last_report = 0
    with ThreadPoolExecutor(max_workers=config["workers"]) as pool:
        for index, group in enumerate(groups):
            if stopping.is_set():
                break
            futures = {pool.submit(worker, cell):cell for cell in group}
            while futures:
                done, _ = wait(futures, timeout=10, return_when=FIRST_COMPLETED)
                for future in done:
                    cell = futures.pop(future)
                    try:
                        future.result()
                    except Exception as error:
                        update(cell, status="interrupted" if stopping.is_set() else "failed", error=str(error), finished_at=now())
                update()
                if time.monotonic() - last_report > 60:
                    try:
                        report(config_path)
                    except (subprocess.SubprocessError, OSError) as error:
                        update(report_error=str(error))
                    last_report = time.monotonic()
            if index == 0:
                checks = [c for c in state["cells"] if c["cell_id"] in smoke]
                passed = all(c["status"] == "complete" and c["measurement"]["feasible"]
                             and (c["arm"] != "review_on" or c["measurement"]["feedback_in_prompt"]) for c in checks)
                if not passed:
                    update(status="smoke_failed")
                    report(config_path)
                    return
    update(status="interrupted" if stopping.is_set() else "complete" if all(c["status"] == "complete" for c in state["cells"]) else "complete_with_failures", finished_at=now())
    report(config_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "refresh-empty-preflight", "run", "report", "archive"))
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--cell-id")
    args = parser.parse_args()
    path = args.config.resolve()
    config = json.loads(path.read_text())
    if args.action == "archive":
        cell = next(c for c in cells_for(config) if c["cell_id"] == args.cell_id)
        print(json.dumps(archive_cell(config, cell)))
    else:
        {"prepare":prepare, "refresh-empty-preflight":refresh_empty_preflight,
         "run":run, "report":lambda p,c:report(p)}[args.action](path, config)
