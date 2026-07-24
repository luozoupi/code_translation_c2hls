#!/usr/bin/env python3
"""Read-only HTTP dashboard for Fir batch_parallel campaigns."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import urllib.error
import urllib.request
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts" / "fir"))

from batch_parallel.config import campaign_paths, load_campaign
from batch_parallel.dashboard_speedup import compute_speedup_summary
from batch_parallel.dashboard_progress import (
    bench_hls_progress,
    cosim_enabled_for_campaign,
    read_llm_in_flight,
    slot_hls_progress,
)
from batch_parallel.dispatch import setup_tag_for_campaign
from batch_parallel.queue import FirBatchParallelQueue

HTML_PATH = Path(__file__).with_name("dashboard.html")


def _parse_timeleft(raw: str) -> int | None:
    t = raw.strip()
    if not t or t in ("NOT_SET", "UNLIMITED", "N/A"):
        return None
    days = 0
    if "-" in t:
        day_part, t = t.split("-", 1)
        days = int(day_part)
    parts = t.split(":")
    if len(parts) == 3:
        hours, minutes, seconds = (int(parts[0]), int(parts[1]), int(parts[2]))
    elif len(parts) == 2:
        hours, minutes, seconds = (0, int(parts[0]), int(parts[1]))
    else:
        return None
    return days * 86400 + hours * 3600 + minutes * 60 + seconds


def _slurm_state(job_id: str | None, *, campaign_status: str | None = None) -> str:
    if not job_id:
        return "none"
    try:
        out = subprocess.check_output(
            ["squeue", "-h", "-j", str(job_id), "-o", "%T"],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
        if out:
            return out.split("\n")[0]
        status = str(campaign_status or "")
        if status == "aborted":
            return "cancelled"
        if status in ("complete", "completing"):
            return "ended"
        return "inactive"
    except subprocess.CalledProcessError:
        status = str(campaign_status or "")
        if status == "aborted":
            return "cancelled"
        if status in ("complete", "completing"):
            return "ended"
        return "inactive"


def _slurm_time_left_sec(job_id: str | None) -> int | None:
    if not job_id:
        return None
    try:
        out = subprocess.check_output(
            ["squeue", "-h", "-j", str(job_id), "-o", "%L"],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
        if not out:
            return None
        return _parse_timeleft(out.split("\n")[0])
    except subprocess.CalledProcessError:
        return None


def _endpoint_health(endpoint_file: Path) -> tuple[bool, dict[str, Any]]:
    if not endpoint_file.is_file():
        return False, {}
    try:
        payload = json.loads(endpoint_file.read_text(encoding="utf-8"))
    except Exception:
        return False, {}
    url = str(payload.get("url") or "").rstrip("/")
    healthy = False
    if url:
        for suffix in ("/models", "/health"):
            try:
                with urllib.request.urlopen(f"{url}{suffix}", timeout=5) as resp:
                    if resp.status == 200:
                        healthy = True
                        break
            except (urllib.error.URLError, TimeoutError, OSError):
                continue
    return healthy, payload


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _tail_events(path: Path, *, limit: int = 40) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    out: list[dict[str, Any]] = []
    for line in lines[-limit:]:
        line = line.strip()
        if not line:
            continue
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return out


def _job_ids_by_bench(db_path: Path) -> dict[str, int]:
    import sqlite3

    if not db_path.is_file():
        return {}
    conn = sqlite3.connect(db_path)
    try:
        rows = conn.execute("SELECT id, bench FROM jobs").fetchall()
        return {str(bench): int(job_id) for job_id, bench in rows}
    finally:
        conn.close()


def _worker_node_index(worker_id: str | None) -> int | None:
    if not worker_id:
        return None
    match = re.match(r"flash-n(\d+)-", worker_id)
    return int(match.group(1)) if match else None


def build_snapshot(campaign_root: Path) -> dict[str, Any]:
    paths = campaign_paths(campaign_root)
    campaign = load_campaign(campaign_root) if paths["campaign"].is_file() else {}
    status = _read_json(paths["status"])
    node_map = _read_json(paths["node_map"])
    healthy, endpoint = _endpoint_health(paths["endpoint"])

    pilot = (campaign.get("config") or {}).get("pilot") or {}
    model_id = str(pilot.get("model") or "")
    setup_tag = setup_tag_for_campaign(campaign)
    cosim_enabled = cosim_enabled_for_campaign(campaign)
    llm_in_flight = read_llm_in_flight(campaign_root)

    queue = FirBatchParallelQueue(paths["queue_db"])
    jobs = queue.all_jobs()
    id_by_bench = _job_ids_by_bench(paths["queue_db"])
    job_rows = []
    summary = {"total": 0, "done": 0, "claimed": 0, "pending": 0, "failed": 0}
    slot_by_node: dict[int, dict[str, Any]] = {}
    for slot in node_map.get("slots") or []:
        slot_by_node[int(slot["node_index"])] = slot

    for row in jobs:
        st = str(row.get("status") or "pending")
        summary["total"] += 1
        if st in summary:
            summary[st] += 1
        bench = str(row["bench"])
        job_id = id_by_bench.get(bench)
        node_index = _worker_node_index(row.get("worker_id"))
        slurm_job_id = None
        if node_index is not None and node_index in slot_by_node:
            slurm_job_id = str(slot_by_node[node_index].get("slurm_job_id") or "") or None
        hls = bench_hls_progress(
            campaign_root,
            bench=bench,
            status=st,
            model_id=model_id,
            setup_tag=setup_tag,
            node_index=node_index,
            slurm_job_id=slurm_job_id,
            cosim_enabled=cosim_enabled,
            llm_in_flight=llm_in_flight,
        )
        job_rows.append({
            "id": job_id,
            "bench": bench,
            "status": st,
            "worker_id": row.get("worker_id"),
            "error": row.get("error"),
            "result_path": row.get("result_path"),
            "hls": hls,
        })

    done_benches = [str(row["bench"]) for row in jobs if str(row.get("status") or "") == "done"]
    speedup_summary = compute_speedup_summary(
        campaign_root,
        benches=done_benches,
        model_id=model_id,
        setup_tag=setup_tag,
    )
    per_bench_speedup = speedup_summary.get("per_bench") or {}
    per_bench_latency = speedup_summary.get("per_bench_latency") or {}
    per_bench_cosim = speedup_summary.get("per_bench_cosim") or {}
    per_bench_issues = speedup_summary.get("per_bench_issues") or {}
    for row in job_rows:
        row["speedup"] = per_bench_speedup.get(str(row["bench"]))
        row["latency"] = per_bench_latency.get(str(row["bench"]))
        row["cosim"] = per_bench_cosim.get(str(row["bench"]))
        row["run_issues"] = per_bench_issues.get(str(row["bench"]))

    bench_by_id = {row["id"]: row["bench"] for row in job_rows if row.get("id") is not None}
    serving_job = str(endpoint.get("job_id") or campaign.get("gpu_job_id") or "")
    campaign_status = str(campaign.get("campaign_status") or "")
    slots = []
    for slot in node_map.get("slots") or []:
        node_index = int(slot["node_index"])
        slurm_job_id = str(slot.get("slurm_job_id") or "")
        active_id = slot.get("active_job_id")
        bench = bench_by_id.get(active_id) if active_id is not None else None
        hls = slot_hls_progress(
            campaign_root,
            node_index=node_index,
            slurm_job_id=slurm_job_id,
            bench=bench,
            cosim_enabled=cosim_enabled,
            llm_in_flight=llm_in_flight,
        )
        slots.append({
            **slot,
            "bench": bench,
            "hls": hls,
            "slurm_state": _slurm_state(slurm_job_id, campaign_status=campaign_status),
        })

    return {
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "campaign_root": str(campaign_root),
        "campaign_name": campaign_root.name,
        "cosim_enabled": cosim_enabled,
        "campaign": {
            "campaign_status": campaign.get("campaign_status"),
            "compute_state": campaign.get("compute_state"),
            "gpu_mode": campaign.get("gpu_mode"),
            "gpu_job_id": campaign.get("gpu_job_id"),
            "dedicated_gpu_job_id": campaign.get("dedicated_gpu_job_id"),
        },
        "summary": summary,
        "speedup_summary": {
            "n": speedup_summary.get("n"),
            "best_geomean": speedup_summary.get("best_geomean"),
            "avg_geomean": speedup_summary.get("avg_geomean"),
            "worst_geomean": speedup_summary.get("worst_geomean"),
            "latency_mean": speedup_summary.get("latency_mean"),
            "cosim_speedup_geomean": speedup_summary.get("cosim_speedup_geomean"),
        },
        "jobs": job_rows,
        "status": status,
        "slots": slots,
        "endpoint": {
            "healthy": healthy,
            "url": endpoint.get("url"),
            "job_id": endpoint.get("job_id"),
            "host": endpoint.get("host"),
        },
        "gpu": {
            "job_id": serving_job or None,
            "slurm_state": _slurm_state(serving_job, campaign_status=campaign_status),
            "time_left_sec": _slurm_time_left_sec(serving_job),
            "dedicated_job_id": campaign.get("dedicated_gpu_job_id"),
            "dedicated_slurm_state": _slurm_state(
                str(campaign.get("dedicated_gpu_job_id") or ""),
                campaign_status=campaign_status,
            ),
        },
        "recent_events": _tail_events(paths["events"]),
    }


class DashboardHandler(BaseHTTPRequestHandler):
    campaign_root: Path = Path(".")
    html_bytes: bytes = b""

    def log_message(self, fmt: str, *args: Any) -> None:
        print(f"[dashboard] {self.address_string()} - {fmt % args}")

    def do_GET(self) -> None:
        if self.path in ("/", "/index.html"):
            self._send(200, self.html_bytes, "text/html; charset=utf-8")
            return
        if self.path == "/api/snapshot":
            payload = json.dumps(build_snapshot(self.campaign_root), indent=2).encode("utf-8")
            self._send(200, payload, "application/json; charset=utf-8")
            return
        self._send(404, b"not found\n", "text/plain; charset=utf-8")

    def _send(self, code: int, body: bytes, content_type: str) -> None:
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)


def main() -> int:
    parser = argparse.ArgumentParser(description="Fir batch_parallel progress dashboard")
    parser.add_argument("--campaign-root", required=True, type=Path)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()

    campaign_root = args.campaign_root.resolve()
    if not campaign_root.is_dir():
        print(f"ERROR: campaign root not found: {campaign_root}", file=sys.stderr)
        return 2

    html_bytes = HTML_PATH.read_bytes()
    handler = DashboardHandler
    handler.campaign_root = campaign_root
    handler.html_bytes = html_bytes

    server = ThreadingHTTPServer((args.host, args.port), handler)
    print(f"dashboard: http://{args.host}:{args.port}/")
    print(f"campaign: {campaign_root}")
    print("Ctrl+C to stop")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
