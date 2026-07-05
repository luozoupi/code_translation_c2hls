#!/usr/bin/env python3
"""Read/write artifacts/pc2/session.json for supervised batch runs."""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path


def _repo_root() -> Path:
    return Path(os.environ.get("C2HLS_ROOT", Path(__file__).resolve().parents[2]))


def session_path() -> Path:
    root = _repo_root() / "artifacts" / "pc2"
    session_id = os.environ.get("PC2_SESSION_ID", "").strip()
    if session_id:
        return root / "sessions" / session_id / "session.json"
    return root / "session.json"


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def default_session() -> dict:
    return {
        "created_at": _now(),
        "updated_at": _now(),
        "session_id": os.environ.get("PC2_SESSION_ID", "").strip() or None,
        "gpu_partition": os.environ.get("PC2_GPU_PARTITION", "gpu_h100"),
        "compute_partition": os.environ.get("PC2_COMPUTE_PARTITION", "fpga"),
        "walltime": os.environ.get("PC2_WALLTIME", "1-00:00:00"),
        "gpu_job_id": None,
        "compute_job_id": None,
        "gpu_state": "queued",
        "compute_state": "waiting_for_gpu",
        "worker_cmd": os.environ.get(
            "PC2_WORKER_CMD",
            f"{os.environ.get('C2HLS_PYTHON', 'python3')} run_agentic_sweep.py --pc2",
        ),
        "restarts": {"gpu": 0, "compute": 0},
        "last_error": None,
        "gpu_borrowed": False,
        "borrowed_from": None,
    }


def load_session() -> dict:
    path = session_path()
    if not path.exists():
        return default_session()
    return json.loads(path.read_text())


def save_session(data: dict) -> None:
    path = session_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    data["updated_at"] = _now()
    path.write_text(json.dumps(data, indent=2) + "\n")


def _reset_run_fields(data: dict) -> None:
    """Clear job/runtime state so a new start_session does not inherit a prior completion."""
    data["gpu_job_id"] = None
    data["compute_job_id"] = None
    data["gpu_state"] = "queued"
    data["compute_state"] = "waiting_for_gpu"
    data["last_error"] = None
    data["gpu_borrowed"] = False
    data["borrowed_from"] = None
    data.setdefault("restarts", {})["gpu"] = 0
    data["restarts"]["compute"] = 0


def cmd_init(args: argparse.Namespace) -> int:
    path = session_path()
    if path.exists() and not args.reset:
        print(path, file=sys.stderr)
        return 0
    data = load_session() if path.exists() and args.reset else default_session()
    if args.reset:
        worker = data.get("worker_cmd")
        _reset_run_fields(data)
        if worker:
            data["worker_cmd"] = worker
    save_session(data)
    print(path)
    return 0


def cmd_reset_run(_: argparse.Namespace) -> int:
    data = load_session() if session_path().exists() else default_session()
    worker = data.get("worker_cmd")
    _reset_run_fields(data)
    if worker:
        data["worker_cmd"] = worker
    save_session(data)
    print(session_path())
    return 0


def cmd_get(args: argparse.Namespace) -> int:
    data = load_session()
    keys = args.key.split(".")
    cur: object = data
    for key in keys:
        if not isinstance(cur, dict) or key not in cur:
            return 1
        cur = cur[key]
    print(cur)
    return 0


def cmd_set(args: argparse.Namespace) -> int:
    data = load_session()
    keys = args.key.split(".")
    cur = data
    for key in keys[:-1]:
        cur = cur.setdefault(key, {})
    if args.json:
        cur[keys[-1]] = json.loads(args.value)
    elif args.value in ("", "null", "None"):
        cur[keys[-1]] = None
    else:
        cur[keys[-1]] = args.value
    save_session(data)
    return 0


def cmd_bump_restart(args: argparse.Namespace) -> int:
    data = load_session()
    data.setdefault("restarts", {}).setdefault(args.which, 0)
    data["restarts"][args.which] = int(data["restarts"][args.which]) + 1
    save_session(data)
    print(data["restarts"][args.which])
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_init = sub.add_parser("init")
    p_init.add_argument(
        "--reset",
        action="store_true",
        help="reset job/runtime fields when session.json already exists",
    )
    p_init.set_defaults(func=cmd_init)

    p_reset = sub.add_parser("reset-run")
    p_reset.set_defaults(func=cmd_reset_run)

    p_get = sub.add_parser("get")
    p_get.add_argument("key")
    p_get.set_defaults(func=cmd_get)

    p_set = sub.add_parser("set")
    p_set.add_argument("key")
    p_set.add_argument("value")
    p_set.add_argument("--json", action="store_true")
    p_set.set_defaults(func=cmd_set)

    p_bump = sub.add_parser("bump-restart")
    p_bump.add_argument("which", choices=["gpu", "compute"])
    p_bump.set_defaults(func=cmd_bump_restart)

    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
