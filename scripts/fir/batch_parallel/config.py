"""Campaign configuration for Fir batch_parallel harness."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent.parent
DEFAULT_PILOT_JSON = SCRIPT_DIR / "batch_parallel_pilot.json"


@dataclass
class FirBatchParallelConfig:
    compute_nodes: int = 2
    compute_nodes_match_benches: bool = False
    workers_per_node: int = 2
    worker_cpus: int = 8
    worker_mem_gb: int = 32
    poll_sec: float = 5.0
    coordinator_poll_sec: float = 30.0
    auto_stop_delay_sec: int = 120
    gpu_policy: str = "batch_park"  # batch_park | always_on
    gpu_batch_threshold: int = 2
    gpu_batch_flush_s: int = 3600
    park_threshold_s: float = 1800.0
    long_vitis_park_s: float = 3600.0
    park_grace_s: float = 300.0
    llm_burst_grace_s: float = 300.0
    long_vitis_benches: list[str] = field(default_factory=list)
    job_prefix: str = "firbp"
    artifact_prefix: str = "batch_parallel"
    pilot_benches: list[str] = field(
        default_factory=lambda: ["hlsfactory_2mm", "hlsfactory_lu", "hlsfactory_3mm"]
    )
    pilot_workflow: str = "flash"
    pilot_variant: str = ""
    pilot_run_cosim: bool = False
    model: str = "mistralai/Devstral-2-123B-Instruct-2512"
    turns: int = 4

    @property
    def worker_slots(self) -> int:
        return self.compute_nodes * self.workers_per_node

    def node_slurm_cpus(self) -> int:
        return self.workers_per_node * self.worker_cpus

    def node_slurm_mem_gb(self) -> int:
        return self.workers_per_node * self.worker_mem_gb

    def to_dict(self) -> dict[str, Any]:
        return {
            "compute_nodes": self.compute_nodes,
            "compute_nodes_match_benches": self.compute_nodes_match_benches,
            "workers_per_node": self.workers_per_node,
            "worker_cpus": self.worker_cpus,
            "worker_mem_gb": self.worker_mem_gb,
            "poll_sec": self.poll_sec,
            "coordinator_poll_sec": self.coordinator_poll_sec,
            "auto_stop_delay_sec": self.auto_stop_delay_sec,
            "gpu_policy": self.gpu_policy,
            "gpu_batch_threshold": self.gpu_batch_threshold,
            "gpu_batch_flush_s": self.gpu_batch_flush_s,
            "park_threshold_s": self.park_threshold_s,
            "long_vitis_park_s": self.long_vitis_park_s,
            "park_grace_s": self.park_grace_s,
            "llm_burst_grace_s": self.llm_burst_grace_s,
            "long_vitis_benches": list(self.long_vitis_benches),
            "job_prefix": self.job_prefix,
            "artifact_prefix": self.artifact_prefix,
            "pilot": {
                "benches": list(self.pilot_benches),
                "workflow": self.pilot_workflow,
                "variant": self.pilot_variant,
                "run_cosim": self.pilot_run_cosim,
                "model": self.model,
                "turns": self.turns,
            },
        }


def load_config(json_path: Path | None = None) -> FirBatchParallelConfig:
    config_path = os.getenv("BATCH_PARALLEL_CONFIG", "").strip()
    path = Path(config_path) if config_path else (json_path or DEFAULT_PILOT_JSON)
    data = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}
    pilot = data.pop("pilot", {}) or {}
    cfg = FirBatchParallelConfig()
    for key, value in data.items():
        if hasattr(cfg, key):
            setattr(cfg, key, value)
    if pilot.get("benches"):
        cfg.pilot_benches = [str(b) for b in pilot["benches"]]
    if pilot.get("workflow"):
        cfg.pilot_workflow = str(pilot["workflow"])
    if pilot.get("variant"):
        cfg.pilot_variant = str(pilot["variant"])
    if "run_cosim" in pilot:
        raw = pilot["run_cosim"]
        cfg.pilot_run_cosim = raw if isinstance(raw, bool) else str(raw).strip().lower() in (
            "1",
            "true",
            "yes",
            "on",
        )
    if pilot.get("model"):
        cfg.model = str(pilot["model"])
    if pilot.get("turns"):
        cfg.turns = int(pilot["turns"])
    if data.get("long_vitis_benches"):
        cfg.long_vitis_benches = [str(b) for b in data["long_vitis_benches"]]
    env_model = os.getenv("C2HLS_MODEL", "").strip()
    if env_model:
        cfg.model = env_model
    if os.getenv("C2HLS_TURNS", "").strip():
        cfg.turns = int(os.getenv("C2HLS_TURNS", "4"))
    env_prefix = os.getenv("FIR_BATCH_JOB_PREFIX", "").strip()
    if env_prefix:
        cfg.job_prefix = env_prefix
    if cfg.compute_nodes_match_benches and cfg.pilot_benches:
        cfg.compute_nodes = len(cfg.pilot_benches)
    return cfg


def campaign_paths(campaign_root: Path) -> dict[str, Path]:
    root = campaign_root.resolve()
    flow = root / "flow"
    return {
        "root": root,
        "campaign": root / "campaign.json",
        "queue_db": root / "queue.db",
        "endpoint": root / "llm_endpoint.json",
        "events": flow / "events.jsonl",
        "status": flow / "status.json",
        "node_map": flow / "node_map.json",
        "coordinator_pid": root / "coordinator.pid",
        "watch_log": flow / "watch.log",
        "matrix": root / "matrix.json",
        "manifest": root / "manifest.json",
    }


def gpu_policy_from_campaign(campaign: dict[str, Any], cfg: FirBatchParallelConfig | None = None) -> str:
    if cfg is None:
        cfg = load_config()
    stored = campaign.get("config") or {}
    return str(stored.get("gpu_policy") or cfg.gpu_policy or "batch_park")


def gpu_parking_enabled(campaign: dict[str, Any], cfg: FirBatchParallelConfig | None = None) -> bool:
    return gpu_policy_from_campaign(campaign, cfg) != "always_on"


def campaign_job_prefix(campaign: dict[str, Any], *, default: str = "firbp") -> str:
    stored = (campaign.get("config") or {}).get("job_prefix")
    if stored:
        return str(stored)
    return str(campaign.get("job_prefix") or default)


def init_campaign_json(
    campaign_root: Path,
    cfg: FirBatchParallelConfig,
    *,
    stamp: str,
) -> None:
    paths = campaign_paths(campaign_root)
    paths["root"].mkdir(parents=True, exist_ok=True)
    paths["events"].parent.mkdir(parents=True, exist_ok=True)
    doc = {
        "stamp": stamp,
        "site": "fir",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "campaign_status": "running",
        "compute_state": "waiting_for_gpu",
        "gpu_mode": "up",
        "gpu_job_id": None,
        "gpu_borrowed": False,
        "parked_flash_since": None,
        "park_pending_at": None,
        "park_pending_reason": None,
        "compute_jobs": [],
        "config": cfg.to_dict(),
    }
    paths["campaign"].write_text(json.dumps(doc, indent=2) + "\n", encoding="utf-8")


def load_campaign(campaign_root: Path) -> dict[str, Any]:
    path = campaign_paths(campaign_root)["campaign"]
    return json.loads(path.read_text(encoding="utf-8"))


def save_campaign(campaign_root: Path, doc: dict[str, Any]) -> None:
    campaign_paths(campaign_root)["campaign"].write_text(
        json.dumps(doc, indent=2) + "\n",
        encoding="utf-8",
    )


def benches_for_campaign(campaign: dict[str, Any], cfg: FirBatchParallelConfig | None = None) -> list[str]:
    if cfg is None:
        cfg = load_config()
    pilot = (campaign.get("config") or {}).get("pilot") or {}
    benches = [str(b) for b in (pilot.get("benches") or cfg.pilot_benches)]
    return benches
