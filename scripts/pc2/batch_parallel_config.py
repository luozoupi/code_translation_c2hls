"""Campaign configuration for batch_parallel harness."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    import tomllib
except ModuleNotFoundError:  # Python < 3.11
    tomllib = None  # type: ignore[assignment]

REPO = Path(__file__).resolve().parents[2]
SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_PILOT_TOML = SCRIPT_DIR / "batch_parallel_pilot.toml"
DEFAULT_PILOT_JSON = SCRIPT_DIR / "batch_parallel_pilot.json"

# Short-first bench order (pilot); profile CSV can override later.
SHORT_FIRST_BENCHES = [
    "jacobi-1d",
    "gesummv",
    "correlation",
    "fdtd-2d",
    "atax-medium",
    "bicg",
    "mvt-medium",
]


@dataclass
class BatchParallelConfig:
    synth_nodes_per_variant: int = 2
    synth_workers_per_node: int = 4
    cosim_nodes_per_variant: int = 4
    cosim_workers_per_node: int = 2
    worker_cpus: int = 8
    worker_mem_gb: int = 32
    gpu_batch_threshold: int = 5
    gpu_batch_flush_s: int = 3600
    gpu_policy: str = "batch_park"  # batch_park | always_on
    # If true, synth-role workers also claim cosim jobs (cosim_nodes_per_variant=0).
    combined_hls_nodes: bool = False
    park_threshold_s: float = 7200.0
    long_cosim_park_s: float = 3600.0
    park_grace_s: float = 1800.0
    long_cosim_benches: list[str] = field(default_factory=list)
    long_cosim_profile_min_s: float = 3600.0
    cosim_profile_csv: str = ""
    cosim_timeout_s: int = 43200
    # Requeue claimed jobs whose worker heartbeat is older than this (seconds).
    stale_claim_s: float = 1800.0
    # Hard ceiling on repair attempt index (0-based). Cosim failures also increment.
    max_repair_attempt: int = 7
    bench_order: str = "short_first"
    bench_seeding: str = "short_first_waves"
    max_inflight_benches: int = 3
    poll_sec: float = 2.0
    coordinator_poll_sec: float = 15.0
    job_prefix: str = "bpcplx"
    pilot_variant: str = "aav_n"
    pilot_benches: list[str] = field(default_factory=lambda: list(SHORT_FIRST_BENCHES[:4]))
    pilot_workflow: str = "flash"
    pilot_corpus: str = ""
    pilot_failure_policy: str = "ignore"
    model: str = "devstral2"
    turns: int = 4

    @property
    def synth_slots_per_variant(self) -> int:
        return self.synth_nodes_per_variant * self.synth_workers_per_node

    @property
    def cosim_slots_per_variant(self) -> int:
        return self.cosim_nodes_per_variant * self.cosim_workers_per_node

    def node_slurm_cpus(self, role: str) -> int:
        workers = (
            self.synth_workers_per_node if role == "synth" else self.cosim_workers_per_node
        )
        return workers * self.worker_cpus

    def node_slurm_mem_gb(self, role: str) -> int:
        workers = (
            self.synth_workers_per_node if role == "synth" else self.cosim_workers_per_node
        )
        return workers * self.worker_mem_gb

    def sort_benches(self, benches: list[str]) -> list[str]:
        if self.bench_order == "listed":
            order = {b: i for i, b in enumerate(self.pilot_benches)}
            return sorted(benches, key=lambda b: order.get(b, 9999))
        if self.bench_order != "short_first":
            return sorted(benches)
        order = {b: i for i, b in enumerate(self.pilot_benches)}
        if order:
            return sorted(benches, key=lambda b: order.get(b, 9999))
        fallback = {b: i for i, b in enumerate(SHORT_FIRST_BENCHES)}
        return sorted(benches, key=lambda b: fallback.get(b, fallback.get(b.replace("hlsfactory_", ""), 9999)))

    def to_dict(self) -> dict[str, Any]:
        return {
            "synth_nodes_per_variant": self.synth_nodes_per_variant,
            "synth_workers_per_node": self.synth_workers_per_node,
            "cosim_nodes_per_variant": self.cosim_nodes_per_variant,
            "cosim_workers_per_node": self.cosim_workers_per_node,
            "worker_cpus": self.worker_cpus,
            "worker_mem_gb": self.worker_mem_gb,
            "gpu_batch_threshold": self.gpu_batch_threshold,
            "gpu_batch_flush_s": self.gpu_batch_flush_s,
            "gpu_policy": self.gpu_policy,
            "combined_hls_nodes": self.combined_hls_nodes,
            "park_threshold_s": self.park_threshold_s,
            "long_cosim_park_s": self.long_cosim_park_s,
            "park_grace_s": self.park_grace_s,
            "long_cosim_benches": list(self.long_cosim_benches),
            "long_cosim_profile_min_s": self.long_cosim_profile_min_s,
            "cosim_profile_csv": self.cosim_profile_csv,
            "cosim_timeout_s": self.cosim_timeout_s,
            "stale_claim_s": self.stale_claim_s,
            "max_repair_attempt": self.max_repair_attempt,
            "bench_order": self.bench_order,
            "bench_seeding": self.bench_seeding,
            "max_inflight_benches": self.max_inflight_benches,
            "poll_sec": self.poll_sec,
            "coordinator_poll_sec": self.coordinator_poll_sec,
            "job_prefix": self.job_prefix,
            "pilot": {
                "variant": self.pilot_variant,
                "benches": self.pilot_benches,
                "workflow": self.pilot_workflow,
                "corpus": self.pilot_corpus,
                "failure_policy": self.pilot_failure_policy,
                "model": self.model,
                "turns": self.turns,
            },
        }


def benches_for_config(cfg: BatchParallelConfig) -> list[str]:
    raw = os.getenv("C2HLS_AUTOSA_DSE_FLASH_BENCHES", "").strip()
    if raw:
        return [item.strip() for item in raw.split(",") if item.strip()]
    raw = os.getenv("C2HLS_AUTOSA_FLASH_BENCHES", "").strip()
    if raw:
        return [item.strip() for item in raw.split(",") if item.strip()]
    raw = os.getenv("C2HLS_TIER_B_GOLD_BENCHES", "").strip()
    if raw:
        return [item.strip() for item in raw.split(",") if item.strip()]
    raw = os.getenv("C2HLS_TIER_A_FLASH_BENCHES", "").strip()
    if raw:
        return [item.strip() for item in raw.split(",") if item.strip()]
    return list(cfg.pilot_benches)


def seed_kwargs_for_workflow(workflow: str) -> dict[str, str]:
    if workflow in (
        "chathls_multistep",
        "tier_a_multistep",
        "tier_b_multistep",
    ):
        # Default queue seed is codegen/phase_b/translate — correct for multistep.
        return {}
    if workflow in (
        "tier_a_flash",
        "autosa_flash",
        "autosa_dse_flash",
        "tier_b_gold",
        "tier_b_flash",
        "chathls_flash",
        "c2hlsc_flash",
    ):
        return {
            "initial_kind": "synth",
            "initial_phase": "reference",
            "initial_stage": "gold_gate",
        }
    return {}


def _load_toml(path: Path) -> dict[str, Any]:
    if not path.is_file() or tomllib is None:
        return {}
    return tomllib.loads(path.read_text(encoding="utf-8"))


def load_config(toml_path: Path | None = None) -> BatchParallelConfig:
    config_path = os.getenv("BATCH_PARALLEL_CONFIG", "").strip()
    if config_path:
        json_path = Path(config_path)
    else:
        json_path = DEFAULT_PILOT_JSON
    if json_path.is_file():
        data = json.loads(json_path.read_text(encoding="utf-8"))
    else:
        path = toml_path or DEFAULT_PILOT_TOML
        data = _load_toml(path)
    pilot = data.pop("pilot", {}) or {}
    cfg = BatchParallelConfig()
    for key, value in data.items():
        if hasattr(cfg, key):
            setattr(cfg, key, value)
    if pilot.get("variant"):
        cfg.pilot_variant = str(pilot["variant"])
    if pilot.get("benches"):
        cfg.pilot_benches = [str(b) for b in pilot["benches"]]
    if pilot.get("workflow"):
        cfg.pilot_workflow = str(pilot["workflow"])
    if pilot.get("corpus"):
        cfg.pilot_corpus = str(pilot["corpus"])
    if pilot.get("failure_policy"):
        cfg.pilot_failure_policy = str(pilot["failure_policy"])
    if pilot.get("model"):
        cfg.model = str(pilot["model"])
    if pilot.get("turns"):
        cfg.turns = int(pilot["turns"])
    ext_model = os.getenv("BATCH_PARALLEL_EXTERNAL_MODEL", "").strip()
    if ext_model:
        cfg.model = ext_model
    else:
        env_model = os.getenv("C2HLS_MODEL", "").strip()
        if env_model:
            cfg.model = env_model
    if os.getenv("C2HLS_TURNS", "").strip():
        cfg.turns = int(os.getenv("C2HLS_TURNS", "4"))
    if os.getenv("C2HLS_MAX_REPAIR_ATTEMPT", "").strip():
        cfg.max_repair_attempt = int(os.getenv("C2HLS_MAX_REPAIR_ATTEMPT", "7"))
    if os.getenv("C2HLS_STALE_CLAIM_S", "").strip():
        cfg.stale_claim_s = float(os.getenv("C2HLS_STALE_CLAIM_S", "1800"))
    env_prefix = os.getenv("PC2_BATCH_JOB_PREFIX", "").strip()
    if env_prefix:
        cfg.job_prefix = env_prefix
    return cfg


def campaign_job_prefix(campaign: dict[str, Any], *, default: str = "bpcplx") -> str:
    """Slurm job-name prefix stored on the campaign (e.g. bpfcosim, bpcplx)."""
    top = str(campaign.get("job_prefix") or "").strip()
    if top:
        return top
    cfg = campaign.get("config") or {}
    nested = str(cfg.get("job_prefix") or "").strip()
    if nested:
        return nested
    return default


def campaign_benches(campaign: dict[str, Any], cfg: BatchParallelConfig | None = None) -> list[str]:
    stored = campaign.get("config") or {}
    pilot = stored.get("pilot") or {}
    benches = [str(b) for b in (pilot.get("benches") or [])]
    if cfg is None:
        cfg = BatchParallelConfig()
        for key, value in stored.items():
            if key != "pilot" and hasattr(cfg, key):
                setattr(cfg, key, value)
    if pilot.get("variant"):
        cfg.pilot_variant = str(pilot["variant"])
    if benches:
        cfg.pilot_benches = benches
    elif not cfg.pilot_benches:
        cfg = load_config()
    return cfg.sort_benches(list(cfg.pilot_benches))


def gpu_policy_from_campaign(campaign: dict[str, Any], cfg: BatchParallelConfig | None = None) -> str:
    if cfg is None:
        cfg = load_config()
    stored = campaign.get("config") or {}
    return str(stored.get("gpu_policy") or cfg.gpu_policy or "batch_park")


def gpu_parking_enabled(campaign: dict[str, Any], cfg: BatchParallelConfig | None = None) -> bool:
    if campaign.get("external_llm"):
        return False
    return gpu_policy_from_campaign(campaign, cfg) != "always_on"


def campaign_artifact_prefix() -> str:
    return os.getenv("BATCH_PARALLEL_ARTIFACT_PREFIX", "batch_parallel").strip() or "batch_parallel"


def campaign_dir_name(stamp: str) -> str:
    return f"{campaign_artifact_prefix()}_{stamp}"


def default_campaign_root(stamp: str | None = None) -> Path:
    suffix = stamp or datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    return REPO / "artifacts" / "pc2" / campaign_dir_name(suffix)


def campaign_paths(campaign_root: Path) -> dict[str, Path]:
    root = campaign_root.resolve()
    flow = root / "flow"
    return {
        "root": root,
        "campaign_json": root / "campaign.json",
        "queue_db": root / "queue.db",
        "endpoint": root / "llm_endpoint.json",
        "session_json": root / "session.json",
        "coordinator_pid": root / "coordinator.pid",
        "coordinator_log": flow / "coordinator.log",
        "events": flow / "events.jsonl",
        "gpu_events": flow / "by_scope" / "gpu.jsonl",
        "status": flow / "snapshots" / "status.json",
        "node_map": flow / "snapshots" / "node_map.json",
        "reports": root / "reports",
        "complete_marker": root / "CAMPAIGN_COMPLETE",
    }


def load_campaign(campaign_root: Path) -> dict[str, Any]:
    path = campaign_root / "campaign.json"
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def save_campaign(campaign_root: Path, data: dict[str, Any]) -> None:
    path = campaign_root / "campaign.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def init_campaign_json(
    campaign_root: Path,
    cfg: BatchParallelConfig,
    *,
    stamp: str,
    active_variants: list[str] | None = None,
) -> dict[str, Any]:
    variants = active_variants or [cfg.pilot_variant]
    prefix = campaign_artifact_prefix()
    job_prefix = os.getenv("PC2_BATCH_JOB_PREFIX", "").strip() or cfg.job_prefix or "bpcplx"
    doc = {
        "campaign_status": "running",
        "stamp": stamp,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "completed_at": None,
        "job_prefix": job_prefix,
        "gpu_mode": "up",
        "gpu_job_id": None,
        "gpu_session_id": f"{prefix}_{stamp}",
        "coordinator_pid": None,
        "parked_codegen_since": None,
        "park_pending_at": None,
        "park_pending_reason": None,
        "config": cfg.to_dict(),
        "active_variants": variants,
        "compute_jobs": [],
        "compute_state": "waiting_for_gpu",
        "no_gpu": False,
    }
    save_campaign(campaign_root, doc)
    return doc
