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
    gpu_renew_before_s: int = 600
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
            "gpu_renew_before_s": self.gpu_renew_before_s,
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
        "autosa_multistep",
    ):
        # Default queue seed is codegen/phase_b/translate — correct for multistep.
        return {}
    if workflow in (
        "tier_a_flash",
        "autosa_flash",
        "autosa_gold",
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
    if workflow == "zero_shot_direct":
        return {
            "initial_kind": "codegen",
            "initial_phase": "flash",
            "initial_stage": "optimize",
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
    raw_enf = os.getenv("C2HLS_ENFORCEMENT", "").strip().lower()
    if raw_enf in {"1", "true", "yes", "on"}:
        doc["enforcement"] = True
        try:
            doc["enforcement_rounds"] = max(
                1, int(os.getenv("C2HLS_ENFORCEMENT_ROUNDS") or "20")
            )
        except ValueError:
            doc["enforcement_rounds"] = 20
    raw_flow = os.getenv("C2HLS_AUTOSA_FLOW", "").strip().lower()
    if raw_flow in {"1", "true", "yes", "on"}:
        doc["autosa_flow"] = True
        doc["enforcement"] = False

        def _flag(name: str, *, default: bool) -> bool:
            raw = os.getenv(name, "").strip().lower()
            if not raw:
                return default
            return raw in {"1", "true", "yes", "on"}

        doc["post_flash_dse"] = _flag("C2HLS_POST_FLASH_DSE", default=True)
        doc["dse_chain_flash"] = _flag("C2HLS_DSE_CHAIN_FLASH", default=True)
        doc["dse_v2"] = _flag("C2HLS_DSE_V2", default=False)
        doc["dse_v2_chain_flash"] = _flag("C2HLS_DSE_V2_CHAIN_FLASH", default=False)
        doc["post_flash_stream"] = _flag("C2HLS_POST_FLASH_STREAM", default=True)
        doc["stream_chain_flash"] = _flag("C2HLS_STREAM_CHAIN_FLASH", default=True)
        if doc["dse_v2"]:
            # DSE 2.0 replaces v1 and stops before stream.
            doc["post_flash_dse"] = False
            doc["dse_chain_flash"] = False
            doc["post_flash_stream"] = False
            doc["stream_chain_flash"] = False
            doc["dse_v2_chain_flash"] = True
        if _flag("C2HLS_POST_FLASH_NO_SKILLS", default=False):
            doc["post_flash_no_skills"] = True
        dse_v2_grid = os.getenv("C2HLS_DSE_V2_GRID", "").strip()
        if dse_v2_grid:
            doc["dse_v2_grid"] = dse_v2_grid
        if _flag("C2HLS_SKIP_PHASE_B", default=False):
            doc["skip_phase_b"] = True
        if _flag("C2HLS_ONE_SHOT", default=False):
            doc["one_shot"] = True
            doc["skip_phase_b"] = True
        if _flag("C2HLS_REFERENCE_ONLY", default=False):
            doc["reference_only"] = True
    if os.getenv("C2HLS_REFERENCE_ONLY", "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }:
        doc["reference_only"] = True
    dse_skills = os.getenv("C2HLS_DSE_SKILL_ENTRIES_JSON", "").strip()
    if dse_skills:
        doc["dse_skill_entries_json"] = dse_skills
    stream_skills = os.getenv("C2HLS_STREAM_SKILL_ENTRIES_JSON", "").strip()
    if stream_skills:
        doc["stream_skill_entries_json"] = stream_skills
    raw_dse_min = os.getenv("C2HLS_DSE_MIN_DSP", "").strip()
    if raw_dse_min.lstrip("-").isdigit():
        doc["dse_min_dsp"] = int(raw_dse_min)
    flavor = os.getenv("C2HLS_MM_FLOW_FLAVOR", "").strip()
    if flavor:
        doc["mm_flow_flavor"] = flavor
    mode = os.getenv("C2HLS_FLASH_OPT_PROMPT_MODE", "").strip()
    if mode:
        doc["flash_opt_prompt_mode"] = mode
    raw_turns = os.getenv("C2HLS_TURNS", "").strip()
    if raw_turns:
        try:
            doc["turns"] = max(1, int(raw_turns))
        except ValueError:
            pass
    raw_to = os.getenv("C2HLS_SYNTH_TIMEOUT", "").strip()
    if raw_to:
        try:
            doc["synth_timeout"] = max(1, int(raw_to))
        except ValueError:
            pass
    if os.getenv("C2HLS_SKIP_FLASH", "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }:
        doc["skip_flash"] = True
        doc["skip_phase_b"] = True
        seed_dir = os.getenv("C2HLS_FLASH_SEED_DIR", "").strip()
        if seed_dir:
            doc["flash_seed_dir"] = seed_dir
    pe_recipe = os.getenv("C2HLS_PE_RECIPE", "").strip()
    if pe_recipe:
        doc["pe_recipe"] = pe_recipe
    raw_dsp = os.getenv("C2HLS_FLASH_MIN_DSP", "").strip()
    if raw_dsp.isdigit():
        doc["flash_min_dsp"] = int(raw_dsp)
    raw_max = os.getenv("C2HLS_FLASH_MAX_DSP", "").strip()
    if raw_max.isdigit():
        doc["flash_max_dsp"] = int(raw_max)
    if os.getenv("C2HLS_FLASH_DSP_REDO", "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }:
        doc["flash_dsp_redo"] = 1
    raw_fill = os.getenv("C2HLS_FLASH_DSP_FILL_PCT", "").strip()
    if raw_fill.isdigit():
        doc["flash_dsp_fill_pct"] = int(raw_fill)
    cands = os.getenv("C2HLS_CANDIDATES_PER_STEP", "").strip()
    if cands:
        doc["candidates_per_step"] = cands
    raw_row = os.getenv("C2HLS_FLASH_ROW_UF", "").strip()
    if raw_row.isdigit():
        doc["flash_row_uf"] = int(raw_row)
    raw_pe = os.getenv("C2HLS_FLASH_PE_BLK", "").strip()
    if raw_pe.isdigit():
        doc["flash_pe_blk"] = int(raw_pe)
    raw_kt = os.getenv("C2HLS_FLASH_K_TILE", "").strip()
    if raw_kt.isdigit():
        doc["flash_k_tile"] = int(raw_kt)
    raw_pp = os.getenv("C2HLS_FLASH_TILE_PP", "").strip().lower()
    if raw_pp in {"1", "true", "yes", "on"}:
        doc["flash_tile_pp"] = 1
    raw_onchip = os.getenv("C2HLS_FLASH_ONCHIP", "").strip().lower()
    if raw_onchip in {"1", "true", "yes", "on"}:
        doc["flash_onchip"] = 1
    raw_ot = os.getenv("C2HLS_FLASH_ONCHIP_TILE", "").strip().lower()
    if raw_ot in {"1", "true", "yes", "on"}:
        doc["flash_onchip_tile"] = 1
    skill_bin = os.getenv("C2HLS_FLASH_SKILL_BIN", "").strip().lower().replace("-", "_")
    if skill_bin:
        doc["flash_skill_bin"] = skill_bin
    pack = os.getenv("C2HLS_PACKAGED_SKILLS_JSON", "").strip()
    if pack:
        doc["packaged_skills_json"] = pack
    order = os.getenv("C2HLS_SKILL_PROMPT_ORDER_JSON", "").strip()
    if order:
        doc["skill_prompt_order_json"] = order
    if os.getenv("C2HLS_PACKAGED_SKILLS_ONLY", "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }:
        doc["packaged_skills_only"] = 1
    if os.getenv("C2HLS_FLASH_ONLY", "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }:
        doc["flash_only"] = True
        doc["post_flash_dse"] = False
        doc["dse_chain_flash"] = False
        doc["post_flash_stream"] = False
        doc["stream_chain_flash"] = False
        doc["dse_v2"] = False
        doc["dse_v2_chain_flash"] = False
    for env, key in (
        ("C2HLS_FLASH_MAX_TOKENS", "flash_max_tokens"),
        ("C2HLS_LLM_MAX_TOKENS", "llm_max_tokens"),
        ("C2HLS_CPP_CONTINUATIONS", "cpp_continuations"),
    ):
        raw = os.getenv(env, "").strip()
        if raw.isdigit():
            doc[key] = int(raw)
    thinking = (os.getenv("C2HLS_THINKING") or "").strip().lower()
    if thinking:
        doc["thinking"] = thinking
    save_campaign(campaign_root, doc)
    return doc


def apply_campaign_synth_timeout(campaign: dict[str, Any] | None = None) -> int | None:
    """Force ``C2HLS_SYNTH_TIMEOUT`` from campaign.json after meta defaults.

    AutoSA kernel meta sets ``synth_timeout_s: 14400`` and would otherwise
    overwrite a launcher timeout (e.g. 1h for mm enforcement).
    """
    raw = None
    if isinstance(campaign, dict) and campaign.get("synth_timeout") is not None:
        raw = campaign.get("synth_timeout")
    else:
        root = os.getenv("BATCH_PARALLEL_CAMPAIGN_ROOT", "").strip()
        if root:
            path = Path(root) / "campaign.json"
            if path.is_file():
                try:
                    doc = json.loads(path.read_text(encoding="utf-8"))
                except (OSError, json.JSONDecodeError):
                    doc = {}
                if isinstance(doc, dict):
                    raw = doc.get("synth_timeout")
    if raw is None:
        return None
    try:
        val = max(1, int(raw))
    except (TypeError, ValueError):
        return None
    os.environ["C2HLS_SYNTH_TIMEOUT"] = str(val)
    return val


_AUTOSA_FLOW_ENV_KEYS = (
    ("autosa_flow", "C2HLS_AUTOSA_FLOW"),
    ("post_flash_dse", "C2HLS_POST_FLASH_DSE"),
    ("dse_chain_flash", "C2HLS_DSE_CHAIN_FLASH"),
    ("dse_v2", "C2HLS_DSE_V2"),
    ("dse_v2_chain_flash", "C2HLS_DSE_V2_CHAIN_FLASH"),
    ("post_flash_stream", "C2HLS_POST_FLASH_STREAM"),
    ("stream_chain_flash", "C2HLS_STREAM_CHAIN_FLASH"),
    ("post_flash_no_skills", "C2HLS_POST_FLASH_NO_SKILLS"),
    ("skip_phase_b", "C2HLS_SKIP_PHASE_B"),
    ("one_shot", "C2HLS_ONE_SHOT"),
    ("reference_only", "C2HLS_REFERENCE_ONLY"),
)


def apply_autosa_flow_from_campaign(campaign: dict[str, Any] | None = None) -> None:
    """Honor campaign.json AutoSA-flow flags even when Slurm dropped the env."""
    if not isinstance(campaign, dict):
        return
    for key, env in _AUTOSA_FLOW_ENV_KEYS:
        if key not in campaign or campaign[key] is None:
            continue
        val = campaign[key]
        if isinstance(val, bool):
            os.environ[env] = "1" if val else "0"
        else:
            raw = str(val).strip().lower()
            os.environ[env] = "1" if raw in {"1", "true", "yes", "on"} else "0"
    if os.environ.get("C2HLS_AUTOSA_FLOW") == "1":
        os.environ["C2HLS_ENFORCEMENT"] = "0"
    if os.environ.get("C2HLS_DSE_V2") == "1":
        os.environ["C2HLS_POST_FLASH_STREAM"] = "0"
        os.environ["C2HLS_STREAM_CHAIN_FLASH"] = "0"
        os.environ["C2HLS_POST_FLASH_DSE"] = "0"
        os.environ["C2HLS_DSE_CHAIN_FLASH"] = "0"
        os.environ.setdefault("C2HLS_DSE_V2_CHAIN_FLASH", "1")
    if os.environ.get("C2HLS_ONE_SHOT") == "1":
        os.environ["C2HLS_FLASH_ONLY"] = "1"
        os.environ["C2HLS_POST_FLASH_DSE"] = "0"
        os.environ["C2HLS_DSE_CHAIN_FLASH"] = "0"
        os.environ["C2HLS_POST_FLASH_STREAM"] = "0"
        os.environ["C2HLS_STREAM_CHAIN_FLASH"] = "0"
        os.environ["C2HLS_DSE_V2"] = "0"
        os.environ["C2HLS_DSE_V2_CHAIN_FLASH"] = "0"
    grid = campaign.get("dse_v2_grid")
    if isinstance(grid, str) and grid.strip():
        os.environ["C2HLS_DSE_V2_GRID"] = grid.strip()
    pe_recipe = campaign.get("pe_recipe")
    if isinstance(pe_recipe, str) and pe_recipe.strip():
        os.environ["C2HLS_PE_RECIPE"] = pe_recipe.strip()
    flash_min = campaign.get("flash_min_dsp")
    if flash_min is not None and str(flash_min).strip().lstrip("-").isdigit():
        os.environ["C2HLS_FLASH_MIN_DSP"] = str(int(flash_min))
    flash_max = campaign.get("flash_max_dsp")
    if flash_max is not None and str(flash_max).strip().lstrip("-").isdigit():
        os.environ["C2HLS_FLASH_MAX_DSP"] = str(int(flash_max))
    flash_redo = campaign.get("flash_dsp_redo")
    if flash_redo is not None:
        raw_redo = str(flash_redo).strip().lower()
        if raw_redo in {"1", "true", "yes", "on"}:
            os.environ["C2HLS_FLASH_DSP_REDO"] = "1"
    flash_fill = campaign.get("flash_dsp_fill_pct")
    if flash_fill is not None and str(flash_fill).strip().lstrip("-").isdigit():
        os.environ["C2HLS_FLASH_DSP_FILL_PCT"] = str(int(flash_fill))
    cands = campaign.get("candidates_per_step")
    if cands is not None and str(cands).strip():
        os.environ["C2HLS_CANDIDATES_PER_STEP"] = str(cands).strip()
    flash_row = campaign.get("flash_row_uf")
    if flash_row is not None and str(flash_row).strip().lstrip("-").isdigit():
        os.environ["C2HLS_FLASH_ROW_UF"] = str(int(flash_row))
    flash_pe = campaign.get("flash_pe_blk")
    if flash_pe is not None and str(flash_pe).strip().lstrip("-").isdigit():
        os.environ["C2HLS_FLASH_PE_BLK"] = str(int(flash_pe))
    flash_kt = campaign.get("flash_k_tile")
    if flash_kt is not None and str(flash_kt).strip().lstrip("-").isdigit():
        os.environ["C2HLS_FLASH_K_TILE"] = str(int(flash_kt))
    flash_pp = campaign.get("flash_tile_pp")
    if flash_pp is not None:
        raw_pp = str(flash_pp).strip().lower()
        if raw_pp in {"1", "true", "yes", "on"}:
            os.environ["C2HLS_FLASH_TILE_PP"] = "1"
    flash_onchip = campaign.get("flash_onchip")
    if flash_onchip is not None:
        raw_oc = str(flash_onchip).strip().lower()
        if raw_oc in {"1", "true", "yes", "on"}:
            os.environ["C2HLS_FLASH_ONCHIP"] = "1"
    flash_ot = campaign.get("flash_onchip_tile")
    if flash_ot is not None:
        raw_ot = str(flash_ot).strip().lower()
        if raw_ot in {"1", "true", "yes", "on"}:
            os.environ["C2HLS_FLASH_ONCHIP_TILE"] = "1"
    skill_bin = campaign.get("flash_skill_bin")
    if isinstance(skill_bin, str) and skill_bin.strip():
        os.environ["C2HLS_FLASH_SKILL_BIN"] = skill_bin.strip()
    pack = campaign.get("packaged_skills_json")
    if isinstance(pack, str) and pack.strip():
        os.environ["C2HLS_PACKAGED_SKILLS_JSON"] = pack.strip()
    order = campaign.get("skill_prompt_order_json")
    if isinstance(order, str) and order.strip():
        os.environ["C2HLS_SKILL_PROMPT_ORDER_JSON"] = order.strip()
    if campaign.get("packaged_skills_only"):
        os.environ["C2HLS_PACKAGED_SKILLS_ONLY"] = "1"
    mode = campaign.get("flash_opt_prompt_mode")
    if isinstance(mode, str) and mode.strip():
        os.environ["C2HLS_FLASH_OPT_PROMPT_MODE"] = mode.strip()
    turns = campaign.get("turns")
    if turns is not None and str(turns).strip():
        try:
            os.environ["C2HLS_TURNS"] = str(int(turns))
        except (TypeError, ValueError):
            pass
    flavor = campaign.get("mm_flow_flavor")
    if isinstance(flavor, str) and flavor.strip():
        os.environ["C2HLS_MM_FLOW_FLAVOR"] = flavor.strip()
    dse_pack = campaign.get("dse_skill_entries_json")
    if isinstance(dse_pack, str) and dse_pack.strip():
        os.environ["C2HLS_DSE_SKILL_ENTRIES_JSON"] = dse_pack.strip()
    stream_pack = campaign.get("stream_skill_entries_json")
    if isinstance(stream_pack, str) and stream_pack.strip():
        os.environ["C2HLS_STREAM_SKILL_ENTRIES_JSON"] = stream_pack.strip()
    dse_min = campaign.get("dse_min_dsp")
    if dse_min is not None and str(dse_min).strip().lstrip("-").isdigit():
        os.environ["C2HLS_DSE_MIN_DSP"] = str(int(dse_min))
    for key, env in (
        ("flash_max_tokens", "C2HLS_FLASH_MAX_TOKENS"),
        ("llm_max_tokens", "C2HLS_LLM_MAX_TOKENS"),
        ("cpp_continuations", "C2HLS_CPP_CONTINUATIONS"),
    ):
        val = campaign.get(key)
        if val is None:
            continue
        raw = str(val).strip()
        if raw.isdigit():
            os.environ[env] = raw
    thinking = campaign.get("thinking")
    if isinstance(thinking, str) and thinking.strip():
        os.environ["C2HLS_THINKING"] = thinking.strip()

