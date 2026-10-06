#!/usr/bin/env python3
"""autosa_mm variant sweep: manifest, DAG submit, unique prefixes.

43 base configs + 43 mem siblings × N replicates. Later stages seed from
completed parents so flash LLM+csynth is not re-run. Never writes frozen
campaign trees.

Usage:
  python scripts/pc2/autosa_mm_variant_sweep.py --dry-run
  python scripts/pc2/autosa_mm_variant_sweep.py --wave flash,oneshot --submit \\
      --endpoint-url http://login5:18092/v1
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

REPO = Path(__file__).resolve().parents[2]
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(SCRIPT_DIR))

DEFAULT_ENDPOINT = "http://login5:18092/v1"
DEFAULT_MODEL = "deepseek-v4-flash"
DEFAULT_MAX_INFLIGHT = 3
DEFAULT_REPS = 10
CLOCK_NS = "3.33"
PART = "xcu280-fsvh2892-2L-e"
FLASH_FLOOR_DSP = 300

SKILLS = ("noskills", "aav_n_90", "aav_n_gf")
SKILL_CODE = {"noskills": "n", "aav_n_90": "9", "aav_n_gf": "g"}
SKILL_FLAVOR = {"noskills": "noskills", "aav_n_90": "aav_n_90", "aav_n_gf": "skills"}
FLOORS = (0, 1)

WAVES = ("flash", "dse", "stream", "enf", "oneshot", "mem")
WAVE_FAMILIES = {
    "flash": frozenset({"flash"}),
    "oneshot": frozenset({"oneshot"}),
    "dse": frozenset({"dse_v1", "dse_v2"}),
    "stream": frozenset({"stream"}),
    "enf": frozenset({"enf"}),
    "mem": frozenset({"mem"}),
}
WAVE_RANK = {
    "flash": 0,
    "oneshot": 0,
    "dse_v1": 1,
    "dse_v2": 1,
    "stream": 2,
    "enf": 3,
    "mem": 3.5,
}

CSV_FIELDS = (
    "family",
    "skill",
    "floor",
    "dse",
    "stream",
    "enf",
    "mem",
    "rep",
    "stage",
    "latency_cycles",
    "latency_cycles_worst",
    "dsp",
    "bram",
    "lut",
    "ff",
    "csim",
    "campaign_root",
)

FROZEN_MARKERS = (
    "20260830_mmflow",
    "20260830_mm32x8",
    "20260909_",
    "20260910_234214",
    "20260911_075217",
    "20260915_dse13160_pp",
    "flash16k_v41_",
    "20260916_redo",
    "20260916_flash_",
    "20260917_flash_",
    "_repro2",
    "_repro3",
    "_repro4",
    "autosa_mm_variant_sweep_20260918",
)


def unique_config_count() -> int:
    """43 base families + 43 mem siblings."""
    return (3 + 3 + 6 + 6 + 12 + 12 + 1) * 2


def refuse_frozen(path: Path | str) -> None:
    text = str(path)
    for marker in FROZEN_MARKERS:
        if marker in text:
            raise ValueError(f"refusing to write frozen tree matching {marker!r}: {text}")


def sweep_root_for(date: str, *, repo: Path = REPO) -> Path:
    root = repo / "artifacts" / "pc2" / f"autosa_mm_variant_sweep_{date}"
    refuse_frozen(root)
    return root


def campaign_root_for(sweep_root: Path, cell_id: str) -> Path:
    path = sweep_root / "cells" / f"{cell_id}_camp"
    refuse_frozen(path)
    return path


def artifact_prefix_for_cell(cell: dict[str, Any], *, repo: Path = REPO) -> str:
    camp = Path(cell["campaign_root"]).resolve()
    base = (Path(repo) / "artifacts" / "pc2").resolve()
    try:
        rel = camp.relative_to(base)
    except ValueError:
        rel = Path(f"autosa_mm_variant_sweep_{cell.get('stamp')}") / "cells" / f"{cell['cell_id']}_camp"
    text = str(rel)
    if text.endswith("_camp"):
        text = text[: -len("_camp")]
    return text


def job_prefix_for(
    *,
    family: str,
    skill: str,
    floor: int,
    dse: str,
    stream: int,
    enf: int,
    rep: int,
    parent_prefix: str = "",
    prefix_tag: str = "",
) -> str:
    """Compact unique Slurm prefix (watch/drain/coord collide if two cells share one)."""
    if family == "mem":
        base = parent_prefix.strip()
        if not base:
            raise ValueError("mem job prefix requires parent_prefix")
        if base.startswith("v"):
            return "vm" + base[1:]
        return "vm" + base
    if family == "oneshot":
        prefix = f"vsos{rep:02d}"
    else:
        code = SKILL_CODE[skill]
        fl = "1" if int(floor) else "0"
        rr = f"{int(rep):02d}"
        if family == "flash":
            prefix = f"vsf{code}{fl}{rr}"
        elif family == "dse_v1":
            prefix = f"vs1{code}{fl}{rr}"
        elif family == "dse_v2":
            prefix = f"vs2{code}{fl}{rr}"
        elif family == "stream" and dse == "v1":
            prefix = f"vt1{code}{fl}{rr}"
        elif family == "stream" and dse == "v2":
            prefix = f"vt2{code}{fl}{rr}"
        elif family == "enf" and dse == "v1":
            prefix = f"ve1{code}{fl}{rr}"
        elif family == "enf" and dse == "v2":
            prefix = f"ve2{code}{fl}{rr}"
        else:
            raise ValueError(f"cannot encode job prefix for family={family} dse={dse}")
    tag = str(prefix_tag or "").strip()
    return f"{tag}{prefix}" if tag else prefix


def _skill_tag(skill: str) -> str:
    return {"noskills": "ns", "aav_n_90": "90", "aav_n_gf": "gf"}[skill]


def _flash_cell_id(skill: str, floor: int, rep: int) -> str:
    return f"flash_{_skill_tag(skill)}_f{int(floor)}_r{int(rep):02d}"


def _dse_cell_id(version: str, skill: str, floor: int, rep: int) -> str:
    return f"dse{version}_{_skill_tag(skill)}_f{int(floor)}_r{int(rep):02d}"


def _follow_cell_id(kind: str, dse: str, skill: str, floor: int, rep: int) -> str:
    return f"{kind}_dse{dse[-1]}_{_skill_tag(skill)}_f{int(floor)}_r{int(rep):02d}"


def _base_cell(
    *,
    cell_id: str,
    family: str,
    skill: str,
    floor: int,
    dse: str,
    stream: int,
    enf: int,
    rep: int,
    stamp: str,
    parent_cell: str,
    sweep_root: Path,
    mem: int = 0,
    parent_prefix: str = "",
    prefix_tag: str = "",
) -> dict[str, Any]:
    prefix = job_prefix_for(
        family=family,
        skill=skill,
        floor=floor,
        dse=dse,
        stream=stream,
        enf=enf,
        rep=rep,
        parent_prefix=parent_prefix,
        prefix_tag=prefix_tag,
    )
    return {
        "cell_id": cell_id,
        "family": family,
        "skill": skill,
        "floor": int(floor),
        "dse": dse,
        "stream": int(stream),
        "enf": int(enf),
        "mem": int(mem),
        "rep": int(rep),
        "stamp": stamp,
        "parent_cell": parent_cell,
        "job_prefix": prefix,
        "campaign_root": str(campaign_root_for(sweep_root, cell_id)),
        "status": "pending",
    }


def build_manifest(
    *,
    reps: int = DEFAULT_REPS,
    date: str | None = None,
    repo: Path = REPO,
    sweep_root: Path | None = None,
    prefix_tag: str = "",
) -> list[dict[str, Any]]:
    if reps < 1:
        raise ValueError("reps must be >= 1")
    day = date or datetime.now(timezone.utc).strftime("%Y%m%d")
    if sweep_root is None:
        sweep_root = sweep_root_for(day, repo=repo)
    else:
        sweep_root = Path(sweep_root)
        refuse_frozen(sweep_root)
    cells: list[dict[str, Any]] = []
    for rep in range(1, reps + 1):
        for skill in SKILLS:
            for floor in FLOORS:
                flash_id = _flash_cell_id(skill, floor, rep)
                cells.append(
                    _base_cell(
                        cell_id=flash_id,
                        family="flash",
                        skill=skill,
                        floor=floor,
                        dse="",
                        stream=0,
                        enf=0,
                        rep=rep,
                        stamp=day,
                        parent_cell="",
                        sweep_root=sweep_root,
                        prefix_tag=prefix_tag,
                    )
                )
                for ver, family in (("1", "dse_v1"), ("2", "dse_v2")):
                    dse_id = _dse_cell_id(ver, skill, floor, rep)
                    dse_tag = f"v{ver}"
                    cells.append(
                        _base_cell(
                            cell_id=dse_id,
                            family=family,
                            skill=skill,
                            floor=floor,
                            dse=dse_tag,
                            stream=0,
                            enf=0,
                            rep=rep,
                            stamp=day,
                            parent_cell=flash_id,
                            sweep_root=sweep_root,
                            prefix_tag=prefix_tag,
                        )
                    )
                    cells.append(
                        _base_cell(
                            cell_id=_follow_cell_id("stream", dse_tag, skill, floor, rep),
                            family="stream",
                            skill=skill,
                            floor=floor,
                            dse=dse_tag,
                            stream=1,
                            enf=0,
                            rep=rep,
                            stamp=day,
                            parent_cell=dse_id,
                            sweep_root=sweep_root,
                            prefix_tag=prefix_tag,
                        )
                    )
                    cells.append(
                        _base_cell(
                            cell_id=_follow_cell_id("enf", dse_tag, skill, floor, rep),
                            family="enf",
                            skill=skill,
                            floor=floor,
                            dse=dse_tag,
                            stream=0,
                            enf=1,
                            rep=rep,
                            stamp=day,
                            parent_cell=dse_id,
                            sweep_root=sweep_root,
                            prefix_tag=prefix_tag,
                        )
                    )
        cells.append(
            _base_cell(
                cell_id=f"oneshot_r{rep:02d}",
                family="oneshot",
                skill="one_shot",
                floor=0,
                dse="",
                stream=0,
                enf=0,
                rep=rep,
                stamp=day,
                parent_cell="",
                sweep_root=sweep_root,
                prefix_tag=prefix_tag,
            )
        )
    mem_cells: list[dict[str, Any]] = []
    for parent in cells:
        mem_cells.append(
            _base_cell(
                cell_id=f"mem_{parent['cell_id']}",
                family="mem",
                skill=parent["skill"],
                floor=int(parent["floor"]),
                dse=parent.get("dse") or "",
                stream=int(parent.get("stream") or 0),
                enf=int(parent.get("enf") or 0),
                mem=1,
                rep=int(parent["rep"]),
                stamp=day,
                parent_cell=parent["cell_id"],
                sweep_root=sweep_root,
                parent_prefix=parent["job_prefix"],
                prefix_tag="",
            )
        )
    cells.extend(mem_cells)
    return cells


def unique_configs(cells: Iterable[dict[str, Any]]) -> set[tuple[Any, ...]]:
    keys: set[tuple[Any, ...]] = set()
    for cell in cells:
        keys.add(
            (
                cell["family"],
                cell["skill"],
                int(cell["floor"]),
                cell.get("dse") or "",
                int(cell.get("stream") or 0),
                int(cell.get("enf") or 0),
                int(cell.get("mem") or 0),
            )
        )
    return keys


def write_cells(path: Path, cells: list[dict[str, Any]]) -> None:
    refuse_frozen(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for cell in cells:
            fh.write(json.dumps(cell, sort_keys=True) + "\n")


def load_cells(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        rows.append(json.loads(line))
    return rows


def flavor_for_skill(skill: str) -> str:
    if skill == "one_shot":
        return "one_shot"
    return SKILL_FLAVOR[skill]


def launch_env_for_cell(cell: dict[str, Any]) -> dict[str, str]:
    """Env the launcher should export. Empty string means unset."""
    family = cell["family"]
    env: dict[str, str] = {
        "C2HLS_MODEL": DEFAULT_MODEL,
        "C2HLS_PART": PART,
        "C2HLS_CLOCK_NS": CLOCK_NS,
        "C2HLS_FLASH_MAX_TOKENS": "65536",
        "C2HLS_LLM_MAX_TOKENS": "65536",
        "C2HLS_SWEEP_JOB_PREFIX": cell["job_prefix"],
        "PC2_BATCH_JOB_PREFIX": cell["job_prefix"],
        "C2HLS_FLASH_DSP_REDO": "",
        "C2HLS_FLASH_MAX_DSP": "",
    }
    if family in {"flash", "oneshot"}:
        env["C2HLS_FLASH_ONLY"] = "1"
        env["C2HLS_POST_FLASH_DSE"] = "0"
        env["C2HLS_POST_FLASH_STREAM"] = "0"
        env["C2HLS_DSE_V2"] = "0"
        env["C2HLS_DSE_CHAIN_FLASH"] = "0"
        env["C2HLS_STREAM_CHAIN_FLASH"] = "0"
    ready = os.getenv("C2HLS_AUTOSA_READY_ROOT", "").strip()
    if ready:
        env["C2HLS_AUTOSA_READY_ROOT"] = ready
    if family == "oneshot":
        env["C2HLS_ONE_SHOT"] = "1"
        env["C2HLS_SKIP_PHASE_B"] = "1"
        env["C2HLS_FLASH_OPT_PROMPT_MODE"] = "zero_shot"
        env["C2HLS_FLASH_MIN_DSP"] = ""
        env["C2HLS_POST_FLASH_NO_SKILLS"] = "1"
        env["C2HLS_PACKAGED_SKILLS_JSON"] = ""
    elif family == "flash":
        env["C2HLS_ONE_SHOT"] = "0"
        if int(cell.get("floor") or 0):
            env["C2HLS_FLASH_MIN_DSP"] = str(FLASH_FLOOR_DSP)
        else:
            env["C2HLS_FLASH_MIN_DSP"] = ""
    elif family == "dse_v1":
        env["C2HLS_POST_FLASH_DSE"] = "1"
        env["C2HLS_DSE_CHAIN_FLASH"] = "1"
        env["C2HLS_POST_FLASH_STREAM"] = "0"
        env["C2HLS_DSE_V2"] = "0"
        env["C2HLS_FLASH_ONLY"] = "0"
    elif family == "dse_v2":
        env["C2HLS_DSE_V2"] = "1"
        env["C2HLS_DSE_V2_CHAIN_FLASH"] = "1"
        env["C2HLS_POST_FLASH_DSE"] = "0"
        env["C2HLS_POST_FLASH_STREAM"] = "0"
        env["C2HLS_FLASH_ONLY"] = "0"
    elif family == "stream":
        env["C2HLS_POST_FLASH_STREAM"] = "1"
        env["C2HLS_STREAM_CHAIN_FLASH"] = "1"
        env["C2HLS_FLASH_ONLY"] = "0"
    elif family == "enf":
        env["C2HLS_SKIP_FLASH"] = "1"
        env["C2HLS_SKIP_PHASE_B"] = "1"
        env["C2HLS_PP_LOAD_B_IN_DF"] = "1"
        env["C2HLS_ENFORCEMENT"] = "1"
        env["C2HLS_FLASH_ONLY"] = "0"
        env["C2HLS_POST_FLASH_DSE"] = "0"
        env["C2HLS_POST_FLASH_STREAM"] = "0"
    if family in {"flash", "oneshot", "dse_v1", "dse_v2", "stream"}:
        # One billed completion must be allowed to finish. A 600s client
        # timeout aborts the proxy, then a retry starts a second billed call.
        env["C2HLS_LLM_TIMEOUT"] = "3600"
        env["C2HLS_LLM_EMPTY_RETRIES"] = "1"
        wall = os.getenv("C2HLS_COMPUTE_WALLTIME", "").strip()
        if wall:
            env["PC2_BATCH_PARALLEL_WALLTIME"] = wall
            env["PC2_FORCE_WALLTIME"] = wall
            env["C2HLS_SYNTH_TIMEOUT"] = "86400"
            env["C2HLS_CSIM_TIMEOUT"] = "86400"
    elif family == "mem":
        env["C2HLS_MEM_ITER"] = "1"
        env["C2HLS_MEM_ITER_ROUNDS"] = "50"
        env["C2HLS_MEM_ITER_PROMPT_TOKENS"] = "32768"
        env["C2HLS_MEM_ITER_MAX_TOKENS"] = "65536"
        env["C2HLS_MEM_ITER_CONTEXT_TOKENS"] = "131072"
        env["C2HLS_FLASH_ONLY"] = "0"
        env["C2HLS_POST_FLASH_DSE"] = "0"
        env["C2HLS_POST_FLASH_STREAM"] = "0"
        env["C2HLS_DSE_V2"] = "0"
        env["C2HLS_ONE_SHOT"] = "0"
        env["C2HLS_MEM_PARENT_FAMILY"] = mem_parent_family(cell)
        env["C2HLS_LLM_TIMEOUT"] = "3600"
        env["C2HLS_LLM_TIMEOUT_RETRIES"] = "8"
        env["C2HLS_LLM_RETRY_BACKOFF_S"] = "30"
    return env


def _load_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def csim_status(report: dict[str, Any]) -> str:
    csim = report.get("csim")
    if isinstance(csim, dict):
        if csim.get("passed") is True or csim.get("success") is True or csim.get("ok") is True:
            return "pass"
        status = str(csim.get("status") or "").strip().lower()
        if status in {"pass", "passed", "ok", "success"}:
            return "pass"
        if (
            csim.get("passed") is False
            or csim.get("success") is False
            or status in {"fail", "failed", "error"}
        ):
            return "fail"
    if isinstance(csim, bool):
        return "pass" if csim else "fail"
    if report.get("csim_passed") is True:
        return "pass"
    if report.get("csim_passed") is False:
        return "fail"
    if report.get("success") is False:
        return "fail"
    if report.get("success") is True:
        return "pass"
    return ""


def qor_from_report(report: Any) -> dict[str, Any] | None:
    if not isinstance(report, dict):
        return None
    payload = report
    nested = report.get("report")
    if isinstance(nested, dict) and (
        "latency_cycles" in nested or "dsp" in nested
    ):
        payload = nested
    lat = payload.get("latency_cycles")
    if lat is None:
        lat = payload.get("latency_min")
    if lat is None and payload.get("dsp") is None:
        return None
    worst = payload.get("latency_cycles_worst", lat)
    return {
        "latency_cycles": lat,
        "latency_cycles_worst": worst,
        "dsp": payload.get("dsp"),
        "bram": payload.get("bram"),
        "lut": payload.get("lut"),
        "ff": payload.get("ff"),
        "csim": csim_status(report) or csim_status(payload),
    }


def find_variant_cell(campaign_root: Path) -> Path | None:
    variants = campaign_root / "variants"
    search_roots = [variants] if variants.is_dir() else []
    if campaign_root.is_dir():
        search_roots.append(campaign_root)
    names = (
        "autosa_mm_flash_opt_report.json",
        "autosa_mm_selected_report.json",
        "autosa_mm_dse_report.json",
        "autosa_mm_stream_report.json",
        "autosa_mm_enforcement.json",
        "autosa_mm_mem_report.json",
    )
    for root in search_roots:
        for name in names:
            hits = sorted(root.rglob(name))
            if hits:
                return hits[0].parent
    return None


def stage_report_paths(cell_dir: Path) -> dict[str, Path]:
    mapping: dict[str, Path] = {}
    candidates = {
        "flash": (
            "autosa_mm_flash_opt_report.json",
            "autosa_mm_flash_seed_report.json",
        ),
        "dse": ("autosa_mm_dse_report.json",),
        "dse_v2": (
            "autosa_mm_dse_v2_leaderboard.json",
            "autosa_mm_post_flash_dse_v2.json",
        ),
        "stream": ("autosa_mm_stream_report.json",),
        "enf": ("autosa_mm_enforcement.json",),
        "oneshot": (
            "autosa_mm_flash_opt_report.json",
            "autosa_mm_selected_report.json",
        ),
        "mem": ("autosa_mm_mem_report.json",),
    }
    for stage, names in candidates.items():
        for name in names:
            path = cell_dir / name
            if path.is_file():
                mapping[stage] = path
                break
            hits = list(cell_dir.glob(name.replace("autosa_mm_", "*_")))
            if hits:
                mapping[stage] = hits[0]
                break
    return mapping


def _report_complete(path: Path) -> bool:
    data = _load_json(path)
    if not isinstance(data, dict):
        return False
    qor = qor_from_report(data)
    if qor is None:
        return False
    status = qor.get("csim") or ""
    if status == "fail":
        return False
    return qor.get("latency_cycles") is not None or qor.get("dsp") is not None


def cell_complete(cell: dict[str, Any]) -> bool:
    root = Path(cell.get("campaign_root") or "")
    variant = find_variant_cell(root)
    if variant is None:
        return False
    reports = stage_report_paths(variant)
    family = cell.get("family")
    if family in {"flash", "oneshot"}:
        path = reports.get("flash") or reports.get("oneshot")
    elif family == "dse_v1":
        path = reports.get("dse")
    elif family == "dse_v2":
        marker = reports.get("dse_v2")
        if marker is None:
            return False
        path = reports.get("dse")
    elif family == "stream":
        path = reports.get("stream")
    elif family == "enf":
        path = reports.get("enf")
    elif family == "mem":
        path = reports.get("mem")
    else:
        return False
    return bool(path and _report_complete(path))


def parent_complete(cell: dict[str, Any], cells_by_id: dict[str, dict[str, Any]] | None = None) -> bool:
    parent_id = cell.get("parent_cell") or ""
    if not parent_id:
        return True
    parent = (cells_by_id or {}).get(parent_id)
    if parent is None:
        return False
    return cell_complete(parent)


def copy_parent_variants(parent_root: Path, child_root: Path) -> Path:
    refuse_frozen(child_root)
    refuse_frozen(parent_root)
    src = parent_root / "variants"
    if not src.is_dir():
        raise FileNotFoundError(f"parent variants missing: {src}")
    dst = child_root / "variants"
    child_root.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return dst
    shutil.copytree(src, dst)
    return dst


def make_enforcement_seed(dse_root: Path, seed_dir: Path) -> Path:
    refuse_frozen(seed_dir)
    cell = find_variant_cell(dse_root)
    if cell is None:
        raise FileNotFoundError(f"no HLS cell under {dse_root}")
    cpp = cell / "autosa_mm_selected.cpp"
    rpt = cell / "autosa_mm_selected_report.json"
    if not cpp.is_file():
        cpp = cell / "autosa_mm_dse.cpp"
    if not rpt.is_file():
        rpt = cell / "autosa_mm_dse_report.json"
    if not cpp.is_file() or not rpt.is_file():
        raise FileNotFoundError(f"DSE selected kernel missing in {cell}")
    seed_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(cpp, seed_dir / "autosa_mm_flash_opt.cpp")
    shutil.copy2(rpt, seed_dir / "autosa_mm_flash_opt_report.json")
    return seed_dir


def parse_waves(raw: str | None) -> set[str]:
    if not raw:
        return set(WAVES)
    waves: set[str] = set()
    for part in raw.split(","):
        name = part.strip().lower()
        if not name:
            continue
        if name not in WAVE_FAMILIES:
            raise ValueError(f"unknown wave {part!r} (use flash|dse|stream|enf|oneshot|mem)")
        waves.add(name)
    return waves


def mem_parent_family(cell: dict[str, Any]) -> str:
    """Parent family of a mem_* cell from copied flags."""
    if int(cell.get("enf") or 0):
        return "enf"
    if int(cell.get("stream") or 0):
        return "stream"
    dse = str(cell.get("dse") or "").strip()
    if dse in {"v2", "2"}:
        return "dse_v2"
    if dse:
        return "dse_v1"
    if str(cell.get("skill") or "") == "one_shot":
        return "oneshot"
    return "flash"


def _cell_rank(cell: dict[str, Any]) -> float:
    family = cell.get("family") or ""
    if family == "mem":
        return float(WAVE_RANK.get(mem_parent_family(cell), 0)) + 0.5
    return float(WAVE_RANK.get(family, 9))


def cells_for_waves(cells: list[dict[str, Any]], waves: set[str]) -> list[dict[str, Any]]:
    families: set[str] = set()
    for wave in waves:
        families |= set(WAVE_FAMILIES[wave])
    selected = [c for c in cells if c["family"] in families]
    selected.sort(key=lambda c: (_cell_rank(c), int(c.get("rep") or 0), c["cell_id"]))
    return selected


def squeue_job_names() -> list[str] | None:
    try:
        proc = subprocess.run(
            ["squeue", "-u", os.environ.get("USER", ""), "-h", "-o", "%j"],
            check=False,
            capture_output=True,
            text=True,
            timeout=30,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        return None
    if proc.returncode != 0:
        return None
    return [line.strip() for line in proc.stdout.splitlines() if line.strip()]


def prefix_in_squeue(prefix: str, names: list[str]) -> bool:
    for name in names:
        if name == prefix or name.startswith(prefix + "-"):
            return True
    return False


def reap_dead_cells(cells: list[dict[str, Any]], names: list[str] | None) -> int:
    """Reset submitted/running cells whose prefix is gone and have no report.

    If squeue is unavailable (names is None), do not reap.
    """
    if names is None:
        return 0
    n = 0
    for cell in cells:
        if cell.get("status") not in {"submitted", "running"}:
            continue
        if cell_complete(cell):
            cell["status"] = "complete"
            continue
        if prefix_in_squeue(cell.get("job_prefix") or "", names):
            continue
        cell["fail_count"] = int(cell.get("fail_count") or 0) + 1
        cell["error"] = "job vanished without complete report"
        cell["status"] = "failed" if cell["fail_count"] >= 3 else "pending"
        n += 1
    return n


def inflight_count(cells: list[dict[str, Any]], names: list[str] | None = None) -> int:
    """Count unique campaign prefixes in squeue, not helper/synth job rows."""
    prefixes = {c["job_prefix"] for c in cells}
    if names is None:
        names = squeue_job_names()
    if names is None:
        return sum(1 for c in cells if c.get("status") in {"submitted", "running"} and not cell_complete(c))
    active: set[str] = set()
    for name in names:
        for prefix in prefixes:
            if name == prefix or name.startswith(prefix + "-"):
                active.add(prefix)
                break
    return len(active)


def _export_env(base: dict[str, str], extra: dict[str, str]) -> dict[str, str]:
    env = dict(base)
    for key, val in extra.items():
        if val == "":
            env.pop(key, None)
        else:
            env[key] = val
    return env


def _write_sidecar(campaign_root: Path, cell: dict[str, Any]) -> None:
    refuse_frozen(campaign_root)
    campaign_root.mkdir(parents=True, exist_ok=True)
    (campaign_root / "sweep_cell.json").write_text(
        json.dumps(cell, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def submit_sbatch(
    *,
    job_name: str,
    script: Path,
    env: dict[str, str],
    extra_args: list[str] | None = None,
    dry_run: bool,
) -> str:
    cmd = [
        "sbatch",
        "--parsable",
        "--job-name",
        job_name,
        "--chdir",
        str(REPO),
        "--export=ALL",
        str(script),
    ]
    if extra_args:
        cmd[6:6] = extra_args
    if dry_run:
        return "dry-run"
    proc = subprocess.run(cmd, check=True, capture_output=True, text=True, env=env, cwd=str(REPO))
    return proc.stdout.strip().split(";")[0]


def submit_flash_or_oneshot(
    cell: dict[str, Any],
    *,
    endpoint: str,
    dry_run: bool,
    env: dict[str, str],
) -> str:
    campaign_root = Path(cell["campaign_root"])
    refuse_frozen(campaign_root)
    artifact_prefix = artifact_prefix_for_cell(cell, repo=REPO)
    flavor = flavor_for_skill(cell["skill"])
    launch_env = _export_env(env, launch_env_for_cell(cell))
    launch_env["C2HLS_SWEEP_ARTIFACT_PREFIX"] = artifact_prefix
    launch_env["BATCH_PARALLEL_ARTIFACT_PREFIX"] = artifact_prefix
    launch_env["BATCH_PARALLEL_STAMP"] = "camp"
    launch_env["C2HLS_FLASH_ONLY"] = "1"
    if not dry_run:
        # The flow script deletes campaign_root before seeding the queue.
        # Keep the proxy beside that directory so the delete does not take it down.
        proxy_dir = campaign_root.parent / f".{campaign_root.name}_llm"
        endpoint = _dedicated_llm_endpoint(proxy_dir) or endpoint
    launch_env["OPENAI_BASE_URL"] = endpoint
    launch_env["C2HLS_OPENAI_HOSTED_URL"] = endpoint
    cmd = [
        str(SCRIPT_DIR / "start_autosa_mm_flow.sh"),
        "--stamp",
        "camp",
        "--flavor",
        flavor,
        "--endpoint-url",
        endpoint,
    ]
    if dry_run:
        cmd.append("--dry-run")
        print(" ".join(cmd))
        print(f"  prefix={cell['job_prefix']} campaign={campaign_root}")
        return "dry-run"
    proc = subprocess.run(cmd, check=False, capture_output=True, text=True, env=launch_env, cwd=str(REPO))
    out = (proc.stdout or "").strip()
    err = (proc.stderr or "").strip()
    if out:
        print(out)
    if proc.returncode != 0:
        if err:
            print(err)
        raise RuntimeError(
            f"start_autosa_mm_flow.sh failed rc={proc.returncode} cell={cell['cell_id']}: {err or out[-500:]}"
        )
    return out


def _dedicated_llm_endpoint(session_dir: Path) -> str:
    """Start a per-campaign login-node proxy. Empty string if it cannot start."""
    script = SCRIPT_DIR / "start_dedicated_deepseek_proxy.sh"
    if not script.is_file():
        return ""
    proc = subprocess.run(
        ["bash", str(script), str(session_dir)],
        check=False,
        capture_output=True,
        text=True,
        cwd=str(REPO),
    )
    url = (proc.stdout or "").strip().splitlines()
    if proc.returncode != 0 or not url:
        err = (proc.stderr or proc.stdout or "").strip()
        raise RuntimeError(f"dedicated deepseek proxy failed: {err[-500:]}")
    return url[-1].strip()


def submit_dse_or_stream(
    cell: dict[str, Any],
    parent: dict[str, Any],
    *,
    endpoint: str,
    dry_run: bool,
    env: dict[str, str],
) -> str:
    child = Path(cell["campaign_root"])
    parent_root = Path(parent["campaign_root"])
    refuse_frozen(child)
    if dry_run:
        print(
            f"sbatch {cell['family']} prefix={cell['job_prefix']} "
            f"parent={parent['cell_id']} campaign={child}"
        )
        return "dry-run"
    copy_parent_variants(parent_root, child)
    _write_sidecar(child, cell)
    launch_env = _export_env(env, launch_env_for_cell(cell))
    launch_env["C2HLS_POST_FLASH_MATRIX_ROOT"] = str(child)
    launch_env["C2HLS_POST_FLASH_BENCHES"] = "autosa_mm"
    endpoint = _dedicated_llm_endpoint(child / "node_llm_proxy") or endpoint
    launch_env["OPENAI_BASE_URL"] = endpoint
    launch_env["C2HLS_OPENAI_HOSTED_URL"] = endpoint
    launch_env["CHATHLS_API_BASE"] = endpoint
    launch_env["C2HLS_MODEL"] = DEFAULT_MODEL
    if cell["family"] == "dse_v2":
        script = SCRIPT_DIR / "post_flash_dse_v2.sbatch.sh"
    elif cell["family"] == "dse_v1":
        script = SCRIPT_DIR / "post_flash_dse.sbatch.sh"
    else:
        script = SCRIPT_DIR / "post_flash_stream.sbatch.sh"
    wall = os.getenv("C2HLS_COMPUTE_WALLTIME", "").strip()
    job = submit_sbatch(
        job_name=cell["job_prefix"],
        script=script,
        env=launch_env,
        extra_args=[f"--time={wall}"] if wall else None,
        dry_run=False,
    )
    (child / "slurm_job_id").write_text(job + "\n", encoding="utf-8")
    return job


def submit_enforcement(
    cell: dict[str, Any],
    parent: dict[str, Any],
    *,
    endpoint: str,
    dry_run: bool,
    env: dict[str, str],
) -> str:
    child = Path(cell["campaign_root"])
    refuse_frozen(child)
    seed_dir = child / "seed_flash"
    artifact_prefix = str(
        Path("autosa_mm_variant_sweep_" + str(cell["stamp"])) / "cells" / cell["cell_id"]
    )
    if dry_run:
        print(
            f"enforcement prefix={cell['job_prefix']} parent={parent['cell_id']} "
            f"seed={seed_dir} campaign={child}"
        )
        return "dry-run"
    make_enforcement_seed(Path(parent["campaign_root"]), seed_dir)
    _write_sidecar(child, cell)
    launch_env = _export_env(env, launch_env_for_cell(cell))
    launch_env["C2HLS_SWEEP_JOB_PREFIX"] = cell["job_prefix"]
    launch_env["PC2_BATCH_JOB_PREFIX"] = cell["job_prefix"]
    launch_env["C2HLS_SWEEP_ARTIFACT_PREFIX"] = artifact_prefix
    launch_env["BATCH_PARALLEL_ARTIFACT_PREFIX"] = artifact_prefix
    launch_env["BATCH_PARALLEL_STAMP"] = "camp"
    launch_env["C2HLS_FLASH_SEED_DIR"] = str(seed_dir)
    launch_env["C2HLS_SKIP_FLASH"] = "1"
    launch_env["C2HLS_PP_LOAD_B_IN_DF"] = "1"
    cmd = [
        str(SCRIPT_DIR / "start_autosa_mm_enforcement.sh"),
        "--stamp",
        "camp",
        "--seed-flash",
        str(seed_dir),
        "--load-b-in-df",
        "--endpoint-url",
        endpoint,
    ]
    proc = subprocess.run(cmd, check=False, capture_output=True, text=True, env=launch_env, cwd=str(REPO))
    out = (proc.stdout or "").strip()
    err = (proc.stderr or "").strip()
    if out:
        print(out)
    if proc.returncode != 0:
        if err:
            print(err)
        raise RuntimeError(
            f"start_autosa_mm_enforcement.sh failed rc={proc.returncode} "
            f"cell={cell['cell_id']}: {err or out[-500:]}"
        )
    return out


def submit_mem(
    cell: dict[str, Any],
    parent: dict[str, Any],
    *,
    endpoint: str,
    dry_run: bool,
    env: dict[str, str],
) -> str:
    child = Path(cell["campaign_root"])
    parent_root = Path(parent["campaign_root"])
    refuse_frozen(child)
    if dry_run:
        print(
            f"sbatch mem prefix={cell['job_prefix']} "
            f"parent={parent['cell_id']} campaign={child}"
        )
        return "dry-run"
    copy_parent_variants(parent_root, child)
    _write_sidecar(child, cell)
    launch_env = _export_env(env, launch_env_for_cell(cell))
    launch_env["C2HLS_POST_FLASH_MATRIX_ROOT"] = str(child)
    launch_env["C2HLS_POST_FLASH_BENCHES"] = "autosa_mm"
    launch_env["OPENAI_BASE_URL"] = endpoint
    launch_env["C2HLS_OPENAI_HOSTED_URL"] = endpoint
    launch_env["CHATHLS_API_BASE"] = endpoint
    launch_env["C2HLS_MODEL"] = DEFAULT_MODEL
    launch_env["C2HLS_MEM_PARENT_FAMILY"] = mem_parent_family(cell)
    job = submit_sbatch(
        job_name=cell["job_prefix"],
        script=SCRIPT_DIR / "post_flash_mem_iter.sbatch.sh",
        env=launch_env,
        dry_run=False,
    )
    (child / "slurm_job_id").write_text(job + "\n", encoding="utf-8")
    return job


def submit_cell(
    cell: dict[str, Any],
    cells_by_id: dict[str, dict[str, Any]],
    *,
    endpoint: str,
    dry_run: bool,
) -> str:
    refuse_frozen(cell["campaign_root"])
    parent_id = cell.get("parent_cell") or ""
    parent = cells_by_id.get(parent_id) if parent_id else None
    env = os.environ.copy()
    family = cell["family"]
    if family in {"flash", "oneshot"}:
        return submit_flash_or_oneshot(cell, endpoint=endpoint, dry_run=dry_run, env=env)
    if parent is None:
        raise ValueError(f"{cell['cell_id']} missing parent {parent_id}")
    if family in {"dse_v1", "dse_v2", "stream"}:
        return submit_dse_or_stream(
            cell, parent, endpoint=endpoint, dry_run=dry_run, env=env
        )
    if family == "enf":
        return submit_enforcement(cell, parent, endpoint=endpoint, dry_run=dry_run, env=env)
    if family == "mem":
        return submit_mem(cell, parent, endpoint=endpoint, dry_run=dry_run, env=env)
    raise ValueError(f"unknown family {family}")


def format_manifest_row(cell: dict[str, Any]) -> str:
    return (
        f"{cell['cell_id']}\t{cell['family']}\t{cell['skill']}\tfloor={cell['floor']}\t"
        f"dse={cell['dse'] or '-'}\tstream={cell['stream']}\tenf={cell['enf']}\t"
        f"mem={cell.get('mem', 0)}\trep={cell['rep']:02d}\tprefix={cell['job_prefix']}\t"
        f"parent={cell['parent_cell'] or '-'}"
    )


def run_sweep(
    *,
    date: str,
    reps: int,
    waves: set[str],
    endpoint: str,
    max_inflight: int,
    dry_run: bool,
    submit: bool,
    repo: Path = REPO,
    sweep_root: Path | None = None,
    prefix_tag: str = "",
) -> list[dict[str, Any]]:
    cells = build_manifest(
        reps=reps, date=date, repo=repo, sweep_root=sweep_root, prefix_tag=prefix_tag
    )
    if sweep_root is None:
        sweep_root = sweep_root_for(date, repo=repo)
    else:
        sweep_root = Path(sweep_root)
    refuse_frozen(sweep_root)
    sweep_root.mkdir(parents=True, exist_ok=True)
    cells_path = sweep_root / "cells.jsonl"
    existing = {c["cell_id"]: c for c in load_cells(cells_path)}
    for cell in cells:
        prev = existing.get(cell["cell_id"])
        if prev:
            if prev.get("status") not in {None, "", "pending"}:
                cell["status"] = prev["status"]
            if prev.get("job_id"):
                cell["job_id"] = prev["job_id"]
            if prev.get("fail_count"):
                cell["fail_count"] = prev["fail_count"]
            if prev.get("error"):
                cell["error"] = prev["error"]
    write_cells(cells_path, cells)
    selected = cells_for_waves(cells, waves)
    print(f"sweep_root={sweep_root}")
    print(f"cells={len(cells)} unique={len(unique_configs(cells))} wave={sorted(waves)} selected={len(selected)}")
    if dry_run or not submit:
        for cell in selected:
            print(format_manifest_row(cell))
        if dry_run:
            print("dry-run: zero sbatch")
        return cells
    cells_by_id = {c["cell_id"]: c for c in cells}
    reaped = reap_dead_cells(cells, squeue_job_names())
    if reaped:
        write_cells(cells_path, cells)
        print(f"reaped {reaped} dead submitted cells")
    for cell in selected:
        if cell_complete(cell):
            cell["status"] = "complete"
            continue
        if cell.get("status") in {"submitted", "running", "complete"}:
            continue
        if int(cell.get("fail_count") or 0) >= 3:
            print(f"skip {cell['cell_id']}: fail_count={cell.get('fail_count')}")
            continue
        if not parent_complete(cell, cells_by_id):
            print(f"skip {cell['cell_id']}: parent not complete")
            continue
        while inflight_count(cells) >= max_inflight:
            print(f"max_inflight={max_inflight} reached; waiting")
            time.sleep(30)
        try:
            job = submit_cell(cell, cells_by_id, endpoint=endpoint, dry_run=False)
        except Exception as exc:
            cell["fail_count"] = int(cell.get("fail_count") or 0) + 1
            cell["status"] = "pending"
            cell["error"] = str(exc)
            write_cells(cells_path, cells)
            print(f"FAIL {cell['cell_id']}: {exc}")
            continue
        cell["status"] = "submitted"
        cell["job_id"] = job
        write_cells(cells_path, cells)
        print(f"submitted {cell['cell_id']} job={job} prefix={cell['job_prefix']}")
    write_cells(cells_path, cells)
    return cells


def follow_sweep(
    *,
    date: str,
    reps: int,
    waves: set[str],
    endpoint: str,
    max_inflight: int,
    repo: Path = REPO,
    sleep_s: int = 120,
) -> None:
    sweep_root = sweep_root_for(date, repo=repo)
    sweep_root.mkdir(parents=True, exist_ok=True)
    print(f"follow sweep_root={sweep_root} sleep={sleep_s}s max_inflight={max_inflight}")
    while True:
        cells = run_sweep(
            date=date,
            reps=reps,
            waves=waves,
            endpoint=endpoint,
            max_inflight=max_inflight,
            dry_run=False,
            submit=True,
            repo=repo,
        )
        selected = cells_for_waves(cells, waves)
        for cell in selected:
            if cell_complete(cell):
                cell["status"] = "complete"
        write_cells(sweep_root / "cells.jsonl", cells)
        n_complete = sum(1 for c in selected if c.get("status") == "complete" or cell_complete(c))
        n_failed = sum(1 for c in selected if int(c.get("fail_count") or 0) >= 3)
        n_pending = len(selected) - n_complete - n_failed
        print(
            f"follow status complete={n_complete} pending={n_pending} "
            f"failed={n_failed} inflight={inflight_count(cells)}"
        )
        if n_pending <= 0:
            print("follow: no pending cells left")
            return
        time.sleep(max(15, int(sleep_s)))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--submit", action="store_true")
    parser.add_argument(
        "--follow",
        action="store_true",
        help="Keep submitting DSE/stream/enf as parents finish (long-running)",
    )
    parser.add_argument("--wave", default="", help="flash|dse|stream|enf|oneshot|mem (comma-separated)")
    parser.add_argument("--stamp", default="", help="YYYYMMDD sweep date (default: today UTC)")
    parser.add_argument("--reps", type=int, default=DEFAULT_REPS)
    parser.add_argument("--max-inflight", type=int, default=DEFAULT_MAX_INFLIGHT)
    parser.add_argument("--sleep", type=int, default=120, help="Follow poll interval in seconds")
    parser.add_argument("--endpoint-url", default=DEFAULT_ENDPOINT)
    args = parser.parse_args()
    date = args.stamp.strip() or datetime.now(timezone.utc).strftime("%Y%m%d")
    if args.follow and not args.wave:
        args.wave = "flash,oneshot,dse,stream,enf,mem"
        args.submit = True
    waves = parse_waves(args.wave or None) if args.wave else (
        set(WAVES) if args.dry_run or not args.submit else set()
    )
    if args.submit and not args.dry_run and not args.wave and not args.follow:
        raise SystemExit("refusing to submit all waves; pass --wave flash,oneshot,mem (etc)")
    if not waves:
        waves = parse_waves(args.wave)
    if args.follow and not args.dry_run:
        follow_sweep(
            date=date,
            reps=args.reps,
            waves=waves,
            endpoint=args.endpoint_url,
            max_inflight=args.max_inflight,
            sleep_s=args.sleep,
        )
        return 0
    run_sweep(
        date=date,
        reps=args.reps,
        waves=waves,
        endpoint=args.endpoint_url,
        max_inflight=args.max_inflight,
        dry_run=args.dry_run,
        submit=args.submit and not args.dry_run,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
