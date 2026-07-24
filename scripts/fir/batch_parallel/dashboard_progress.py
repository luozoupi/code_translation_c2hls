"""Infer csynth / csim / cosim progress from Fir batch_parallel compute logs."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "scripts" / "explorer"))

from batch_parallel.config import campaign_paths
from batch_parallel.dispatch import cell_dir, model_cell_tag
from metrics import bench_cosim_metrics_from_multistep_doc, bench_run_issues_from_multistep_doc


def _cosim_chip_from_doc(doc: dict[str, Any], *, bench_short: str) -> str | None:
    metrics = bench_cosim_metrics_from_multistep_doc(
        doc,
        bench_short_name=bench_short,
    )
    if not metrics:
        return None
    status = str(metrics.get("status") or "not_run")
    if status == "not_run":
        return None
    return status


def cosim_enabled_for_campaign(campaign: dict[str, Any]) -> bool:
    pilot = (campaign.get("config") or {}).get("pilot") or {}
    if "run_cosim" in pilot:
        return bool(pilot.get("run_cosim"))
    workflow = str(pilot.get("workflow") or "")
    if workflow.startswith("zero_shot"):
        return True
    config = campaign.get("config") or {}
    prefix = str(config.get("artifact_prefix") or "")
    if "cosim" in prefix.lower():
        return True
    stamp = str(campaign.get("stamp") or "")
    if "cosim" in stamp.lower():
        return True
    return os.getenv("C2HLS_RUN_COSIM", "0").strip().lower() in ("1", "true", "yes")


def slurm_stderr_path(campaign_root: Path, *, node_index: int, slurm_job_id: str) -> Path:
    return campaign_root / f"slurm-compute-n{node_index}-{slurm_job_id}.err"


def tail_text(path: Path, *, max_bytes: int = 96_000) -> str:
    if not path.is_file():
        return ""
    try:
        size = path.stat().st_size
        with path.open("rb") as handle:
            if size > max_bytes:
                handle.seek(size - max_bytes)
            raw = handle.read()
        return raw.decode("utf-8", errors="replace")
    except OSError:
        return ""


def _blank_hls(cosim_enabled: bool) -> dict[str, str]:
    return {
        "phase": "—",
        "csynth": "—",
        "csim": "—",
        "cosim": "off" if not cosim_enabled else "—",
    }


def _find_flash_step(doc: dict[str, Any]) -> dict[str, Any] | None:
    for step in doc.get("steps") or []:
        if step.get("step_name") == "flash":
            return step
    return None


def _csim_chip_from_generated(
    doc: dict[str, Any],
    flash_step: dict[str, Any] | None,
) -> str:
    for src in (doc.get("csim"), (flash_step or {}).get("csim")):
        if not isinstance(src, dict) or not src.get("ran"):
            continue
        if src.get("passed") is True or src.get("success") is True:
            return "pass"
        if str(src.get("status") or "").lower() in ("pass", "passed", "ok"):
            return "pass"
        return "fail"
    return "—"


def _csynth_csim_from_failed_flash(flash_step: dict[str, Any]) -> tuple[str, str]:
    attempts = flash_step.get("attempt_results") or []
    stage = str((attempts[-1] if attempts else {}).get("stage") or "").lower()
    if stage in ("compile_check", "compile", "abi_preflight"):
        return "fail", "—"
    if stage == "csim":
        return "pass", "fail"
    if stage == "cosim":
        return "pass", "pass"
    return "fail", "—"


def hls_chips_from_artifacts(
    multistep_doc: dict[str, Any] | None,
    *,
    manifest_doc: dict[str, Any] | None = None,
) -> tuple[str, str]:
    """Return (csynth, csim) from generated flash/phase-B outcomes (not gold reference)."""
    if not multistep_doc:
        return "—", "—"

    phase = str(multistep_doc.get("phase") or "")
    if phase == "reference":
        return "—", "—"

    flash_step = _find_flash_step(multistep_doc)
    if flash_step is None and phase == "B":
        return "fail", "—"

    flash_ok: bool | None = None
    if isinstance(manifest_doc, dict) and "flash_step_success" in manifest_doc:
        flash_ok = bool(manifest_doc.get("flash_step_success"))
    elif flash_step is not None:
        flash_ok = bool(flash_step.get("success"))
    elif phase == "flash":
        flash_ok = False

    if flash_ok is None:
        return "—", "—"
    if flash_ok:
        return "pass", _csim_chip_from_generated(multistep_doc, flash_step)
    if flash_step is None:
        return "fail", "—"
    return _csynth_csim_from_failed_flash(flash_step)


def _apply_artifact_hls_chips(
    out: dict[str, str],
    *,
    cell: Path,
    bench: str,
    cosim_enabled: bool,
) -> None:
    manifest_doc: dict[str, Any] | None = None
    multistep_doc: dict[str, Any] | None = None
    manifest = cell / f"{bench}_flow_manifest.json"
    results = cell / f"{bench}_multistep_results.json"
    if manifest.is_file():
        try:
            manifest_doc = json.loads(manifest.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            manifest_doc = None
    if results.is_file():
        try:
            multistep_doc = json.loads(results.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            multistep_doc = None

    csynth, csim = hls_chips_from_artifacts(multistep_doc, manifest_doc=manifest_doc)
    if csynth != "—":
        out["csynth"] = csynth
    if csim != "—":
        out["csim"] = csim

    if multistep_doc:
        bench_short = bench.removeprefix("hlsfactory_")
        cosim_chip = _cosim_chip_from_doc(multistep_doc, bench_short=bench_short)
        if cosim_chip:
            out["cosim"] = cosim_chip
        issues = bench_run_issues_from_multistep_doc(multistep_doc)
        if issues:
            out["issues"] = issues
    elif not cosim_enabled:
        out["cosim"] = "off"


def infer_hls_progress(log_text: str, *, cosim_enabled: bool) -> dict[str, str]:
    """Parse worker stderr tail into phase + csynth/csim/cosim status chips."""
    out = _blank_hls(cosim_enabled)
    if not log_text.strip():
        return out

    in_flash = False
    for line in log_text.splitlines():
        low = line.lower()

        if "=== [phase a]" in low:
            out["phase"] = "compile"
            in_flash = False
        elif "=== [phase b]" in low and "translating" in low:
            out["phase"] = "phase_b LLM"
            in_flash = False
            out["csynth"] = "—"
            out["csim"] = "—"
        elif "[phase b] synthesis attempt" in low:
            out["phase"] = "csynth"
            out["csynth"] = "running"
        elif "[phase b] synthesis success" in low or (
            not in_flash and "synthesis success!" in low
        ):
            out["csynth"] = "pass"
            if out["csim"] == "—":
                out["phase"] = "csim"
                out["csim"] = "next"
        elif "synthesis failed" in low:
            out["csynth"] = "fail"
        elif "running c-simulation" in low:
            out["phase"] = "csim"
            out["csim"] = "running"
        elif "[phase b] csim passed" in low or "csim passed" in low:
            out["csim"] = "pass"
            if cosim_enabled and out["cosim"] in ("—", "off"):
                out["phase"] = "cosim"
                out["cosim"] = "next"
        elif "csim failed" in low:
            out["csim"] = "fail"
        elif cosim_enabled and ("cosim passed" in low or "co-simulation passed" in low):
            out["cosim"] = "pass"
        elif cosim_enabled and ("cosim failed" in low or "co-simulation failed" in low):
            out["cosim"] = "fail"
        elif cosim_enabled and "running co-simulation" in low:
            out["phase"] = "cosim"
            out["cosim"] = "running"
        elif "=== [step: flash]" in low or "applying optimization" in low:
            in_flash = True
            out["phase"] = "flash LLM"
            out["csynth"] = "—"
            out["csim"] = "—"
        elif in_flash and ("chat/completions" in low or "retrying request" in low):
            out["phase"] = "flash LLM"
        elif in_flash and "synthesis attempt" in low:
            out["phase"] = "csynth"
            out["csynth"] = "running"
        elif in_flash and "synthesis success" in low:
            out["phase"] = "csim"
            out["csynth"] = "pass"
            if out["csim"] == "—":
                out["csim"] = "next"
        elif in_flash and "running c-simulation" in low:
            out["phase"] = "csim"
            out["csim"] = "running"
        elif in_flash and "csim passed" in low:
            out["csim"] = "pass"
            if cosim_enabled and out["cosim"] in ("—", "off"):
                out["phase"] = "cosim"
                out["cosim"] = "next"
        elif in_flash and "csim failed" in low:
            out["csim"] = "fail"

    return out


def read_llm_in_flight(campaign_root: Path) -> dict[str, Any] | None:
    path = campaign_paths(campaign_root)["root"] / "flow" / "gpu_llm.json"
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    inflight = payload.get("in_flight")
    return dict(inflight) if isinstance(inflight, dict) else None


def slot_hls_progress(
    campaign_root: Path,
    *,
    node_index: int,
    slurm_job_id: str,
    bench: str | None,
    cosim_enabled: bool,
    llm_in_flight: dict[str, Any] | None,
) -> dict[str, str]:
    if not bench:
        return _blank_hls(cosim_enabled)

    err = slurm_stderr_path(campaign_root, node_index=node_index, slurm_job_id=slurm_job_id)
    progress = infer_hls_progress(tail_text(err), cosim_enabled=cosim_enabled)

    if llm_in_flight and str(llm_in_flight.get("bench") or "") == bench:
        worker = str(llm_in_flight.get("worker") or "")
        if worker == f"flash-n{node_index}-s0" or not worker:
            progress["phase"] = "LLM"

    return progress


def bench_hls_progress(
    campaign_root: Path,
    *,
    bench: str,
    status: str,
    model_id: str,
    setup_tag: str,
    node_index: int | None,
    slurm_job_id: str | None,
    cosim_enabled: bool,
    llm_in_flight: dict[str, Any] | None,
) -> dict[str, str]:
    if status == "claimed" and node_index is not None and slurm_job_id:
        return slot_hls_progress(
            campaign_root,
            node_index=node_index,
            slurm_job_id=slurm_job_id,
            bench=bench,
            cosim_enabled=cosim_enabled,
            llm_in_flight=llm_in_flight,
        )
    if status in ("done", "failed"):
        out = _blank_hls(cosim_enabled)
        out["phase"] = "done" if status == "done" else "fail"
        tag = model_cell_tag(model_id)
        cell = cell_dir(campaign_root, bench, tag, setup_tag)
        _apply_artifact_hls_chips(
            out,
            cell=cell,
            bench=bench,
            cosim_enabled=cosim_enabled,
        )
        return out
    return _blank_hls(cosim_enabled)
