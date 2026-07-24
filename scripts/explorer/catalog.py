"""Scan artifact campaigns and build a normalized experiment catalog."""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from benchmark_cosim_baseline import load_benchmark_cosim_baseline
from metrics import (
    bench_cosim_metrics_from_multistep_doc,
    bench_csynth_latency_from_multistep_doc,
    bench_run_issues_from_multistep_doc,
    bench_speedup_from_multistep_doc,
    cosim_status_counts_from_benches,
    geomean_cosim_speedup_from_benches,
    geomean_from_bench_speedups,
    mean_latency_from_benches,
)

_COSIM_BASELINE_MAP: dict[str, int] | None = None


def _cosim_baseline_map() -> dict[str, int]:
    global _COSIM_BASELINE_MAP
    if _COSIM_BASELINE_MAP is None:
        _COSIM_BASELINE_MAP = load_benchmark_cosim_baseline()
    return _COSIM_BASELINE_MAP

REPO = Path(__file__).resolve().parents[2]

SITE_DIRS = ("fir", "pc2", "flash_api", "team")

SKIP_TOP_LEVEL = frozenset({
    "sessions",
    "analysis",
    "flash_selected_bundle",
    "multistep_cosim",
    "reports",
})

STAMP_RE = re.compile(r"_(\d{8}_\d{6})$")

FIXED_COSIM_PREFIX_RE = re.compile(r"(?:flash|multistep)_fixed_cosim_(.+)$", re.IGNORECASE)
TRAILING_STAMP_RE = re.compile(r"_\d{8}(?:_\d{6})?$")

SKILL_MODE_FROM_SETUP = {
    "all_skills_avoids_global": "all_skills_avoids_global",
    "all_skills_no_avoids_global": "all_skills_no_avoids_global",
    "noskills": "noskills",
    "bottleneck": "bottleneck",
    "zero_shot_cosim": "zero_shot",
}


def bench_short(name: str) -> str:
    return name.removeprefix("hlsfactory_")


def bench_corpus(name: str) -> str:
    if name.startswith("hlsfactory_"):
        return "hlsfactory"
    if name.startswith("forgebench_") or name.startswith("spector_hls_"):
        return "tier_a"
    return "external"


def corpus_from_benches(benches: dict[str, Any]) -> str:
    kinds = {bench_corpus(b) for b in benches}
    if len(kinds) == 1:
        return next(iter(kinds))
    if not kinds:
        return "unknown"
    return "mixed"


def parse_stamp(dirname: str) -> str | None:
    match = STAMP_RE.search(dirname)
    return match.group(1) if match else None


def parse_skill_variant(dirname: str) -> str | None:
    match = FIXED_COSIM_PREFIX_RE.search(dirname)
    if not match:
        return None
    variant = TRAILING_STAMP_RE.sub("", match.group(1))
    return variant or None


def parse_workflow(dirname: str, *, mode: str | None = None) -> str:
    if (mode or "").lower() == "multistep":
        return "multistep"
    if (mode or "").lower() == "flash":
        return "flash"
    low = dirname.lower()
    if "multistep" in low:
        return "multistep"
    if "hls_baseline" in low or "baseline_smoke" in low:
        return "baseline"
    if "dataflow" in low:
        return "flash_dataflow"
    if "flash" in low:
        return "flash"
    if "batch_parallel" in low:
        return "flash"
    return "unknown"


def parse_cosim(
    dirname: str,
    *,
    setup: str | None = None,
    campaign: dict[str, Any] | None = None,
) -> str:
    low = dirname.lower()
    setup_low = (setup or "").lower()
    if "fixed_cosim" in low or "fixed_cosim" in setup_low:
        return "on"
    if "cosim" in low and "no_cosim" not in low:
        return "on"
    pilot = ((campaign or {}).get("config") or {}).get("pilot") or {}
    if "run_cosim" in pilot:
        return "on" if pilot.get("run_cosim") else "off"
    if dirname.startswith("batch_parallel") or dirname.startswith("flash_smoke"):
        return "off"
    return "unknown"


def parse_skills_mode(dirname: str, *, setup: str | None = None, manifest: dict | None = None) -> str:
    if manifest:
        if manifest.get("flash_opt_prompt_mode") == "zero_shot":
            return "zero_shot"
        if manifest.get("skill_prompt_mode"):
            return str(manifest["skill_prompt_mode"])
    if setup:
        for key, value in SKILL_MODE_FROM_SETUP.items():
            if key in setup:
                return value
        if "zero_shot" in setup:
            return "zero_shot"
        if "noskills" in setup:
            return "noskills"
    low = dirname.lower()
    if "zero_shot" in low:
        return "zero_shot"
    if "noskills" in low:
        return "noskills"
    if "all_skills_avoids_global" in dirname.lower():
        return "all_skills_avoids_global"
    if "all_skills_no_avoids" in dirname.lower():
        return "all_skills_no_avoids_global"
    if "curated" in dirname.lower():
        return "bottleneck"
    return "unknown"


def _read_json(path: Path) -> Any:
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _matrix_ok_status(status: str | None) -> bool:
    return str(status or "").lower() in ("ok", "done", "success")


def _bench_entry_from_result(
    bench: str,
    *,
    status: str,
    result_path: str | Path | None,
) -> dict[str, Any]:
    entry: dict[str, Any] = {
        "bench": bench,
        "bench_short": bench_short(bench),
        "corpus": bench_corpus(bench),
        "status": "ok" if _matrix_ok_status(status) else str(status or "unknown"),
        "result_path": str(result_path) if result_path else None,
        "speedup": None,
        "latency": None,
        "cosim": None,
        "run_issues": None,
    }
    if entry["status"] == "ok" and result_path:
        path = Path(result_path)
        if not path.is_file():
            path = Path(str(result_path))
        if path.is_file():
            try:
                doc = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                doc = None
            if isinstance(doc, dict):
                sp = bench_speedup_from_multistep_doc(doc)
                if sp:
                    entry["speedup"] = sp
                lat = bench_csynth_latency_from_multistep_doc(doc)
                if lat:
                    entry["latency"] = lat
                cosim = bench_cosim_metrics_from_multistep_doc(
                    doc,
                    baseline_map=_cosim_baseline_map(),
                    bench_short_name=bench_short(bench),
                )
                if cosim:
                    entry["cosim"] = cosim
                issues = bench_run_issues_from_multistep_doc(doc)
                if issues:
                    entry["run_issues"] = issues
    return entry


def _benches_from_matrix(matrix: list[dict[str, Any]], campaign_root: Path) -> dict[str, Any]:
    benches: dict[str, Any] = {}
    for row in matrix:
        bench = str(row.get("bench") or "")
        if not bench:
            continue
        result_path = row.get("result_path")
        path = Path(str(result_path)) if result_path else None
        if path is None or not path.is_file():
            alt = campaign_root / bench
            if alt.is_dir():
                for found in alt.glob(f"**/{bench}_multistep_results.json"):
                    result_path = found
                    break
        entry = _bench_entry_from_result(
            bench,
            status=str(row.get("status") or "unknown"),
            result_path=result_path,
        )
        benches[bench_short(bench)] = entry
    return benches


def _benches_from_glob(campaign_root: Path, manifest: dict[str, Any]) -> dict[str, Any]:
    benches: dict[str, Any] = {}
    listed = manifest.get("benches") or []
    for bench in listed:
        bench = str(bench)
        found = None
        for path in campaign_root.glob(f"{bench}/**/{bench}_multistep_results.json"):
            found = path
            break
        if not found:
            for path in campaign_root.glob(f"**/{bench}_multistep_results.json"):
                found = path
                break
        status = "ok" if found and found.is_file() else "missing"
        entry = _bench_entry_from_result(bench, status=status, result_path=found)
        benches[bench_short(bench)] = entry
    if benches:
        return benches
    for path in sorted(campaign_root.glob("hlsfactory_*/*/*_multistep_results.json")):
        bench = path.parts[-3]
        entry = _bench_entry_from_result(
            bench,
            status="ok",
            result_path=path,
        )
        benches[bench_short(bench)] = entry
    return benches


def _counts_from_benches(benches: dict[str, Any]) -> dict[str, int]:
    total = len(benches)
    ok = sum(1 for b in benches.values() if b.get("status") == "ok")
    fail = total - ok
    return {"total": total, "ok": ok, "fail": fail}


def _has_dataflow_reports(campaign_root: Path) -> bool:
    reports = campaign_root / "reports"
    if not reports.is_dir():
        return False
    return any(reports.glob("post_flash_dataflow*"))


def _skills_meta_from_cells(campaign_root: Path) -> tuple[str | None, int | None]:
    for skills_path in sorted(campaign_root.glob("**/skills_source.json")):
        doc = _read_json(skills_path)
        if isinstance(doc, dict):
            src = doc.get("skills_json") or doc.get("source")
            if src:
                path = Path(str(src))
                if path.is_file():
                    skills_doc = _read_json(path)
                    if isinstance(skills_doc, list):
                        return path.name, len(skills_doc)
                    if isinstance(skills_doc, dict) and "skills" in skills_doc:
                        return path.name, len(skills_doc["skills"])
                return path.name, None
    return None, None


def build_experiment_record(
    campaign_root: Path,
    *,
    site: str,
    dirname: str,
) -> dict[str, Any] | None:
    matrix = _read_json(campaign_root / "matrix.json")
    manifest = _read_json(campaign_root / "manifest.json")
    campaign = _read_json(campaign_root / "campaign.json")

    if matrix is None and manifest is None and campaign is None:
        return None
    if matrix is None and manifest is None and campaign is not None:
        if not any(campaign_root.glob("hlsfactory_*")) and not any(
            campaign_root.glob("**/*_multistep_results.json")
        ):
            return None

    exp_id = f"{site}/{dirname}"
    setup = None
    model = None
    mode = None

    if isinstance(manifest, dict):
        setup = str(manifest.get("setup") or "") or None
        model = manifest.get("model")

    if isinstance(matrix, list) and matrix:
        first = matrix[0]
        mode = first.get("mode")
        if not model:
            model = first.get("model")
        if not setup and first.get("cell_dir"):
            cell = Path(str(first["cell_dir"]))
            if cell.name.count("__") >= 1:
                setup = cell.name.split("__", 1)[1] if "__" in cell.name else cell.name

    if isinstance(campaign, dict) and not model:
        pilot = (campaign.get("config") or {}).get("pilot") or {}
        model = pilot.get("model")

    workflow = parse_workflow(dirname, mode=str(mode) if mode else None)
    if _has_dataflow_reports(campaign_root) and workflow == "flash":
        workflow = "flash_dataflow"

    if isinstance(matrix, list):
        benches = _benches_from_matrix(matrix, campaign_root)
    elif isinstance(manifest, dict):
        benches = _benches_from_glob(campaign_root, manifest)
    else:
        benches = {}
        for path in sorted(campaign_root.glob("hlsfactory_*/*/*_multistep_results.json")):
            bench = path.parts[-3]
            entry = _bench_entry_from_result(bench, status="ok", result_path=path)
            benches[bench_short(bench)] = entry

    counts = _counts_from_benches(benches)
    geomean = geomean_from_bench_speedups(benches)
    latency_mean = mean_latency_from_benches(benches)
    cosim_speedup_geomean = geomean_cosim_speedup_from_benches(benches)
    cosim_status_counts = cosim_status_counts_from_benches(benches)
    skills_json, skills_count = _skills_meta_from_cells(campaign_root)

    label = dirname.replace("_", " ")
    if setup:
        label = f"{dirname} ({setup})"

    return {
        "id": exp_id,
        "label": label,
        "site": site,
        "stamp": parse_stamp(dirname),
        "dirname": dirname,
        "workflow": workflow,
        "cosim": parse_cosim(dirname, setup=setup, campaign=campaign if isinstance(campaign, dict) else None),
        "corpus": corpus_from_benches(benches),
        "skills_mode": parse_skills_mode(dirname, setup=setup, manifest=manifest if isinstance(manifest, dict) else None),
        "skills_json": skills_json,
        "skills_count": skills_count,
        "skill_variant": parse_skill_variant(dirname),
        "model": model,
        "path": str(campaign_root.resolve()),
        "planned": False,
        "notes": None,
        "counts": counts,
        "benches": benches,
        "geomean": geomean,
        "latency_mean": latency_mean,
        "cosim_speedup_geomean": cosim_speedup_geomean,
        "cosim_status_counts": cosim_status_counts,
    }


def _is_campaign_dir(path: Path) -> bool:
    if not path.is_dir():
        return False
    return any((path / name).is_file() for name in ("matrix.json", "manifest.json", "campaign.json"))


def scan_site(site_root: Path, *, site: str) -> list[dict[str, Any]]:
    if not site_root.is_dir():
        return []
    experiments: list[dict[str, Any]] = []
    for child in sorted(site_root.iterdir()):
        if not child.is_dir():
            continue
        if child.name in SKIP_TOP_LEVEL:
            continue
        if child.name.endswith(".md") or child.name.endswith(".tex"):
            continue
        if not _is_campaign_dir(child):
            continue
        record = build_experiment_record(child, site=site, dirname=child.name)
        if record:
            experiments.append(record)
    return experiments


def scan_all(repo_root: Path | None = None) -> list[dict[str, Any]]:
    root = repo_root or REPO
    experiments: list[dict[str, Any]] = []
    for site in SITE_DIRS:
        site_root = root / "artifacts" / site
        experiments.extend(scan_site(site_root, site=site))
    experiments.sort(key=lambda e: (e.get("site") or "", e.get("stamp") or "", e.get("id") or ""))
    return experiments


def load_registry(registry_path: Path | None = None) -> list[dict[str, Any]]:
    path = registry_path or (REPO / "experiments_registry.json")
    doc = _read_json(path)
    if not isinstance(doc, dict):
        return []
    entries = doc.get("experiments") or []
    return [e for e in entries if isinstance(e, dict)]


def merge_registry(
    scanned: list[dict[str, Any]],
    registry_entries: list[dict[str, Any]],
    *,
    repo_root: Path | None = None,
) -> list[dict[str, Any]]:
    root = repo_root or REPO
    by_id = {exp["id"]: exp for exp in scanned}
    merged = list(scanned)

    for entry in registry_entries:
        exp_id = str(entry.get("id") or "")
        if not exp_id:
            continue
        path_raw = entry.get("path")
        path = Path(str(path_raw)) if path_raw else None
        if path and not path.is_absolute():
            path = root / path

        if exp_id in by_id:
            existing = by_id[exp_id]
            for key in ("label", "notes", "workflow", "cosim", "corpus", "skills_mode"):
                if entry.get(key) and entry.get(key) != "unknown":
                    existing[key] = entry[key]
            if entry.get("notes"):
                existing["notes"] = entry["notes"]
            continue

        if path and path.is_dir():
            site = str(entry.get("site") or exp_id.split("/")[0])
            dirname = path.name
            record = build_experiment_record(path, site=site, dirname=dirname)
            if record:
                record["id"] = exp_id
                if entry.get("label"):
                    record["label"] = entry["label"]
                if entry.get("notes"):
                    record["notes"] = entry["notes"]
                merged.append(record)
                by_id[exp_id] = record
            continue

        planned = {
            "id": exp_id,
            "label": str(entry.get("label") or exp_id),
            "site": str(entry.get("site") or "unknown"),
            "stamp": entry.get("stamp"),
            "dirname": exp_id.split("/", 1)[-1] if "/" in exp_id else exp_id,
            "workflow": str(entry.get("workflow") or "unknown"),
            "cosim": str(entry.get("cosim") or "unknown"),
            "corpus": str(entry.get("corpus") or "unknown"),
            "skills_mode": str(entry.get("skills_mode") or "unknown"),
            "skills_json": entry.get("skills_json"),
            "skills_count": entry.get("skills_count"),
            "skill_variant": entry.get("skill_variant"),
            "model": entry.get("model"),
            "path": None,
            "planned": True,
            "notes": entry.get("notes"),
            "counts": {"total": 0, "ok": 0, "fail": 0},
            "benches": {},
            "geomean": {"n": 0, "best": None, "avg": None, "worst": None},
        }
        merged.append(planned)
        by_id[exp_id] = planned

    merged.sort(key=lambda e: (e.get("site") or "", e.get("stamp") or "", e.get("id") or ""))
    return merged


def build_facets(experiments: list[dict[str, Any]]) -> dict[str, list[str]]:
    keys = ("site", "workflow", "cosim", "corpus", "skills_mode", "skill_variant", "model")
    facets: dict[str, set[str]] = {k: set() for k in keys}
    for exp in experiments:
        for key in keys:
            value = exp.get(key)
            if value and str(value) != "unknown":
                facets[key].add(str(value))
    return {k: sorted(v) for k, v in facets.items()}


def all_benches_union(experiments: list[dict[str, Any]]) -> list[dict[str, str]]:
    seen: dict[str, str] = {}
    for exp in experiments:
        for short, info in (exp.get("benches") or {}).items():
            corpus = str(info.get("corpus") or bench_corpus(f"hlsfactory_{short}"))
            seen[short] = corpus
    return [{"bench": b, "corpus": seen[b]} for b in sorted(seen)]


def build_index(
    repo_root: Path | None = None,
    registry_path: Path | None = None,
) -> dict[str, Any]:
    root = repo_root or REPO
    scanned = scan_all(root)
    registry = load_registry(registry_path or (root / "experiments_registry.json"))
    experiments = merge_registry(scanned, registry, repo_root=root)
    return {
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "repo_root": str(root.resolve()),
        "experiments": experiments,
        "facets": build_facets(experiments),
        "all_benches": all_benches_union(experiments),
    }


def compare_experiments(
    experiments: list[dict[str, Any]],
    *,
    ids: list[str],
    bench_filter: list[str] | None = None,
    include_planned: bool = False,
) -> dict[str, Any]:
    by_id = {exp["id"]: exp for exp in experiments}
    selected = []
    for exp_id in ids:
        exp = by_id.get(exp_id)
        if not exp:
            continue
        if exp.get("planned") and not include_planned:
            continue
        selected.append(exp)

    bench_set = set(bench_filter) if bench_filter else None
    summaries = []
    for exp in selected:
        gm = geomean_from_bench_speedups(exp.get("benches") or {}, bench_filter=bench_set)
        summaries.append({
            "id": exp["id"],
            "label": exp.get("label"),
            "geomean": gm,
            "latency_mean": mean_latency_from_benches(
                exp.get("benches") or {},
                bench_filter=bench_set,
            ),
            "cosim_speedup_geomean": geomean_cosim_speedup_from_benches(
                exp.get("benches") or {},
                bench_filter=bench_set,
            ),
        })

    per_bench_rows: list[dict[str, Any]] = []
    if bench_set is None:
        union_benches: set[str] = set()
        for exp in selected:
            union_benches.update((exp.get("benches") or {}).keys())
        bench_list = sorted(union_benches)
    else:
        bench_list = sorted(bench_set)

    for bench in bench_list:
        row: dict[str, Any] = {"bench": bench, "experiments": {}}
        for exp in selected:
            info = (exp.get("benches") or {}).get(bench)
            if not info:
                row["experiments"][exp["id"]] = None
                continue
            row["experiments"][exp["id"]] = {
                "status": info.get("status"),
                "speedup": info.get("speedup"),
                "latency": info.get("latency"),
                "cosim": info.get("cosim"),
            }
        per_bench_rows.append(row)

    filtered_geomean = {}
    if len(selected) == 2:
        a, b = selected[0], selected[1]
        for kind in ("best", "avg", "worst"):
            ga = geomean_from_bench_speedups(
                a.get("benches") or {}, bench_filter=bench_set,
            ).get(kind)
            gb = geomean_from_bench_speedups(
                b.get("benches") or {}, bench_filter=bench_set,
            ).get(kind)
            if ga and gb and gb > 0:
                filtered_geomean[f"delta_{kind}"] = ga / gb

    return {
        "experiment_ids": [e["id"] for e in selected],
        "bench_filter": sorted(bench_set) if bench_set else None,
        "summaries": summaries,
        "per_bench": per_bench_rows,
        "pairwise_delta": filtered_geomean if filtered_geomean else None,
    }
