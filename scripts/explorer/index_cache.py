"""Disk-backed cache for the experiment explorer catalog index."""

from __future__ import annotations

import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from catalog import SITE_DIRS, SKIP_TOP_LEVEL, _is_campaign_dir, build_index

CACHE_VERSION = 1
# Result markers only — queue.db / campaign.json change during live runs.
FINGERPRINT_MARKERS = ("matrix.json", "manifest.json")


def default_cache_dir(repo_root: Path) -> Path:
    import os

    env = os.environ.get("EXPLORER_CACHE_DIR", "").strip()
    if env:
        return Path(env).expanduser().resolve()
    return (repo_root / "artifacts" / ".explorer").resolve()


def cache_paths(repo_root: Path) -> dict[str, Path]:
    root = default_cache_dir(repo_root)
    return {
        "dir": root,
        "index": root / "catalog_index.json",
        "meta": root / "catalog_index.meta.json",
    }


def collect_fingerprint_sources(
    repo_root: Path,
    registry_path: Path,
) -> list[list[Any]]:
    """Cheap scan of artifact mtimes — used to detect catalog changes."""
    sources: list[list[Any]] = []
    if registry_path.is_file():
        st = registry_path.stat()
        sources.append(["registry", str(registry_path.resolve()), st.st_mtime_ns, st.st_size])

    for site in SITE_DIRS:
        site_root = repo_root / "artifacts" / site
        if not site_root.is_dir():
            continue
        for child in sorted(site_root.iterdir()):
            if not child.is_dir():
                continue
            if child.name in SKIP_TOP_LEVEL or child.name.startswith("."):
                continue
            if not _is_campaign_dir(child):
                continue
            rel = f"{site}/{child.name}"
            st_dir = child.stat()
            sources.append(["campaign", rel, st_dir.st_mtime_ns, 0])
            for name in FINGERPRINT_MARKERS:
                path = child / name
                if not path.is_file():
                    continue
                st = path.stat()
                sources.append(["marker", f"{rel}/{name}", st.st_mtime_ns, st.st_size])
    return sources


def fingerprint_sources(sources: list[list[Any]]) -> str:
    payload = json.dumps(sources, separators=(",", ":"), sort_keys=True).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def compute_fingerprint(repo_root: Path, registry_path: Path) -> str:
    return fingerprint_sources(collect_fingerprint_sources(repo_root, registry_path))


def _read_json(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        doc = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return doc if isinstance(doc, dict) else None


def load_disk_cache(repo_root: Path) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    paths = cache_paths(repo_root)
    meta = _read_json(paths["meta"])
    if not meta or meta.get("version") != CACHE_VERSION:
        return None, meta
    index = _read_json(paths["index"])
    if not index:
        return None, meta
    return index, meta


def save_disk_cache(
    repo_root: Path,
    *,
    index: dict[str, Any],
    fingerprint: str,
    build_ms: float,
) -> None:
    paths = cache_paths(repo_root)
    paths["dir"].mkdir(parents=True, exist_ok=True)
    meta = {
        "version": CACHE_VERSION,
        "fingerprint": fingerprint,
        "built_at": datetime.now(timezone.utc).isoformat(),
        "build_ms": round(build_ms, 1),
        "experiment_count": len(index.get("experiments") or []),
    }
    paths["meta"].write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    paths["index"].write_text(json.dumps(index, indent=2) + "\n", encoding="utf-8")


def build_and_cache_index(
    repo_root: Path,
    registry_path: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    t0 = time.monotonic()
    fingerprint = compute_fingerprint(repo_root, registry_path)
    index = build_index(repo_root, registry_path)
    build_ms = (time.monotonic() - t0) * 1000.0
    save_disk_cache(repo_root, index=index, fingerprint=fingerprint, build_ms=build_ms)
    meta = _read_json(cache_paths(repo_root)["meta"]) or {}
    return index, {
        "source": "rebuild",
        "fingerprint": fingerprint,
        "build_ms": build_ms,
        **meta,
    }


_MEMORY: dict[str, Any] = {"ts": 0.0, "fingerprint": None, "payload": None, "meta": None}


def get_index(
    repo_root: Path,
    registry_path: Path,
    *,
    cache_sec: float = 30.0,
    force_refresh: bool = False,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Return catalog index and cache metadata (source: memory|disk|rebuild)."""
    repo_root = repo_root.resolve()
    registry_path = registry_path.resolve()
    fingerprint = compute_fingerprint(repo_root, registry_path)
    now = time.monotonic()

    if (
        not force_refresh
        and _MEMORY["payload"] is not None
        and _MEMORY["fingerprint"] == fingerprint
        and (now - float(_MEMORY["ts"])) < cache_sec
    ):
        meta = dict(_MEMORY.get("meta") or {})
        meta["source"] = "memory"
        meta["fingerprint"] = fingerprint
        return _MEMORY["payload"], meta

    if not force_refresh:
        disk_index, disk_meta = load_disk_cache(repo_root)
        if (
            disk_index is not None
            and disk_meta is not None
            and disk_meta.get("fingerprint") == fingerprint
        ):
            meta = {
                "source": "disk",
                "fingerprint": fingerprint,
                "built_at": disk_meta.get("built_at"),
                "build_ms": disk_meta.get("build_ms"),
                "experiment_count": disk_meta.get("experiment_count"),
            }
            _MEMORY["ts"] = now
            _MEMORY["fingerprint"] = fingerprint
            _MEMORY["payload"] = disk_index
            _MEMORY["meta"] = meta
            return disk_index, meta

    index, meta = build_and_cache_index(repo_root, registry_path)
    _MEMORY["ts"] = now
    _MEMORY["fingerprint"] = fingerprint
    _MEMORY["payload"] = index
    _MEMORY["meta"] = meta
    return index, meta


def prewarm_index_cache(repo_root: Path, registry_path: Path) -> dict[str, Any]:
    """Build or validate disk cache (for startup scripts)."""
    fingerprint = compute_fingerprint(repo_root, registry_path)
    disk_index, disk_meta = load_disk_cache(repo_root)
    if disk_index is not None and disk_meta and disk_meta.get("fingerprint") == fingerprint:
        return {
            "source": "disk",
            "fingerprint": fingerprint,
            "experiment_count": disk_meta.get("experiment_count"),
            "built_at": disk_meta.get("built_at"),
        }
    _, meta = build_and_cache_index(repo_root, registry_path)
    return meta
