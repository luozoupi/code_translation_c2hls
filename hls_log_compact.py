"""Compact oversized Vitis HLS / XSim simulator logs after cosim."""

from __future__ import annotations

import os
import re
from collections import deque
from pathlib import Path
from typing import Any

TIME_LINE_RE = re.compile(r"^Time:\s*", re.IGNORECASE)
COMPACT_BANNER = "[c2hls] compacted simulator log"


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name, "1" if default else "0").strip().lower()
    return raw not in ("0", "false", "no", "off")


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name, "").strip()
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def cosim_log_compact_enabled() -> bool:
    return _env_bool("C2HLS_COSIM_COMPACT_LOGS", default=True)


def _is_time_line(line: str) -> bool:
    return bool(TIME_LINE_RE.match(line))


def _is_warning_line(line: str) -> bool:
    return line.startswith("Warning:")


class _LineReader:
    def __init__(self, handle):
        self._handle = handle
        self._pushback: str | None = None

    def readline(self) -> str:
        if self._pushback is not None:
            line, self._pushback = self._pushback, None
            return line
        return self._handle.readline()

    def unread(self, line: str) -> None:
        self._pushback = line


def compact_simulator_log_file(
    path: Path,
    *,
    min_bytes: int | None = None,
    header_lines: int | None = None,
    footer_lines: int | None = None,
    max_warnings: int | None = None,
) -> dict[str, Any]:
    """Replace *path* in place with a compacted log when it exceeds *min_bytes*."""
    path = Path(path)
    result: dict[str, Any] = {
        "path": str(path),
        "compacted": False,
        "original_bytes": 0,
        "compacted_bytes": 0,
        "original_lines": 0,
        "warnings_kept": 0,
        "warnings_total": 0,
    }
    if not path.is_file():
        return result

    min_bytes = min_bytes if min_bytes is not None else _env_int("C2HLS_COSIM_COMPACT_LOG_MIN_BYTES", 1 << 20)
    header_lines = header_lines if header_lines is not None else _env_int("C2HLS_COSIM_COMPACT_LOG_HEADER_LINES", 80)
    footer_lines = footer_lines if footer_lines is not None else _env_int("C2HLS_COSIM_COMPACT_LOG_FOOTER_LINES", 50)
    max_warnings = max_warnings if max_warnings is not None else _env_int("C2HLS_COSIM_COMPACT_LOG_MAX_WARNINGS", 10)

    original_bytes = path.stat().st_size
    result["original_bytes"] = original_bytes
    if original_bytes < min_bytes:
        result["compacted_bytes"] = original_bytes
        return result

    header: list[str] = []
    footer: deque[str] = deque(maxlen=max(footer_lines, 1))
    warnings: deque[list[str]] = deque(maxlen=max(max_warnings, 1))
    pending_time: str | None = None
    total_lines = 0
    warnings_total = 0

    def append_footer(line: str) -> None:
        footer.append(line)

    def record_warning(block: list[str]) -> None:
        nonlocal warnings_total
        warnings_total += 1
        warnings.append(block)
        for item in block:
            append_footer(item)

    with path.open("r", encoding="utf-8", errors="replace") as handle:
        reader = _LineReader(handle)
        while True:
            line = reader.readline()
            if not line:
                break
            total_lines += 1

            if len(header) < header_lines:
                header.append(line)
                continue

            if _is_time_line(line):
                pending_time = line
                append_footer(line)
                continue

            if _is_warning_line(line):
                block: list[str] = []
                if pending_time is not None:
                    block.append(pending_time)
                    pending_time = None
                block.append(line)
                next_line = reader.readline()
                if next_line:
                    total_lines += 1
                    if _is_time_line(next_line):
                        block.append(next_line)
                    else:
                        reader.unread(next_line)
                record_warning(block)
                continue

            pending_time = None
            append_footer(line)

    result["original_lines"] = total_lines
    result["warnings_total"] = warnings_total
    result["warnings_kept"] = len(warnings)

    omitted_lines = max(0, total_lines - len(header) - len(footer))
    compacted_parts = [
        *header,
        (
            f"=== {COMPACT_BANNER}: omitted ~{omitted_lines} middle lines "
            f"({original_bytes} bytes before compaction); "
            f"showing last {len(warnings)} of {warnings_total} simulator warnings ===\n"
        ),
    ]
    for block in warnings:
        compacted_parts.extend(block)
    compacted_parts.append(
        f"=== {COMPACT_BANNER}: log tail ({len(footer)} lines) ===\n"
    )
    compacted_parts.extend(footer)

    tmp_path = path.with_suffix(path.suffix + ".compact.tmp")
    with tmp_path.open("w", encoding="utf-8", errors="replace") as out:
        out.writelines(compacted_parts)
    compacted_bytes = tmp_path.stat().st_size
    if compacted_bytes >= original_bytes:
        tmp_path.unlink(missing_ok=True)
        result["compacted_bytes"] = original_bytes
        return result

    os.replace(tmp_path, path)
    result["compacted"] = True
    result["compacted_bytes"] = compacted_bytes
    return result


def cosim_work_dir_log_paths(work_dir: Path) -> list[Path]:
    paths = [work_dir / "logs" / "hls_run_tcl.log"]
    if work_dir.is_dir():
        paths.extend(sorted(work_dir.glob("hls_proj/**/xsim.log")))
    return [path for path in paths if path.is_file()]


def compact_cosim_work_dir_logs(work_dir: str | Path) -> list[dict[str, Any]]:
    """Compact known large Vitis/XSim logs under a cosim work directory."""
    if not cosim_log_compact_enabled():
        return []
    root = Path(work_dir)
    return [compact_simulator_log_file(path) for path in cosim_work_dir_log_paths(root)]
