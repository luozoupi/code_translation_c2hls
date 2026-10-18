"""Resolve packaged inputs independently of the current working directory."""
from pathlib import Path
import json
import os


def project_root():
    configured = os.getenv("C2HLS_DATA_ROOT")
    if configured:
        return Path(configured).expanduser().resolve()
    source = Path(__file__).resolve().parent.parent
    if (source / "pyproject.toml").is_file():
        return source
    return Path(__file__).resolve().parent / "data"


def output_root():
    configured = os.getenv("C2HLS_OUTPUT_ROOT")
    if configured:
        return Path(configured).expanduser().resolve()
    source = Path(__file__).resolve().parent.parent
    return source if (source / "pyproject.toml").is_file() else Path.cwd()


def resolve_historical_path(path, mapping_file=None):
    """Resolve a historical location without modifying its original manifest."""
    value = Path(path).expanduser()
    mapping_file = mapping_file or os.getenv("C2HLS_ARCHIVE_PATH_MAP")
    if mapping_file:
        payload = json.loads(Path(mapping_file).read_text())
        entries = payload.get("paths", payload)
        for old, new in sorted(entries.items(), key=lambda item: -len(item[0])):
            try:
                relative = value.relative_to(Path(old))
            except ValueError:
                continue
            return Path(new).expanduser() / relative
    return value
