"""Command-line entrypoint and conservative identities for CLI resume."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

from hlsmith.resources import project_root


_ROOT_RUNTIME_MODULES = (
    "c2hls", "prompt_c2hls", "paper_baselines", "experiment_budget",
    "experiment_telemetry", "sigmetrics_main_policy", "run_agentic_sweep",
    "export_schema_jsonl", "export_rl_corpus", "export_ml4accel_points",
    "validate_corpus", "verify_corpus_stability", "sanitize_saved_results", "report",
)
_CONFIG_FILE_CONTROLS = (
    "C2HLS_SKILL_LIBRARY_PATH", "C2HLS_SKILLS_PATH", "C2HLS_SKILLS_EXTRA",
    "C2HLS_HLSFACTORY_SHAPE_REGISTRY", "C2HLS_PHASE_B_SEED_MANIFEST",
    "C2HLS_ARCHIVE_PATH_MAP",
    "C2HLS_VITIS_SETTINGS", "C2HLS_EMU_ENV_SCRIPT", "VITIS_SETTINGS",
)
_OTHER_CONTROLS = (
    "OPENAI_BASE_URL", "DEEPSEEK_BASE_URL", "ANTHROPIC_BASE_URL",
    "VITIS_SETTINGS", "VITIS_HLS_TCL_RUNNER",
)
_NON_RUNTIME_DIRECTORIES = {"__pycache__", ".cache", "data", "build", "artifacts", "results", "outputs", "logs"}


def _digest(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _file_identity(path: Path):
    return _digest(path.read_bytes()) if path.is_file() else None


def _benchmark_identity(root: Path) -> dict:
    """Hash declared inputs, without scanning generated experiment trees."""
    metadata_path = root / "metadata.json"
    metadata = json.loads(metadata_path.read_text())
    names = {"metadata.json", "plain.cpp"}
    for key in ("plain_c_file", "header_file", "testbench_file", "cosim_testbench_file",
                "gold_hls_source_file", "gold_hls_baseline_file", "kernel_file"):
        if metadata.get(key):
            names.add(str(metadata[key]))
    for key in ("support_files", "cosim_support_files"):
        names.update(str(name) for name in metadata.get(key) or [])
    for variant in metadata.get("variants") or []:
        for key in ("file", "header_file"):
            if variant.get(key):
                names.add(str(variant[key]))
    names.update(path.name for path in root.glob("*.h"))
    names.update(path.relative_to(root).as_posix()
                 for path in (root / "support").rglob("*") if path.is_file())
    return {name: _file_identity(root / name) for name in sorted(names)}


def resume_identity(benchmark_dir, configuration, *, environ=None, package_root=None, data_root=None):
    """Identify inputs and actual implementation bytes before a CLI run.

    ``package_root`` locates code independently of ``project_root()``, which
    points at packaged data in an installed wheel. Environment values are
    represented only by hashes. Mutable generated outputs and billing/cache
    payloads are deliberately outside this source/input identity.
    """
    environment = os.environ if environ is None else environ
    package = Path(package_root) if package_root is not None else Path(__file__).resolve().parent
    data = Path(data_root) if data_root is not None else project_root()
    runtime = {"hlsmith/" + path.relative_to(package).as_posix(): _file_identity(path)
               for path in sorted(package.rglob("*.py"))
               if not _NON_RUNTIME_DIRECTORIES.intersection(path.relative_to(package).parts)}
    aliases_path = package / "legacy_modules.json"
    aliases = json.loads(aliases_path.read_text()) if aliases_path.is_file() else {}
    runtime["hlsmith/legacy_modules.json"] = _file_identity(aliases_path)
    for name in sorted(set(_ROOT_RUNTIME_MODULES).union(aliases)):
        runtime[name + ".py"] = _file_identity(package.parent / (name + ".py"))
    dataset = package.parent / "dataset_pipeline"
    runtime.update({"dataset_pipeline/" + path.relative_to(dataset).as_posix(): _file_identity(path)
                    for path in sorted(dataset.rglob("*.py")) if "__pycache__" not in path.parts})
    resources = {}
    for directory in ("configs", "skills", "skill_v2", "skill_v3_no_rmw_fc27133",
                      "hls_full_optimization_skills_schema_1_1_package"):
        for path in sorted((data / directory).rglob("*.json")):
            resources[directory + "/" + path.relative_to(data / directory).as_posix()] = _file_identity(path)
    for name in _CONFIG_FILE_CONTROLS:
        for index, value in enumerate(environment.get(name, "").split(os.pathsep)):
            if value:
                # Persist a logical label rather than the possibly private path.
                resources[f"configured:{name}:{index}"] = _file_identity(Path(value).expanduser())
    controls = {name: _digest(str(value).encode()) for name, value in sorted(environment.items())
                if name.startswith("C2HLS_") or name in _OTHER_CONTROLS}
    payload = {"configuration": json.loads(json.dumps(configuration)), "inputs": _benchmark_identity(Path(benchmark_dir)),
               "runtime": runtime, "resources": resources, "control_hashes": controls}
    serialized = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return {"schema_version": "hlsmith.cli-resume.v1", "sha256": _digest(serialized), "payload": payload}


def main(argv=None):
    from hlsmith.pipeline import main as pipeline_main
    return pipeline_main(argv)
