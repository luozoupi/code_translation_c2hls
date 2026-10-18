"""Register and run the prospective, paired DeepSeek cleanup comparison.

Registration and estimation never load credentials or launch tools. Historical
results supply workload estimates, not new observations or guaranteed costs.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shlex
import shutil
import signal
import subprocess
import sys
import tempfile
import time
import xml.etree.ElementTree as ET

from hlsmith.research.cleanup_statistics import summarize_cleanup_statistics

SCHEMA = "hlsmith.cleanup-comparison.v1"
PROVIDER = {
    "provider": "deepseek", "model": "deepseek-flash",
    "declared_release": "DeepSeek-V4.1-Flash-1017",
    "endpoint": "https://api.deepseek.com", "thinking": "disabled",
    "immutable_weights_claim": False, "seed_supported": False,
    "rates_per_million": {"input": 0.30, "cached_input": 0.006, "output": 1.20},
    "rates_captured_on": "2026-10-18", "rates_scope": "peak USD; recheck before launch",
    "rates_source": "https://api-docs.deepseek.com/quick_start/pricing/",
}
BENCHMARKS = tuple("hlsfactory_" + name for name in (
    "gemm", "2mm", "3mm", "doitgen", "syrk", "syr2k", "symm", "trmm", "mvt",
    "gemver", "gesummv", "atax", "bicg", "cholesky", "correlation", "covariance",
    "durbin", "fdtd-2d", "floyd-warshall", "gramschmidt", "jacobi-1d",
    "jacobi-2d", "lu", "ludcmp", "nussinov", "trisolv"))
ARMS = (
    {"id": "oneshot", "entrypoint": "run_benchmark", "skills": "off", "turns": 1,
     "quality_repair_turns": 0, "generation_timeout_seconds": 7200},
    *({"id": mode + "_" + policy,
       "entrypoint": "run_benchmark" if mode == "flash" else "run_benchmark_multistep",
       "skills": policy, "turns": 5, "quality_repair_turns": 4,
       "generation_timeout_seconds": 14400}
      for mode in ("flash", "ms") for policy in ("noskills", "curated", "allpos")),
)
COHORTS = (
    {"id": "frozen-paper", "source": "frozen", "prompt_profile": "paper-sigmetrics-2026"},
    {"id": "clean-paper", "source": "clean", "prompt_profile": "paper-sigmetrics-2026"},
    {"id": "clean-researcher", "source": "clean", "prompt_profile": "researcher-v1"},
)
TREATMENT_WARNING = (
    "Seven requested configurations do not imply seven active treatments: v6 "
    "single-shot run_benchmark does not inject recipes for curated/all-positive "
    "flags. Preserve this behavior in the cleanup comparison; do not attribute "
    "single-shot differences to recipe guidance."
)
# Native helper bounds compilation and execution separately. Both the plain
# oracle and selected candidate run, hence four 180-second native allowances.
VALIDATION_SECONDS = 4 * 180 + 300 + 3600


def now():
    return datetime.now(timezone.utc).isoformat()


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def content_sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def measured_cycles(result):
    value = result.get('kernel_runtime_cycles')
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return int(value) if math.isfinite(value) and value > 0 and float(value).is_integer() else None


def independently_passed(result, golden_sha256=None):
    return (result.get('success') is True and result.get('passed') is True
            and isinstance(result.get('correctness'), dict)
            and result['correctness'].get('passed') is True
            and (golden_sha256 is None or result.get('golden_output_sha256') == golden_sha256))


def test_timed_out(result):
    error = str(result.get('error', '')).lower()
    return (result.get('timed_out') is True or result.get('status') == 'timeout'
            or 'timeout' in error or 'timed out' in error)


def selected_feasibility(runtime, report, target):
    """Qualify recorded synthesis evidence after the independent native gate."""
    result = runtime._paper_candidate_feasibility(report, csim={'passed': True},
                 part=target['part'], clock_ns=target['clock_ns'])
    reasons = list(result['reasons'])
    for key in ('bram', 'dsp', 'ff', 'lut', 'uram', 'latency_cycles_worst'):
        value = report.get(key)
        try:
            number = float(value)
            valid = (not isinstance(value, bool) and math.isfinite(number)
                     and number.is_integer() and number >= 0
                     and (key != 'latency_cycles_worst' or number > 0))
        except (TypeError, ValueError, OverflowError):
            valid = False
        if not valid:
            reasons.append('invalid_or_missing_' + key)
    requested = report.get('requested_clock_period_ns')
    try:
        matched = (not isinstance(requested, bool) and math.isfinite(float(requested))
                   and abs(float(requested) - target['clock_ns']) <= 1e-9)
    except (TypeError, ValueError, OverflowError):
        matched = False
    if not matched:
        reasons.append('requested_clock_identity_mismatch')
    result.update(feasible=not reasons, reasons=reasons,
                  correctness_evidence='independent_native_output_check')
    return result


def fresh_rtl_synthesis(evaluator, rtl):
    """Read the fresh synthesis already executed by co-simulation, without rerunning it."""
    work = rtl.get('work_dir')
    if not work:
        return {}, {'status': 'missing', 'reason': 'RTL tool workspace not recorded'}
    directory = Path(work) / 'hls_proj' / 'sol1' / 'syn' / 'report'
    xml = directory / 'csynth.xml'
    report = directory / 'csynth.rpt'
    source = xml if xml.is_file() else report
    if not source.is_file():
        return {}, {'status': 'missing', 'reason': 'fresh RTL synthesis report missing'}
    try:
        parsed = (evaluator.parse_synthesis_xml(str(xml)) if xml.is_file()
                  else evaluator.parse_synthesis_report(report.read_text()))
    except (OSError, ValueError, TypeError, OverflowError) as error:
        return {}, {'status': 'invalid', 'reason': type(error).__name__, 'report_sha256': sha(source)}
    evidence = {'status': 'recorded', 'path': str(source.resolve()), 'sha256': sha(source),
                'scope': 'fresh synthesis inside the same RTL invocation'}
    # Vitis 2023.2 puts the requested period under UserAssignments, whereas
    # the legacy parser looks under SummaryOfTimingAnalysis. Bind the missing
    # field to actual tool inputs/report values, never the campaign setting.
    clocks = []
    try:
        if xml.is_file():
            root = ET.parse(xml).getroot()
            value = root.findtext('UserAssignments/TargetClockPeriod')
            if value is not None:
                unit = root.findtext('UserAssignments/unit')
                if unit is not None and unit.strip() != 'ns':
                    raise ValueError('unsupported XML target clock unit')
                clocks.append({'kind': 'vitis_xml', 'path': str(xml.resolve()), 'sha256': sha(xml),
                               'field': 'UserAssignments/TargetClockPeriod', 'period_ns': positive(value)})
        if parsed.get('requested_clock_period_ns') is not None:
            clocks.append({'kind': 'parsed_synthesis_report', 'path': str(source.resolve()), 'sha256': sha(source),
                           'field': 'requested_clock_period_ns', 'period_ns': positive(parsed['requested_clock_period_ns'])})
        tcl = Path(work) / 'run_cosim.tcl'
        if tcl.is_file():
            commands = [line.strip() for line in tcl.read_text().splitlines()
                        if re.match(r'^\s*create_clock\b', line)]
            # The frozen and cleaned evaluators emit this literal command.
            # Reject dynamic/multiple clocks rather than guessing Tcl values.
            if len(commands) != 1:
                raise ValueError('missing or ambiguous Tcl create_clock command')
            match = re.fullmatch(r'create_clock\s+-period\s+([0-9]+(?:\.[0-9]*)?(?:[eE][+-]?[0-9]+)?)\s+-name\s+default', commands[0])
            if match is None:
                raise ValueError('unsupported Tcl create_clock command')
            clocks.append({'kind': 'vitis_tcl', 'path': str(tcl.resolve()), 'sha256': sha(tcl),
                           'command': commands[0], 'period_ns': positive(match.group(1))})
        if not clocks:
            raise ValueError('requested clock absent from tool report and Tcl')
        requested = clocks[0]['period_ns']
        if any(abs(clock['period_ns'] - requested) > 1e-9 for clock in clocks):
            raise ValueError('requested clock disagrees between tool report and Tcl')
        parsed['requested_clock_period_ns'] = requested
        estimated = parsed.get('estimated_clock_period_ns')
        if estimated is not None:
            parsed['slack_ns'] = round(requested - positive(estimated), 3)
        evidence['requested_clock'] = {'status': 'verified', 'period_ns': requested, 'sources': clocks}
    except (OSError, ValueError, TypeError, OverflowError, ET.ParseError) as error:
        parsed['requested_clock_period_ns'] = None
        parsed['slack_ns'] = None
        evidence.update(status='invalid', reason=str(error), requested_clock={'status': 'invalid', 'sources': clocks})
    return parsed, evidence


def read(path):
    return json.loads(Path(path).read_text())


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    temporary.replace(path)


def positive(value):
    if isinstance(value, bool):
        raise ValueError("limits must be positive finite numbers")
    number = float(value)
    if not math.isfinite(number) or number <= 0:
        raise ValueError("limits must be positive finite numbers")
    return number


def make_jobs():
    """Rotate arm and cohort order without using any observed outcome."""
    jobs = []
    for repeat in range(1, 4):
        for index, benchmark in enumerate(BENCHMARKS):
            for position in range(len(ARMS)):
                arm = ARMS[(position + index + repeat - 1) % len(ARMS)]
                for offset in range(len(COHORTS)):
                    cohort = COHORTS[(offset + index + position + repeat - 1) % len(COHORTS)]
                    jobs.append({"id": f"{benchmark}__{arm['id']}__r{repeat}__{cohort['id']}",
                                 "benchmark": benchmark, "arm": arm['id'],
                                 "cohort": cohort['id'], "repeat": repeat,
                                 "recipe_injection_eligible": arm['entrypoint'] == 'run_benchmark_multistep' and arm['skills'] not in {'off', 'noskills'},
                                 "effective_single_shot_skill_treatment": False if arm['entrypoint'] == 'run_benchmark' else None,
                                 "maximum_job_seconds": arm['generation_timeout_seconds'] + VALIDATION_SECONDS})
    return jobs


def source_manifest(root):
    root = Path(root).resolve()
    paths = list(root.glob("*.py"))
    for folder in ("hlsmith", "configs", "skills", "skill_v2", "skill_v3_no_rmw_fc27133"):
        paths += [p for p in (root / folder).rglob("*")
                  if p.is_file() and p.suffix in {".py", ".json", ".md", ".txt"}
                  and "__pycache__" not in p.parts]
    if not (root / "c2hls.py").is_file():
        raise ValueError(f"source lacks compatible c2hls.py entrypoint: {root}")
    return {str(p.resolve()): sha(p) for p in sorted(set(paths))}


def public_inputs(root, output):
    """Copy only public files; never expose expert code or reduced RTL inputs."""
    records = []
    whitelist = ("benchmark", "source_repo", "plain_c_file", "header_file", "testbench_file",
                 "kernel_top", "hls_top", "translated_hls_top", "support_files", "include_dirs",
                 "cosim_depths", "part", "clock_ns")
    for benchmark in BENCHMARKS:
        source = Path(root) / benchmark
        original = read(source / "metadata.json")
        meta = {key: original[key] for key in whitelist if key in original}
        meta.update(benchmark=benchmark, supports_csim=True, supports_cosim=False,
                    validation_input_scope="unchanged_public_testbench_no_reduced_cosim_overrides",
                    study_role="prospective_development_corpus_not_held_out")
        names = {meta.get("plain_c_file") or "plain.cpp", meta.get("header_file") or "kernel.h",
                 meta.get("testbench_file") or "testbench.cpp"}
        names.update(meta.get("support_files") or [])
        names.update(p.relative_to(source).as_posix() for p in (source / "support").rglob("*") if p.is_file())
        destination = output / benchmark
        destination.mkdir(parents=True)
        for name in sorted(names):
            rel = Path(name)
            if rel.is_absolute() or ".." in rel.parts or "gold" in rel.name.lower():
                raise ValueError(f"unsafe public input: {benchmark}/{name}")
            origin = source / rel
            target = destination / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(origin, target)
        save(destination / "metadata.json", meta)
        records.append({"benchmark": benchmark, "path": str(destination.resolve()),
                        "original_metadata_sha256": sha(source / "metadata.json"),
                        "files": {str(p.resolve()): sha(p) for p in destination.rglob("*") if p.is_file()}})
    return records


def historical_estimate(payload):
    """Reprice the recorded workload at current peak rates, without cache credit."""
    rows = []
    for path in sorted((Path(payload) / "results").glob("*/matrix.json")):
        rows += read(path)
    if len(rows) != 546:
        raise ValueError(f"expected original 546-cell archive, found {len(rows)}")
    tokens_in = sum(int((r.get("summary", {}).get("llm_usage") or {}).get("input_tokens") or 0) for r in rows)
    tokens_out = sum(int((r.get("summary", {}).get("llm_usage") or {}).get("output_tokens") or 0) for r in rows)
    missing = sum(bool((r.get("summary", {}).get("llm_usage") or {}).get("usage_missing_calls")) for r in rows)
    missing_summaries = sum(not (r.get('summary') or {}).get('llm_usage') for r in rows)
    seconds = sum(float(r.get("wallclock_s") or 0) for r in rows)
    # The historical archive has 3 models x 182 cells. The new study has
    # 3 cohorts x 3 repeats x 182 cells, hence three times that recorded work.
    scale = len(make_jobs()) / len(rows)
    rate = PROVIDER["rates_per_million"]
    return {"historical_rows": len(rows), "historical_models": "Sonnet/Opus/GPT-5.5, not DeepSeek",
            "input_tokens": round(tokens_in * scale), "output_tokens": round(tokens_out * scale),
            "api_estimate_usd_without_cache_credit": (tokens_in * rate['input'] + tokens_out * rate['output']) * scale / 1e6,
            "generation_job_hours": seconds * scale / 3600,
            "source_rows_with_missing_usage": missing,
            "source_rows_without_usage_summary": missing_summaries,
            "source_matrix_hashes": {str(p.resolve()): sha(p) for p in sorted((Path(payload) / 'results').glob('*/matrix.json'))},
            "qualification": "workload extrapolation across different models; excludes new final validation; not a cost/runtime guarantee"}


def register(args):
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    sources = {"frozen": str(args.frozen_source.resolve()), "clean": str(args.clean_source.resolve())}
    identities = {name: source_manifest(path) for name, path in sources.items()}
    inputs = public_inputs(args.benchmark_root, output / "inputs")
    skill = output / "skills.json"
    shutil.copy2(args.skill_library, skill)
    registry = output / "output_shapes.json"
    shutil.copy2(args.shape_registry, registry)
    estimates = historical_estimate(args.historical_payload)
    jobs = make_jobs()
    manifest = {"schema_version": SCHEMA, "created_at": now(), "provider": PROVIDER,
                "sources": sources, "source_files": identities, "cohorts": COHORTS,
                "arms": ARMS, "benchmark_entries": inputs, "jobs": jobs,
                "skill_library": str(skill), "skill_library_sha256": sha(skill),
                "shape_registry": str(registry), "shape_registry_sha256": sha(registry),
                "target": {"vitis_version": "2023.2", "part": "xcu280-fsvh2892-2L-e", "clock_ns": 3.33},
                "max_output_tokens_per_request": 8192, "synthesis_timeout_seconds": 2400,
                "decoding": {"temperature": 0, "top_p_requested": 1, "seed_supported": False, "thinking": "disabled"},
                "validation_timeouts": {"native_compile_or_execute": 180, "native_pair_outer": 720, "csim": 300, "cosim": 3600},
                "estimates": estimates, "maximum_sequential_job_hours": sum(j['maximum_job_seconds'] for j in jobs) / 3600,
                "registration_only": True, "api_or_compute_caps": "required explicitly at launch; no paid work authorized by registration",
                "method": "prospective reference-blind public-only seven-arm comparison; not exact historical replay",
                "budget_policy": "preserve original per-arm turns/quality limits; no invented five-call/five-synthesis search cap",
                "validation_policy": "freeze selected source before independent native/CSim/RTL; never reselect after validation",
                "implementation_note": "historical flash means run_benchmark; it is not multistep strategy=flash"}
    manifest['treatment_warning'] = TREATMENT_WARNING
    save(output / "manifest.json", manifest)
    save(output / "state.json", {"manifest_sha256": sha(output / "manifest.json"), "status": "registered",
                                "jobs": [{"id": j['id'], "status": "pending"} for j in jobs], "job_seconds_used": 0})
    print(json.dumps({"registered": str(output), "jobs": len(jobs), "per_cohort": 546,
                      "estimates": {k: v for k, v in estimates.items() if k != 'source_matrix_hashes'},
                      "treatment_warning": TREATMENT_WARNING,
                      "maximum_sequential_job_hours": manifest['maximum_sequential_job_hours']}))


def verify(output):
    output = Path(output)
    manifest = read(output / "manifest.json")
    state = read(output / "state.json")
    if manifest.get("schema_version") != SCHEMA or state["manifest_sha256"] != sha(output / "manifest.json"):
        raise ValueError("registration identity changed")
    if manifest['jobs'] != make_jobs() or manifest['arms'] != list(ARMS) or manifest['cohorts'] != list(COHORTS):
        raise ValueError("complete fixed 1638-cell configuration required")
    if [row.get('id') for row in state['jobs']] != [job['id'] for job in manifest['jobs']]:
        raise ValueError("state job identities or order differ from registration")
    seconds = state.get('job_seconds_used')
    if isinstance(seconds, bool) or not isinstance(seconds, (int, float)) or not math.isfinite(seconds) or seconds < 0:
        raise ValueError("invalid accumulated compute allowance")
    for name, root in manifest['sources'].items():
        if source_manifest(root) != manifest['source_files'][name]:
            raise ValueError("registered input changed: source file set or content: " + root)
    groups = list(manifest['source_files'].values()) + [e['files'] for e in manifest['benchmark_entries']]
    groups.append(manifest['estimates']['source_matrix_hashes'])
    for group in groups:
        for path, expected in group.items():
            if not Path(path).is_file() or sha(path) != expected:
                raise ValueError("registered input changed: " + path)
    for key in ('skill_library', 'shape_registry'):
        if sha(manifest[key]) != manifest[key + '_sha256']:
            raise ValueError("registered " + key + " changed")
    if len(manifest['jobs']) != 1638 or len({j['id'] for j in manifest['jobs']}) != 1638:
        raise ValueError("complete 1638-cell denominator required")
    return manifest


def active_vitis():
    names = {"vitis_hls", "vitis-run", "xsim", "xsimk", "v++", "xelab", "xvlog"}
    pids = []
    for path in Path('/proc').glob('[0-9]*/comm'):
        try:
            if path.read_text().strip() in names:
                pids.append(int(path.parent.name))
        except (OSError, ProcessLookupError):
            continue
    return pids


def probe_vitis_versions(settings):
    """Bounded version-only queries for both historical and cleaned runners."""
    settings = Path(settings).resolve()
    installation = settings.parent.parent.parent
    tools = {}
    script = '''source "$1" >/dev/null 2>&1 || exit $?
if command -v "$2" >/dev/null 2>&1; then
  command -v "$2"
  exec "$2" "$3"
elif [ -x "$4" ]; then
  printf '%s\\n' "$4"
  exec "$4" "$3"
else
  exit 127
fi
'''
    for name, flags in (('vitis-run', ('--version', '-version')), ('vitis_hls', ('-version', '--version'))):
        fallback = installation / ('Vitis_HLS' if name == 'vitis_hls' else 'Vitis') / '2023.2' / 'bin' / name
        attempts = []
        for flag in flags:
            try:
                with tempfile.TemporaryDirectory(prefix='hlsmith-version-') as directory:
                    result = subprocess.run(['bash', '--noprofile', '--norc', '-c', script,
                        'hlsmith-tool-version', str(settings), name, flag, str(fallback)],
                        capture_output=True, text=True, timeout=10, cwd=directory)
                lines = result.stdout.splitlines()
                binary = Path(lines[0]).resolve() if lines and lines[0].startswith('/') else None
                text = '\n'.join(lines[1:]) + result.stderr
                version = re.search(r'\b(?:vitis-run|vitis HLS)\b[^\n]*?\bv?(20\d{2}\.\d+)\b', text, re.IGNORECASE)
                observed = version.group(1) if version else None
                matched = result.returncode == 0 and observed == '2023.2'
                entry = {'status': 'verified' if matched else 'unverified', 'returncode': result.returncode,
                         'command': [name, flag], 'version_output': text[:4000], 'version_output_sha256': hashlib.sha256(text.encode()).hexdigest(),
                         'binary': str(binary) if binary else None,
                         'binary_sha256': sha(binary) if binary and binary.is_file() else None,
                         'observed_version': observed}
            except subprocess.TimeoutExpired:
                entry = {'status': 'timeout', 'command': [name, flag], 'limit_seconds': 10}
            attempts.append(entry)
            if entry['status'] == 'verified':
                break
        tools[name] = {**attempts[-1], 'attempts': attempts}
    return tools


def preflight(output, settings=None):
    manifest = verify(output)
    issues = []
    if not Path(manifest['sources']['clean'], 'hlsmith', 'prompts', 'researcher_v1.py').is_file():
        issues.append('researcher prompt profile missing')
    registry = read(manifest['shape_registry']).get('benchmarks', {})
    if read(manifest['shape_registry']).get('schema_version') != 'c2hls.hlsfactory-output-shapes.v1':
        issues.append('unsupported authoritative output-shape registry schema')
    for entry in manifest['benchmark_entries']:
        meta = read(Path(entry['path']) / 'metadata.json')
        if str(meta.get('source_repo', '')).lower() != 'hlsfactory':
            issues.append('independent HLSFactory oracle not required:' + entry['benchmark'])
        record = registry.get(entry['benchmark']) or {}
        if not isinstance(record.get('outputs'), dict) or not record['outputs']:
            issues.append('empty authoritative output contract:' + entry['benchmark'])
        if record.get('testbench_sha256') != sha(Path(entry['path']) / meta['testbench_file']):
            issues.append('public harness registry mismatch:' + entry['benchmark'])
    if settings is not None and not Path(settings).is_file():
        issues.append('Vitis settings missing')
    versions = probe_vitis_versions(settings) if settings is not None and Path(settings).is_file() else {}
    for tool, record in versions.items():
        if record['status'] != 'verified':
            issues.append('Vitis 2023.2 version not verified:' + tool)
    if shutil.which('g++') is None:
        issues.append('native compiler g++ missing')
    report = {"checked_at": now(), "complete": not issues, "issues": issues,
              "warnings": [TREATMENT_WARNING],
              "active_vitis_pids": active_vitis(), "provider_calls": 0, "vitis_calls": 0,
              "tool_versions": versions, "tool_version_check": 'version_only' if versions else 'not_requested',
              "vitis_version_queries": sum(len(tool['attempts']) for tool in versions.values()),
              "vitis_job_calls": 0,
              "weights": "hosted alias; no local weights required or immutable version claimed",
              "credentials": "not read during preflight"}
    save(Path(output) / 'preflight.json', report)
    return report


def cell_environment(manifest, job, limits, output):
    # Avoid inherited campaign, model-role, decoding and routing controls.
    env = {k: v for k, v in os.environ.items() if not k.startswith(('C2HLS_', 'HLSMITH_', 'OPENAI_', 'ANTHROPIC_'))}
    cohort = next(c for c in manifest['cohorts'] if c['id'] == job['cohort'])
    arm = next(a for a in manifest['arms'] if a['id'] == job['arm'])
    scratch = Path(limits['scratch_root']) / job['id']
    common_runner = limits.get('common_runner') or limits.get('tool_versions', {}).get('vitis-run', {})
    common_binary = common_runner.get('binary') or 'vitis-run'
    # Frozen v6 spells this executable by name. The cleaned runtime accepts
    # an absolute command; point both at the same verified installation.
    if common_runner.get('binary'):
        env['PATH'] = str(Path(common_binary).parent) + os.pathsep + env.get('PATH', '')
    runner_command = shlex.quote(common_binary) + ' --tcl --input_file'
    env.update({"PYTHON_DOTENV_DISABLED": "1", "PYTHONDONTWRITEBYTECODE": "1", "HLS_TCL_RUNNER": runner_command,
                "OPENAI_BASE_URL": PROVIDER['endpoint'], "C2HLS_MODEL": PROVIDER['model'],
                "C2HLS_MODEL_REVISION": PROVIDER['declared_release'] + '@declared-provider-release-not-immutable',
                "C2HLS_DEEPSEEK_THINKING": "disabled", "C2HLS_DEEPSEEK_REASONING_EFFORT": "high",
                "C2HLS_LLM_TEMPERATURE": "0", "C2HLS_LLM_TOP_P": "1", "C2HLS_PROVIDER_MAX_RETRIES": "0",
                "C2HLS_API_BUDGET_LEDGER": str(Path(output).resolve() / 'api_budget.json'),
                "C2HLS_AGGREGATE_INPUT_TOKEN_BUDGET": str(limits['max_input_tokens_per_cell']),
                "C2HLS_SAMPLE_ID": job['id'], "C2HLS_MAX_COMPLETION_TOKENS": "8192",
                "C2HLS_REFERENCE_BLIND": "1", "C2HLS_REFERENCE_VALIDATE_MODE": "public_only",
                "C2HLS_ORACLE_MODE": "0", "C2HLS_GT_AWARE_REVERT": "0", "C2HLS_GT_COMPARISON_IN_CONTROL": "0",
                "C2HLS_TRANSCRIPT_AUDIT": "1", "C2HLS_REFERENCE_BLIND_FAIL_ON_LEAK": "1",
                "C2HLS_PHASE8_BASELINE_ALIGN": "0", "C2HLS_PHASE5_GT_PREPOP": "0",
                "C2HLS_REFERENCE_COSIM": "0", "C2HLS_COSIM_SELECTED_ONLY": "0", "C2HLS_FORCE_SELECTED_COSIM": "0",
                "C2HLS_COSIM_REQUIRED": "0", "C2HLS_GENERATION_COSIM": "0", "C2HLS_HW_EMU_FINAL": "0",
                "C2HLS_COSIM_SKIP_SLOWER_THAN_GOLD": "0", "C2HLS_FEEDBACK_LLM": "0",
                "C2HLS_SKILL_MODE": "off" if arm['skills'] in {'off', 'noskills'} else "on",
                "C2HLS_SKILL_LIBRARY_FROZEN": "1", "C2HLS_SKILL_LIBRARY_PERSIST": "0", "C2HLS_SKILL_UPDATE_STATS": "0",
                "C2HLS_SKILL_LIBRARY_PATH": manifest['skill_library'],
                "C2HLS_SKILL_PROMPT_SCOPE": 'skillless' if arm['skills'] in {'off', 'noskills'} else 'all_positive' if arm['skills'] == 'allpos' else 'smart_best_fit',
                "C2HLS_FORCE_SKILL_PROMPTS": "0" if arm['skills'] in {'off', 'noskills'} else "1",
                "C2HLS_DYNAMIC_ROUTING": "0" if arm['id'] == 'oneshot' else "1",
                "C2HLS_STRATEGY": "dynamic", "C2HLS_QOR_DESIGN_SWEEP": "0", "C2HLS_QOR_PROFILE_REVIEW": "0",
                "C2HLS_PROMPT_PROFILE": cohort['prompt_profile'], "C2HLS_CPU_GOLDEN_TIMEOUT": "180",
                "C2HLS_SYNTH_TIMEOUT": "2400", "C2HLS_CSIM_TIMEOUT": "180", "C2HLS_COSIM_TIMEOUT": "3600",
                "C2HLS_VITIS_SETTINGS": limits['vitis_settings'], "C2HLS_VITIS_VERSION": "2023.2",
                "C2HLS_PART": "xcu280-fsvh2892-2L-e", "C2HLS_CLOCK_NS": "3.33",
                "C2HLS_HLSFACTORY_SHAPE_REGISTRY": manifest['shape_registry'],
                "C2HLS_TMP_ROOT": str(scratch), "C2HLS_VITIS_USER_HOME": str(scratch / 'vitis_home')})
    if arm['id'] == 'oneshot':
        env['C2HLS_STRATEGY'] = 'static'
    return env


def owned_processes(root_pid, identities, table):
    """Track descendants by PID and start time, including detached sessions."""
    if not identities and root_pid in table:
        identities[root_pid] = table[root_pid]['start_ticks']
    live = {pid for pid, start in identities.items() if table.get(pid, {}).get('start_ticks') == start}
    while True:
        found = {pid for pid, row in table.items() if row['ppid'] in live} - live
        if not found:
            break
        for pid in found:
            identities[pid] = table[pid]['start_ticks']
        live.update(found)
    return live


def run_process(command, *, cwd, env, timeout, log):
    """Reap only observed job descendants; PID reuse never grants ownership."""
    from hlsmith.toolchain.cosim_runtime_monitor import _process_table
    started = time.monotonic()
    identities = {}
    with Path(log).open('w') as stream:
        process = subprocess.Popen(command, cwd=cwd, env=env, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
        try:
            while True:
                owned_processes(process.pid, identities, _process_table())
                returncode = process.poll()
                if returncode is not None:
                    status = 'complete' if returncode == 0 else 'execution_error'
                    break
                remaining = timeout - (time.monotonic() - started)
                if remaining <= 0:
                    status, returncode = 'timeout', None
                    break
                time.sleep(min(.2, remaining))
        finally:
            owned_processes(process.pid, identities, _process_table())
            for sig in (signal.SIGTERM, signal.SIGKILL):
                table = _process_table()
                for pid in sorted(identities, reverse=True):
                    if table.get(pid, {}).get('start_ticks') == identities[pid]:
                        try:
                            os.kill(pid, sig)
                        except ProcessLookupError:
                            pass
                if sig == signal.SIGTERM:
                    time.sleep(.2)
            try:
                process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                raise RuntimeError('owned job did not exit after termination')
    return {'status': status, 'returncode': returncode, 'elapsed_seconds': time.monotonic() - started}


def launch(args):
    import fcntl
    from experiment_budget import initialize
    output = args.output.resolve()
    with (output / 'controller.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        manifest = verify(output)
        report = preflight(output, args.vitis_settings)
        if not report['complete'] or report['active_vitis_pids']:
            raise ValueError('preflight failed or unrelated Vitis jobs are active; nothing launched')
        state = read(output / 'state.json')
        if state['status'] != 'registered':
            raise ValueError('campaign has already started; no implicit rerolls or resume')
        if not args.rates_rechecked:
            raise ValueError('recheck official peak prices and pass --rates-rechecked before paid launch')
        limits = {'api_usd': positive(args.max_api_usd), 'compute_seconds': positive(args.max_compute_hours) * 3600,
                  'max_input_tokens_per_cell': int(positive(args.max_input_tokens_per_cell)),
                  'vitis_settings': str(args.vitis_settings.resolve()), 'scratch_root': str(args.scratch_root.resolve())}
        if limits['max_input_tokens_per_cell'] <= 0:
            raise ValueError('per-cell input token cap must be a positive integer')
        limits['vitis_settings_sha256'] = sha(args.vitis_settings)
        limits['tool_versions'] = report['tool_versions']
        limits['common_runner'] = {**report['tool_versions']['vitis-run'],
                                  'arguments': ['--tcl', '--input_file'],
                                  'scope': 'same verified binary and arguments for frozen and clean cohorts'}
        limits['hls_tcl_runner'] = shlex.quote(limits['common_runner']['binary']) + ' --tcl --input_file'
        key = os.getenv('DEEPSEEK_API_KEY', '').strip()
        if not key:
            raise ValueError('set DEEPSEEK_API_KEY before launch')
        initialize(output / 'api_budget.json', limit_usd=limits['api_usd'], prices={PROVIDER['model']: PROVIDER['rates_per_million']})
        save(output / 'launch.json', {'manifest_sha256': sha(output / 'manifest.json'), 'created_at': now(), 'limits': limits})
        state.update(status='running', started_at=now())
        save(output / 'state.json', state)
        for job, row in zip(manifest['jobs'], state['jobs']):
            if state['job_seconds_used'] + job['maximum_job_seconds'] > limits['compute_seconds']:
                state.update(status='stopped_compute_cap')
                break
            if active_vitis():
                state.update(status='stopped_external_vitis')
                break
            ledger = read(output / 'api_budget.json')
            if any(r['status'] != 'settled' or r.get('overrun') for r in ledger['reservations']):
                state.update(status='stopped_accounting_uncertainty')
                break
            if sum(r['charged_or_reserved_usd'] for r in ledger['reservations']) >= ledger['limit_usd']:
                state.update(status='stopped_api_cap')
                break
            folder = output / 'jobs' / job['id']
            folder.mkdir(parents=True, exist_ok=False)
            row.update(status='running', started_at=now())
            state['active_job'] = job['id']
            save(output / 'state.json', state)
            env = cell_environment(manifest, job, limits, output)
            env['OPENAI_API_KEY'] = key
            env['PYTHONPATH'] = str(Path(__file__).resolve().parents[1])
            command = [sys.executable, '-m', 'hlsmith.experiments', 'worker', '--output', str(output), '--job', job['id']]
            process = run_process(command, cwd=manifest['sources']['clean'], env=env,
                                  timeout=job['maximum_job_seconds'], log=folder / 'worker.log')
            state['job_seconds_used'] += process['elapsed_seconds']
            row.update(process, finished_at=now())
            if (folder / 'measurement.json').is_file():
                row['measurement'] = str(folder / 'measurement.json')
            elif process['status'] == 'complete':
                row.update(status='execution_error', error='worker returned without a terminal measurement')
                state.update(status='stopped_execution_error')
                break
            if process['status'] != 'complete':
                state.update(status='stopped_execution_error')
                break
            ledger = read(output / 'api_budget.json')
            if any(r['status'] != 'settled' or r.get('overrun') for r in ledger['reservations']):
                state.update(status='stopped_accounting_uncertainty')
                break
            save(output / 'state.json', state)
        else:
            state['status'] = 'complete'
        state.pop('active_job', None)
        state['updated_at'] = now()
        save(output / 'state.json', state)
        print(json.dumps({'status': state['status'], 'completed': sum(r['status'] == 'complete' for r in state['jobs'])}))


def worker(args):
    """Private worker: imports one implementation in a fresh interpreter."""
    output = args.output.resolve()
    manifest = verify(output)
    receipt = read(output / 'launch.json')
    state = read(output / 'state.json')
    if (receipt['manifest_sha256'] != sha(output / 'manifest.json')
            or state.get('active_job') != args.job or state.get('status') != 'running'):
        raise ValueError('worker requires an active registered launch')
    limits = receipt.get('limits') or {}
    if not {'api_usd', 'compute_seconds', 'max_input_tokens_per_cell', 'scratch_root',
            'vitis_settings', 'vitis_settings_sha256'}.issubset(limits):
        raise ValueError('worker requires explicit registered API, compute and input limits')
    for name in ('api_usd', 'compute_seconds', 'max_input_tokens_per_cell'):
        positive(limits[name])
    if (isinstance(limits['max_input_tokens_per_cell'], bool)
            or not isinstance(limits['max_input_tokens_per_cell'], int)):
        raise ValueError('per-cell input token cap must be a positive integer')
    ledger = read(output / 'api_budget.json')
    if ledger.get('limit_usd') != limits['api_usd']:
        raise ValueError('worker API ledger differs from registered cap')
    if sha(limits['vitis_settings']) != limits['vitis_settings_sha256']:
        raise ValueError('registered Vitis settings changed')
    for tool in limits.get('tool_versions', {}).values():
        if tool.get('binary_sha256') and sha(tool['binary']) != tool['binary_sha256']:
            raise ValueError('registered Vitis executable changed')
    job = next(j for j in manifest['jobs'] if j['id'] == args.job)
    if state['job_seconds_used'] + job['maximum_job_seconds'] > limits['compute_seconds']:
        raise ValueError('worker exceeds registered remaining compute allowance')
    key = os.environ.get('OPENAI_API_KEY') or os.environ.get('DEEPSEEK_API_KEY')
    controlled = cell_environment(manifest, job, limits, output)
    # Rebuild, rather than trust, caller-provided campaign controls. This also
    # makes a direct private worker unable to omit the shared API ledger.
    for name in tuple(os.environ):
        if name.startswith(('C2HLS_', 'HLSMITH_', 'OPENAI_', 'ANTHROPIC_')):
            os.environ.pop(name)
    os.environ.update(controlled)
    if args.stage == 'validate':
        # hls_eval freezes this value at import time, including when imported
        # transitively by c2hls. Set it before loading either implementation.
        os.environ['C2HLS_CSIM_TIMEOUT'] = '300'
        os.environ.pop('C2HLS_GENERATION_COSIM', None)
    if key:
        os.environ['OPENAI_API_KEY'] = key
    elif args.stage in {'all', 'generate'}:
        raise ValueError('registered generation worker requires its explicit provider key')
    cohort = next(c for c in manifest['cohorts'] if c['id'] == job['cohort'])
    arm = next(a for a in manifest['arms'] if a['id'] == job['arm'])
    entry = next(e for e in manifest['benchmark_entries'] if e['benchmark'] == job['benchmark'])
    folder = output / 'jobs' / job['id']
    # Generation and final validation run in separate child interpreters, so
    # the clean validator is common to both implementation cohorts.
    stage = args.stage
    source = manifest['sources'][cohort['source']] if stage == 'generate' else manifest['sources']['clean']
    if stage == 'all':
        def terminate_worker(signum, frame):
            raise SystemExit(128 + signum)
        signal.signal(signal.SIGTERM, terminate_worker)
        environment = os.environ.copy()
        command = [sys.executable, '-m', 'hlsmith.experiments', 'worker', '--output', str(output), '--job', args.job]
        generation = run_process(command + ['--stage', 'generate'], cwd=source, env=environment,
                                 timeout=arm['generation_timeout_seconds'], log=folder / 'generation.log')
        save(folder / 'generation_process.json', generation)
        if generation['status'] == 'timeout':
            save(folder / 'measurement.json', {'job': job, 'status': 'generation_timeout', 'rtl': {'ran': False}})
            return
        if generation['status'] != 'complete':
            raise RuntimeError('generation worker failed; preserve evidence and inspect')
        validation = run_process(command + ['--stage', 'validate'], cwd=source, env=environment,
                                 timeout=VALIDATION_SECONDS, log=folder / 'validation.log')
        if validation['status'] == 'timeout':
            save(folder / 'measurement.json', {'job': job, 'status': 'validation_timeout', 'rtl': {'ran': False},
                                             'timeout_scope': 'combined_native_csim_rtl_validation'})
            return
        if validation['status'] != 'complete':
            raise RuntimeError('validation worker failed; preserve evidence and inspect')
        return
    sys.path.insert(0, source)
    import c2hls
    if not Path(c2hls.__file__).resolve().is_relative_to(Path(source).resolve()):
        raise ValueError('worker imported a different implementation than its registered source')
    if stage == 'generate':
        from experiment_telemetry import capture_tool_calls, tool_summary
        with capture_tool_calls() as events:
            result = getattr(c2hls, arm['entrypoint'])(entry['path'], str(folder / 'generation'),
                       gpt_model=PROVIDER['model'], turns_limitation=arm['turns'], quality_repair_turns=arm['quality_repair_turns'])
        save(folder / 'generation.json', result)
        save(folder / 'generation_tool_flows.json', tool_summary(events))
        ledger = read(output / 'api_budget.json')
        rows = [r for r in ledger['reservations'] if r['run_id'] == job['id']]
        usage = result.get('llm_usage') or {}
        if (any(r['status'] != 'settled' or r.get('overrun') for r in rows)
                or usage.get('usage_missing_calls')
                or usage.get('candidate_requests', usage.get('calls', 0)) != len(rows)):
            raise RuntimeError('provider/accounting failure is not a generated nonpass')
        errors = json.dumps(result)
        if any(marker in errors for marker in ('BudgetExhausted', 'campaign API budget exhausted', 'aggregate input token reservation exceeds')):
            raise RuntimeError('registered launch allowance exhausted; preserve result and stop dispatch')
        suffix = '_generated.cpp' if arm['entrypoint'] == 'run_benchmark' else '_final.cpp'
        selected = folder / 'generation' / (job['benchmark'] + suffix)
        if selected.is_file() and result.get('success') is True:
            shutil.copy2(selected, folder / 'selected.cpp')
            selected_report = result.get('synth_report') if arm['entrypoint'] == 'run_benchmark' else result.get('final_report')
            selected_report = selected_report or {}
            save(folder / 'selection.json', {'source_sha256': sha(selected), 'frozen_before_validation': True,
                 'synthesis_report': selected_report, 'synthesis_report_sha256': content_sha(selected_report),
                 'prompt_profile': cohort['prompt_profile'], 'source_cohort': cohort['source'],
                 'model_identity': {key: (result.get('run') or {}).get(key) for key in
                                    ('model', 'model_revision', 'prompt_profile')},
                 'common_runner': {key: receipt['limits'].get('common_runner', {}).get(key)
                                   for key in ('binary', 'binary_sha256', 'observed_version', 'arguments')},
                 'declared_provider': manifest['provider']})
        return
    selected = folder / 'selected.cpp'
    if not selected.is_file():
        generation = read(folder / 'generation.json')
        status = 'blocked_input' if generation.get('phase') == 'independent_golden' else 'generation_nonpass'
        save(folder / 'measurement.json', {'job': job, 'status': status, 'rtl': {'ran': False}})
        return
    selection = read(folder / 'selection.json')
    if sha(selected) != selection['source_sha256']:
        raise ValueError('selected source changed before validation')
    if content_sha(selection.get('synthesis_report', {})) != selection.get('synthesis_report_sha256'):
        raise ValueError('selected synthesis report changed before validation')
    os.environ.pop('C2HLS_GENERATION_COSIM', None)
    os.environ['C2HLS_CSIM_TIMEOUT'] = '300'
    import hls_eval
    from golden_output import compare_hlsfactory_dumps
    from experiment_telemetry import capture_tool_calls, tool_summary
    inputs = c2hls._load_benchmark_inputs(entry['path'])
    golden = c2hls._prepare_independent_golden(inputs)
    code = selected.read_text()
    common = {'header_code': inputs['header_code'], 'header_name': inputs['header_name'], 'extra_files': inputs['extra_files']}
    native = hls_eval.run_native_testbench(code, inputs['testbench_code'], **common, timeout=180)
    comparison = compare_hlsfactory_dumps(golden.get('output', ''), native.get('output', ''), specs=golden.get('specs', {})).to_dict()
    record = {'job': job, 'selected_source_sha256': sha(selected), 'native': {k: v for k, v in native.items() if k != 'output'},
              'treatment_warning': TREATMENT_WARNING,
              'golden_provenance': golden.get('provenance', {}),
              'selected_synthesis_report': selection.get('synthesis_report', {}),
              'common_runner': {key: receipt['limits'].get('common_runner', {}).get(key)
                                for key in ('binary', 'binary_sha256', 'observed_version', 'arguments')},
              'native_comparison': comparison, 'status': 'failed_native', 'csim': {'ran': False}, 'rtl': {'ran': False}}
    golden_sha256 = hashlib.sha256(golden.get('output', '').encode()).hexdigest()
    record['golden_output_sha256'] = golden_sha256
    if golden.get('success') is True and golden.get('output') and native.get('passed') is True and comparison.get('passed') is True:
        record['feasibility'] = selected_feasibility(c2hls, selection.get('synthesis_report', {}), manifest['target'])
        if not record['feasibility']['feasible']:
            record['status'] = 'failed_feasibility'
            save(folder / 'measurement.json', record)
            return
        common.update(top_function=inputs['meta'].get('translated_hls_top', 'workload'),
                      part=manifest['target']['part'], clock_ns=manifest['target']['clock_ns'],
                      golden_output_text=golden['output'], golden_output_specs=golden.get('specs', {}))
        with capture_tool_calls() as events:
            record['csim'] = hls_eval.run_csim(code, inputs['testbench_code'], **common)
            if independently_passed(record['csim'], golden_sha256):
                record['rtl'] = hls_eval.run_cosim(code, inputs['testbench_code'], **common, interface_depths=inputs['meta'].get('cosim_depths', {}))
                rtl = record['rtl']
                record['status'] = ('passed' if independently_passed(rtl, golden_sha256) and measured_cycles(rtl) is not None
                                    else 'rtl_timeout' if test_timed_out(rtl)
                                    else 'failed_rtl')
                if record['status'] == 'passed':
                    fresh_report, evidence = fresh_rtl_synthesis(hls_eval, rtl)
                    record['rtl_synthesis_evidence'] = evidence
                    record['rtl_synthesis_report'] = fresh_report
                    record['rtl_feasibility'] = selected_feasibility(c2hls, fresh_report, manifest['target'])
                    if not record['rtl_feasibility']['feasible']:
                        record['status'] = 'unqualified_rtl_synthesis'
            else:
                record['status'] = 'csim_timeout' if test_timed_out(record['csim']) else 'failed_csim'
        record['tool_flows'] = tool_summary(events)
    save(folder / 'measurement.json', record)


def report(output):
    """Keep every planned cell and distinguish generation from measured RTL."""
    output = Path(output).resolve()
    manifest = verify(output)
    state = read(output / 'state.json')
    states = {j['id']: j for j in state['jobs']}
    rows, paired, researcher_pairs = [], [], []
    for job in manifest['jobs']:
        path = output / 'jobs' / job['id'] / 'measurement.json'
        measured = read(path) if path.is_file() else {}
        if measured and measured.get('job') != job:
            raise ValueError('measurement job identity changed: ' + job['id'])
        selected = path.parent / 'selected.cpp'
        if measured.get('selected_source_sha256') and (not selected.is_file() or sha(selected) != measured['selected_source_sha256']):
            raise ValueError('measurement source identity changed: ' + job['id'])
        rtl = measured.get('rtl') or {}
        golden_sha256 = measured.get('golden_output_sha256')
        qualified = (measured.get('status') == 'passed' and bool(measured.get('selected_source_sha256'))
                     and measured.get('native_comparison', {}).get('passed') is True
                     and measured.get('golden_provenance', {}).get('status') == 'passed'
                     and bool(golden_sha256)
                     and measured.get('golden_provenance', {}).get('output_sha256') == golden_sha256
                     and measured.get('feasibility', {}).get('feasible') is True
                     and measured.get('rtl_feasibility', {}).get('feasible') is True
                     and independently_passed(measured.get('csim') or {}, golden_sha256)
                     and independently_passed(rtl, golden_sha256) and measured_cycles(rtl) is not None)
        if measured.get('status') == 'passed' and not qualified:
            raise ValueError('passing measurement lacks independent/timing/resource evidence: ' + job['id'])
        rows.append({**job, 'status': measured.get('status') or states[job['id']]['status'],
                     'rtl_pass': qualified,
                     'rtl_cycles': measured_cycles(rtl) if qualified else None,
                     'selected_source_sha256': measured.get('selected_source_sha256')})
    grouped = {}
    for row in rows:
        grouped.setdefault((row['benchmark'], row['arm'], row['repeat']), {})[row['cohort']] = row
    for identity, cells in grouped.items():
        frozen, clean = cells['frozen-paper'], cells['clean-paper']
        ratio = None
        if frozen['rtl_pass'] and clean['rtl_pass'] and (clean['rtl_cycles'] or 0) > 0 and (frozen['rtl_cycles'] or 0) > 0:
            ratio = frozen['rtl_cycles'] / clean['rtl_cycles']
        paired.append({'benchmark': identity[0], 'arm': identity[1], 'repeat': identity[2],
                       'frozen_status': frozen['status'], 'clean_status': clean['status'],
                       'measured_frozen_over_clean_rtl_cycles': ratio})
        researcher = cells['clean-researcher']
        researcher_ratio = (clean['rtl_cycles'] / researcher['rtl_cycles']
                            if clean['rtl_pass'] and researcher['rtl_pass'] else None)
        researcher_pairs.append({'benchmark': identity[0], 'arm': identity[1], 'repeat': identity[2],
                                 'paper_status': clean['status'], 'researcher_status': researcher['status'],
                                 'measured_paper_over_researcher_rtl_cycles': researcher_ratio})
    ratios = [r['measured_frozen_over_clean_rtl_cycles'] for r in paired if r['measured_frozen_over_clean_rtl_cycles'] is not None]
    budget = read(output / 'api_budget.json') if (output / 'api_budget.json').is_file() else {'reservations': []}
    researcher_ratios = [r['measured_paper_over_researcher_rtl_cycles'] for r in researcher_pairs
                         if r['measured_paper_over_researcher_rtl_cycles'] is not None]
    summary = {'schema_version': 'hlsmith.cleanup-comparison-report.v1', 'created_at': now(),
               'registered_cells': len(rows), 'status_counts': dict(Counter(r['status'] for r in rows)),
               'cohort_counts': {c['id']: dict(Counter(r['status'] for r in rows if r['cohort'] == c['id'])) for c in manifest['cohorts']},
               'paired_paper_denominator': len(paired), 'paired_rtl_passes': len(ratios),
               'paired_rtl_geomean_speedup': math.exp(sum(map(math.log, ratios)) / len(ratios)) if ratios else None,
               'researcher_paired_denominator': len(researcher_pairs), 'researcher_paired_rtl_passes': len(researcher_ratios),
               'researcher_paired_rtl_geomean_speedup': (math.exp(sum(map(math.log, researcher_ratios)) / len(researcher_ratios)) if researcher_ratios else None),
               'api_charged_or_reserved_usd': sum(r['charged_or_reserved_usd'] for r in budget['reservations']),
               'api_reservation_statuses': dict(Counter(r['status'] for r in budget['reservations'])),
               'job_hours_used': state['job_seconds_used'] / 3600, 'treatment_warning': TREATMENT_WARNING,
               'campaign_status': state['status'], 'complete_paper_reproduction': False,
               'evidence_scope': 'Prospective DeepSeek cleanup comparison only; archived hardware replay, original commercial/open-model cohorts, router/adaptation studies and missing GRPO training are separate work.',
               'acceptance': 'No automatic equivalence/noninferiority claim; evaluate frozen thresholds and all registered outcome counts.',
               'qualification': 'paired ratio covers jointly passing RTL only; retain failures/timeouts/pending cells in coverage; hosted runs are not immutable-weight seed-controlled replicas'}
    summary['uncertainty'] = summarize_cleanup_statistics(
        rows, kernels=[entry['benchmark'] for entry in manifest['benchmark_entries']],
        arms=[arm['id'] for arm in manifest['arms']],
        repeats=sorted({job['repeat'] for job in manifest['jobs']}),
        cohorts=[cohort['id'] for cohort in manifest['cohorts']],
        paired_comparisons=(('frozen-paper', 'clean-paper'), ('clean-paper', 'clean-researcher')),
        bootstrap_seed=20261018, draws=1000)
    save(output / 'report.json', summary)
    for filename, data in (('cells.csv', rows), ('paired_paper.csv', paired), ('paired_researcher.csv', researcher_pairs)):
        with (output / filename).open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(data[0]))
            writer.writeheader()
            writer.writerows(data)
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='action', required=True)
    registration = sub.add_parser('register')
    for name in ('frozen-source', 'clean-source', 'benchmark-root', 'skill-library', 'shape-registry', 'historical-payload', 'output'):
        registration.add_argument('--' + name, type=Path, required=True)
    for action in ('estimate', 'preflight', 'report', 'launch', 'worker'):
        command = sub.add_parser(action)
        command.add_argument('--output', type=Path, required=True)
        if action in ('preflight', 'launch'):
            command.add_argument('--vitis-settings', type=Path, required=action == 'launch')
        if action == 'launch':
            command.add_argument('--max-api-usd', type=float, required=True)
            command.add_argument('--max-compute-hours', type=float, required=True)
            command.add_argument('--max-input-tokens-per-cell', type=int, required=True)
            command.add_argument('--scratch-root', type=Path, required=True)
            command.add_argument('--rates-rechecked', action='store_true')
        if action == 'worker':
            command.add_argument('--job', required=True)
            command.add_argument('--stage', choices=('all', 'generate', 'validate'), default='all')
    args = parser.parse_args(argv)
    if args.action == 'register':
        register(args)
    elif args.action == 'estimate':
        manifest = verify(args.output)
        print(json.dumps({'jobs': len(manifest['jobs']), 'estimates': manifest['estimates'],
                          'treatment_warning': manifest['treatment_warning'],
                          'maximum_sequential_job_hours': manifest['maximum_sequential_job_hours']}, indent=2))
    elif args.action == 'preflight':
        preflight_report = preflight(args.output, args.vitis_settings)
        print(json.dumps(preflight_report, indent=2))
        return 0 if preflight_report['complete'] else 2
    elif args.action == 'report':
        print(json.dumps(report(args.output), indent=2))
    elif args.action == 'launch':
        launch(args)
    else:
        worker(args)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
