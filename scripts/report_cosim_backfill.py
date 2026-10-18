#!/usr/bin/env python3
"""Capture attributable RTL evidence and render a reproducible backfill report.

Read-only with respect to every campaign. Each report has immutable copies of
the JSON inputs actually read, their hashes, and an explicit capture interval.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import shutil
import sys
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]
TERMINAL = {'pass', 'fail', 'timeout'}
STATUS_ORDER = ['pass', 'fail', 'timeout', 'running', 'pending', 'interrupted',
                'execution_error', 'evidence_error']


def now():
    return datetime.now(timezone.utc)


class Snapshot:
    def __init__(self, output):
        self.output = Path(output)
        self.cache = {}
        self.sources = {}
        (self.output / 'evidence').mkdir()

    def read(self, path):
        path = Path(path).resolve()
        key = str(path)
        if key not in self.cache:
            raw = path.read_bytes()
            digest = hashlib.sha256(raw).hexdigest()
            self.cache[key] = json.loads(raw)
            destination = self.output / 'evidence' / (digest + '.json')
            if not destination.exists():
                destination.write_bytes(raw)
            self.sources[key] = {'sha256': digest, 'bytes': len(raw),
                                 'snapshot': str(destination.relative_to(self.output)),
                                 'read_at': now().isoformat()}
        return self.cache[key]


def controller_alive(identity):
    """Never interpret a reused PID or a prior boot as a running controller."""
    try:
        if identity['boot_id'] != Path('/proc/sys/kernel/random/boot_id').read_text().strip():
            return False
        proc = Path('/proc') / str(identity['pid'])
        fields = (proc / 'stat').read_text().rsplit(')', 1)[1].split()
        argv = (proc / 'cmdline').read_bytes().split(b'\0')
        return (fields[0] != 'Z' and fields[19] == str(identity['start_ticks'])
                and str(identity['source_runner']).encode() in argv)
    except (OSError, KeyError, IndexError):
        return False


def positive(value):
    return (isinstance(value, (int, float)) and not isinstance(value, bool)
            and math.isfinite(value) and value > 0)


def classify(row, measurement, task=None, live=False):
    """Gate plotted latency on independent correctness and exact source identity."""
    task = task or {}
    issues = []
    task_status = task.get('status', 'pending')
    if measurement is None:
        status = task_status
        if status in TERMINAL:
            status = 'evidence_error'
            issues.append('terminal_state_without_measurement')
        elif status == 'running' and not live:
            status = 'interrupted'
        return {'status': status, 'rtl_cycles': None, 'issues': issues,
                'error': task.get('error', ''), 'recorded_status': None}
    c = measurement.get('cosim') or {}
    status = c.get('status')
    if measurement.get('validation_id') != row['validation_id']:
        issues.append('validation_identity_mismatch')
    if measurement.get('kernel_sha256') != row['kernel_sha256']:
        issues.append('measurement_kernel_mismatch')
    if c.get('target_code_sha256') != row['kernel_sha256']:
        issues.append('rtl_target_kernel_mismatch')
    if row.get('historical_toolchain') and measurement.get('toolchain') != row['historical_toolchain']:
        issues.append('historical_toolchain_mismatch')
    if status not in TERMINAL:
        issues.append('unknown_measurement_status')
    if measurement.get('archive_error'):
        issues.append('archive_error')
    if task_status in TERMINAL and task_status != status:
        issues.append('controller_measurement_status_mismatch')
    if status == 'pass':
        correctness = c.get('correctness') or {}
        csim = measurement.get('csim') or {}
        golden = measurement.get('golden') or {}
        if not (c.get('passed') is True and c.get('success') is True
                and c.get('ran') is True and positive(c.get('kernel_runtime_cycles'))):
            issues.append('pass_without_positive_executed_rtl_latency')
        checks = correctness.get('checks') or []
        if not (correctness.get('passed') is True
                and (correctness.get('rtl_postcheck_dump_count') or 0) >= 1
                and checks and all(x.get('passed') is True for x in checks)):
            issues.append('pass_without_independent_rtl_postcheck')
        if not (csim.get('passed') is True and csim.get('success') is True
                and (csim.get('correctness') or {}).get('passed') is True):
            issues.append('pass_without_independent_csim')
        if not (golden.get('status') == 'passed' and golden.get('output_sha256')
                and golden['output_sha256'] == c.get('golden_output_sha256')
                == csim.get('golden_output_sha256')):
            issues.append('golden_identity_mismatch')
        expected_golden = row.get('historical_golden_output_sha256')
        if expected_golden and golden.get('output_sha256') != expected_golden:
            issues.append('historical_golden_identity_mismatch')
        registry = row.get('golden_registry') or {}
        measured_registry = golden.get('shape_registry') or {}
        for key in ('sha256', 'testbench_sha256'):
            if registry.get(key) and registry[key] != measured_registry.get(key):
                issues.append('golden_registry_' + key + '_mismatch')
    final_status = 'evidence_error' if issues else status
    if task_status == 'execution_error':
        final_status = 'execution_error'
    correctness = c.get('correctness') or {}
    return {'status': final_status,
            'rtl_cycles': c.get('kernel_runtime_cycles') if final_status == 'pass' else None,
            'recorded_status': status, 'issues': issues,
            'worker_seconds': c.get('runtime_seconds'),
            'error': c.get('error', ''), 'failure_reason': correctness.get('reason', ''),
            'started_at': measurement.get('started_at'),
            'finished_at': measurement.get('finished_at')}


def group_rows(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[row['validation_id']].append(row)
    for ident, group in groups.items():
        for field in ('benchmark', 'kernel_sha256', 'input_hashes', 'golden_registry', 'historical_toolchain'):
            if any(r.get(field) != group[0].get(field) for r in group):
                raise ValueError(f'Conflicting deduplicated {field}: {ident}')
    return groups


def measure(snapshot, cohort, rows, raw_root, tasks=None, live=False, original_ids=None, toolchain=None):
    results = []
    for ident, group in sorted(group_rows(rows).items()):
        row = group[0]
        path = Path(raw_root) / 'measurements' / (ident + '.json')
        task = (tasks or {}).get(ident)
        m = snapshot.read(path) if path.is_file() else None
        item = classify(row, m, task, live)
        if m is not None and toolchain is not None and m.get('toolchain') != toolchain:
            item.update(status='evidence_error', rtl_cycles=None)
            item['issues'].append('campaign_toolchain_mismatch')
        if m is not None and task and task.get('measurement_sha256'):
            if snapshot.sources[str(path.resolve())]['sha256'] != task['measurement_sha256']:
                item.update(status='evidence_error', rtl_cycles=None)
                item['issues'].append('controller_measurement_hash_mismatch')
        item.update(cohort=cohort, validation_id=ident, benchmark=row['benchmark'],
                    original_validation_id=(original_ids or {}).get(ident, ident),
                    kernel_sha256=row['kernel_sha256'], historical_row_count=len(group),
                    input_hashes=row.get('input_hashes'),
                    toolchain=m.get('toolchain') if m else row.get('historical_toolchain'),
                    measurement_path=str(path) if m is not None else None,
                    measurement_sha256=snapshot.sources[str(path.resolve())]['sha256'] if m is not None else None)
        results.append(item)
    return results


def merge_recovery(legacy, recovery):
    """Replace unfinished attempts only, preserving the 102-design denominator."""
    merged = {r['validation_id']: dict(r) for r in legacy}
    replaced = set()
    for row in recovery:
        prior = row['prior_replay_id']
        if prior in replaced or prior not in merged:
            raise ValueError('Duplicate or unregistered recovery link')
        if merged[prior]['recorded_status'] in TERMINAL:
            raise ValueError('Recovery would overwrite a terminal replay')
        if (merged[prior]['kernel_sha256'] != row['kernel_sha256']
                or merged[prior]['input_hashes'] != row['input_hashes']):
            raise ValueError('Recovery source/input mismatch')
        merged[prior] = row
        replaced.add(prior)
    return list(merged.values())


def pair_reviewers(rows):
    groups = defaultdict(dict)
    for row in rows:
        key = (row['benchmark'], row['strategy'])
        if row['arm'] in groups[key]:
            raise ValueError('Duplicate reviewer arm')
        groups[key][row['arm']] = row
    pairs = []
    for (benchmark, strategy), arms in sorted(groups.items()):
        if set(arms) != {'review_off', 'review_on'}:
            raise ValueError('Missing reviewer arm')
        off, on = arms['review_off'], arms['review_on']
        matched = (bool(off['input_hashes']) and off['input_hashes'] == on['input_hashes']
                   and off.get('toolchain') == on.get('toolchain')
                   and bool(off.get('toolchain')) and off.get('repeat') == on.get('repeat'))
        ok = matched and off['status'] == on['status'] == 'pass'
        pairs.append({'benchmark': benchmark, 'strategy': strategy,
                      'status': 'paired_pass' if ok else 'unavailable',
                      'off_status': off['status'], 'on_status': on['status'],
                      'inputs_toolchain_repeat_match': matched,
                      'off_validation_id': off['validation_id'], 'on_validation_id': on['validation_id'],
                      'speedup': off['rtl_cycles'] / on['rtl_cycles'] if ok else None})
    return pairs


def count(rows):
    counts = Counter(r['status'] for r in rows)
    if set(counts) - set(STATUS_ORDER):
        raise ValueError(f'Unknown statuses: {set(counts) - set(STATUS_ORDER)}')
    return dict(counts)


def collect(snapshot, root):
    base = root / 'results_sweeps'
    hist_path = base / 'historical_registry_recovery_20260928'
    ab_path = base / 'reviewer_ab_rtl_r1_20261001'
    replay_path = base / 'sigmetrics_timeout_replay_20261001'
    rec_path = base / 'sigmetrics_timeout_recovery_20261006'
    hist = snapshot.read(hist_path / 'manifest.json')
    ab = snapshot.read(ab_path / 'manifest.json')
    legacy_hist = snapshot.read(replay_path / 'historical_timeout_replay_20261001/manifest.json')
    replay_state = snapshot.read(replay_path / 'state.json')
    rec_manifest = snapshot.read(rec_path / 'manifest.json')
    rec_state = snapshot.read(rec_path / 'state.json')
    rec_live = controller_alive(snapshot.read(rec_path / 'controller_identity.json'))
    # Read only once: these are evidence snapshots, not transactional database reads.
    for folder in (hist_path, ab_path, replay_path, rec_path):
        snapshot.read(folder / 'summary.json')
    historical = measure(snapshot, 'historical_1h', hist['rows'], hist['raw_root'], toolchain=hist['toolchain'])
    reviewer = measure(snapshot, 'reviewer_ab_1h', ab['rows'], ab['raw_root'], toolchain=ab['toolchain'])
    id_to_original = {r['validation_id']: r['replay_of_validation_id'] for r in legacy_hist['rows']}
    legacy = measure(snapshot, 'replay_3h', legacy_hist['rows'], legacy_hist['raw_root'],
                     {t['id']: t for t in replay_state['tasks'] if t['kind'] == 'historical'},
                     original_ids=id_to_original, toolchain=legacy_hist['toolchain'])
    hist_by_id = {r['validation_id']: r for r in historical}
    if set(id_to_original.values()) != {r['validation_id'] for r in historical if r['status'] == 'timeout'}:
        raise ValueError('Three-hour replay is not exactly the original timeout subset')
    for row in legacy:
        original = hist_by_id[row['original_validation_id']]
        if row['kernel_sha256'] != original['kernel_sha256'] or row['input_hashes'] != original['input_hashes']:
            raise ValueError('Replay changes original source or public inputs')
    recovery, diagnostic = [], []
    for cohort in ('recovery', 'diagnostic'):
        tasks = [t for t in rec_state['tasks'] if t['cohort'] == cohort]
        paths = {t['historical_output'] for t in tasks}
        for path in sorted(paths):
            m = snapshot.read(Path(path) / 'manifest.json')
            taskmap = {t['id']: t for t in tasks if t['historical_output'] == path}
            if set(taskmap) != set(group_rows(m['rows'])):
                raise ValueError('Recovery manifest/task mismatch')
            original_ids = {k: id_to_original[t['prior_id']] for k, t in taskmap.items()}
            items = measure(snapshot, 'replay_3h' if cohort == 'recovery' else 'diagnostic_6h',
                            m['rows'], m['raw_root'], taskmap, rec_live, original_ids, toolchain=m['toolchain'])
            for item in items:
                item['prior_replay_id'] = taskmap[item['validation_id']]['prior_id']
                item['timeout_seconds'] = taskmap[item['validation_id']]['timeout_seconds']
            (recovery if cohort == 'recovery' else diagnostic).extend(items)
    replay = merge_recovery(legacy, recovery)
    all_validations = historical + replay + diagnostic + reviewer
    cohorts = []
    for cid, label, limit, rows in [
            ('historical_1h', 'Original backfill (1 h)', 3600, historical),
            ('replay_3h', 'Timeout replay (3 h)', 10800, replay),
            ('diagnostic_6h', 'Separate diagnostic (6 h)', 21600, diagnostic),
            ('reviewer_ab_1h', 'Reviewer A/B, repeat 1 (1 h)', 3600, reviewer)]:
        cohorts.append({'id': cid, 'label': label, 'timeout_seconds': limit,
                        'n_unique': len(rows), 'counts': count(rows)})
    ab_by_id = {r['validation_id']: r for r in reviewer}
    reviewer_rows = []
    for row in ab['rows']:
        item = dict(ab_by_id[row['validation_id']])
        item.update({k: row[k] for k in ('strategy', 'arm', 'repeat', 'cell_id')})
        reviewer_rows.append(item)
    pairs = pair_reviewers(reviewer_rows)
    reviewer_arms = []
    for strategy in ('flash', 'dynamic'):
        for arm in ('review_off', 'review_on'):
            subset = [r for r in reviewer_rows if r['strategy'] == strategy and r['arm'] == arm]
            reviewer_arms.append({'strategy': strategy, 'arm': arm, 'n_cells': len(subset), 'counts': count(subset)})
    fails = Counter((r['benchmark'], r.get('failure_reason') or r.get('error') or 'unspecified')
                    for r in historical if r['status'] == 'fail')
    ever_pass = {r['validation_id'] for r in historical if r['status'] == 'pass'}
    ever_pass.update(r['original_validation_id'] for r in replay if r['status'] == 'pass')
    continuation = base / 'sigmetrics_main_continuation_20261006'
    main = snapshot.read(continuation / 'summary.json')
    main_live = controller_alive(snapshot.read(continuation / 'controller_identity.json'))
    parents = [{'benchmark': t['benchmark'], 'status': t['status']}
               for t in replay_state['tasks'] if t['kind'] == 'parent']
    return {'schema_version': 'c2hls.cosim-backfill-report.v1',
            'cohorts': cohorts, 'validations': all_validations,
            'historical_row_counts': dict(sum((Counter({r['status']: r['historical_row_count']}) for r in historical), Counter())),
            'historical_unique_verified_with_replay': len(ever_pass),
            'reviewer_rows': reviewer_rows, 'reviewer_pairs': pairs, 'reviewer_arms': reviewer_arms,
            'historical_failures': [{'benchmark': b, 'reason': reason, 'n_unique': n} for (b, reason), n in sorted(fails.items())],
            'parent_replay': parents, 'main_continuation': main,
            'main_controller_alive': main_live, 'recovery_controller_alive': rec_live,
            'recovery_counts': count(recovery), 'diagnostic_policy': rec_manifest['diagnostic_timeout_basis'],
            'evidence_issues': [{'cohort': r['cohort'], 'validation_id': r['validation_id'], 'issues': r['issues']}
                                for r in all_validations if r['issues']]}


def write_csv(path, rows):
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with Path(path).open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: json.dumps(v, sort_keys=True) if isinstance(v, (list, dict)) else v
                             for k, v in row.items()})


def write_report(output, data):
    lines = ['# RTL co-simulation backfill report',
             f"\nSnapshot: **{data['captured_at_local']}**. Capture completed {data['capture_finished_at']}.",
             '\nCounts below are distinct validation identities within each cohort. Cohorts overlap; do not add their denominators.',
             '\n| Cohort | Distinct validations | RTL pass | Validation failure | Timeout | Running | Pending | Other |',
             '|---|---:|---:|---:|---:|---:|---:|---:|']
    for c in data['cohorts']:
        n = c['counts']; other = sum(n.get(s, 0) for s in ('interrupted', 'execution_error', 'evidence_error'))
        lines.append(f"| {c['label']} | {c['n_unique']} | {n.get('pass',0)} | {n.get('fail',0)} | {n.get('timeout',0)} | {n.get('running',0)} | {n.get('pending',0)} | {other} |")
    lines += [f"\nThe original 405 validations map to 535 historical rows: {data['historical_row_counts']}. "
              'Repeated rows share validations and are not independent executions.',
              f"\nAcross the original backfill and its three-hour timeout sensitivity replay, **{data['historical_unique_verified_with_replay']}/405 designs** "
              'now have a passing independent RTL check on the saved public testbench. This is retrospective validation coverage across different time limits, not a common-budget success rate or an optimization gain.',
              '\nThe three-hour cohort includes 102 original timeouts. Interrupted attempts are linked to their recovery identities; terminal original results are never replaced. The six-hour diagnostic has its own denominator and is excluded from three-hour results.',
              '\n## Figures',
              '\nEach chart is exported as vector PDF and SVG plus a 300-dpi PNG. [Combined figure PDF](cosim_backfill_figures.pdf) contains the complete figure set. Numerical inputs are in `validations.csv`, `reviewer_cells.csv`, and `reviewer_pairs.csv`.']
    captions = [
        ('cosim_cohort_outcomes', 'Validation outcomes by timeout policy', 'Distinct validations within each cohort. Bars show percentages with counts. The replay is a selected subset of original timeouts; the six-hour diagnostic is separate. Outcomes are not independent across cohorts.'),
        ('cosim_benchmark_outcomes', 'Historical validation outcomes by kernel', 'Original one-hour backfill and three-hour replay use separate denominators. The displayed failures are validation failures, which can include harness or parser errors as well as numerical mismatches.'),
        ('reviewer_flash_rtl_cycles', 'Measured RTL latency: flash mode', 'DeepSeek reviewer A/B, repeat 1, public testbench. Bars use measured cycles only after independent C-output and RTL-output checks and source-identity checks. T denotes timeout and F denotes validation failure; neither has a latency value.'),
        ('reviewer_multistep_rtl_cycles', 'Measured RTL latency: multistep mode', 'Dynamic in saved records denotes multistep mode. Reviewer on/off use the same benchmark inputs. A single repeat provides descriptive paired observations, not confidence intervals.'),
        ('reviewer_paired_rtl_speedup', 'Paired RTL speedup from enabling the reviewer', 'Ratio of measured reviewer-off cycles to reviewer-on cycles; above one favors reviewer on. Only pairs passing both validations with matching inputs, toolchain, and repeat have a ratio. Missing pairs remain visible. This is a within-mode reviewer comparison, not a flash-versus-multistep effect.')]
    for name, title, caption in captions:
        if (output / (name + '.pdf')).exists():
            lines += [f'\n### {title}', f'\n![{title}]({name}.png)',
                      f'\n{caption} [PDF]({name}.pdf) · [SVG]({name}.svg)']
    lines += ['\n## Reviewer A/B cell coverage',
              '\nThere are 108 cells (27 kernels × two modes × reviewer on/off), linked to 102 distinct validations. A shared validation is displayed in each corresponding cell, but counted once in the cohort-level chart.',
              '\n| Mode | Reviewer | Cells | Pass | Failure | Timeout |', '|---|---|---:|---:|---:|---:|']
    for r in data['reviewer_arms']:
        c = r['counts']
        lines.append(f"| {'Multistep' if r['strategy']=='dynamic' else 'Flash'} | {r['arm']} | {r['n_cells']} | {c.get('pass',0)} | {c.get('fail',0)} | {c.get('timeout',0)} |")
    complete_pairs = Counter(p['strategy'] for p in data['reviewer_pairs'] if p['status'] == 'paired_pass')
    lines += [f"\nMeasured reviewer-off/on latency ratios are available for **{complete_pairs['flash']}/27 flash pairs** "
              f"and **{complete_pairs['dynamic']}/27 multistep pairs**. These incomplete subsets must not replace full-cohort outcome reporting."]
    trmm = [r for r in data['reviewer_rows'] if r['benchmark'] == 'hlsfactory_trmm']
    if trmm:
        lines += ['\nTRMM case-study check: ' + '; '.join(
            f"{r['strategy']}/{r['arm']}: {r['rtl_cycles']:,} RTL cycles" if r['status'] == 'pass'
            else f"{r['strategy']}/{r['arm']}: {r['status']}" for r in trmm) +
            '. Missing multistep measurements cannot establish an RTL-verified multistep advantage.']
    lines += ['\n## Failure triage', '\nOriginal failure labels are retained; these are not all demonstrated algorithmic failures.',
              '\n| Kernel | Recorded reason | Distinct validations |', '|---|---|---:|']
    for f in data['historical_failures']:
        lines.append(f"| {f['benchmark'].removeprefix('hlsfactory_')} | {f['reason'].replace('|','/')} | {f['n_unique']} |")
    lines += ['\nFDTD failures include previously diagnosed malformed/early-ended output dumps. They require a separately versioned harness repair and revalidation before assigning an algorithmic-correctness cause. Numerical mismatches also need candidate-specific diagnosis.',
              '\n## Interpretation and next evidence',
              '\n- These historical passes cover the saved public testbench. They do not establish all-input equivalence or the five-case main-study validation scope.',
              '- All 535 original historical rows retain a false reference-audit flag. RTL validation does not resolve generation/reference leakage; use these as historical validation evidence, not an uncontaminated causal comparison of models or skills.',
              '- Historical reference numbers are synthesis estimates. This report does not divide an estimated reference latency by measured candidate cycles and call it measured RTL speedup.',
              '- Timeout is a censored tool-wall-time outcome, not a numerical correctness failure. The cap includes candidate synthesis and co-simulation setup, execution, and output checking. Kernel latency cycles are not a wall-time deadline.',
              '- The replay is ordered by synthesis estimates and still incomplete. Current pass proportions are interim, selected-sample observations; keep timeout and pending counts visible.',
              '- Continue the existing three-hour replay and separate six-hour diagnostic. Use their completed-transaction and phase timing evidence to distinguish slow simulation from unresolved stalls before setting future limits.',
              '- For paper claims, use matched RTL/RTL pairs with all nonpasses reported and add the remaining registered repeats. Keep the historical audit separate from the frozen recipe-representation main study.',
              f"\nMain-study continuation (separate experiment): `{json.dumps(data['main_continuation']['cells'], sort_keys=True)}`. "
              f"Controller alive at capture: {data['main_controller_alive']}. These are cell execution states, not an asserted count of new optimized RTL designs.",
              f"\nThree-hour public-C parent replay (also separate): `{json.dumps(data['parent_replay'], sort_keys=True)}`. It does not retroactively qualify blocked frozen main-study parents.",
              f"\nEvidence consistency issues: **{len(data['evidence_issues'])}**. Any inconsistent pass is excluded from latency bars and retained as an evidence error.",
              '\n## Reproduce',
              '\nRun `python scripts/report_cosim_backfill.py` from this worktree to create a new timestamped snapshot without changing campaigns. Run `python scripts/plot_cosim_backfill_report.py --report <snapshot>/report.json` to redraw this snapshot. The source JSON files actually read are copied into `evidence/`; `sources.json` records SHA-256 hashes and read times. The original measurements remain unchanged.']
    (output / 'report.md').write_text('\n'.join(lines) + '\n')
    latex = [
        '% Copy the desired figure PDF(s) beside this snippet; requires graphicx.',
        '% Descriptive snapshot; retain timeout-policy and incomplete-pair caveats.',
        '\\begin{figure}[t]', '\\centering',
        '\\includegraphics[width=\\linewidth]{cosim_cohort_outcomes.pdf}',
        '\\caption{RTL co-simulation outcomes by validation policy. Counts are distinct validations within each cohort. '
        'The three-hour replay is a selected subset of the original one-hour timeouts, and the six-hour diagnostic is separate. '
        'Validation failures include harness and numerical-check failures. Cohort denominators must not be added.}',
        '\\label{fig:cosim-backfill-coverage}', '\\end{figure}', '',
        '\\begin{figure}[t]', '\\centering',
        '\\includegraphics[width=\\linewidth]{reviewer_flash_rtl_cycles.pdf}',
        '\\caption{Measured RTL latency for the flash-mode reviewer ablation (repeat 1). '
        'Bars require successful independent C-output and RTL-output checks for the selected source. '
        'T and F mark timeout and validation failure; neither is assigned a latency. '
        'The public testbench validation scope differs from the five-case main study.}',
        '\\label{fig:reviewer-flash-rtl}', '\\end{figure}', '',
        '\\begin{figure}[t]', '\\centering',
        '\\includegraphics[width=\\linewidth]{reviewer_multistep_rtl_cycles.pdf}',
        '\\caption{Measured RTL latency for the multistep reviewer ablation (repeat 1), using the same conventions as '
        'Figure~\\ref{fig:reviewer-flash-rtl}. All nonpassing configurations remain visible. '
        'Single-repeat measurements support descriptive comparisons, not statistical significance claims.}',
        '\\label{fig:reviewer-multistep-rtl}', '\\end{figure}', '']
    (output / 'figure_captions.tex').write_text('\n'.join(latex))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--no-plots', action='store_true')
    args = parser.parse_args()
    start = now()
    output = args.output or args.root / 'artifacts/sigmetrics2027/backfill_reports' / start.strftime('%Y%m%dT%H%M%S%fZ')
    output.mkdir(parents=True, exist_ok=False)
    snapshot = Snapshot(output)
    data = collect(snapshot, args.root.resolve())
    data.update(captured_at=start.isoformat(), captured_at_local=start.astimezone(ZoneInfo('America/Chicago')).isoformat(),
                capture_finished_at=now().isoformat())
    (output / 'report.json').write_text(json.dumps(data, indent=2) + '\n')
    (output / 'sources.json').write_text(json.dumps(snapshot.sources, indent=2) + '\n')
    write_csv(output / 'validations.csv', data['validations'])
    write_csv(output / 'reviewer_cells.csv', data['reviewer_rows'])
    write_csv(output / 'reviewer_pairs.csv', data['reviewer_pairs'])
    write_csv(output / 'failure_and_timeout_index.csv', [r for r in data['validations'] if r['status'] != 'pass'])
    shutil.copy2(Path(__file__), output / 'report_cosim_backfill.py')
    if not args.no_plots:
        from plot_cosim_backfill_report import render_report
        render_report(data, output)
        shutil.copy2(Path(__file__).with_name('plot_cosim_backfill_report.py'), output / 'plot_cosim_backfill_report.py')
    write_report(output, data)
    print(json.dumps({'output': str(output), 'cohorts': data['cohorts'], 'evidence_issues': len(data['evidence_issues'])}, indent=2))
    if data['evidence_issues']:
        sys.exit(2)


if __name__ == '__main__':
    main()
