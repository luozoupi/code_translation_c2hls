#!/usr/bin/env python3
"""Write a bounded, metadata-only map of experiment directories for handoff."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
HOME = Path('/home/luo00466')
OLD = HOME / 'code_translation-c2hls'
NAS2 = Path('/mnt/data2/luo00466')
NAS1 = Path('/mnt/data1/luo00466')


def command(args):
    p = subprocess.run(args, cwd=REPO, text=True, capture_output=True, timeout=30)
    return dict(returncode=p.returncode, stdout=p.stdout.strip(), stderr=p.stderr.strip())


def entry(path):
    p = Path(path)
    result = dict(path=str(p), name=p.name, exists=p.exists(), symlink=p.is_symlink())
    if p.is_symlink():
        result['link_target'] = os.readlink(p)
        result['resolved_path'] = str(p.resolve())
    if p.exists():
        stat = p.stat()
        result.update(kind='directory' if p.is_dir() else 'file',
                      bytes=None if p.is_dir() else stat.st_size,
                      modified_at=datetime.fromtimestamp(stat.st_mtime, timezone.utc).isoformat())
        if p.is_dir() and (p / 'payload.tar.zst').is_file():
            result.update(kind='archive_bundle_directory',
                          archive_metadata=str(p / 'metadata.json'),
                          transparent_historical_directory=False)
    return result


def inventory(root):
    value = entry(root)
    value['entries'] = []
    if root.is_dir():
        for path in sorted(root.iterdir()):
            if path.name in {'__pycache__', '.pytest_cache'}:
                continue
            item = entry(path)
            # List campaign contents, but never descend into Vitis work or checkpoints.
            if path.is_dir() and not path.is_symlink():
                item['immediate_child_names'] = sorted(p.name for p in path.iterdir())
            value['entries'].append(item)
    return value


def safe_campaign_snapshot(path):
    fields = {'status', 'updated_at', 'finished_at', 'pid', 'counts', 'planned_runs',
              'audit_valid_runs', 'feasible_runs', 'planned_pairs', 'valid_pairs',
              'benchmark_clusters', 'effects', 'geomean_speedup_on_over_off',
              'benchmark_cluster_bootstrap_95_interval', 'arm_counts', 'active', 'pending',
              'completed_this_invocation', 'resource_gate', 'historical_rows',
              'unique_queued_validations', 'finished_new_validations', 'row_status_counts',
              'measurement_status_counts', 'schema_validation'}
    if not path.is_file():
        return entry(path)
    value = json.loads(path.read_text())
    return dict(path=str(path), snapshot={k: v for k, v in value.items() if k in fields})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=REPO / 'artifacts/handoff')
    args = parser.parse_args()
    roots = [REPO / rel for rel in ('results_sweeps', 'artifacts', 'artifacts/experiment_matrix',
        'artifacts/setup_router', 'artifacts/analysis', 'artifacts/phase_b_seeds',
        'artifacts/model_comparison', 'artifacts/qwen_cosim_enrichment', 'results', 'results_multistep')]
    roots += [OLD / rel for rel in ('artifacts', 'artifacts/experiment_matrix', 'artifacts/rl_corpus',
        'artifacts/vllm_baseline', 'artifacts/vllm_vitis_smoke', 'artifacts/vllm_cosim_compare',
        'results_sweeps', 'results', 'results_multistep')]
    roots += [NAS2 / rel for rel in ('c2hls_rl', 'c2hls_rl/skill_v3_fc27133',
        'c2hls_rl/july_cosim_backfill_20260928', 'c2hls_rl/deepseek_flash_reviewer_ab27_20260925',
        'c2hls_sft', 'c2hls_sft/runs', 'c2hls_sft/corpus', 'c2hls_sft/eval',
        'c2hls_sft/serving', 'code_translation-c2hls_archive_20260706_1245',
        'home_cleanup_20260706_approved')]
    roots += [NAS1 / rel for rel in ('c2hls_archive_20260811', 'research_archive_20260917')]
    branches = [REPO, OLD, HOME / 'code_translation-c2hls-enhanced-fc27133', HOME / 'code_translation_c2hls']
    data = dict(schema_version='c2hls.session-handoff-index.v1',
                captured_at=datetime.now(timezone.utc).isoformat(),
                scope='Root entries plus one level of child names; not a recursive file/size audit.',
                secrets_policy='No credentials, prompt text, raw logs, or arbitrary manifest contents are read.',
                worktrees=command(['git', 'worktree', 'list']),
                repositories=[dict(path=str(p), branch=command(['git', '-C', str(p), 'branch', '--show-current']),
                    head=command(['git', '-C', str(p), 'rev-parse', 'HEAD'])) for p in branches if p.is_dir()],
                git_status=command(['git', 'status', '--short']),
                tmux_sessions=command(['tmux', 'list-sessions']),
                roots=[inventory(root) for root in roots],
                reference_files=[entry(p) for p in [HOME / 'schema_records.jsonl',
                    OLD / 'csynth_vitis_2023.2__device_xilinx_u280_gen3x16_xdma_1_202211_1.jsonl']],
                live_snapshots=[safe_campaign_snapshot(REPO / 'results_sweeps' / rel) for rel in (
                    'july_cosim_backfill_20260928/state.json', 'july_cosim_backfill_20260928/summary.json',
                    'deepseek_flash_reviewer_ab27_20260925/campaign_state.json',
                    'deepseek_flash_reviewer_ab27_20260925/summary.json')])
    tracked = command(['git', 'diff', '--name-only'])['stdout'].splitlines()
    untracked = command(['git', 'ls-files', '--others', '--exclude-standard'])['stdout'].splitlines()
    data['changed_source_hashes'] = {name: hashlib.sha256((REPO / name).read_bytes()).hexdigest()
        for name in sorted(set(tracked + untracked)) if Path(name).suffix in {'.py', '.md'} and (REPO / name).is_file()}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.output_dir / 'handoff_index_20260928.json'
    pending = json_path.with_suffix('.json.tmp')
    pending.write_text(json.dumps(data, indent=2, sort_keys=True) + '\n')
    pending.replace(json_path)
    lines = ['# Handoff Directory Index', '', 'Generated: ' + data['captured_at'], '',
             data['scope'], '', 'See `docs/session_handoff_20260928.md` for interpretation and operations.', '',
             '| Root | Direct entries | Kind |', '|---|---:|---|']
    for root in data['roots']:
        lines.append(f"| `{root['path']}` | {len(root['entries'])} | {root.get('kind', 'missing')} |")
    lines += ['', '## Result Campaigns', '']
    for root in data['roots']:
        if root['path'] == str(REPO / 'results_sweeps'):
            lines += ['- `' + x['path'] + '`' for x in root['entries'] if x.get('kind') == 'directory']
    lines += ['', '## Live Session Snapshot', '', '```text', data['tmux_sessions']['stdout'] or '(none)', '```', '',
              'Re-read the live state files before acting; this index is a snapshot.']
    (args.output_dir / 'handoff_index_20260928.md').write_text('\n'.join(lines) + '\n')
    print(json.dumps(dict(index=str(json_path), roots=len(roots),
                         entries=sum(len(r['entries']) for r in data['roots']), captured_at=data['captured_at'])))


if __name__ == '__main__':
    main()
