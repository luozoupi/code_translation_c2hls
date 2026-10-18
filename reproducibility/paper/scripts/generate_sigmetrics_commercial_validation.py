#!/usr/bin/env python3
"""SIGMETRICS companion figure: fixed synthesis selections and RTL validation.

The original plotting inputs and outputs remain immutable. Both plotted measures
use recipe-disabled cycles / selected-design cycles within one evidence tier.
C simulation establishes functional test outcomes, not hardware cycle latency;
the pre-existing cycle bars are HLS synthesis estimates.

Run with --freeze-review-join once to import/verify the frozen local audit. Normal
reproduction reads only committed CSV inputs, checks their SHA-256 identities,
and never depends on live experiment folders or reselects a design using RTL.
The optional --freeze-historical-synthesis import verifies the original July-30
CSynth/CoSim exports and freezes their source identifiers and worst-latency fields.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

import generate_aligned_model_expertise as historical
import generate_july30_results as july30

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'generated/sigmetrics'
JOIN_SHA256 = 'e6bb3768d784ea954854fc2dc5b6b2f25d166ed8309b105bc54a4cfc4eb4a8af'
EVIDENCE_SHA256 = '2cf37dd742d06855403cd106096867c9ae2590e182cdbf48e8932361b40d5db9'
EVIDENCE = OUT / 'commercial_validation_evidence.csv'
RECEIPT = OUT / 'commercial_validation_evidence_receipt.json'
HISTORICAL_SYNTHESIS = OUT / 'commercial_validation_historical_synthesis.csv'
HISTORICAL_RECEIPT = OUT / 'commercial_validation_historical_synthesis_receipt.json'
HISTORICAL_SYNTHESIS_SHA256 = '5ae5364da54a3cf33f65bfd9e130646a70215c8d32632d03ca88a209cb664074'
MODELS = {'SONNET': 'Claude Sonnet 4.6', 'OPUS': 'Claude Opus 4.8', 'GPT55': 'GPT-5.5'}
FROZEN_COUNTS = {'SONNET': (25, 24), 'OPUS': (26, 25), 'GPT55': (23, 23)}
EVIDENCE_FIELDS = (
    'model', 'kernel', 'role', 'arm', 'paper_cycles', 'cosim_cycles', 'status',
    'job', 'source_sha1', 'source_sha256', 'source_report_binding',
    'original_common_cohort', 'original_partial_grid', 'rtl_evidence_tier',
    'vitis_pass_log_marker', 'testbench_hash_recorded',
    'independent_reference_output_check', 'timeout_s', 'record_sha256',
)
SYNTH_COLOR = '#648DB1'
RTL_COLOR = '#BC6840'


def rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline='', encoding='utf-8') as stream:
        return list(csv.DictReader(stream))


def write_csv(path: Path, records: list[dict[str, Any]], fields=None) -> None:
    with path.open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields or list(records[0]), lineterminator='\n')
        writer.writeheader()
        writer.writerows(records)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def positive(value: str | float) -> float:
    result = float(value)
    require(math.isfinite(result) and result > 0, f'Invalid cycle count: {value!r}')
    return result


def freeze_review_join(path: Path) -> None:
    """Verify the archived source and Vitis evidence before freezing a small CSV."""
    require(historical.sha256(path) == JOIN_SHA256, 'Review join changed; audit again before import')
    source = rows(path)
    frozen = []
    for row in source:
        require(row['hash_verifies'] == row['archive_cpp_matches_plan'] == row['cycle_match'] == 'True',
                'Original source/report identity is unresolved')
        code = Path(row['source_path']).read_bytes()
        require(hashlib.sha1(code).hexdigest() == row['source_sha1'], 'Source SHA-1 mismatch')
        require(hashlib.sha256(code).hexdigest() == row['source_sha256'], 'Source SHA-256 mismatch')
        record_path = Path(row['record_path'])
        record = json.loads(record_path.read_text())
        require(record['code_sha1'] == row['source_sha1'], 'Vitis record has different source')
        require(record['status'] == row['status'], 'Vitis status changed')
        require(float(record['csynth_cycles']) == float(row['paper_cycles']), 'Synthesis selection changed')
        if row['status'] == 'passed':
            require(positive(record['cosim_cycles']) == positive(row['cosim_cycles']), 'RTL cycles changed')
            require('co-simulation finished: PASS' in record.get('log_tail', ''), 'No Vitis PASS evidence')
            if row['rtl_evidence_tier'] == 'embedded_report_and_PASS_log':
                report = record.get('measured', {})
                require(report.get('rpt_status') == 'Pass' and report.get('report_text') and
                        positive(report['latency_cycles_avg']) == positive(row['cosim_cycles']),
                        'Embedded report missing or inconsistent')
        frozen.append({**{field: row[field] for field in EVIDENCE_FIELDS if field != 'record_sha256'},
                       'record_sha256': historical.sha256(record_path)})
    OUT.mkdir(parents=True, exist_ok=True)
    write_csv(EVIDENCE, frozen, EVIDENCE_FIELDS)
    RECEIPT.write_text(json.dumps({
        'source': str(path.relative_to(ROOT)), 'source_sha256': JOIN_SHA256,
        'frozen_csv_sha256': historical.sha256(EVIDENCE), 'role_records': len(frozen),
        'verified': ['source SHA-1 and SHA-256', 'original synthesis cycles', 'Vitis record source identity',
                     'Vitis completion status', 'positive RTL cycles and PASS marker for passes',
                     'embedded report status and cycles where available'],
        'validation_scope': 'Reported Vitis C/RTL consistency',
        'independent_reference_output_check': 'not_recorded_in_this_bundle',
        'testbench_hash_recorded': False,
    }, indent=2) + '\n')


def validate_evidence(records: list[dict[str, str]]) -> None:
    require(len(records) == 156, 'Expected 156 frozen selected/off records')
    seen = set()
    for row in records:
        key = (row['model'], row['kernel'], row['role'])
        require(key not in seen, f'Duplicate role: {key}')
        seen.add(key)
        require(row['model'] in MODELS and row['role'] in {'selected', 'off'}, f'Unexpected role: {key}')
        require(row['source_report_binding'] == 'source_report_and_csim_linked', f'Unlinked source: {key}')
        positive(row['paper_cycles'])
        require(row['status'] in {'passed', 'timeout', 'env_error'}, f'Unexpected status: {key}')
        if row['status'] == 'passed':
            positive(row['cosim_cycles'])
            require(row['vitis_pass_log_marker'] == 'True', f'Pass lacks log marker: {key}')
            require(row['rtl_evidence_tier'] in {'embedded_report_and_PASS_log', 'parsed_cycles_and_PASS_log'},
                    f'Unrecognized pass evidence tier: {key}')
        else:
            require(not row['cosim_cycles'] and row['rtl_evidence_tier'] == 'no_completed_RTL',
                    f'Incomplete RTL outcome must have no plotted cycles: {key}')


def freeze_historical_synthesis(results_dir: Path) -> None:
    """Import the checksum-bound synthesis reports for the existing RTL selections."""
    frozen, inputs = [], {}
    for model, rtl_name, synth_name in (
        ('DeepSeek V4 Flash', july30.DEEPSEEK_FILE, july30.DEEPSEEK_CSYNTH_FILE),
        ('Grok 4.5', july30.GROK_FILE, july30.GROK_CSYNTH_FILE),
    ):
        for name in (rtl_name, synth_name):
            inputs[name] = july30.require_hash(results_dir / name)
        rtl = july30.read_cosim_csv(results_dir / rtl_name)
        july30.cross_check_csynth_cosim(results_dir / synth_name, rtl)
        rtl_by_bench = {row.bench: row for row in rtl}
        for row in rows(results_dir / synth_name):
            old = rtl_by_bench[row['bench']]
            frozen.append({
                'model': model, 'kernel': row['bench'], 'cohort': row['set'],
                'selected_synthesis_cycles': row['skills_latency_worst'],
                'off_synthesis_cycles': row['noskills_latency_worst'],
                'selected_synthesis_field': 'skills_latency_worst',
                'off_synthesis_field': 'noskills_latency_worst',
                'selected_rtl_cycles': row['skills_flash_cosim_cycles'],
                'off_rtl_cycles': row['noskills_flash_cosim_cycles'],
                'selected_pass': row['skills_flash_pass'], 'off_pass': row['noskills_flash_pass'],
                'selected_source': row['skills_source'], 'off_source': row['noskills_source'],
                'rtl_selected_source': old.enabled_source, 'rtl_off_source': old.disabled_source,
                'selected_report': row['skills_report'], 'off_report': row['noskills_report'],
            })
    OUT.mkdir(parents=True, exist_ok=True)
    write_csv(HISTORICAL_SYNTHESIS, frozen)
    HISTORICAL_RECEIPT.write_text(json.dumps({
        'inputs': inputs, 'frozen_csv_sha256': historical.sha256(HISTORICAL_SYNTHESIS),
        'records': len(frozen),
        'verified': ['pinned July-30 input SHA-256 identities', 'same benchmark and cohort',
                     'same selected/off campaign source identifiers',
                     'same reported RTL cycles and pass flags', 'original selected-report paths'],
        'synthesis_fields': ['skills_latency_worst', 'noskills_latency_worst'],
        'evidence_scope': 'Compact reported synthesis and RTL tables; raw generated sources and tool reports unavailable',
        'selection': 'Original selected implementations retained; no selection by synthesis or RTL cycles during import',
    }, indent=2) + '\n')


def historical_synthesis_panels(records, rtl_models, original_rtl_rows):
    """Join reported CSynth values only to the exact previously plotted RTL cells."""
    require(len(records) == 56, 'Expected 56 historical synthesis records')
    archived = {(r['model_label'], r['bench']): r for r in original_rtl_rows}
    panels, seen = {model: {} for model in rtl_models}, set()
    for record in records:
        model, bench = record['model'], record['kernel']
        key = (model, bench)
        require(key in archived and key not in seen, f'Unexpected or duplicate historical cell: {key}')
        seen.add(key)
        row, original = rtl_models[model][bench], archived[key]
        require(record['cohort'] == original['cohort'], f'{key}: historical cohort mismatch')
        synth_cycles = {}
        for role, prefix in [('selected', 'skills'), ('off', 'noskills')]:
            expected_source = record[f'rtl_{role}_source']
            if expected_source.startswith('best(') and expected_source.endswith(')'):
                expected_source = expected_source[5:-1]
            require(record[f'{role}_source'] == expected_source, f'{key}: historical source identity mismatch')
            require(record[f'{role}_synthesis_field'] == f'{prefix}_latency_worst',
                    f'{key}: historical synthesis metric changed')
            report = record[f'{role}_report']
            require(report.endswith(f'/hlsfactory_{bench}_selected_report.json'),
                    f'{key}: historical selected-report link mismatch')
            synthesis = record[f'{role}_synthesis_cycles']
            synth_cycles[role] = positive(synthesis) if synthesis else None
            enabled = role == 'selected'
            require(record[f'{role}_pass'] == str(row['enabled_pass' if enabled else 'disabled_pass']),
                    f'{key}: historical RTL pass mismatch')
            recorded_rtl = record[f'{role}_rtl_cycles']
            require((positive(recorded_rtl) if recorded_rtl else None) ==
                    row['enabled_cycles' if enabled else 'disabled_cycles'],
                    f'{key}: historical RTL cycles mismatch')
        paired_synthesis = all(value is not None for value in synth_cycles.values())
        panels[model][bench] = {
            'synthesis_ratio': synth_cycles['off'] / synth_cycles['selected'] if paired_synthesis else None,
            'rtl_ratio': row['ratio'], 'partial': False, 'asymmetric': row['asymmetric'],
            'eligible': row['baseline_cycles'] is not None,
            'common': row['primary'] and row['baseline_cycles'] is not None and row['ratio'] is not None,
            'selected_status': 'passed' if row['enabled_pass'] else 'not_passed',
            'off_status': 'passed' if row['disabled_pass'] else 'not_passed',
            'selected_rtl_cycles': row['enabled_cycles'], 'off_rtl_cycles': row['disabled_cycles'],
            'selected_synthesis_cycles': synth_cycles['selected'], 'off_synthesis_cycles': synth_cycles['off'],
            'synthesis_evidence': record,
        }
    for model, panel in panels.items():
        require(sum(p['common'] for p in panel.values()) == (22 if model.startswith('DeepSeek') else 21),
                'Historical RTL table cohort changed')
    return panels


def commercial_panels(records: list[dict[str, str]], original: dict[str, list[dict[str, str]]]):
    validate_evidence(records)
    grouped = defaultdict(dict)
    for row in records:
        grouped[row['model'], row['kernel']][row['role']] = row
    columns = {
        'SONNET': ('benchmark', 'best_arm', 'best_cycles', 'best_noskill_arm', 'best_noskill_cycles'),
        'OPUS': ('bench', 'selected_setup', 'selected_cycles', 'best_expertise_disabled_setup', 'best_expertise_disabled_cycles'),
        'GPT55': ('benchmark', 'selected_setup', 'selected_cycles', 'best_no_skills_setup', 'best_no_skills_cycles'),
    }
    panels = {}
    for model, source in original.items():
        b, sa, sc, oa, oc = columns[model]
        panel = {}
        require(len(source) == 26, f'{model}: original selection coverage changed')
        for archived in source:
            bench = archived[b]
            pair = grouped[model, bench]
            require(set(pair) == {'selected', 'off'}, f'{model}/{bench}: missing role')
            selected, off = pair['selected'], pair['off']
            require(selected['arm'] == archived[sa] and off['arm'] == archived[oa],
                    f'{model}/{bench}: original selected/off arm changed')
            require(positive(selected['paper_cycles']) == positive(archived[sc]) and
                    positive(off['paper_cycles']) == positive(archived[oc]),
                    f'{model}/{bench}: original synthesis cycles changed')
            partial = bench == 'doitgen' if model == 'SONNET' else (
                archived['partial_setup_grid'] == 'True' if model == 'GPT55' else False)
            require(selected['original_common_cohort'] == off['original_common_cohort'] == str(not partial),
                    f'{model}/{bench}: common cohort changed')
            require(selected['original_partial_grid'] == off['original_partial_grid'] == str(partial),
                    f'{model}/{bench}: partial grid changed')
            both_pass = selected['status'] == off['status'] == 'passed'
            panel[bench] = {
                'synthesis_ratio': positive(off['paper_cycles']) / positive(selected['paper_cycles']),
                'rtl_ratio': positive(off['cosim_cycles']) / positive(selected['cosim_cycles']) if both_pass else None,
                'selected': selected, 'off': off, 'partial': partial, 'asymmetric': False,
                'eligible': not partial, 'common': not partial and both_pass,
                'selected_status': selected['status'], 'off_status': off['status'],
            }
        require(sum(p['eligible'] for p in panel.values()) == FROZEN_COUNTS[model][0], 'Eligible cohort drift')
        require(sum(p['common'] for p in panel.values()) == FROZEN_COUNTS[model][1], 'Paired cohort drift')
        panels[MODELS[model]] = panel
    return panels


def load_data():
    require(historical.sha256(EVIDENCE) == EVIDENCE_SHA256, 'Frozen evidence CSV changed')
    files = {name: ROOT / 'generated' / name for name in historical.EXPECTED_HASHES}
    input_hashes = {str(path.relative_to(ROOT)): historical.require_hash(path) for path in files.values()}
    original = {model: rows(files[name]) for model, name in (
        ('SONNET', 'newskills_sonnet_snapshot.csv'), ('OPUS', 'july30_opus_selection.csv'),
        ('GPT55', 'newskills_gpt55_snapshot.csv'))}
    evidence = rows(EVIDENCE)
    panels = commercial_panels(evidence, original)
    cosim = historical.read_cosim(files['july30_cosim_expertise.csv'])
    require(historical.sha256(HISTORICAL_SYNTHESIS) == HISTORICAL_SYNTHESIS_SHA256,
            'Frozen historical synthesis CSV changed')
    panels.update(historical_synthesis_panels(rows(HISTORICAL_SYNTHESIS), cosim['models'],
                                             rows(files['july30_cosim_expertise.csv'])))
    input_hashes[str(EVIDENCE.relative_to(ROOT))] = historical.sha256(EVIDENCE)
    input_hashes[str(HISTORICAL_SYNTHESIS.relative_to(ROOT))] = historical.sha256(HISTORICAL_SYNTHESIS)
    return panels, evidence, input_hashes


def summary(panels):
    result = {}
    for model, panel in panels.items():
        common = [p for p in panel.values() if p['common']]
        rtl = [p['rtl_ratio'] for p in common]
        hls = [p['synthesis_ratio'] for p in common if p['synthesis_ratio'] is not None]
        result[model] = {
            'eligible': sum(p['eligible'] for p in panel.values()), 'paired': len(common),
            'synthesis_paired': len(hls),
            'synthesis_off_over_selected_gm': historical.geometric_mean(hls) if len(hls) == len(common) else None,
            'rtl_off_over_selected_gm': historical.geometric_mean(rtl),
            'rtl_wins': sum(r > 1 for r in rtl), 'rtl_ties': sum(r == 1 for r in rtl),
            'rtl_losses': sum(r < 1 for r in rtl),
            'included_kernels': [b for b, p in panel.items() if p['common']],
            'omitted_kernels': [b for b, p in panel.items() if not p['common']],
            'missing_synthesis_kernels': [b for b, p in panel.items() if p['synthesis_ratio'] is None],
        }
    return result


def ratio_bar(ax, x, ratio, width, color, excluded=False):
    if ratio == 1:
        ax.scatter(x, 0, marker='o', s=7, facecolor='white', edgecolor=color, linewidth=.6, zorder=5)
    else:
        ax.bar(x, math.log2(ratio), width=width, color=color, edgecolor='#333333', linewidth=.25,
               hatch='///' if excluded else None, alpha=.55 if excluded else 1, zorder=3)


def draw(panels, summaries, output):
    font_path = font_manager.findfont('Times New Roman', fallback_to_default=False)
    plt.rcParams.update({'font.family': 'Times New Roman', 'font.size': 6.4,
                         'pdf.fonttype': 42, 'ps.fonttype': 42, 'axes.linewidth': .5,
                         'hatch.linewidth': .4, 'xtick.labelsize': 5.5, 'ytick.labelsize': 5.4})
    fig, axes = plt.subplots(5, 1, figsize=(7.08, 4.80), sharex=True,
                             gridspec_kw={'left': .080, 'right': .992, 'top': .863, 'bottom': .135, 'hspace': .49})
    for index, ((model, panel), ax) in enumerate(zip(panels.items(), axes)):
        commercial = index < 3
        if commercial:
            limits, ticks, labels = (-2.3, 6.65), [-2, 0, 2, 4, 6], ['¼', '1', '4', '16', '64']
        elif index == 3:
            limits, ticks, labels = (-2.5, 10.7), [-2, 0, 4, 8], ['¼', '1', '16', '256']
        else:
            limits, ticks, labels = (-7.3, 5.7), [-6, -2, 0, 4], ['1/64', '¼', '1', '16']
        plotted = [math.log2(p[key]) for p in panel.values()
                   for key in ('synthesis_ratio', 'rtl_ratio') if p[key] is not None]
        limits = (min(limits[0], min(plotted) - .65), max(limits[1], max(plotted) + .85))
        ax.set_ylim(*limits)
        ax.set_yticks(ticks, labels)
        ax.axhline(0, color='#333333', linewidth=.6, zorder=2)
        ax.grid(axis='y', color='#DEDEDE', linewidth=.35, zorder=0)
        ax.spines[['top', 'right']].set_visible(False)
        ax.tick_params(axis='y', length=2, pad=2)
        ax.tick_params(axis='x', length=2 if index == 4 else 0)
        ax.set_facecolor('#FAFAFA')
        sm = summaries[model]
        detail = f"paired {sm['paired']}/{sm['eligible']}"
        if sm['synthesis_paired'] != sm['paired']:
            detail += f"; CSynth {sm['synthesis_paired']}/{sm['paired']}"
        ax.set_title(f'({chr(97+index)}) {model}   |   {detail}', loc='left', fontsize=7.2, fontweight='bold', pad=2)
        for x, bench in enumerate(historical.CANONICAL_28):
            p = panel.get(bench)
            if p is None:
                for offset in (-.19, .19):
                    ax.text(x+offset, .60, '—', ha='center', va='center', fontsize=6, color='#777777')
                continue
            excluded = p['partial'] or p['asymmetric'] or (not commercial and not p['eligible'])
            if p['synthesis_ratio'] is not None:
                ratio_bar(ax, x-.19, p['synthesis_ratio'], .34, SYNTH_COLOR, excluded)
            else:
                ax.text(x-.19, .68, '—', ha='center', color=SYNTH_COLOR, fontsize=6)
            if p['rtl_ratio'] is not None:
                ratio_bar(ax, x+.19, p['rtl_ratio'], .34, RTL_COLOR, excluded)
            else:
                flag = 'T' if 'timeout' in [p['selected_status'], p['off_status']] else (
                    'E' if 'env_error' in [p['selected_status'], p['off_status']] else 'F')
                ax.text(x+.19, .68, flag, ha='center', color='#B32025', fontsize=6.5, fontweight='bold')
            if excluded and p['rtl_ratio'] is not None:
                ratios = [r for r in [p['synthesis_ratio'], p['rtl_ratio']] if r is not None]
                top = max([0] + [math.log2(r) for r in ratios])
                ax.text(x, top+.25, '†' if p['partial'] else ('‡' if p['asymmetric'] else '*'),
                        fontsize=6, ha='center', va='bottom', color='#555555')
        ax.set_xlim(-.65, 27.65)
    axes[-1].set_xticks(range(28), historical.CANONICAL_28, rotation=51, ha='right', rotation_mode='anchor')
    fig.supylabel('Recipe-disabled / selected cycles  (log scale; >1 favors selected)', x=.013, fontsize=6.5)
    handles = [Patch(facecolor=SYNTH_COLOR, edgecolor='#333333', label='CSynth estimate (left)'),
               Patch(facecolor=RTL_COLOR, edgecolor='#333333', label='CoSim measured cycles (right)'),
               Line2D([], [], marker='o', markerfacecolor='white', markeredgecolor='#444444', linestyle='none',
                      markersize=3.5, label='Equal cycles'),
               Patch(facecolor='#B8B8B8', edgecolor='#333333', hatch='///', label='Excluded from summary')]
    fig.legend(handles=handles, loc='upper center', bbox_to_anchor=(.53, .997), ncol=4,
               fontsize=6.1, frameon=False, handlelength=1.5, columnspacing=.9)
    fig.text(.53, .928, 'T: timeout    E: tool / environment error    F: no C/RTL pass    —: unavailable / not run', ha='center', fontsize=5.9)
    fig.text(.53, .904, '† Partial setup grid    ‡ Asymmetric rerun    * Outside original table cohort; y-ranges differ', ha='center', fontsize=5.9)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output.with_suffix('.pdf'), metadata={'Title': 'Fixed-selection HLS and RTL cycle ratios',
                'Author': 'Anonymous', 'Creator': Path(__file__).name, 'CreationDate': None, 'ModDate': None})
    fig.savefig(output.with_suffix('.png'), dpi=250)
    plt.close(fig)
    return font_path


def table(summaries, path):
    lines = [r'% AUTO-GENERATED by scripts/generate_sigmetrics_commercial_validation.py.',
             r'\begin{table}[t]', r'\centering',
             r'\caption{Paired-cycle summary of Fig.~\ref{fig:aligned-model-expertise}. '
             r'Available geometric means use the same passing pairs. Ratios above one favor the selected design. '
             r'Commercial coverage is paired/complete setup grids. DeepSeek and Grok retain the original 26-kernel table cohort. '
             r'Incomplete grids and five asymmetric Grok reruns are excluded. '
             r'DeepSeek has synthesis estimates for 21 of its 22 pairs, so its HLS aggregate is omitted.}',
             r'\label{tab:aligned-model-expertise}', r'\scriptsize',
             r'\setlength{\tabcolsep}{4pt}', r'\renewcommand{\arraystretch}{1.05}',
             r'\begin{tabular}{@{}llcrrr@{}}', r'\toprule',
             r'Model & Evidence & Paired & \shortstack{HLS off/\\selected} & '
             r'\shortstack{RTL off/\\selected} & \shortstack{RTL wins/\\ties/losses} \\', r'\midrule']
    for model, sm in summaries.items():
        hls = sm['synthesis_off_over_selected_gm']
        hls_tex = f"{hls:.2f}$\\times$" if hls is not None else '---'
        lines.append(f"{model} & {'HLS + RTL' if sm['synthesis_paired'] else 'RTL'} & "
                     f"{sm['paired']}/{sm['eligible']} & {hls_tex} & {sm['rtl_off_over_selected_gm']:.2f}$\\times$ & "
                     f"{sm['rtl_wins']}/{sm['rtl_ties']}/{sm['rtl_losses']} \\\\")
    lines.extend([r'\bottomrule', r'\end{tabular}', r'\end{table}', ''])
    path.write_text('\n'.join(lines))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--freeze-review-join', type=Path)
    parser.add_argument('--freeze-historical-synthesis', type=Path,
                        help='Checksum-verify and import the original July-30 synthesis/RTL tables')
    args = parser.parse_args()
    if args.freeze_review_join:
        freeze_review_join(args.freeze_review_join.resolve())
        print(f'Frozen evidence SHA-256: {historical.sha256(EVIDENCE)}')
        return
    if args.freeze_historical_synthesis:
        freeze_historical_synthesis(args.freeze_historical_synthesis.resolve())
        print(f'Frozen historical synthesis SHA-256: {historical.sha256(HISTORICAL_SYNTHESIS)}')
        return
    panels, evidence, inputs = load_data()
    summaries = summary(panels)
    OUT.mkdir(parents=True, exist_ok=True)
    output = ROOT / 'figures/sigmetrics/model_expertise_validation'
    font = draw(panels, summaries, output)
    table_path = OUT / 'commercial_validation_table.tex'
    table(summaries, table_path)
    points = []
    for model, panel in panels.items():
        for bench in historical.CANONICAL_28:
            p = panel.get(bench)
            points.append({'model': model, 'kernel': bench,
                           **{key: '' if p is None or p[key] is None else p[key] for key in (
                               'synthesis_ratio', 'rtl_ratio', 'selected_status', 'off_status',
                               'partial', 'asymmetric', 'eligible', 'common')}})
    csv_path = OUT / 'commercial_validation_points.csv'
    write_csv(csv_path, points)
    macros_path = OUT / 'commercial_validation_macros.tex'
    macros = ['% AUTO-GENERATED paired-cohort commercial results.']
    for name, model in zip(('Sonnet', 'Opus', 'GPT'), MODELS.values()):
        sm = summaries[model]
        for suffix, value in [('Pairs', sm['paired']), ('Eligible', sm['eligible']),
                              ('HLSRatio', f"{sm['synthesis_off_over_selected_gm']:.2f}"),
                              ('RTLRatio', f"{sm['rtl_off_over_selected_gm']:.2f}")]:
            macros.append(f'\\newcommand{{\\Commercial{name}{suffix}}}{{{value}}}')
    macros.extend([f"\\newcommand{{\\CommercialRTLPassRoles}}{{{sum(r['status'] == 'passed' for r in evidence)}}}",
                   f'\\newcommand{{\\CommercialRTLTotalRoles}}{{{len(evidence)}}}'])
    macros_path.write_text('\n'.join(macros) + '\n')
    outputs = [output.with_suffix('.pdf'), output.with_suffix('.png'), table_path, csv_path, macros_path]
    manifest = {
        'schema': 'hlsmith.sigmetrics.commercial-validation.v2', 'inputs': inputs,
        'source_join_sha256': JOIN_SHA256,
        'outputs': {str(path.relative_to(ROOT)): historical.sha256(path) for path in outputs},
        'font': {'family': 'Times New Roman', 'path': font}, 'figure_inches': [7.08, 4.80],
        'metric': 'recipe-disabled cycles / selected-design cycles; numerator and denominator from the same evidence tier',
        'commercial_selection': 'Freeze both original HLS-synthesis-selected arms; no RTL reselection',
        'commercial_summary': 'Intersection of original complete setup grids with both-arm RTL passes; same pairs for HLS and RTL',
        'older_RTL_summary': 'Preserve original 22/26 DeepSeek and 21/26 Grok cohorts, including original baseline-availability filter; five asymmetric Grok reruns excluded',
        'older_synthesis_fields': ['skills_latency_worst', 'noskills_latency_worst'],
        'older_synthesis_binding': 'Checksum-bound companion exports match original RTL source identifiers, cohort, cycles, and pass flags; original selected-report paths retained',
        'older_synthesis_coverage': 'DeepSeek has 26/28 displayed pairs, 21/22 summary pairs; Grok has 27/28 displayed pairs, all 21 summary pairs. No HLS geometric mean is computed over a smaller subset than the RTL mean.',
        'older_evidence_scope': 'Compact reported synthesis and RTL tables; raw generated source and tool reports unavailable',
        'validation_scope': 'Reported Vitis C/RTL consistency; independent original-reference output check not recorded in the commercial bundle',
        'baseline': 'No benchmark-baseline cycles are used in any ratio',
        'model_identity': 'gpt55 deployment alias is retained under the historical GPT-5.5 paper label',
        'evidence_tier_counts': dict(Counter(row['rtl_evidence_tier'] for row in evidence)),
        'status_counts_by_role': dict(Counter(row['status'] for row in evidence)),
        'models': summaries,
    }
    (OUT / 'commercial_validation_manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps(summaries, indent=2))


if __name__ == '__main__':
    main()
