"""One controller: translate, check, optionally improve, then revalidate."""
import hashlib
from importlib.resources import files
import json
import math
import os
from pathlib import Path
import re
import shutil
import tempfile

from .model import Model, ModelError, now, sha, write_json
from . import tools
from . import lightningsim as trace_backend
from . import bambu as bambu_backend

SCHEMA = 'hlsmith.minimal.run.v1'
PROMPT_PROFILE = 'minimal-researcher-v1'


def positive(value, name, *, zero=False):
    if type(value) not in (int, float) or not math.isfinite(value) or value < 0 or (not zero and value == 0):
        raise ValueError(name + ' must be finite and ' + ('nonnegative' if zero else 'positive'))
    return value


def load_project(path):
    path = Path(path).resolve()
    data = json.loads(path.read_text())
    if not isinstance(data, dict):
        raise ValueError('kernel.json must contain an object')
    def resolve(value):
        source = (path.parent / value).resolve()
        if not source.is_file():
            raise ValueError('input file missing: ' + str(source))
        return source
    source, testbench = resolve(data['source']), resolve(data['testbench'])
    if source.suffix != '.c':
        raise ValueError('the independent reference must be a C source (.c)')
    if testbench.suffix not in {'.c', '.cpp', '.cc', '.cxx'}:
        raise ValueError('testbench must be a C or C++ source file')
    reserved = {'reference.c', 'candidate.cpp', 'manifest.json', 'testbench' + testbench.suffix,
                'testbench.c', 'testbench.cpp', 'candidate.o', 'reference.o', 'testbench.o',
                'reference.exe', 'candidate.exe'}
    assets = {}
    for group in ('headers', 'test_files'):
        assets[group] = {}
        for name, filename in data.get(group, {}).items():
            relative = Path(name)
            if (relative.is_absolute() or '..' in relative.parts or not relative.parts
                    or name in reserved):
                raise ValueError('asset names must be safe relative paths')
            if name in assets.get('headers', {}) and group == 'test_files':
                raise ValueError('public and test assets overlap')
            assets[group][name] = resolve(filename)
    validation = {'atol': 0.0, 'rtol': 0.0, 'compile_timeout': 30, 'run_timeout': 5,
                  **data.get('validation', {})}
    for name, value in validation.items():
        positive(value, name, zero=name in {'atol', 'rtol'})
    if any(k in data.get('provider', {}) for k in ('api_key', 'key', 'token')):
        raise ValueError('credentials belong in the provider key environment variable')
    top = data.get('top')
    if top is not None and not re.fullmatch(r'[A-Za-z_][A-Za-z0-9_]*', top):
        raise ValueError('top must be a C identifier')
    return {'path': path, 'data': data, 'source': source, 'testbench': testbench,
            **assets, 'validation': validation, 'top': top}


def snapshot(project, out):
    directory = out / 'inputs'
    directory.mkdir()
    result = dict(project)
    identities = {}
    def copy(source, name):
        target = directory / name
        target.parent.mkdir(parents=True, exist_ok=True)
        raw = source.read_bytes()
        target.write_bytes(raw)
        identities[name] = {'original_path': str(source), 'sha256': hashlib.sha256(raw).hexdigest()}
        return target
    result['source'] = copy(project['source'], 'reference.c')
    result['testbench'] = copy(project['testbench'], 'testbench' + project['testbench'].suffix)
    for group in ('headers', 'test_files'):
        result[group] = {name: copy(source, name) for name, source in project[group].items()}
    config = {**project['data'], 'source': 'reference.c', 'testbench': result['testbench'].name,
              **{group: {name: name for name in project[group]} for group in ('headers', 'test_files')}}
    for section in ('bambu', 'vitis', 'lightningsim'):
        if section in config:
            settings = config[section] = dict(config[section])
            for key in ('executable', 'hls_root'):
                value = settings.get(key)
                if value and (key == 'hls_root' or '/' in value):
                    settings[key] = str((project['path'].parent / value).resolve())
    project_snapshot = 'kernel.json'
    while (directory / project_snapshot).exists():
        project_snapshot = '_' + project_snapshot
    write_json(directory / project_snapshot, config)
    write_json(directory / 'manifest.json', {'project': str(project['path']), 'project_sha256': sha(project['path']),
                                             'project_snapshot': project_snapshot, 'files': identities})
    return result, identities


def _interface_headers(project):
    return sorted(project['headers'], key=lambda name: name != 'hlsmith_task.h')


def _reference_check(project, out):
    provenance = project['data'].get('testbench_provenance')
    if not provenance:
        return None
    checked = tools.validate_native(project['source'], None, project['testbench'], out=out / 'reference-check',
        include_files={**project['headers'], **project['test_files']},
        interface_headers=_interface_headers(project), sanitize=provenance['kind'] == 'generated',
        **project['validation'])
    write_json(out / 'reference-check.json', checked)
    return checked


def extract_source(text):
    fences = re.findall(r'```(?:c\+\+|cpp|cxx|c)?[ \t]*\n(.*?)```', text, re.DOTALL)
    if '```' in text:
        if len(fences) != 1:
            raise ValueError('expected one complete source block')
        text = fences[0]
    if not text.strip():
        raise ValueError('empty candidate')
    return text.strip() + '\n'


def _messages(project, prompt, current=None, feedback=None, improve=False, backend='native'):
    public = '\n\n'.join('Interface file ' + name + ':\n' + path.read_text()
                          for name, path in project['headers'].items())
    content = 'Original C kernel:\n' + project['source'].read_text() + '\n\n' + public
    settings = project['data'].get('vitis', {})
    if backend == 'bambu':
        content += '\n\nCompiler backend: Bambu. Use portable C++ and the supplied public interface. RTL is checked with Verilator.'
    elif settings:
        content += '\n\nHardware settings:\n' + json.dumps({
            'top': project['top'], **{k: settings[k] for k in ('part', 'clock_ns', 'resource_limits') if k in settings}}, sort_keys=True)
    if current is not None:
        content += '\n\nCurrent candidate:\n' + current
    if feedback:
        content += '\n\nCompiler/check feedback:\n' + feedback
    if improve:
        content += ('\n\nReduce executed RTL cycles for the same fixed testbench. Resource usage and physical timing are not measured in this backend.'
                    if backend == 'bambu' else '\n\nReduce the recorded HLS latency while meeting the configured clock and resource budgets.')
    return [{'role': 'system', 'content': prompt}, {'role': 'user', 'content': content}]


def _feedback(native, synthesis=None, eligibility=None):
    # No oracle output, expected value, hidden test input or testbench is sent
    # back to the model. Compiler diagnostics and hardware estimates suffice.
    if not native.get('passed'):
        diagnostic = '\n'.join(row.get('stderr', '') for row in native.get('commands', [])
                               if row.get('status') != 'passed' and '-c' in row.get('argv', [])
                               and any(Path(arg).name == 'candidate.cpp' for arg in row.get('argv', [])))
        return ('Native compilation/execution failed.\n' + diagnostic[-4000:]
                if diagnostic else 'Candidate outputs do not match the independent C check, or execution failed.')
    return json.dumps({'hls_estimate': (synthesis or {}).get('metrics', {}).get('hls_estimate', {}),
                       'synthesis_status': (synthesis or {}).get('status'), 'eligibility': eligibility,
                       'diagnostics': ((synthesis or {}).get('execution', {}).get('stderr', '')
                                       + (synthesis or {}).get('execution', {}).get('stdout', ''))[-4000:]
                       if synthesis and not synthesis.get('passed') else None}, sort_keys=True)


def qualify_synthesis(synthesis, settings):
    metrics = synthesis.get('metrics', {}).get('hls_estimate', {})
    reasons = []
    if not synthesis.get('passed'):
        reasons.append('synthesis_failed')
    for field in ('latency_cycles_worst', 'estimated_clock_period_ns', 'requested_clock_period_ns'):
        value = metrics.get(field)
        if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
            reasons.append('missing_or_invalid_' + field)
    if type(metrics.get('latency_cycles_worst')) is not int:
        reasons.append('latency_must_be_integer_cycles')
    if not reasons:
        if abs(metrics['requested_clock_period_ns'] - settings['clock_ns']) > 1e-9:
            reasons.append('requested_clock_mismatch')
        if metrics['estimated_clock_period_ns'] > settings['clock_ns']:
            reasons.append('estimated_clock_exceeds_budget')
    limits = settings.get('resource_limits', {})
    for name, limit in limits.items():
        value = metrics.get(name)
        if type(value) is not int or value < 0:
            reasons.append('missing_or_invalid_' + name)
        elif value > limit:
            reasons.append(name + '_exceeds_budget')
    return {'passed': not reasons, 'reasons': reasons, 'resource_limits': limits,
            'scope': 'configured resource/clock budgets; not physical implementation or paper eligibility'}


def _trace_settings(project):
    config = dict(project['data'].get('lightningsim', {}))
    if set(config) - {'executable', 'timeout', 'hls_root'}:
        raise ValueError('lightningsim settings support executable, timeout and hls_root only')
    executable = config.get('executable', 'lightningsim')
    if '/' in executable:
        executable = str((project['path'].parent / executable).resolve())
    config['executable'] = executable
    config.setdefault('timeout', 120)
    positive(config['timeout'], 'lightningsim.timeout')
    if config.get('hls_root'):
        config['hls_root'] = str((project['path'].parent / config['hls_root']).resolve())
    elif not os.environ.get('XILINX_HLS') and project['data'].get('vitis', {}).get('executable'):
        compiler = shutil.which(project['data']['vitis']['executable'])
        if compiler and Path(compiler).name == 'vitis_hls':
            root = Path(compiler).resolve().parent.parent
            if (root / 'include/Makefile.sysc.rules').is_file():
                config['hls_root'] = str(root)
    return config


def _bambu_settings(project):
    config = dict(project['data'].get('bambu', {}))
    if set(config) - {'executable', 'timeout', 'clock_ns'}:
        raise ValueError('bambu settings support executable, timeout and clock_ns only')
    executable = config.get('executable', 'bambu')
    if '/' in executable:
        executable = str((project['path'].parent / executable).resolve())
    config['executable'] = executable
    config.setdefault('timeout', 120)
    positive(config['timeout'], 'bambu.timeout')
    if config.get('clock_ns') is not None:
        positive(config['clock_ns'], 'bambu.clock_ns')
    if not project['top']:
        raise ValueError('Bambu requires a top function in kernel.json')
    if not shutil.which(executable):
        raise ValueError('Bambu executable unavailable; configure bambu.executable')
    if not shutil.which('verilator'):
        raise ValueError('Bambu co-simulation requires Verilator; no model request made')
    return config


def run(project_path, out, *, responses=None, max_calls=3, optimize=0, synthesize=False, max_api_usd=None,
        lightningsim=False, backend=None, top=None, testbench=None, headers=(), sizes=(), values=(), config=None):
    if Path(project_path).suffix.lower() == '.c':
        from .task import prepare
        with tempfile.TemporaryDirectory(prefix='hlsmith_task_') as directory:
            prepared = prepare(project_path, directory, top=top, testbench=testbench, headers=headers,
                               sizes=sizes, values=values, config=config)
            return run(prepared, out, responses=responses, max_calls=max_calls, optimize=optimize,
                       synthesize=synthesize, max_api_usd=max_api_usd, lightningsim=lightningsim,
                       backend=backend or ('vitis' if synthesize else 'bambu'))
    if top is not None or testbench is not None or headers or sizes or values or config is not None:
        raise ValueError('source options require a .c input; JSON projects contain their own settings')
    project = load_project(project_path)
    if not shutil.which('gcc') or not shutil.which('g++'):
        raise ValueError('native checks require gcc and g++; no model request made')
    if type(optimize) is not int or optimize < 0 or type(max_calls) is not int or max_calls < 1:
        raise ValueError('max_calls must be positive and optimize nonnegative integers')
    backend = backend or ('vitis' if synthesize else 'native')
    if backend not in {'native', 'vitis', 'bambu'}:
        raise ValueError('backend must be native, vitis or bambu')
    if synthesize and backend != 'vitis':
        raise ValueError('--synthesize selects Vitis; use --backend bambu without that flag')
    synthesize = backend == 'vitis'
    if optimize and backend == 'native':
        raise ValueError('optimization needs --synthesize; CPU wall time is not an HLS objective')
    if lightningsim and not synthesize:
        raise ValueError('--lightningsim requires --synthesize and its Vitis solution')
    trace_settings = _trace_settings(project) if lightningsim else None
    if lightningsim and not shutil.which(trace_settings['executable']):
        raise ValueError('LightningSim executable unavailable; configure lightningsim.executable')
    bambu_settings = _bambu_settings(project) if backend == 'bambu' else None
    settings = project['data'].get('vitis', {})
    if synthesize:
        if not project['top'] or not settings.get('part'):
            raise ValueError('synthesis needs top and vitis.part in kernel.json')
        positive(settings.get('clock_ns'), 'vitis.clock_ns')
        if not shutil.which(settings.get('executable', 'vitis_hls')):
            raise ValueError('Vitis HLS executable unavailable; set vitis.executable or use native checks')
        limits = settings.get('resource_limits', {})
        if optimize and not limits:
            raise ValueError('optimization needs explicit vitis.resource_limits')
        for name, limit in limits.items():
            if name not in {'lut', 'ff', 'bram', 'dsp', 'uram'} or type(limit) is not int or limit < 0:
                raise ValueError('resource limits need supported names and nonnegative integers')
    out = Path(out).resolve()
    out.mkdir(parents=True, exist_ok=False)
    project, identities = snapshot(project, out)
    prompt = files('hlsmith_minimal').joinpath('prompt.md').read_text()
    result = {'schema': SCHEMA, 'started_at': now(), 'status': 'failed', 'backend': backend,
        'scope': 'new minimal runner; not paper reproduction',
        'prompt': {'profile': PROMPT_PROFILE, 'sha256': hashlib.sha256(prompt.encode()).hexdigest()},
        'input_manifest': identities, 'limits': {'max_calls': max_calls, 'optimization_attempts': optimize},
        'attempts': [], 'selected': None, 'warnings': [], 'hardware': {'rtl_cycles': None,
        'trace_estimate_cycles': None, 'physical_implementation': None}}
    write_json(out / 'result.json', result)
    preflight = _reference_check(project, out)
    if preflight is not None:
        result['testbench'] = {**project['data']['testbench_provenance'],
            'sha256': sha(project['testbench']), 'reference_preflight_passed': preflight['passed']}
        if not preflight['passed']:
            result.update(status='blocked_input', finished_at=now(),
                          api_accounting={'calls': [], 'charged_or_reserved_usd': 0, 'mode': 'not_started'})
            result['warnings'].append('Original C/testbench preflight failed; no model request made.')
            write_json(out / 'result.json', result)
            return result
    model = Model(out / 'model', responses=responses, config=project['data'].get('provider'),
                  max_calls=max_calls, max_usd=max_api_usd)
    include_files = {**project['headers'], **project['test_files']}
    best = None
    current = feedback = None
    improvements = 0
    for index in range(max_calls):
        improve = best is not None
        if improve and improvements >= optimize:
            break
        if improve:
            improvements += 1
            current = Path(best['source']).read_text()
            feedback = best['feedback']
        folder = out / 'attempts' / f'{index:03d}'
        folder.mkdir(parents=True)
        entry = {'index': index, 'stage': 'improve' if improve else 'translate' if index == 0 else 'repair',
                 'status': 'failed', 'source': None, 'source_sha256': None}
        result['attempts'].append(entry)
        try:
            response = model.complete(_messages(project, prompt, current, feedback, improve, backend))
        except ModelError as error:
            entry.update(status='stopped', reason=str(error))
            result['warnings'].append(str(error))
            break
        try:
            current = extract_source(response)
        except ValueError as error:
            entry['reason'] = feedback = str(error)
            write_json(out / 'result.json', result)
            continue
        candidate = folder / 'candidate.cpp'
        candidate.write_text(current)
        entry.update(source=str(candidate), source_sha256=sha(candidate))
        native = tools.validate_native(project['source'], candidate, project['testbench'], out=folder / 'native',
                                       include_files=include_files, interface_headers=_interface_headers(project),
                                       **project['validation'])
        write_json(folder / 'native.json', native)
        entry['native'] = {'passed': native['passed'], 'status': native['status'], 'record': str(folder / 'native.json')}
        if (native.get('reference', {}).get('status') != 'passed'
                or not native.get('reference', {}).get('stdout', '').strip()):
            entry.update(status='blocked_input', reason='original C oracle or supplied testbench failed')
            result['warnings'].append('original C oracle unavailable; no additional model requests')
            break
        synthesis = None
        rtl_check = None
        qualification = None
        eligible = native['passed']
        if eligible and backend == 'bambu':
            rtl_check = bambu_backend.evaluate(candidate, project['testbench'], out=folder / 'bambu',
                top=project['top'], include_files=include_files, **bambu_settings)
            write_json(folder / 'bambu.json', rtl_check)
            eligible = rtl_check['passed']
            entry['bambu'] = rtl_check
            entry['eligibility'] = {'passed': eligible, 'resource_eligibility': None,
                'physical_timing_fit': None, 'scope': 'native and generated-RTL correctness for supplied inputs only'}
        if eligible and synthesize:
            extra = {'testbench': project['testbench'], 'test_files': project['test_files']} if lightningsim else {}
            synthesis = tools.synthesize_vitis(candidate, out=folder / 'synthesis', top=project['top'],
                part=settings['part'], clock_ns=settings['clock_ns'], include_files=project['headers'],
                executable=settings.get('executable'), timeout=settings.get('timeout', 300), **extra)
            write_json(folder / 'synthesis.json', synthesis)
            if lightningsim and synthesis['passed']:
                entry['lightningsim'] = trace_backend.estimate(synthesis['solution_dir'], out=folder / 'lightningsim',
                    workspace=synthesis['workspace'], provenance={'candidate_sha256': entry['source_sha256'],
                    'testbench_sha256': sha(project['testbench'])}, **trace_settings)
                write_json(folder / 'lightningsim.json', entry['lightningsim'])
                if not entry['lightningsim'].get('estimate_available'):
                    result['warnings'].append('LightningSim estimate unavailable for attempt ' + str(index))
            qualification = qualify_synthesis(synthesis, settings)
            entry.update(synthesis=synthesis, eligibility=qualification)
            eligible = qualification['passed']
        feedback = (json.dumps({'backend':'bambu','rtl_cosim_status':rtl_check['status'],
            'rtl_cycles':rtl_check.get('metrics', {}).get('rtl_cycles'),
            'scope':'fixed-testbench cycles; FPGA resource/timing eligibility not established'},sort_keys=True)
            if rtl_check else _feedback(native, synthesis, qualification))
        entry['feedback'] = feedback
        if eligible:
            entry['status'] = 'passed'
            score = (rtl_check['metrics']['rtl_cycles'] if rtl_check else
                     synthesis['metrics']['hls_estimate']['latency_cycles_worst'] if synthesize else None)
            entry['objective'] = {'metric': 'rtl_cosim_total_cycles' if rtl_check else
                                  'hls_estimated_worst_cycles' if synthesize else None, 'value': score}
            if best is None or (backend != 'native' and score < best['objective']['value']):
                best = entry
        elif (native['status'] == 'unavailable' or (synthesis and synthesis['status'] == 'unavailable')
              or (rtl_check and rtl_check['status'] == 'unavailable')):
            result['warnings'].append('required tool unavailable; no further model requests')
            break
        write_json(out / 'result.json', result)
    if best is not None:
        source = Path(best['source'])
        if sha(source) != best['source_sha256']:
            raise ValueError('winning candidate changed after evaluation')
        selected = out / 'selected.cpp'
        selected.write_bytes(source.read_bytes())
        final = tools.validate_native(project['source'], selected, project['testbench'], out=out / 'final-validation',
                                      include_files=include_files, interface_headers=_interface_headers(project),
                                      **project['validation'])
        write_json(out / 'final-validation.json', final)
        result['selected'] = {'path': str(selected), 'sha256': sha(selected), 'attempt': best['index'],
                              'objective': best['objective'], 'independent_native_passed': final['passed']}
        if backend == 'bambu':
            result['bambu'] = best['bambu']
            result['selected']['generated_rtl_passed'] = best['bambu']['passed']
            result['hardware'].update(rtl_cycles=best['bambu']['metrics']['rtl_cycles'],
                cycle_scope='total for fixed testbench invocations', resource_eligibility=None,
                physical_timing_fit=None)
        if lightningsim:
            result['lightningsim'] = best.get('lightningsim')
            result['hardware']['trace_estimate_cycles'] = (best.get('lightningsim') or {}).get('metrics', {}).get('trace_estimate_cycles')
        if sha(selected) != best['source_sha256']:
            raise ValueError('selected source differs from evaluated source')
        result['status'] = 'passed' if final['passed'] else 'failed_final_validation'
        if lightningsim and final['passed'] and not (best.get('lightningsim') or {}).get('estimate_available'):
            result['status'] = 'incomplete_estimation'
    result['finished_at'] = now()
    result['api_accounting'] = json.loads((out / 'model/calls.json').read_text())
    write_json(out / 'result.json', result)
    return result


def validate(project_path, candidate, out, *, rtl_config=None, lightningsim_solution=None, lightningsim_workspace=None,
             backend='native', top=None, testbench=None, headers=(), sizes=(), values=(), config=None):
    if Path(project_path).suffix.lower() == '.c':
        from .task import prepare
        with tempfile.TemporaryDirectory(prefix='hlsmith_task_') as directory:
            prepared = prepare(project_path, directory, top=top, testbench=testbench, headers=headers,
                               sizes=sizes, values=values, config=config)
            return validate(prepared, candidate, out, rtl_config=rtl_config, backend=backend,
                            lightningsim_solution=lightningsim_solution,
                            lightningsim_workspace=lightningsim_workspace)
    if top is not None or testbench is not None or headers or sizes or values or config is not None:
        raise ValueError('source options require a .c input; JSON projects contain their own settings')
    if lightningsim_workspace is not None and lightningsim_solution is None:
        raise ValueError('--lightningsim-workspace requires --lightningsim-solution')
    project = load_project(project_path)
    if backend not in {'native','bambu'}:
        raise ValueError('validation backend must be native or bambu')
    if backend == 'bambu' and (rtl_config is not None or lightningsim_solution is not None):
        raise ValueError('Bambu generates and checks its own RTL; supplied-RTL/LightningSim options use the native backend')
    bambu_settings = _bambu_settings(project) if backend == 'bambu' else None
    candidate = Path(candidate).resolve()
    if not candidate.is_file():
        raise ValueError('candidate source missing')
    out = Path(out).resolve()
    out.mkdir(parents=True, exist_ok=False)
    project, identities = snapshot(project, out)
    source = out / 'candidate.cpp'
    source.write_bytes(candidate.read_bytes())
    preflight = _reference_check(project, out)
    native = preflight if preflight is not None and not preflight['passed'] else tools.validate_native(project['source'], source, project['testbench'], out=out / 'native',
        include_files={**project['headers'], **project['test_files']}, interface_headers=_interface_headers(project),
        **project['validation'])
    record = {'schema': SCHEMA, 'started_at': now(), 'status': 'passed' if native['passed'] else 'failed',
              'backend': backend,
              'scope': 'explicit candidate validation; not paper reproduction',
              'input_manifest': identities, 'candidate_sha256': sha(source), 'native': native, 'rtl': None}
    if project['data'].get('testbench_provenance'):
        record['testbench'] = {**project['data']['testbench_provenance'], 'sha256': sha(project['testbench']),
                               'reference_preflight_passed': preflight['passed']}
        if not preflight['passed']:
            record['status'] = 'blocked_input'
    if backend == 'bambu' and native['passed']:
        record['bambu'] = bambu_backend.evaluate(source,project['testbench'],out=out/'bambu',top=project['top'],
            include_files={**project['headers'],**project['test_files']},**bambu_settings)
        record['status'] = 'passed' if record['bambu']['passed'] else 'failed_rtl_validation'
    if rtl_config is not None and native['passed']:
        path = Path(rtl_config).resolve()
        rtl = json.loads(path.read_text())
        directory = out / 'rtl-inputs'
        directory.mkdir()
        rtl_hashes = {}
        def copy_rtl(source, name):
            destination = directory / name
            if destination.exists():
                raise ValueError('RTL input filename collision: ' + name)
            destination.parent.mkdir(parents=True, exist_ok=True)
            raw = source.read_bytes()
            destination.write_bytes(raw)
            rtl_hashes[name] = {'original_path': str(source), 'sha256': hashlib.sha256(raw).hexdigest()}
            return str(destination)
        rtl['rtl_files'] = [copy_rtl((path.parent / name).resolve(), Path(name).name) for name in rtl['rtl_files']]
        harness = (path.parent / rtl['harness']).resolve()
        rtl['harness'] = copy_rtl(harness, harness.name)
        for name, source in {**project['headers'], **project['test_files']}.items():
            copy_rtl(source, name)
        rtl['include_dirs'] = [str((path.parent / name).resolve()) for name in rtl.get('include_dirs', [])]
        rtl['include_dirs'].insert(0, str(directory))
        rtl.update(expected_output=native['reference']['stdout'], atol=project['validation']['atol'],
                   rtol=project['validation']['rtol'])
        record['rtl'] = tools.simulate_verilator(rtl, out=out / 'verilator')
        record['rtl'].update(origin=rtl.get('origin', 'user_supplied'), candidate_rtl_binding='not_established',
            qualification='Supplied RTL checked against original-C outputs; not proof this candidate compiled to that RTL.')
        record['rtl']['input_manifest'] = rtl_hashes
        record['rtl_config_sha256'] = sha(path)
        record['status'] = 'passed' if record['rtl']['passed'] else 'failed'
    if lightningsim_solution is not None and native['passed']:
        record['lightningsim'] = trace_backend.estimate(lightningsim_solution, out=out / 'lightningsim',
            workspace=lightningsim_workspace, provenance={'candidate_binding': 'not_established'}, **_trace_settings(project))
        if record['status'] == 'passed' and not record['lightningsim'].get('estimate_available'):
            record['status'] = 'incomplete_estimation'
    record['finished_at'] = now()
    write_json(out / 'result.json', record)
    return record


def report(path):
    path = Path(path)
    result = json.loads((path / 'result.json' if path.is_dir() else path).read_text())
    selected = result.get('selected') or {}
    ledger = result.get('api_accounting') or {}
    bambu = result.get('bambu') or {}
    return {'status': result['status'], 'scope': result['scope'], 'backend': result.get('backend'),
            'attempts': len(result.get('attempts', [])),
            'selected_source': selected.get('path'), 'selected_sha256': selected.get('sha256'),
            'objective': selected.get('objective'), 'api_charged_or_reserved_usd': ledger.get('charged_or_reserved_usd'),
            'testbench': result.get('testbench'),
            'native_passed': selected.get('independent_native_passed', result.get('native', {}).get('passed')),
            'rtl_passed': bambu.get('passed', (result.get('rtl') or {}).get('passed')),
            'rtl_cycles': bambu.get('metrics', {}).get('rtl_cycles', (result.get('rtl') or {}).get('metrics', {}).get('rtl_cycles')),
            'resource_eligibility': None if bambu else (result.get('hardware') or {}).get('resource_eligibility'),
            'candidate_rtl_binding': ('bambu_generated_and_cosim_checked' if bambu.get('passed') else
                                      (result.get('rtl') or {}).get('candidate_rtl_binding')),
            'lightningsim_status': (result.get('lightningsim') or {}).get('status'),
            'trace_estimate_cycles': (result.get('lightningsim') or {}).get('metrics', {}).get('trace_estimate_cycles'),
            'warnings': result.get('warnings', [])}
