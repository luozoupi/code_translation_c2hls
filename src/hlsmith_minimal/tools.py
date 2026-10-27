"""Small native, Vitis HLS and explicit-RTL tool functions. No legacy imports."""
from decimal import Decimal, DecimalException, localcontext
import hashlib
import math
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import tempfile
import time
import xml.etree.ElementTree as ET

_MAX_OUTPUT = 2 * 1024 * 1024
_NUMBER = re.compile(r'[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?')
_NONFINITE = re.compile(r'[+-]?(?:nan|inf|infinity)', re.I)
_CYCLES = re.compile(r'HLSMITH_CYCLES:\s*(\d+)\s*')


def run_command(argv, *, cwd, timeout, env=None):
    """Run argv without a shell; timeout cleanup targets only its new group."""
    if not math.isfinite(float(timeout)) or timeout <= 0:
        raise ValueError('timeout must be positive and finite')
    start = time.monotonic()
    result = {'argv': [str(arg) for arg in argv], 'status': 'error', 'returncode': None,
              'stdout': '', 'stderr': '', 'wall_seconds': 0.0}
    with tempfile.TemporaryFile() as stdout, tempfile.TemporaryFile() as stderr:
        try:
            process = subprocess.Popen(result['argv'], cwd=cwd, env=env,
                stdin=subprocess.DEVNULL, stdout=stdout, stderr=stderr, start_new_session=True)
        except FileNotFoundError:
            result.update(status='unavailable', error='executable or working directory unavailable')
            return result
        try:
            process.wait(timeout=timeout)
            result['status'] = 'passed' if process.returncode == 0 else 'failed'
        except BaseException as error:
            # This also cleans up the compiler/simulator group on interruption.
            for sig, grace in ((signal.SIGTERM, .3), (signal.SIGKILL, .3)):
                try:
                    os.killpg(process.pid, sig)
                except ProcessLookupError:
                    pass
                try:
                    process.wait(timeout=grace)
                except subprocess.TimeoutExpired:
                    pass
            if not isinstance(error, subprocess.TimeoutExpired):
                raise
            result['status'] = 'timeout'
        result['returncode'] = process.returncode
        for name, stream in (('stdout', stdout), ('stderr', stderr)):
            stream.seek(0)
            raw = stream.read(_MAX_OUTPUT + 1)
            result[name] = raw[:_MAX_OUTPUT].decode('utf-8', errors='replace')
            if len(raw) > _MAX_OUTPUT:
                result[name + '_truncated'] = True
                result['status'] = 'error'
                result['error'] = 'tool output exceeds comparison limit'
    result['wall_seconds'] = time.monotonic() - start
    return result


def compare_outputs(expected, actual, *, atol=0, rtol=0):
    """Compare whitespace tokens, using Decimal so large integers stay exact."""
    absolute, relative = Decimal(str(atol)), Decimal(str(rtol))
    if not all(value.is_finite() and value >= 0 for value in (absolute, relative)):
        raise ValueError('comparison tolerances must be finite and nonnegative')
    left, right = expected.split(), actual.split()
    if not left:
        return {'passed': False, 'reason': 'reference output is empty'}
    if len(left) != len(right):
        return {'passed': False, 'reason': 'output token counts differ',
                'expected_tokens': len(left), 'actual_tokens': len(right)}
    for index, (want, got) in enumerate(zip(left, right)):
        if _NONFINITE.fullmatch(want) or _NONFINITE.fullmatch(got):
            return {'passed': False, 'reason': 'nonfinite output', 'token': index}
        if re.fullmatch(r'[+-]?\d+', want) and re.fullmatch(r'[+-]?\d+', got):
            equal = int(want) == int(got)
        elif _NUMBER.fullmatch(want) and _NUMBER.fullmatch(got):
            try:
                with localcontext() as context:
                    context.prec = max(50, len(want) + len(got) + 10)
                    expected_number, actual_number = Decimal(want), Decimal(got)
                    equal = abs(expected_number - actual_number) <= absolute + relative * abs(expected_number)
            except (DecimalException, OverflowError):
                equal = False
        else:
            equal = want == got
        if not equal:
            return {'passed': False, 'reason': 'output value differs', 'token': index}
    return {'passed': True, 'reason': 'all output tokens match', 'tokens_compared': len(left),
            'atol': str(absolute), 'rtol': str(relative)}


def _new_output(out):
    out = Path(out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    if any(out.iterdir()):
        raise ValueError('tool output directory must be empty; refusing stale artifacts')
    return out


def _copy_inputs(out, include_files):
    out = _new_output(out)
    reserved = {'reference.c', 'candidate.cpp', 'testbench.c', 'testbench.cpp'}
    for name, source in (include_files or {}).items():
        relative = Path(name)
        if not relative.parts or relative.is_absolute() or '..' in relative.parts or relative.name in reserved:
            raise ValueError('include file must have a safe, nonreserved relative name')
        destination = out / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(Path(source), destination)
    return out


def _tool(name):
    return shutil.which(str(name)) if name else None


def _version(executable, cwd, flags=('--version',)):
    result = run_command([executable, *flags], cwd=cwd, timeout=5)
    lines = (result['stdout'] or result['stderr']).splitlines()
    return lines[0] if lines else None


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def validate_native(reference_source, candidate_source=None, testbench=None, *, out, include_files=None,
                    compile_timeout=30, run_timeout=5, atol=0, rtol=0, interface_headers=(), sanitize=False):
    """Check the original alone, or compare a candidate using one testbench object."""
    if testbench is None:
        raise ValueError('a testbench source is required')
    if sanitize and candidate_source is not None:
        raise ValueError('sanitizers apply to reference-only preflight')
    out = _new_output(out)
    gcc, cpp = _tool('gcc'), _tool('g++')
    tb_cpp = Path(testbench).suffix.lower() in ('.cc', '.cpp', '.cxx')
    result = {'status': 'unavailable', 'passed': False, 'backend': 'native',
              'reference': {}, 'candidate': {}, 'comparison': None, 'commands': [],
              'qualification': 'Original-C output comparison for these inputs; not RTL correctness.'}
    if candidate_source is None:
        result['qualification'] = 'Reference-only original-C/testbench preflight; no candidate comparison or RTL correctness checked.'
    if not gcc or ((candidate_source is not None or tb_cpp) and not cpp):
        result['error'] = 'gcc and g++ are required' if candidate_source is not None or tb_cpp else 'gcc is required'
        return result
    result['versions'] = {'gcc': _version(gcc, out)}
    if cpp:
        result['versions']['g++'] = _version(cpp, out)
    result['input_hashes'] = {'reference': _sha(reference_source)}
    if candidate_source is not None:
        result['input_hashes']['candidate'] = _sha(candidate_source)
    result['input_hashes']['testbench'] = _sha(testbench)
    result['input_hashes']['include_files'] = {str(name): _sha(path) for name, path in (include_files or {}).items()}
    if any(name not in (include_files or {}) for name in interface_headers):
        raise ValueError('interface_headers must name supplied public include files')
    result['interface_headers'] = list(interface_headers)
    checks = ['-fsanitize=address,undefined', '-fno-sanitize-recover=all', '-fno-omit-frame-pointer'] if sanitize else []
    if sanitize:
        result['sanitizers'] = ['address', 'undefined']
    tb_name = 'testbench.cpp' if tb_cpp else 'testbench.c'
    def compile_one(command, cwd):
        record = run_command(command, cwd=cwd, timeout=compile_timeout)
        result['commands'].append(record)
        if record['status'] != 'passed':
            result.update(status=record['status'], error='native compilation or linking failed')
            return False
        return True
    # Oracle outputs stay in memory. No reference source, executable or output
    # file is supplied in the generated candidate's directory.
    with tempfile.TemporaryDirectory(prefix='hlsmith_oracle_') as temporary:
        oracle = _copy_inputs(temporary, include_files)
        shutil.copyfile(reference_source, oracle / 'reference.c')
        shutil.copyfile(testbench, oracle / tb_name)
        commands = [
            [gcc, '-std=c11', '-O0', *checks, '-I', str(oracle),
             *[item for name in interface_headers for item in ('-include', str(oracle/name))],
             '-c', 'reference.c', '-o', 'reference.o'],
            [cpp if tb_cpp else gcc, '-std=c++17' if tb_cpp else '-std=c11', '-O0', *checks, '-I', str(oracle),
             '-c', tb_name, '-o', 'testbench.o'],
            [cpp if tb_cpp else gcc, *checks, 'reference.o', 'testbench.o', '-lm', '-o', 'reference.exe'],
        ]
        for command in commands:
            if not compile_one(command, oracle):
                return result
        environment = {**os.environ, 'ASAN_OPTIONS': 'detect_leaks=0'} if sanitize else None
        result['reference'] = run_command([str(oracle/'reference.exe')], cwd=oracle, timeout=run_timeout, env=environment)
        if result['reference']['status'] != 'passed':
            result.update(status=result['reference']['status'], error='reference execution failed')
            return result
        if candidate_source is not None:
            shutil.copyfile(oracle/'testbench.o', out/'testbench.o')
    if candidate_source is None:
        if not result['reference']['stdout'].strip():
            result.update(status='failed', error='reference output is empty')
        elif not compare_outputs(result['reference']['stdout'], result['reference']['stdout'])['passed']:
            result.update(status='failed', error='reference output contains nonfinite values')
        else:
            result.update(status='passed', passed=True)
        return result
    candidate_dir = _copy_inputs(out/'candidate', include_files)
    shutil.copyfile(candidate_source, candidate_dir/'candidate.cpp')
    for command in ([cpp, '-std=c++17', '-O0', '-I', str(candidate_dir),
                     *[item for name in interface_headers for item in ('-include', str(candidate_dir/name))],
                     '-c', 'candidate.cpp', '-o', 'candidate.o'],
                    [cpp, 'candidate.o', str(out/'testbench.o'), '-lm', '-o', 'candidate.exe']):
        if not compile_one(command, candidate_dir):
            return result
    result['candidate'] = run_command([str(candidate_dir/'candidate.exe')], cwd=candidate_dir, timeout=run_timeout)
    if result['candidate']['status'] != 'passed':
        result.update(status=result['candidate']['status'], error='candidate execution failed')
        return result
    comparison = compare_outputs(result['reference']['stdout'], result['candidate']['stdout'], atol=atol, rtol=rtol)
    result.update(comparison=comparison, passed=comparison['passed'],
                  status='passed' if comparison['passed'] else 'failed')
    return result


def simulate_verilator(config, *, out, timeout=300):
    """Simulate explicitly supplied RTL and a C++ harness including Vdut.h."""
    out = _new_output(out)
    result = {'status': 'unavailable', 'passed': False, 'backend': 'verilator',
              'metrics': {'rtl_cycles': None}, 'measurement_origin': 'executed_rtl',
              'cycle_measurement_qualified': False,
              'qualification': 'Supplied RTL/interface/memory harness only; no area, Fmax or board claim.'}
    executable = _tool(config.get('executable', 'verilator'))
    if not executable:
        result['error'] = 'Verilator is unavailable'
        return result
    expected = config.get('expected_output')
    top = config.get('top_module', '')
    if not isinstance(expected, str) or not expected.strip() or not re.fullmatch(r'[A-Za-z_]\w*', top):
        result.update(status='error', error='nonempty oracle output and a valid top_module are required')
        return result
    rtl = [Path(path).resolve() for path in config.get('rtl_files', [])]
    harness = Path(config['harness']).resolve()
    if not rtl or not all(path.is_file() for path in [*rtl, harness]):
        result.update(status='error', error='explicit RTL files and harness must exist')
        return result
    result['version'] = _version(executable, out)
    result['input_hashes'] = {str(path): _sha(path) for path in [*rtl, harness]}
    result['expected_output_sha256'] = hashlib.sha256(expected.encode()).hexdigest()
    command = [executable, '--cc', '--exe', '--build', '-j', '1', '--top-module', top,
               '--prefix', 'Vdut', '--Mdir', str(out / 'obj_dir'), '-o', 'rtl_model']
    for directory in config.get('include_dirs', []):
        command.append('-I' + str(Path(directory).resolve()))
    command.extend([str(harness), *map(str, rtl)])
    build_env = os.environ.copy()
    build_env['CCACHE_DISABLE'] = '1'
    build_env['CCACHE_TEMPDIR'] = str(out/'compiler_tmp')
    (out/'compiler_tmp').mkdir()
    build = run_command(command, cwd=out, timeout=config.get('build_timeout', timeout), env=build_env)
    result['build'] = build
    if build['status'] != 'passed':
        result.update(status=build['status'], error='RTL build failed; inspect missing modules or interface support')
        return result
    execution = run_command([str(out / 'obj_dir/rtl_model')], cwd=out,
                            timeout=config.get('run_timeout', timeout))
    result['execution'] = execution
    if execution['status'] != 'passed':
        result.update(status=execution['status'], error='RTL harness execution failed')
        return result
    cycles, output = [], []
    for line in execution['stdout'].splitlines():
        match = _CYCLES.fullmatch(line)
        if match:
            cycles.append(int(match.group(1)))
        elif line.startswith('HLSMITH_CYCLES'):
            result.update(status='error', error='malformed cycle marker')
            return result
        else:
            output.append(line)
    if len(cycles) > 1:
        result.update(status='error', error='multiple ambiguous cycle markers')
        return result
    result['metrics']['rtl_cycles'] = cycles[0] if cycles and cycles[0] > 0 else None
    result['cycle_measurement_qualified'] = result['metrics']['rtl_cycles'] is not None
    comparison = compare_outputs(expected, '\n'.join(output), atol=config.get('atol', 0), rtol=config.get('rtol', 0))
    result.update(comparison=comparison, passed=comparison['passed'],
                  status='passed' if comparison['passed'] else 'failed')
    return result


def _tcl(value):
    # Tcl double-quoted words: escape substitutions, quotes and control bytes.
    text = str(value).replace('\\', '\\\\').replace('$', '\\$').replace('[', '\\[')
    return '"' + text.replace('"', '\\"').replace('\n', '\\n').replace('\r', '\\r') + '"'


def synthesize_vitis(candidate, *, out, top, part, clock_ns, include_files=None,
                     executable=None, timeout=300, testbench=None, test_files=None):
    """Optional Vitis C-to-RTL synthesis; every reported metric is an estimate."""
    result = {'status': 'unavailable', 'passed': False, 'backend': 'vitis_hls',
              'measurement_origin': 'hls_estimate', 'metrics': {'hls_estimate': {}}, 'rtl_files': []}
    executable = _tool(executable or 'vitis_hls')
    if not executable:
        result['error'] = 'Vitis HLS is unavailable; no C-to-RTL or hardware estimate was produced'
        return result
    if not re.fullmatch(r'[A-Za-z_]\w*', top) or not str(part) or not math.isfinite(float(clock_ns)) or clock_ns <= 0:
        raise ValueError('valid top, part and positive finite clock_ns are required')
    inputs = {**(include_files or {}), **(test_files or {})}
    out = _copy_inputs(out, inputs)
    shutil.copyfile(candidate, out / 'candidate.cpp')
    testbench_file = None
    if testbench is not None:
        testbench_file = out / ('testbench' + Path(testbench).suffix)
        shutil.copyfile(testbench, testbench_file)
    tcl = out / 'synthesize.tcl'
    commands = ['open_project hls_project', 'set_top ' + _tcl(top), 'add_files ' + _tcl(out / 'candidate.cpp')]
    if testbench_file is not None:
        commands.append('add_files -tb ' + _tcl(testbench_file))
    tcl.write_text('\n'.join(commands + ['open_solution solution -flow_target vitis',
        'set_part ' + _tcl(part), 'create_clock -period ' + str(float(clock_ns)) + ' -name default',
        'csynth_design', 'exit', '']))
    is_runner = Path(executable).name == 'vitis-run'
    flags = ['--tcl', '--input_file'] if is_runner else ['-f']
    result['version'] = _version(executable, out, ('--version',) if is_runner else ('-version',))
    result['input_hashes'] = {'candidate': _sha(candidate), 'tcl': _sha(tcl),
        'include_files': {str(name): _sha(path) for name, path in inputs.items()}}
    if testbench_file is not None:
        result['input_hashes']['testbench'] = _sha(testbench_file)
    result.update(workspace=str(out), solution_dir=str(out / 'hls_project/solution'))
    execution = run_command([executable, *flags, str(tcl)], cwd=out, timeout=timeout)
    result['execution'] = execution
    if execution['status'] != 'passed':
        result.update(status=execution['status'], error='HLS synthesis failed')
        return result
    reports = sorted((out / 'hls_project/solution/syn/report').glob('*csynth.xml'))
    if not reports:
        result.update(status='error', error='synthesis did not produce its report')
        return result
    result['report'] = {'path': str(reports[0]), 'sha256': _sha(reports[0])}
    try:
        xml = ET.parse(reports[0]).getroot()
    except ET.ParseError:
        result.update(status='error', error='invalid synthesis XML report')
        return result
    def numeric(path, integer=False):
        try:
            value = float(xml.findtext(path, default=''))
            if not math.isfinite(value) or value < 0 or (integer and not value.is_integer()):
                return None
            return int(value) if integer else value
        except ValueError:
            return None
    periods = [numeric(path) for path in ('.//UserAssignments/TargetClockPeriod',
                                          './/SummaryOfTimingAnalysis/TargetClockPeriod')
               if xml.find(path) is not None]
    if not periods or any(value is None or value <= 0 or not math.isclose(value, float(clock_ns), abs_tol=1e-9, rel_tol=0)
                          for value in periods):
        result.update(status='error', error='missing or contradictory reported target clock')
        return result
    metrics = {'latency_cycles_worst': numeric('.//SummaryOfOverallLatency/Worst-caseLatency', True),
               'estimated_clock_period_ns': numeric('.//SummaryOfTimingAnalysis/EstimatedClockPeriod'),
               'requested_clock_period_ns': periods[0]}
    for name, tag in [('bram', 'BRAM_18K'), ('dsp', 'DSP'), ('ff', 'FF'), ('lut', 'LUT'), ('uram', 'URAM')]:
        metrics[name] = numeric('.//AreaEstimates/Resources/' + tag, True)
    result.update(status='passed', passed=True, metrics={'hls_estimate': metrics},
        rtl_files=[str(path) for path in sorted((out / 'hls_project/solution/syn/verilog').glob('*.v'))],
        report={'path': str(reports[0]), 'sha256': _sha(reports[0])},
        qualification='HLS estimates and emitted files only; RTL dependency closure and physical implementation are unverified.')
    return result
