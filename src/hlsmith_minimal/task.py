"""Prepare plain C inputs without model calls or a general C parser.

Generated harnesses support numeric scalars and separately allocated fixed-size
arrays. Shapes and domain constraints remain the caller's responsibility.
"""
import ctypes
from decimal import Decimal, InvalidOperation
import json
import math
from pathlib import Path
import re
import shutil

from .model import sha, write_json
from .tools import run_command

_SAFE = re.compile(r'[A-Za-z0-9_./+\-]+')
_IDENT = re.compile(r'[A-Za-z_]\w*')
_LIMITS = {'cases': 3, 'parameters': 32, 'dimensions': 4,
           'elements_per_argument': 4096, 'total_elements': 16384,
           'headers': 64, 'header_bytes': 8 * 1024 * 1024, 'clang_seconds': 15}
_TYPES = {'char': ctypes.c_char, 'signed char': ctypes.c_byte,
          'unsigned char': ctypes.c_ubyte, 'short': ctypes.c_short,
          'unsigned short': ctypes.c_ushort, 'int': ctypes.c_int,
          'unsigned int': ctypes.c_uint, 'long': ctypes.c_long,
          'unsigned long': ctypes.c_ulong, 'long long': ctypes.c_longlong,
          'unsigned long long': ctypes.c_ulonglong, '_Bool': ctypes.c_bool,
          'float': ctypes.c_float, 'double': ctypes.c_double,
          'long double': ctypes.c_longdouble}


def _name(value):
    value = str(value)
    path = Path(value)
    if not _SAFE.fullmatch(value) or path.is_absolute() or '..' in path.parts or not path.parts:
        raise ValueError('dependency names must be safe relative paths')
    return path.as_posix()


def _dependencies(entry, explicit, known=None):
    """Discover only quoted includes; testbench-only includes stay private."""
    found, known = {}, known or {}
    queue = [(Path(entry).resolve(), '')]
    for path in explicit:
        path = Path(path).resolve()
        try:
            name = path.relative_to(Path(entry).resolve().parent).as_posix()
        except ValueError:
            name = path.name
        queue.append((path, _name(name)))
    total = 0
    while queue:
        path, name = queue.pop(0)
        if name:
            if name in known or name in found:
                previous = known.get(name, found.get(name))
                if Path(previous) != path and sha(previous) != sha(path):
                    raise ValueError('dependency filename collision: ' + name)
                continue
            found[name] = path
        if not path.is_file():
            raise ValueError('missing input dependency: ' + str(path))
        total += path.stat().st_size
        if len(found) > _LIMITS['headers'] or total > _LIMITS['header_bytes']:
            raise ValueError('input dependency collection exceeds its bounded limits')
        for include in re.findall(r'^\s*#\s*include\s*"([^"\n]+)"', path.read_text(), re.M):
            alias = _name((Path(name).parent / include).as_posix())
            dependency = (path.parent / include).resolve()
            if not dependency.is_file():
                dependency = Path(known.get(alias, found.get(alias, dependency)))
                if not dependency.is_file():
                    # An explicitly supplied external header can satisfy the
                    # same quoted name even before its queue item is visited.
                    matches = [p for p, n in queue if n == alias]
                    if not matches:
                        raise ValueError('quoted include missing; supply --header: ' + include)
                    dependency = matches[0]
            queue.append((dependency, alias))
    return found


def _settings(config, source):
    if config is None:
        return {}
    if isinstance(config, dict):
        data, base = json.loads(json.dumps(config)), source.parent
    else:
        path = Path(config).resolve()
        data, base = json.loads(path.read_text()), path.parent
    allowed = {'provider', 'bambu', 'vitis', 'lightningsim', 'validation', 'test_files'}
    if not isinstance(data, dict) or set(data) - allowed:
        raise ValueError('task config may contain provider/backend/validation/test_files settings only')
    if any(not isinstance(value, dict) for value in data.values()):
        raise ValueError('task config sections must be objects')
    for name in ('bambu', 'vitis', 'lightningsim'):
        settings = data.get(name, {})
        for key in ('executable', 'hls_root'):
            value = settings.get(key)
            if value and (key == 'hls_root' or '/' in value):
                settings[key] = str((base / value).resolve())
    data['test_files'] = {_name(name): str((base / value).resolve())
                          for name, value in data.get('test_files', {}).items()}
    if not all(Path(path).is_file() for path in data['test_files'].values()):
        raise ValueError('configured test file missing')
    return data


def _options(items, kind):
    parsed = {}
    for item in items:
        name, separator, value = str(item).partition('=')
        if len(value) > 128:
            raise ValueError('task option literal exceeds its bounded length')
        if not separator or not _IDENT.fullmatch(name) or name in parsed:
            raise ValueError(kind + ' options require unique PARAM=VALUE entries')
        if kind == 'size':
            if not re.fullmatch(r'[1-9]\d*(?:x[1-9]\d*)*', value):
                raise ValueError('sizes require positive extents such as a=8x8')
            value = [int(part) for part in value.split('x')]
            if len(value) > _LIMITS['dimensions'] or math.prod(value) > _LIMITS['elements_per_argument']:
                raise ValueError('array extent exceeds bounded harness limits')
        else:
            try:
                numeric = Decimal(value)
            except InvalidOperation:
                raise ValueError('scalar values must be finite numeric literals') from None
            if not re.fullmatch(r'[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?', value) or not numeric.is_finite():
                raise ValueError('scalar values must be finite numeric literals')
        parsed[name] = value
    return parsed


def _signature(source, top, headers):
    clang = shutil.which('clang')
    if not clang:
        raise ValueError('Clang is required to generate a testbench; supply --testbench instead')
    argv = [clang, '-Xclang', '-ast-dump=json', '-fsyntax-only', '-std=c11', str(source)]
    for directory in sorted({source.parent, *(path.parent for path in headers.values())}):
        argv.extend(['-I', str(directory)])
    result = run_command(argv, cwd=source.parent, timeout=_LIMITS['clang_seconds'])
    if result['status'] != 'passed':
        raise ValueError('Clang could not derive the interface; supply a testbench: ' + result['stderr'][-2000:])
    tree = json.loads(result['stdout'])
    aliases = {node['name']: node['type'].get('desugaredQualType', node['type']['qualType'])
               for node in tree.get('inner', []) if node.get('kind') == 'TypedefDecl'}
    def canonical(value):
        for _ in range(12):
            expanded = re.sub(r'\b[A-Za-z_]\w*\b', lambda m: aliases.get(m[0], m[0]), value)
            if expanded == value:
                return ' '.join(value.split())
            value = expanded
        raise ValueError('unsupported recursive typedef')
    matches = [node for node in tree.get('inner', []) if node.get('kind') == 'FunctionDecl'
               and node.get('name') == top and any(c.get('kind') == 'CompoundStmt' for c in node.get('inner', []))]
    if len(matches) != 1 or matches[0].get('storageClass') == 'static' or matches[0].get('variadic'):
        raise ValueError('top must identify one external, nonvariadic C function definition')
    node = matches[0]
    mutable = [item.get('name') for item in tree.get('inner', [])
               if item.get('kind') == 'VarDecl' and item.get('storageClass') != 'extern'
               and 'const' not in item.get('type', {}).get('qualType', '').split()]
    # A helper's static state is just as relevant as the top's own state.
    pending = [item for item in tree.get('inner', []) if item.get('kind') == 'FunctionDecl']
    while pending:
        item = pending.pop()
        if item.get('kind') == 'VarDecl' and item.get('storageClass') == 'static' and 'const' not in item.get('type', {}).get('qualType', '').split():
            mutable.append(item.get('name'))
        pending.extend(item.get('inner', []))
    result_type = canonical(node['type']['qualType'].split('(', 1)[0].strip())
    if result_type not in {*_TYPES, 'void'}:
        raise ValueError('unsupported return type; supply a testbench')
    parameters = []
    for argument in node.get('inner', []):
        if argument.get('kind') != 'ParmVarDecl':
            continue
        spelling = canonical(argument['type'].get('desugaredQualType', argument['type']['qualType']))
        spelling = re.sub(r'\b(?:restrict|__restrict|__restrict__)\b', '', spelling).strip()
        spelling = re.sub(r'(\*)\s*const\b', r'\1', spelling)
        spelling = re.sub(r'^(.+?)((?:\[\d+\])+)\s*\*$', r'\1 (*)\2', spelling)
        pointer = '*' in spelling
        inner = list(map(int, re.findall(r'\[(\d+)\]', spelling)))
        base = re.sub(r'\(\s*\*\s*\)|\*|\[\d+\]', '', spelling).strip()
        const = 'const' in base.split()
        base = ' '.join(part for part in base.split() if part != 'const')
        if (spelling.count('*') > 1 or base not in _TYPES or not argument.get('name')
                or ('(' in spelling and not re.fullmatch(r'.+\(\s*\*\s*\)(?:\[\d+\])+', spelling))):
            raise ValueError('unsupported parameter type; supply a testbench: ' + spelling)
        parameters.append({'name': argument['name'], 'base': base, 'pointer': pointer,
                           'const': const, 'inner': inner})
    if len(parameters) > _LIMITS['parameters']:
        raise ValueError('too many parameters for the bounded generated harness')
    version = run_command([clang, '--version'], cwd=source.parent, timeout=5)
    return result_type, parameters, {'executable': clang, 'version': version['stdout'].splitlines()[0], 'mutable_state': mutable}


def _header(directory, public, private, prototype=''):
    header = directory/'hlsmith_task.h'
    if header.name in public or header.name in private:
        raise ValueError('reserved generated header name collision')
    header.write_text('#ifndef HLSMITH_TASK_H\n#define HLSMITH_TASK_H\n#include <stdbool.h>\n#ifdef __cplusplus\nextern "C" {\n#endif\n' + ''.join('#include "'+name+'"\n' for name in public) + prototype + '#ifdef __cplusplus\n}\n#endif\n#endif\n')
    return {header.name: header, **public}


def _declaration(parameter, name):
    base = ('const ' if parameter['const'] else '') + parameter['base'].replace('_Bool', 'bool')
    if parameter['inner']:
        return base + ' (*' + name + ')' + ''.join('[' + str(n) + ']' for n in parameter['inner'])
    return base + (' *' if parameter['pointer'] else ' ') + name


def _print(expression, base):
    if base in ('float', 'double', 'long double'):
        return 'printf(" %.21Lg", (long double)(' + expression + '));'
    unsigned = base.startswith('unsigned') or base == '_Bool'
    return 'printf(" %' + ('ju' if unsigned else 'jd') + '", (' + ('uintmax_t' if unsigned else 'intmax_t') + ')(' + expression + '));'


def _loops(shape, body):
    indices = ['__hlsmith_i' + str(i) for i in range(len(shape))]
    prefix = ''.join('for (size_t ' + i + '=0; ' + i + '<' + str(n) + '; ++' + i + ') {\n' for i, n in zip(indices, shape))
    access = ''.join('[' + i + ']' for i in indices)
    flat = '0'
    for i, extent in zip(indices, shape):
        flat = '(' + flat + '*' + str(extent) + '+' + i + ')'
    return prefix + body(access, flat) + '\n' + '}\n' * len(shape)


def prepare(source, directory, *, top, testbench=None, headers=(), sizes=(), values=(), config=None):
    """Return a conventional kernel.json; no hosted model or hardware runs."""
    source, directory = Path(source).resolve(), Path(directory).resolve()
    if not source.is_file() or source.suffix != '.c' or not _IDENT.fullmatch(top or '') or top == 'main' or top.startswith('__hlsmith_'):
        raise ValueError('plain input requires a .c source and a named external --top other than main')
    directory.mkdir(parents=True, exist_ok=True)
    if any(directory.iterdir()):
        raise ValueError('task preparation directory must be empty')
    data = _settings(config, source)
    public = _dependencies(source, headers)
    private = {name: Path(path) for name, path in data.get('test_files', {}).items()}
    shapes, fixed = _options(sizes, 'size'), _options(values, 'value')
    testbench = Path(testbench).resolve() if testbench is not None else None
    if testbench:
        if testbench.suffix not in ('.c', '.cpp', '.cc', '.cxx') or not testbench.is_file():
            raise ValueError('supplied testbench must be an existing C/C++ source')
        if fixed:
            raise ValueError('--value applies to generated testbenches; supplied testbench values stay unchanged')
        discovered = _dependencies(testbench, (), public)
        for name, path in discovered.items():
            if name in private and private[name] != path:
                raise ValueError('private dependency filename collision: ' + name)
            private[name] = path
    if set(public) & set(private):
        raise ValueError('public headers and private test files overlap')
    metadata = {'kind': 'provided' if testbench else 'generated',
        'version': 'provided-v1' if testbench and not shapes else 'clang-numeric-v1',
        'cases': None if testbench else 3, 'limits': _LIMITS,
        'alias_policy': 'controlled by supplied testbench; not inferred' if testbench else 'distinct generated buffers; pointer aliasing requires a supplied testbench',
        'domain_policy': 'controlled by supplied testbench; values unchanged' if testbench else 'small positive defaults; explicit sizes/values do not prove application preconditions',
        'extents': shapes, 'fixed_values': fixed}
    if testbench:
        metadata['provided_testbench_sha256'] = sha(testbench)
    if testbench is None or shapes:
        result_type, parameters, compiler = _signature(source, top, public)
        if testbench is None and compiler['mutable_state']:
            raise ValueError('mutable global/static state requires a supplied testbench')
        names = {p['name'] for p in parameters}
        if (set(shapes) | set(fixed)) - names:
            raise ValueError('size/value option names do not match top parameters')
        for p in parameters:
            name = p['name']
            if p['pointer']:
                if name not in shapes or len(shapes[name]) != len(p['inner']) + 1 or shapes[name][1:] != p['inner']:
                    raise ValueError('explicit --size required with matching full shape for pointer: ' + name)
            elif name in shapes:
                raise ValueError('--size names a scalar parameter: ' + name)
            if name in fixed:
                if p['pointer']:
                    raise ValueError('--value requires a scalar parameter: ' + name)
                value = Decimal(fixed[name])
                if p['base'] not in ('float', 'double', 'long double'):
                    bits = ctypes.sizeof(_TYPES[p['base']]) * 8
                    unsigned = p['base'].startswith('unsigned') or p['base'] == '_Bool'
                    low, high = (0, 1) if p['base'] == '_Bool' else ((0, 2**bits-1) if unsigned else (-2**(bits-1), 2**(bits-1)-1))
                    if value != value.to_integral_value() or not low <= value <= high:
                        raise ValueError('fixed scalar value is outside its integer type')
        if sum(math.prod(s) for s in shapes.values()) > _LIMITS['total_elements']:
            raise ValueError('total array storage exceeds bounded harness limits')
        if testbench is None and result_type == 'void' and not any(p['pointer'] and not p['const'] for p in parameters):
            raise ValueError('generated interface has no observable return or writable buffer; supply a testbench')
        prototype = result_type.replace('_Bool', 'bool') + ' ' + top + '(' + ', '.join(_declaration(p, p['name']) for p in parameters) + ('' if parameters else 'void') + ');\n'
        public = _header(directory, public, private, prototype)
        metadata.update(compiler=compiler, prototype=prototype, parameters=parameters)
        call_args = ', '.join('__hlsmith_a' + str(i) for i in range(len(parameters)))
        allocations = '\n'.join('m_param_alloc('+str(i)+', sizeof('+p['base'].replace('_Bool','bool')+''.join('['+str(n)+']' for n in shapes[p['name']])+'));' for i,p in enumerate(parameters) if p['pointer'])
        if testbench:
            if 'hlsmith_original_testbench.inc' in private or 'hlsmith_original_testbench.inc' in public:
                raise ValueError('reserved original testbench alias collision')
            private['hlsmith_original_testbench.inc'] = testbench
            wrapper = directory/('hlsmith_testbench.cpp' if testbench.suffix != '.c' else 'hlsmith_testbench.c')
            wrapper.write_text('#include "hlsmith_task.h"\n#ifdef __BAMBU_SIM__\n#include <mdpi/mdpi_user.h>\nstatic ' + result_type.replace('_Bool','bool') + ' __hlsmith_call(' + ', '.join(_declaration(p,'__hlsmith_a'+str(i)) for i,p in enumerate(parameters)) + ') {\n' + allocations + '\n' + ('return ' if result_type!='void' else '') + top + '('+call_args+');\n}\n#define '+top+' __hlsmith_call\n#endif\n#include "hlsmith_original_testbench.inc"\n')
            testbench = wrapper
            metadata['kind'] = 'provided_with_extents'
        else:
            lines = ['#include "hlsmith_task.h"', '#include <stdio.h>', '#include <stdint.h>', '#include <inttypes.h>', '#include <stddef.h>', '#ifdef __BAMBU_SIM__', '#include <mdpi/mdpi_user.h>', '#endif', 'int main(void) {', 'for (int __hlsmith_case=0; __hlsmith_case<3; ++__hlsmith_case) {']
            for i,p in enumerate(parameters):
                name, base = '__hlsmith_a'+str(i), p['base'].replace('_Bool','bool')
                initial = '(('+str(i*5)+'+__hlsmith_case*7+{flat}*3)%17+1)'
                if p['base'] in ('float','double','long double'):
                    initial += '/8.0L'
                if p['base'] == '_Bool':
                    initial = '((__hlsmith_case+{flat})%2)'
                if p['pointer']:
                    shape = shapes[p['name']]
                    lines.append(base+' '+name+''.join('['+str(n)+']' for n in shape)+';')
                    lines.append(_loops(shape,lambda access,flat: name+access+'=('+base+')('+initial.format(flat=flat)+');'))
                else:
                    value = fixed.get(p['name'], initial.format(flat='0'))
                    if p['name'] in fixed:
                        if p['base'] in ('float','double','long double'):
                            value += 'L'
                        else:
                            value = format(Decimal(value), 'f').split('.')[0]
                            value = '(-9223372036854775807LL-1LL)' if value == '-9223372036854775808' else value + ('ULL' if p['base'].startswith('unsigned') or p['base']=='_Bool' else 'LL')
                    lines.append(base+' '+name+'=('+base+')('+value+');')
            lines += ['#ifdef __BAMBU_SIM__', allocations, '#endif']
            call = top+'('+call_args+')'
            lines.append((result_type.replace('_Bool','bool')+' __hlsmith_result='+call+';') if result_type!='void' else call+';')
            if result_type!='void':
                lines.append(_print('__hlsmith_result',result_type))
            for i,p in enumerate(parameters):
                if p['pointer']:
                    lines.append(_loops(shapes[p['name']],lambda access,flat: _print('__hlsmith_a'+str(i)+access,p['base'])))
            lines += ['printf("\\n");', '}', 'return 0;', '}']
            testbench = directory/'hlsmith_testbench.c'
            testbench.write_text('\n'.join(lines)+'\n')
    elif public:
        # No type inference is needed to give existing C headers C linkage.
        # Force this umbrella header before the originals when compiling C++.
        public = _header(directory, public, private)
    data.update(source=str(source), top=top, testbench=str(testbench),
                headers={name:str(path) for name,path in public.items()},
                test_files={name:str(path) for name,path in private.items()}, testbench_provenance=metadata)
    data['testbench_provenance']['testbench_sha256'] = sha(testbench)
    write_json(directory/'kernel.json',data)
    return directory/'kernel.json'
