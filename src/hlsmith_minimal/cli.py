"""The entire command surface: run, validate, report."""
import argparse
import json
from pathlib import Path
import sys

from . import __version__
from . import core


def main(argv=None):
    parser = argparse.ArgumentParser(prog='hlsmith', description='Translate C to HLS and independently check the result.')
    parser.add_argument('--version', action='version', version=__version__)
    sub = parser.add_subparsers(dest='command', required=True)
    run = sub.add_parser('run', help='translate, repair and optionally optimize one kernel')
    run.add_argument('project', type=Path, help='kernel.json')
    run.add_argument('--out', required=True, type=Path, help='new output directory')
    run.add_argument('--responses', type=Path, help='saved responses; never falls back to the API')
    run.add_argument('--max-calls', type=int, default=3)
    run.add_argument('--max-api-usd', type=float, help='required for live provider calls')
    run.add_argument('--synthesize', action='store_true', help='also require Vitis synthesis and configured budgets')
    run.add_argument('--optimize', type=int, default=0, metavar='N', help='bounded improvement attempts; requires synthesis')
    validate = sub.add_parser('validate', help='check an explicit C++ candidate and optional supplied RTL')
    validate.add_argument('project', type=Path)
    validate.add_argument('--candidate', required=True, type=Path)
    validate.add_argument('--out', required=True, type=Path)
    validate.add_argument('--rtl-config', type=Path, help='explicit RTL files, C++ harness and top module')
    report = sub.add_parser('report', help='show a compact result')
    report.add_argument('path', type=Path)
    args = parser.parse_args(argv)
    try:
        if args.command == 'run':
            core.run(args.project, args.out, responses=args.responses, max_calls=args.max_calls,
                     optimize=args.optimize, synthesize=args.synthesize, max_api_usd=args.max_api_usd)
            summary = core.report(args.out)
        elif args.command == 'validate':
            core.validate(args.project, args.candidate, args.out, rtl_config=args.rtl_config)
            summary = core.report(args.out)
        else:
            summary = core.report(args.path)
    except (OSError, ValueError, KeyError, TypeError) as error:
        print('hlsmith: ' + str(error), file=sys.stderr)
        return 2
    print(json.dumps(summary, indent=2, allow_nan=False))
    return 0 if summary['status'] == 'passed' else 1
