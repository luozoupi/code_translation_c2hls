#!/usr/bin/env python3
"""Compatibility entrypoint for the HLSmith pipeline."""
import importlib as _importlib
import sys as _sys

if __name__ == "__main__":
    from hlsmith.cli import main
    raise SystemExit(main())
else:
    # Share the module object so existing patches and callers keep their state.
    _sys.modules[__name__] = _importlib.import_module("hlsmith.pipeline")
