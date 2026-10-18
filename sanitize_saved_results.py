#!/usr/bin/env python3
"""Compatibility import for :mod:`hlsmith.research.sanitize_saved_results`."""
import importlib as _importlib
import sys as _sys

if __name__ == "__main__":
    import runpy
    runpy.run_module("hlsmith.research.sanitize_saved_results", run_name="__main__")
else:
    _sys.modules[__name__] = _importlib.import_module("hlsmith.research.sanitize_saved_results")
