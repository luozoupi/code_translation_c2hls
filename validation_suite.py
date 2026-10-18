#!/usr/bin/env python3
"""Compatibility import for :mod:`hlsmith.validation.validation_suite`."""
import importlib as _importlib
import sys as _sys

if __name__ == "__main__":
    import runpy
    runpy.run_module("hlsmith.validation.validation_suite", run_name="__main__")
else:
    _sys.modules[__name__] = _importlib.import_module("hlsmith.validation.validation_suite")
