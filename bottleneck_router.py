#!/usr/bin/env python3
"""Compatibility import for :mod:`hlsmith.optimization.bottleneck_router`."""
import importlib as _importlib
import sys as _sys

if __name__ == "__main__":
    import runpy
    runpy.run_module("hlsmith.optimization.bottleneck_router", run_name="__main__")
else:
    _sys.modules[__name__] = _importlib.import_module("hlsmith.optimization.bottleneck_router")
