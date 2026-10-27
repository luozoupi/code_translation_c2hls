# Tests

With the package and `pytest` available, run from the project root:

```sh
PYTHONPATH=src python -m pytest -q tests
```

Five focused modules cover the run/select loop, model accounting, native and
RTL tools, Bambu artifact checks, and the optional LightningSim protocol.
They use saved or synthetic model responses and never call hosted models,
Vitis, Bambu synthesis, or GPUs. Native checks use `gcc` and `g++`; a small
Verilator build/simulation runs when Verilator is available and otherwise skips.

Incorrect candidates, malformed reports, and other negative cases are generated
in temporary test directories. The suite checks meaningful failures without
adding demonstration or debug scripts to the source tree.
