"""Regression tests for ChatHLS mobilenet/transformer packaging bugs.

1. flash cosim must stage support headers (weights.h), not only .cpp extras.
2. transformer.h top prototype must use extern \"C\" to match gold/flash defs.
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
READY = (
    REPO
    / "related_work"
    / "benchmarks"
    / "HLSFactory_benchmarks"
    / "chathls_ready"
)


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def test_ensure_extern_c_header_decl_wraps_transformer_prototype():
    prep = _load_module(
        "prepare_chathls_ready_under_test",
        REPO / "scripts" / "prepare_chathls_ready.py",
    )
    raw = (
        '#include <ap_fixed.h>\n'
        "typedef ap_fixed<16, 5> data_t;\n"
        "void transformer(data_t a[8][32], data_t b[8][32]);\n"
    )
    fixed = prep._ensure_extern_c_header_decl(raw, "transformer")
    assert 'extern "C"' in fixed
    assert "void transformer(data_t a[8][32], data_t b[8][32]);" in fixed
    # Idempotent
    assert prep._ensure_extern_c_header_decl(fixed, "transformer") == fixed


def test_ready_transformer_header_has_extern_c_decl():
    header = (READY / "chathls_transformer" / "transformer.h").read_text(encoding="utf-8")
    assert 'extern "C"' in header
    assert "void transformer(" in header


def test_load_cosim_inputs_stages_mobilenet_weights_h():
    flash_cosim = _load_module(
        "flash_cosim_lib_under_test",
        REPO / "scripts" / "pc2" / "flash_cosim_lib.py",
    )
    bench = READY / "chathls_mobilenet"
    assert (bench / "weights.h").is_file()
    assert "weights.h" in json.loads((bench / "metadata.json").read_text())["support_files"]

    inputs = flash_cosim.load_cosim_inputs(bench, full_size=True)
    paths = {item.get("path") for item in inputs["extra_files"]}
    assert "weights.h" in paths
    weights = next(i for i in inputs["extra_files"] if i.get("path") == "weights.h")
    # Kernel #include needs design-side staging (tb=False for headers).
    assert weights.get("tb") is False
    assert "WEIGHTS_QUANTIZED_H" in weights.get("content", "")


def test_load_cosim_inputs_stages_transformer_support_files():
    flash_cosim = _load_module(
        "flash_cosim_lib_under_test",
        REPO / "scripts" / "pc2" / "flash_cosim_lib.py",
    )
    bench = READY / "chathls_transformer"
    inputs = flash_cosim.load_cosim_inputs(bench, full_size=True)
    paths = {item.get("path") for item in inputs["extra_files"]}
    assert "DRAM_attn_input.txt" in paths
    # Header content used for cosim must keep C linkage on the top decl.
    assert 'extern "C"' in inputs["header_code"]
