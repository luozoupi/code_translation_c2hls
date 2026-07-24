from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts" / "pc2"))

from c2hls_port_loop_labels import inject_loop_labels, build_kernel_info


SAMPLE = '''
void kernel_atax(double A[38][42], double x[42], double y[42], double tmp[38]) {
  int i, j;
  for (i = 0; i < 42; i++)
    y[i] = 0;
  for (i = 0; i < 38; i++) {
    tmp[i] = 0.0;
    for (j = 0; j < 42; j++)
      tmp[i] = tmp[i] + A[i][j] * x[j];
    for (j = 0; j < 42; j++)
      y[j] = y[j] + A[i][j] * tmp[i];
  }
}
'''


def test_inject_loop_labels_is_deterministic_and_numbered():
    out1, n1 = inject_loop_labels(SAMPLE, top="kernel_atax")
    out2, n2 = inject_loop_labels(SAMPLE, top="kernel_atax")
    assert n1 == 4 and n2 == 4
    assert out1 == out2
    assert "L1:" in out1 and "L4:" in out1
    assert out1.index("L1:") < out1.index("L2:")


MACHSUITE_STYLE = '''
void gemm(double m1[64][64], double m2[64][64], double prod[64][64]) {
  int i, j, k;
  outer:for(i=0;i<64;i++) {
    middle:for(j=0;j<64;j++) {
      prod[i][j] = 0.0;
      inner:for(k=0;k<64;k++) {
        prod[i][j] += m1[i][k] * m2[k][j];
      }
    }
  }
}
'''


def test_inject_rewrites_c_named_loops_to_ln():
    out, n = inject_loop_labels(MACHSUITE_STYLE, top="gemm")
    assert n == 3
    assert "L1:" in out and "L3:" in out
    assert "outer:" not in out  # stripped
    info = build_kernel_info(out, top="gemm")
    assert sum(1 for l in info.splitlines() if ",loop," in l) == 3


MACHSUITE_SPACED_LABEL = '''
void spmv(TYPE val[NNZ], int32_t cols[NNZ], int32_t rowDelimiters[N+1], TYPE vec[N], TYPE out[N]){
    int i, j;
    spmv_1 : for(i = 0; i < N; i++){
        spmv_2 : for (j = 0; j < N; j++){
            out[i] = val[j];
        }
    }
}
'''


def test_inject_rewrites_spaced_c_labels_to_ln():
    out, n = inject_loop_labels(MACHSUITE_SPACED_LABEL, top="spmv")
    assert n == 2
    assert "L1: for(i = 0; i < N; i++)" in out
    assert "L2: for (j = 0; j < N; j++)" in out
    assert "spmv_1" not in out
    assert "spmv_2" not in out
    info = build_kernel_info(out, top="spmv")
    assert sum(1 for l in info.splitlines() if ",loop," in l) == 2


def test_build_kernel_info_lists_loops_and_top():
    labeled, _ = inject_loop_labels(SAMPLE, top="kernel_atax")
    info = build_kernel_info(labeled, top="kernel_atax")
    lines = info.strip().splitlines()
    assert lines[0] == "kernel_atax"
    assert any(l.startswith("L1,loop,") for l in lines)
    assert sum(1 for l in lines if ",loop," in l) == 4


def test_build_kernel_info_unique_arrays_with_forward_decl():
    src = '''
void kernel_atax(double A[38][42], double x[42], double y[42], double tmp[38]);
void kernel_atax(double A[38][42], double x[42], double y[42], double tmp[38]) {
  for (int i = 0; i < 42; i++) y[i] = 0;
}
'''
    labeled, n = inject_loop_labels(src, top="kernel_atax")
    assert n == 1
    info = build_kernel_info(labeled, top="kernel_atax")
    array_names = [l.split(",")[0] for l in info.splitlines() if ",array," in l]
    assert array_names == ["A", "x", "y", "tmp"]


def test_export_hlsfactory_atax_writes_chathls_layout(tmp_path):
    from export_c2hls_bench_to_chathls import export_bench

    src = REPO / "benchmarks" / "hlsfactory_atax"
    out_root = tmp_path / "benchmark_optimization"
    manifest = export_bench(src, out_root)
    dest = out_root / "hlsfactory_atax"
    assert dest.is_dir()
    top = manifest["top"]
    assert (dest / f"{top}.cpp").is_file()
    assert (dest / "kernel_info.txt").is_file()
    assert (dest / "run_hls.tcl").is_file()
    assert (dest / "port_manifest.json").is_file()
    info = (dest / "kernel_info.txt").read_text().splitlines()
    assert info[0] == top
    assert any(",loop," in line for line in info)
