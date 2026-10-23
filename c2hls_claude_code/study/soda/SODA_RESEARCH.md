# SODA research notes (stencil compiler as the expert tool for stencil kernels)

Repo read: /scratch/hpc-prf-llmfpga/asa582/projects/workspaces/soda_src/soda (UCLA-VAST/soda, commit 720e4e2, read-only). sodac was run from
/scratch/hpc-prf-llmfpga/asa582/projects/workspaces/soda_src/venv. Generated artifacts, build scripts and logs are in
/home/asa582/.claude/jobs/72130906/tmp/soda_research/ (gen/*.cpp, gen/*.host.cpp, bin/, build.sh).
Papers: ICCAD'18 (https://about.blaok.me/pub/iccad18.pdf, 8 pp) and DAC'20 SODA-CR (https://about.blaok.me/pub/dac20-soda-cr.pdf, 6 pp);
both fetched and read via pdftotext. No Vitis/Vivado was run; no Slurm job was submitted.

Everything marked VERIFIED was run in this session (sodac codegen + g++ csim of the generated kernel against the golden that
sodac itself generates). Everything marked UNVERIFIED needs Vitis, TAPA/tapacc or FRT, none of which was run.

---------------------------------------------------------------------------------------------------------------------------

## 0. Headline findings (read this first)

1. All 11 repo kernels generate with sodac and PASS a native g++ csim at the repo default sizes, using a 1-line stub `frt.h`
   (VERIFIED; the headers come from /scratch/hpc-prf-llmfpga/asa582/projects/tmp/c2hls/vitis_hls_2023.2_include). Csim of 1024x1024 or 128^3
   problems takes 1 to 7 s each (build 3 to 50 s; big-expression kernels sobel/denoise/contrast/xcorr build slowest).
2. The generated host golden is the exact C reference, and it is the definitive spec for valid regions (section 2). The generated
   `main()` initialises the int kernels (blur, erosion, xcorr, sobel2d) with the ramp `p+q`, which is a weak test; use random data.
3. The generated multi-tile host code is BUGGY at this commit: jacobi2d (tile 32, iterate 2) on a 64x64 input FAILS (1920
   mismatch lines) (VERIFIED). Input tiling steps by `tile - StencilDim` and output untiling by `tile - StencilDim + 1`
   (frt/host.py region around lines 212-245 and 354-424). Use ONE tile: set `--tile-size` equal to the image width.
   Single-tile runs pass for every config tried (section 3.5).
4. `--replication-factor` is broken at this commit (AttributeError 'str' has no attribute 'window' in core.py:803; dataflow.py:390
   still uses removed `stencil.input`) (VERIFIED). Drop it from the DSE.
5. `border: ignore|preserve` is parsed but "Reserved" (core.py:29-30, 56-57); nothing in the active path reads it (xilinx/host.py
   references stencil.output but that is the deprecated host path). The border region of the output is simply garbage; the host golden only
   checks the interior. Treat output border values as undefined.
6. `iterate: N` is NOT a ping-pong loop in the kernel. sodac statically unrolls N chained copies of the stage
   (tensors named `<in>_iter1..`, core.py:~327-352) inside one dataflow region, so N iterations cost ~zero extra DRAM traffic and
   roughly N times the compute modules (VERIFIED by module counts, section 3.4). The valid region shrinks by the stencil halo per iteration.
7. The `--xocl-kernel` file is NOT a monolithic-HLS-friendly kernel: every module is a data-driven `for (bool enable=true; enable;)` loop
   polling `.empty()` on hls::stream (hls_kernel.py:723, VERIFIED in output). csynth cannot give a trip count, so the top-level latency
   will be reported as `?` (UNVERIFIED, no Vitis). Cycles must come from analytic formula, cosim, or LightningSim (section 3.6).
8. `--xocl-hw-xo` goes through TAPA (`tapac`), which hard-codes `vitis_hls`, runs HLS per module, reads each `report/<module>_csynth.xml`
   and then assembles RTL itself (section 3.2). It also needs `tapacc` (clang tool) and FRT's `frt_get_xlnx_env`; none are on this box
   (`which` fails, VERIFIED). The `--xocl-hls` flag only switches pragma style; it does not choose the tool.

---------------------------------------------------------------------------------------------------------------------------

## 1. Papers

### 1.1 ICCAD 2018, "SODA: Stencil with Optimized Dataflow Architecture" (Chi, Cong, Wei, Zhou)

Platform (Sec. 5.1): Alpha Data ADM-PCIE-KU3, Xilinx Kintex UltraScale XCKU060, 2x8 GB DDR3-1600 (25.6 GB/s theoretical).
Tools: Vivado HLS (per the paper "[5,33]"), Xilinx SDAccel 2017.2. Target clock 250 MHz; across the 75 synthesized configs the
average achieved clock was 245.66 MHz, lowest 229.1 MHz (Sec. 5.2). DRAM interface Wb = 512 bit (Sec. 3.3, Sec. 4.5). CPU baseline:
2x Xeon E5-2620 v3, 24 threads, 4x16 GB DDR4-2133 (68.3 GB/s), Halide-written, tiled/vectorised. FPGA baselines: [3] (Cong/Zou
non-iterative methodology), [17] (Natale et al., iterative, multiple stages), [34] (Zohouri et al., replicated PEs + replicated buffers).

Design space (Sec. 4.2, 4.5): tile sizes T0..T(m-2) (last dim is runtime), unroll factor k (power of 2, so at most 6 points; with
the 8-bit minimum type k <= 64), iterate factor q (bounded by resources: "for the simplest PEs, kq <= 102"). Throughput model
H_ideal(k,q,T) = k*q*f*prod_d (T_d - q*(S_d - 1))/T_d (Eq. 2); resource model per module class (compute / forwarding / auxiliary,
Eq. 5); search is branch-and-bound, "less than 10^6" points, minutes. Model error vs post-synthesis (Table 2): BRAM 1.84%,
DSP 0%, LUT 6.23%, FF 7.58%, throughput 4.22%. 75 configs were synthesised (fixed unroll factor, explore tile size) and on-board
throughput came from linear regression over 4 input sizes (<1% measurement error).

Table 1 (benchmarks and the "Optimal Configuration" the DSE found; paper's wording is "Optimal Configuration"):

| Benchmark | Iterative | Data type | Tile size | Unroll k | Iterate q | Ref |
|---|---|---|---|---|---|---|
| SOBEL 2D | No | uint16_t | 524302 | 32 | n/a | [3] |
| DENOISE 2D | No | float x2 in, float x1 out | 21846 | 8 | n/a | [3] |
| DENOISE 3D | No | float x2 in, float x1 out | 156 x 157 | 4 | n/a | [3] |
| JACOBI 2D | Yes | float | 16392 | 8 | 10 | [17] |
| JACOBI 3D | Yes | float | 181 x 182 | 8 | 6 | [17] |
| SEIDEL 2D | Yes | float | 32768 | 8 | 9 | [17] |
| HEAT 3D | Yes | float | 256 x 257 | 8 | 5 | [17] |

(Tile 524302 for SOBEL is copied verbatim from the extracted text; it is the T0 of a 2-D kernel and looks odd but I could not
cross-check it against the figure.) The paper does not state the problem (image) size, the stencil-window per kernel (it points to
[3],[17]), or per-config resources in absolute numbers (only the model-error table). Listing 2 gives the DSL example: jacobi2d,
tile (3000,*), unroll 3, iterate 2, `out(0,0) = (in(0,-1)+in(-1,0)+in(0,0)+in(1,0)+in(0,1))*0.2f`.

Table 3 (performance of the optimal config; throughput pixel/ns and Op/ns; normalised to 24-thread CPU; #Op = ops per iteration):

| Benchmark (#Op) | Platform | pixel/ns | Op/ns | Norm. to CPU |
|---|---|---|---|---|
| SOBEL 2D (16) | CPU / [3] / SODA | 6.66 / 0.25 / 5.37 | 106.59 / 4.00 / 85.86 | 1 / 0.04 / 0.81 |
| DENOISE 2D (45) | CPU / [3] / SODA | 1.86 / 0.25 / 1.05 | 83.55 / 11.07 / 47.07 | 1 / 0.13 / 0.56 |
| DENOISE 3D (57) | CPU / [3] / SODA | 1.91 / 0.25 / 0.93 | 109.01 / 14.24 / 53.16 | 1 / 0.13 / 0.49 |
| JACOBI 2D (5) | CPU / [17] / [34] / SODA | 5.49 / 16.67 / 17.20 / 18.01 | 27.44 / 83.34 / 86.01 / 90.04 | 1 / 3.04 / 3.13 / 3.28 |
| JACOBI 3D (7) | CPU / [17] / [34] / SODA | 4.24 / 4.72 / 9.86 / 12.00 | 29.66 / 33.01 / 69.04 / 83.98 | 1 / 1.11 / 2.33 / 2.83 |
| SEIDEL 2D (6) | CPU / [17] / [34] / SODA | 5.82 / 15.03 / 15.99 / 16.22 | 34.90 / 90.18 / 95.95 / 97.34 | 1 / 2.58 / 2.75 / 2.79 |
| HEAT 3D (15) | CPU / [17] / [34] / SODA | 4.21 / 4.70 / 6.65 / 8.99 | 63.18 / 70.57 / 99.70 / 134.91 | 1 / 1.12 / 1.58 / 2.14 |

Claims: up to 3.28x over 24-thread CPU (abstract); average 2.76x over CPU on iterative kernels; 9.82x average over [3];
1.08x (2D) / 2.23x (3D) over [17]; 1.03x (2D) / 1.28x (3D) over [34]. Non-iterative kernels are DRAM-bound and the CPU wins
(5.37 vs 6.66 pixel/ns for sobel) because the FPGA board has 25.6 GB/s vs 68.3 GB/s.
"Best configuration" in the paper = the DSE-found optimal config of Table 1 (there is no other named best).

Microarchitecture facts useful for the study (Sec. 3): reuse buffer size = Dr + k - 1 with Dr = max a(s) - min a(s) + 1 (minimal,
Thm 3.1/3.2); k consecutive outputs per cycle; reuse chains split by index mod k; FIFOs > 1024 bit use BRAM, smaller use SRL;
all modules fully pipelined at II=1; burst width 512 requires sub-word types to be coalesced.

### 1.2 DAC 2020, "Exploiting Computation Reuse for Stencil Accelerators" (SODA-CR; Chi, Cong)

Platform: Vivado 2019.1 targeting Alveo U250 (the paper's text; HLS-based prototypes, designs "scaled up to use the available DSPs",
run at 100 to 125 MHz, footnote 5 says they are not tuned for frequency). Host/CPU/GPU comparators: Xeon Gold 6130, Xeon Phi 7250,
Nvidia P100 (numbers taken from [4]). Types: 32-bit float except the 8-bit-int ones marked (3). Unroll factor, burst width, tile size,
iteration count and problem size are NOT given per kernel in the paper ("fixed to produce 1 output element per clock cycle and
all off-chip communication fully reused" in Sec. VI-A; Fig. 6 compares scaled-up iterative kernels on TFlops). Table I lists 16
kernels (name, computation, window): s2d5pt (5-pt, 3x3), s2d33pt (33, 17x17), f2d9pt (3x3), f2d81pt (9x9), s3d7pt (3x3x3),
s3d25pt (9x9x9), f3d27pt (3x3x3), f3d125pt (5x5x5), contrast3 (weighted sum of 197, 17x17), erosion3 (minimum, 19x19),
xcorr3 (sum except center, 19x19), smoother (weighted sum of 25, 25x25), bigbiharm (25, 7x7), lilbiharm (13, 5x5)
(footnote: "Marked benchmarks use 8-bit integers; others use 32-bit float"; the marker positions did not survive pdftotext, so I cannot
say from the text which of contrast/erosion/xcorr are 8-bit. The repo versions use float / int16 / int16).
Table II (operations per output, SODA baseline vs DCMI/HSBR): contrast 197 pointwise -> 30 (-85%), 196 reductions -> 113 (-42%);
erosion 360 reductions -> 12 (-97%); xcorr 359 -> 13 (-96%); average reduction-op saving 58.2%.
Headline: HSBR gives 58.1% fewer LUTs and 54.6% fewer DSPs on average (post-synthesis) and 2.3x average speedup over baseline SODA
(DCMI 1.6x); 10.9x vs Xeon Gold, 3.17x vs Xeon Phi, 1.53x vs P100 on the 8 iterative kernels. BRAM goes UP (123.7% of baseline avg
with exceptions: for contrast/erosion/xcorr the baseline has very deep pipelines so BRAM drops with CR).
Fig. 7 resource ratios exist only as a figure (not extracted). HSBR is `--computation-reuse` (grep: optimization/computation_reuse.py,
1861 lines, github.com/Blaok/soda-cr is the paper code).

### 1.3 Repo kernels vs. papers

| Repo kernel (tests/src) | In ICCAD'18? | In DAC'20? | Repo parameters (burst/unroll/tile/iterate) vs paper |
|---|---|---|---|
| sobel2d | yes (SOBEL 2D) | no | repo 64/4/32/1, int16 in, uint16 out; paper uint16, k=32, tile 524302 |
| denoise2d | yes | no | repo 128/4/32/1, float; paper k=8, tile 21846 |
| denoise3d | yes | no | repo 64/2/(32,32)/1; paper k=4, tile 156x157 |
| jacobi2d | yes (also Listing 2) | s2d5pt-like | repo 64/2/32/2; paper k=8, tile 16392, q=10 |
| jacobi3d | yes | s3d7pt-like | repo 64/2/(32,32)/2; paper k=8, tile 181x182, q=6 |
| seidel2d | yes (3x3 box average, Jacobi-style) | f2d9pt-like | repo 64/2/32/2; paper k=8, tile 32768, q=9 |
| heat3d | yes | s3d7pt-like | repo 64/2/(32,32)/2; paper k=8, tile 256x257, q=5 |
| contrast | no | yes (contrast3, 17x17, 197 taps) | repo 64/2/480/1, float; DAC gives no k/tile |
| erosion | no | yes (erosion3, 19x19 min) | repo 64/4/480/1, int16; DAC gives no k/tile |
| xcorr | no | yes (xcorr3, 19x19) | repo 64/4/480/1, int16 |
| blur | no | no | README example only; repo 256/16/2000/1, uint16 |

In the papers but missing from the repo: s2d33pt, f2d81pt, s3d25pt, f3d27pt, f3d125pt, smoother, bigbiharm, lilbiharm (DAC'20);
nothing from ICCAD'18 is missing. The repo default params are tiny-tile smoke-test values, not the paper's optima.
Note repo's `jacobi2d.soda` uses burst width 64 (= k*32 bit), whereas the paper's optimum uses a 512-bit burst with k=8 (sodac
supports burst > k*width: II=2 read module, VERIFIED in gen/j_b512_u8_t512_i1.cpp Module0Func).

---------------------------------------------------------------------------------------------------------------------------

## 2. SODA DSL semantics (exact reference)

Grammar: src/soda/grammar.py:12-33. Global directives in fixed order: `border`(opt), `burst width`, `cluster`(opt), `iterate`,
`kernel`, `unroll factor`, then `input`+, `param`*, `local`*, `output`+. Comments start with `#`.

### 2.1 Indexing, '*' and layout

* `input dram B <type>: name(T0, T1, ..., *)`. The index tuple is (x0, x1, ..., x_{m-1}); x0 is the FASTEST varying (innermost, "column"),
  the `*` dimension is the LAST (slowest, "row"/plane), its extent is a runtime value. Linearisation: addr = x0 + x1*T0 + x2*T0*T1 + ...
  (src/soda/util.py:9-12 `serialize`; docs/data-layout.md "column-major" in SODA indices). In plain C this is `a[x2][x1][x0]`, i.e. for 2D
  `a[y*W + x]` with `name(dx, dy)`; `u(0,1)` is the NEXT ROW, `u(1,0)` is the next pixel on the same row.
* Statement `stmt(r0, r1, ...) = expr(loads...)` : the LHS index is the anchor. The value stored at position p is
  `f(load at p + l - r)` for each load offset l and LHS offset r. So `blur_x(0,0) = in(0,0)+in(0,1)+in(0,2)` means
  `bx[x][y] = in[x][y]+in[x][y+1]+in[x][y+2]` (anchored at the window's min corner), while `tmp(0,9) = min(in(0,0..18))`
  means `tmp[x][y] = min_{k=-9..9} in[x][y+k]` (centered). Confirmed by the generated golden (e.g. erosion: `q-9 ... q+9`).
* Only the first (innermost) dims have a tile size in the DSL; `--tile-size T0 T1` overrides them (sodac.py:59-64).
* Multiple inputs: exactly one input statement gives the tile size (the others share it), e.g. denoise2d has `f` and `u(32,*)`.
  Each input has its own `dram B`, becomes its own `bank_B_<name>` m_axi port.
* Types: float, uint16, int16, int32... (Vitis type names via haoda). Expressions follow C/C++ promotion and the generated code
  casts the result to the declared type of the statement (grammar.py:propagate_type): e.g. `uint16_t((uint16_t)((a+b+c)/3))`
  (gen/blur.cpp:274). So for uint/int types: arithmetic is done in `int` after integer promotion (no 16-bit wrap in intermediates),
  `/` is C integer division (truncation toward zero), and the value wraps (mod 2^16) only when stored into a `local`/`output` of
  16-bit type. Assigning a `local` of 16-bit type truncates BEFORE the next stage (xcorr tmp1/tmp2, sobel mag_y).
  `int32(x)` is an explicit cast. `min(...)` -> nested std::min; `sqrt` -> unqualified C `sqrt` on a float (may be evaluated in double and
  rounded; use a float tolerance).
* Float order of evaluation: sodac calls `inline.rebalance` (optimization/inline.py:175) which splits long `+` reductions of
  float/int into `cr_var_*` locals. contrast is split into 6 partial-sum locals + output (visible in gen/contrast.cpp header), so
  the float summation order is not left-to-right. Use a relative tolerance. The generated golden uses the same split, so it matches bitwise
  to the kernel. A plain 17x17 double-accumulate reference differs by rounding only.
* `--computation-reuse` (CR, default off): rewrites exprs with `cr_var_` locals (VERIFIED: erosion body shrinks 121 KB -> 78 KB).
  It reorders float sums; for ints it is exact (except 16-bit-store wraps).

### 2.2 border: ignore | preserve

Declared in grammar (grammar.py:32) and stored (core.py:56-57), documented in the class docstring as "Reserved". No code path
consumes `preserve_border` except the deprecated `codegen/xilinx/host.py` (stale `stencil.output`). Both values produce the SAME kernel
(VERIFIED by reading; not diffed). The kernel never handles the halo: docs/data-layout.md says "the halo region of the output is
invalid and needs to be handled by the host". So the output at pixels whose (transitive) stencil window leaves the tile is undefined
(shifted garbage from neighbouring rows / uninitialised BRAM / the padded void elements).

### 2.3 iterate: N

* Requires #inputs == #outputs with identical types (core.py:~91-95). `iterate: 1` has no constraint.
* sodac builds the tensor graph by repeating the local/output statements N times, renaming the input of iteration i to
  `<input>_iter<i>` and the last output keeps its name (core.py:~327-352). The result is one dataflow graph with N chained copies;
  no ping-pong buffer in DRAM, no loop over the kernel. For jacobi2d iterate 2, tile 32: 22 module instances (11 module types);
  iterate 4 at tile 256/unroll 8: 138 instances (VERIFIED).
* Valid output region after N iterations (what the generated golden checks, VERIFIED): with per-iteration offset range
  [lo_d, hi_d] in dimension d (lo_d <= 0 <= hi_d after anchoring), valid index i_d satisfies  -N*lo_d <= i_d < size_d - N*hi_d.
  Jacobi/heat/seidel (radius 1): [N, size-N) in every dim, i.e. a border of width N is garbage. Golden for iterate 2 computes
  iteration 1 on [1,size-1) and iteration 2 on [2,size-2) in each dimension and compares only [2,size-2).
* Nothing is written back to the input array. Chain further iterations by re-invoking the kernel with the previous output
  (valid region shrinks again; tile halo must be re-fetched by the host).

### 2.4 Data layout and the stream interface (docs/data-layout.md + generated host)

* The kernel is a pure stream pipeline: it reads `coalesced_data_num` words of `burst_width` bits from each input port and writes
  the same number of words to each output port. There are no row/column/tile-count arguments. The row length is baked in at
  codegen (`tile_size`), the number of rows only enters through `coalesced_data_num`.
* Input stream = the flattened row-major tile (x0 fastest), zero-padded at the end to a multiple of elements-per-word, plus
  `stencil distance D` extra void elements to drain the pipeline. Output stream = the same length; output pixel with linear
  index L (anchor position) appears at stream index L + D - serialize(window_offset) (frt/host.py:404-416; jacobi2d iterate 2: D=130,
  shift 64). `stencil distace` (sic) is printed in the generated kernel header (core.py:647).
* Host in the generated code computes (VERIFIED, gen/jacobi2d.host.cpp):
  `cycle_count = (H*T0*tile_count + D - 1)/elems_per_word + 1` and passes it as `coalesced_data_num`; input buffer =
  `tile_count*aligned_tile + round_up(D, elems_per_word)` elements.
* Stencil distance examples (VERIFIED from generated headers): jacobi2d tile 32 iterate 2: 130 (= 2*(2*32+1)); tile 512 iterate 1: 1025; heat3d tile 128x128 iterate 4: 131588. Reuse buffer ~ D elements per input stream.

### 2.5 Exact outputs and valid regions (W = extent0 = tile width, H = extent1, Dz = extent2; all indices (x,y[,z]))

Valid region = what the generated golden compares (these rectangles are conservative boxes; tested VERIFIED). For anything outside, SODA output is garbage.

1. **blur** (uint16, tile 2000, burst 256, u16, iterate 1)
   `bx[x][y] = (uint16)((in[x][y] + in[x][y+1] + in[x][y+2]) / 3)` (sum in int, C division, store as uint16)
   `out[x][y] = (uint16)((bx[x][y] + bx[x+1][y] + bx[x+2][y]) / 3)`. Valid: x in [0, W-2), y in [0, H-2). (anchored at min corner, no centring.)
2. **contrast** (float, tile 480, burst 64, u2)
   `out[x][y] = sum_{dy=0..16, dx} w[dy][dx] * in[x+dx][y+dy]`, taps/weights exactly as in tests/src/contrast.soda lines 4-44 (17 rows, symmetric
   diamond pattern, e.g. row dy=0 only dx=8 weight -106, row dy=8 has dx=0..16 with 127 at dx=8). Valid x in [0, W-16), y in [0, H-16).
   Float order differs from naive due to rebalance (tolerance 1e-5 abs & rel as in the generated golden, or 1e-3 as in bench_wide).
3. **denoise2d** (float, inputs f and u(32,*), tile 32, burst 128, u4)
   `du=u[x][y]-u[x][y-1], dd=u[x][y]-u[x][y+1], dl=u[x][y]-u[x-1][y], dr=u[x][y]-u[x+1][y]`;
   `g[x][y] = 1.0f/sqrt(1.0f + du*du + dd*dd + dl*dl + dr*dr)`;
   `r0 = u*f*4.9f`; `r1 = (r0*(2.5f + r0*(10.2f + r0))) * (4.3f + r0*(5.4f + r0*(6.3f + r0)))`;
   `out = (u + 7.7f*(u[x][y+1]*g[x][y+1] + u[x][y-1]*g[x][y-1] + u[x-1][y]*g[x-1][y] + u[x+1][y]*g[x+1][y] + 5.7f*f*r1))
          * (11.1f + 7.7f*(g[x][y+1] + g[x][y-1] + g[x-1][y] + g[x+1][y] + 5.7f))`. Valid x,y in [2, size-2).
4. **denoise3d** (float, f and u(32,32,*), burst 64, u2): same structure in 3D with 6 face neighbours:
   `g = 1.0f/sqrt(0.00005f + sum6 diff^2)`; `r0 = u*f*(1.0f/0.03f)`;
   `r1 = (r0*(2.38944f + r0*(0.950037f + r0))) / (4.65314f + r0*(2.57541f + r0*(1.48937f + r0)))` (tests/src/denoise3d.soda);
   `out = (u + 5.0f*(sum6 u_n*g_n + (1.0f/0.03f)*f*r1)) / (1.0f + 5.0f*(sum6 g_n + (1.0f/0.03f)))`. Valid [2,size-2) in all 3 dims.
5. **erosion** (int16, tile 480, burst 64, u4): `tmp[x][y] = min_{k=-9..9} in[x][y+k]`; `out[x][y] = min_{k=-9..9} tmp[x+k][y]`.
   Valid x,y in [9, size-9). (19x19 min filter, centred.)
6. **heat3d** (float, in(32,32,*), burst 64, u2, iterate 2)
   one iteration: `o = .125f*in(x+1) + .125f*in(x-1) + .125f*in(y+1) + .125f*in(y-1) + .125f*in(z+1) + .125f*in(z-1) + .25f*in(c)`
   left to right as written (x is dim0, y dim1, z dim2). Apply twice. Valid [2, size-2) in all dims.
7. **jacobi2d** (float, t1(32,*), burst 64, u2, iterate 2): `o = (t(x,y+1) + t(x+1,y) + t(x,y) + t(x,y-1) + t(x-1,y)) * 0.2f` (order as written), twice. Valid [2,size-2)^2.
8. **jacobi3d** (float, burst 64, u2, iterate 2): `o = (c + x+ + x- + y+ + y- + z+ + z-) * 0.142857142f`, twice. Valid [2,size-2)^3.
9. **seidel2d** (float, burst 64, u2, iterate 2): despite the name it is Jacobi-style (double-buffer): 3x3 box
   `o = (t(-1,-1)+t(-1,0)+t(-1,1)+t(0,-1)+t(0,0)+t(0,1)+t(1,-1)+t(1,0)+t(1,1)) * .1111111f` (first index = x), twice. Valid [2,size-2)^2.
10. **sobel2d** (int16 in, uint16 out, tile 32, burst 64, u4)
    `mx = (int16)((img[x+1][y-1]-img[x-1][y-1]) + (img[x+1][y]-img[x-1][y])*3 + (img[x+1][y+1]-img[x-1][y+1]))`;
    `my = (uint16)((img[x-1][y+1]-img[x-1][y-1]) + (img[x][y+1]-img[x][y-1])*3 + (img[x+1][y+1]-img[x+1][y-1]))`;
    `mag = (uint16)(65535 - (mx*mx + my*my))`. `mx*mx` is int*int; `my*my` is promoted-uint16 squared (signed-int overflow UB above 46340):
    compute the sum in uint32 for a defined reference; only the low 16 bits matter and are identical. Valid x,y in [1,size-1).
11. **xcorr** (int16, tile 480, burst 64, u4): `t1[x][y] = (int16) sum_{k=-9..9} in[x][y+k]`; `t2[x][y] = (int16) sum_{k=-9..9} t1[x+k][y]`
    (both stored to int16: they WRAP); `out[x][y] = (int16)( ((int32_t)t2[x][y] - in[x][y]) * in[x][y] / 256 )` (int arithmetic, C truncating
    division). Valid [9, size-9)^2.

Test-data hygiene: the generated main() uses `p+q` for blur/erosion/xcorr/sobel2d (weak: sobel of a ramp is constant) and
uniform(0,1) for float kernels. For a fair check use a seeded random array (ints: small range to avoid UB, e.g. 0..255 for sobel).

---------------------------------------------------------------------------------------------------------------------------

## 3. Tooling

### 3.1 sodac outputs and options (src/soda/sodac.py)

`sodac X.soda` writes only what is asked: `--xocl-kernel F.cpp` (Vitis-HLS C++ dataflow kernel, default interface m_axi),
`--frt-host F.cpp` (host + golden; also compiled with -DSODA_CPP_BINDING to call the kernel directly), `--iocl-kernel` (Intel),
`--xocl-hw-xo F.xo` (via TAPA, section 3.2), `--xocl-connectivity`, `--xocl-constraint`, `--estimation-file/--model-file` (resource
model JSON; needs `XILINX_BRAM_MODEL_PATH/query`, UNVERIFIED), and design parameter overrides `--burst-width`, `--unroll-factor`,
`--tile-size T0 [T1]`, `--iterate`, `--border`, `--cluster {none,fine,coarse,full}`, `--inline`, `--computation-reuse {yes,greedy,optimal,glore,built-in,...}`,
`--replication-factor` (BROKEN), `--dram-in/--dram-out`. Codegen takes 0 to 2 s per kernel (VERIFIED, all 11).
`--cluster fine` works offline (fuses forwarders into compute modules; jacobi2d: 22 -> 8 module instances, kernel 29 KB -> 20 KB, VERIFIED);
`coarse`/`full` call tapac + AutoBridge + HLS from inside sodac (optimization/cluster.py:97-157, UNVERIFIED).

### 3.2 --xocl-kernel vs --xocl-hw-xo

* `--xocl-kernel`: pure source-to-source. Emits `extern "C" void <app>_kernel(...)` with `#pragma HLS interface m_axi/s_axilite`, a
  `#pragma HLS dataflow` region, hls::stream<Data<T>> FIFOs with `#pragma HLS stream depth` (depths computed by sodac, capped small; reuse
  buffers are arrays inside the delay modules with `bind_storage ram_s2p`), `#pragma HLS aggregate bit` on every stream.
  The file contains `#ifdef __SYNTHESIS__ #warning this file should be used for simulation only / #warning synthesis result may be sub-optimal #endif`
  (hls_kernel.py:355-361). It is only a preprocessor #warning (non-fatal; the file still goes through csynth). Plausible reasons for
  "sub-optimal" (my inference, not stated in the repo): the vendor DATAFLOW pass inserts handshake/FIFO logic around ~100-400 tiny
  processes, uses default (non-optimised) FIFO depths/implementations for deep chains, and has no floorplan; the TAPA flow
  uses per-task HLS + hand-assembled RTL top with computed FIFO depths and optional AutoBridge floorplanning.
* `--xocl-hw-xo`: rtl_kernel.py (src/soda/codegen/xilinx/rtl_kernel.py). It (1) writes the same kernel with interface `tapa::mmap`
  (`tapa::mmap<tapa::vec_t<float,K>>`, `tapa::istream/ostream`, `.invoke<tapa::detach>(ModuleNFunc,...)`), (2) calls `tapac.main(
  [file, --top=<app>_kernel, --part-num=..., --clock-period=..., --run-tapacc, --run-hls, --generate-task-rtl])` which in
  tapa/core.py:201-262 runs **vitis_hls** (hard-coded `hls='vitis_hls'`, `-std=c++17`, one HLS run per unique module, parallel across cores,
  `config_rtl -reset_level low -module_auto_prefix`), (3) reads each module's `report/<module>_csynth.xml`, logs `II` and `Depth` (loop
  pipeline depth) and sums resources x instance count into `total_resource` (rtl_kernel.py:56-76), (4) writes FIFO depths = sodac depth + 3
  into graph.json, (5) `tapac --generate-task-rtl --generate-top-rtl --pack-xo` (needs Vivado) to produce the .xo. With `--xocl-constraint`
  + `--xocl-connectivity` it also floorplans (AutoBridge). Requirements: `vitis_hls` and `vivado` in PATH, the `tapacc` binary,
  `frt_get_xlnx_env` (FRT) for vendor include paths (tapa/util.py:~113-130). None of these exist on the login node (VERIFIED: `which tapacc
  frt_get_xlnx_env vivado vitis_hls` all missing); the `tapa` 0.0.20240825.1 Python package is in the venv. Whether its tapac.main API matches
  rtl_kernel.py (written against an older tapa) is UNVERIFIED.
* Part/clock options: global absl flags defined in xilinx/opencl.py:22-36 and used by `parse_device_info_from_flags`
  (haoda/backend/xilinx.py:401-452): `--xocl-part-num xcu280-fsvh2892-2L-e` and `--xocl-clock-period 3.33` (ns, passed verbatim to
  `--clock-period`), or `--xocl-platform <shell>` (looked up under /opt/xilinx/platforms or $XILINX_VITIS/platforms) from which part and
  period are read, with explicit flags overriding. If no platform, BOTH part and period are mandatory.
* `--xocl-hls {vivado_hls,vitis_hls}` (opencl.py:128-134): the global `hls` defaults to `'vitis_hls'` (opencl.py:42) and only changes the
  pragma spelling `aggregate bit` vs `data_pack` (hls_kernel.py:53,672). The argparse default text says vivado_hls but the Action does not run
  unless the flag is given, so the output is Vitis-style by default (VERIFIED: generated code has `aggregate bit` and `#ifdef __VITIS_HLS__`).
  It does NOT select which HLS tool the xo flow runs (always vitis_hls via TAPA).

### 3.3 How SODA tests correctness (tests/*.sh)

* test-cpp-host.sh: for each tests/src/*.soda: `sodac --xocl-kernel k.cpp --frt-host h.cpp`, then
  `g++ k.cpp h.cpp -DSODA_TEST_MAIN -DSODA_CPP_BINDING -I$XILINX_VIVADO/include -lOpenCL -fsanitize=address` and run `exe ''` = C-simulation of the
  generated kernel (the kernel function is called as a normal C++ function) compared with a golden C loop nest inside the generated `main()`.
  `<frt.h>` is only needed to compile; the FPGA code path is `#else`'d out. I replaced frt.h by an empty stub (VERIFIED works).
* test-frt-host.sh: same plus FRT host and Intel OpenCL (aoc emulator) when present.
* test-kernel-compilation.sh: `g++ -std=c++11 -fsyntax-only` on the kernel + `aoc -c`.
* test-rtl-sim.sh: `sodac --xocl-hw-xo` then `v++ --link --target hw_emu` and FRT host run against hw_emu (this is the cosim-like check).
  Needs `XCL_PLATFORM` (env.sh). The sw_emu variant is in test-frt-host.sh.
* Golden: generated main() computes the stage-by-stage plain C reference (section 2.5), uses, for floats, a mismatch only if
  `err^2 > thr^2 && err^2/ref^2 > thr^2` with thr = 1e-5 (env THRESHOLD), exact `!=` for ints.
* Input sizes: default `dims = {tile_width, StencilDim1+1}` (very small), larger via argv[2], argv[3] (argv[4] for 3D). Larger widths than
  the tile trigger the broken multi-tile path.

### 3.4 What the kernel top looks like (VERIFIED, generated jacobi2d)

```
extern "C" {
void jacobi2d_kernel(ap_uint<64>* bank_1_t0,   // OUTPUT first (dram 1)
                     ap_uint<64>* bank_0_t1,   // then input (dram 0)
                     uint64_t coalesced_data_num)
#pragma HLS interface m_axi port=bank_1_t0 offset=slave bundle=t0_bank_1
#pragma HLS interface m_axi port=bank_0_t1 offset=slave bundle=t1_bank_0
#pragma HLS interface s_axilite port={bank_1_t0,bank_0_t1,coalesced_data_num,return} bundle=control
  ... hls::stream<Data<ap_uint<64>>> bank_*_buf depth=32; ~77 hls::stream<Data<float>> FIFOs;
#pragma HLS dataflow
  BurstRead(...); Module0Func(...); ... BurstWrite(bank_1_t0, bank_1_t0_buf, coalesced_data_num);
}}
```
* Ports: one m_axi pointer per (tensor, DRAM bank), type `ap_uint<burst_width>*` (coalesced word, NOT float*), own bundle per port; output
  ports listed before input ports; inputs sorted by statement order (denoise2d: out `bank_2_output`, `bank_0_f`, `bank_1_u`).
  Plus a single `uint64_t coalesced_data_num` (word count of the whole padded stream; used for BOTH read and write loops).
  No tile, row, iteration or size arguments. The C++ symbol is `extern "C"` (differs from the dse_v4 convention of a C++ `kernel0`).
* Element/word mapping: word bit field (W-1 .. 0) holds elements highest-index-first (`buf(63,32)` = element 0).
* `Data<T>{T data; bool ctrl;}` streams carry an end flag; compute modules loop `for (bool enable = true; enable;)` with `.empty()` polling.
* Module instances (VERIFIED): jacobi2d default 22 instances of 10 module types; at tile 256, b512, u8, iterate 4: 138 instances/328 streams;
  tile 1024, u16, iterate 3: 200/478; heat3d 128x128 u8 iterate 4: 202/462; sobel2d u32 tile 1024: 264/789; denoise2d u16 tile 1024: 361/811.

### 3.5 csim runs done here (native g++, -O2, -fopenmp, one machine)

All PASS (generated golden compares interior only): defaults of all 11 kernels (<= 5 s each); blur 2000x1000; jacobi2d tile 256 b512 u8 iterate 4
dims 256x300; jacobi2d tile 1024 b512 u16 iterate 3 dims 1024x1024 (3 s); heat3d tile 64x64 b512 u8 iterate 2 dims 64x64x40; heat3d 128^3 b512 u8 iterate 4
(7 s run, 8 s build); sobel2d b512 u32 tile 1024 dims 1024^2 (2 s run, 47 s build); denoise2d b512 u16 tile 1024 (3 s run, 37 s build);
erosion 480x480; seidel2d b512 u8 tile 1024 iterate 4 (4 s); blur b512 u32 tile 1024 (1 s).
FAIL (host bug): jacobi2d defaults (tile 32, iterate 2) on 64x64 (tile_count 2).
(xcorr b512 u32 tile 1024 was still compiling when I wrote this; default xcorr passes.)
Recipe: scripts in the scratch dir (`build.sh`) = `g++ -O2 -std=c++17 -fopenmp k.cpp k.host.cpp -DSODA_TEST_MAIN -DSODA_CPP_BINDING -I<stub frt.h> -I<vitis include>`;
run `./exe '' <W> <H> [<D>]`.

### 3.6 Getting a csynth number comparable to a single-kernel csynth

Options, in order of fidelity to the "Vitis HLS csynth of one kernel" metric the study uses:

A. **csynth of the --xocl-kernel file as-is** (top = `<app>_kernel`, part xcu280-fsvh2892-2L-e, 3.33 ns, std c++11/14 is enough; needs `ap_int.h`,
   `hls_stream.h`; headers are in the Vitis 2023.2 install). Gives full-design resources (DSP/FF/LUT/BRAM/URAM) incl. all FIFOs and AXI adapters,
   i.e. the closest analog of an LLM kernel's csynth. Latency: UNVERIFIED but expected `?` because every module is a `while`-style loop with data
   dependent exit (and the BurstRead loop trip count is the runtime `coalesced_data_num`). To get a number, either (i) add
   `#pragma HLS loop_tripcount` to the module loops and BurstRead/Write by post-processing the generated file (a sed over `for (bool enable`),
   (ii) run cosim with a testbench (generated main() can be adapted; cosim must be run through your Vitis flow), or (iii) LightningSim from
   memory notes (fork with the v2023 m_axi model) which needs a csynth project first.
B. **Analytic cycle count** (valid for the ideal pipeline, UNVERIFIED against cosim): cycles ~ coalesced_data_num * II_word + sum of module
   pipeline depths, where coalesced_data_num = ceil((H*T0*tiles + D)/elems_per_word) + 1 and II_word = 1 when k*width == burst_width else
   burst_width/(k*width) (=2 for 512-bit with k=8 floats). Equivalent to paper Eq. 2: output rate k pixels/cycle. E.g. jacobi2d 1024x1024, tile 1024,
   b512 u16, iterate q: ~ (1048576 + D)/16 ~= 65.8K cycles, independent of q (until resources bind).
C. **Per-module reports from the hw-xo flow** (rtl_kernel.py): gives `II` and `Depth` per module and a SUM of module resources x instance count
   but excludes top-level FIFOs, `BurstRead/Write` AXI adapters (TAPA replaces them with its own async_mmap). Needs the TAPA toolchain
   (section 3.2). Use only if A is infeasible; resource totals will be lower than A.
D. Cosim/hw_emu per test-rtl-sim.sh is the SODA-native route but needs the full v++ flow.

For the cross-tool comparison keep three things equal: same top interface class (m_axi, 512-bit words, one bank per tensor), same clock (3.33 ns),
same functional region (valid interior only), same problem size. Report SODA resources from A and cycles from B (analytic) and/or cosim;
mark which. The csynth clock estimate is not a criterion (user's memory note).

---------------------------------------------------------------------------------------------------------------------------

## 4. Recommended DSE space on U280 and problem sizes

Mirroring the paper's choices (Sec. 4.5): unroll factor k power of two, burst width 512 (U280 AXI is 512-bit; HBM or DDR), iterate q bounded by
DSP/LUT (floats: k*q small), tile as large as BRAM/URAM allows, single DRAM bank per tensor. Use sodac overrides so the .soda files stay unmodified.

| Parameter | Values | Notes |
|---|---|---|
| burst width | 512 fixed | paper's optimum; sodac handles k*width < 512 with II=2.. read/write modules (VERIFIED for float k=8) |
| unroll k | 1,2,4,8,16 for float/32-bit (k*32 <= 512); 8,16,32 for 16-bit (k*16 <= 512, paper sobel optimum k=32); paper optima: 8 (jacobi/heat/seidel/denoise2d), 4 (denoise3d), 32 (sobel) | k=16 float means II=1 on 512-bit; larger k means more fp DSPs (each stencil op x k x q) |
| iterate q (iterative kernels only: jacobi2d/3d, seidel2d, heat3d) | 1,2,4,8 (paper: 10, 6, 9, 5 on KU060) | DSP bound: float add 2 DSP, mult 3 DSP; U280 has 9024 DSPs; the repo defaults are 2. q changes valid region (border q) and module count linearly |
| tile (T0 [,T1]) | 2D: T0 = image width (256..2048); 3D: T0=T1 in {64,128,256} | paper: 2D 16392..32768, 3D 156x157..256x257. Must equal image width because the multi-tile host is broken; buffer ~ 2*T0 (2D) or 2*T0*T1 (3D) elements per input stream per iteration |
| computation reuse | off, `--computation-reuse` (greedy/yes) for contrast, erosion, xcorr (and large-window kernels) | the DAC'20 result: -58% LUT, -55% DSP avg; for erosion/xcorr op count drops ~97%; BRAM may rise except for deep-pipeline kernels |
| cluster | none (default), fine | fine reduces module count (no tool needed); coarse/full require AutoBridge/HLS |
| inline | no/yes | optional; `--inline` untested here |
| replication factor | skip | broken at this commit |
| dram banks | 1 in + 1 out (default dram 0 / 1), or HBM bank split with `--dram-in/--dram-out` for multi-input kernels | denoise has 3 ports |

Space size: for a single kernel about 5 (k) x 4 (q) x 3 (tile) x 2 (CR) ~ 100 points, each sodac run ~1 s; csynth cost is the bottleneck
(2D kernels are cheap, 3D/denoise/contrast modules are large).

Sizes for a fair LLM-vs-SODA comparison (single tile; csim seconds, VERIFIED on the larger ones):
* 2D float (jacobi2d iterate 2/4, seidel2d iterate 2/4, denoise2d): W=H=1024 (4 MB arrays). Throughput-limited by DRAM: ~1.05M pixels in
  ~66K cycles at k=16. Larger (2048^2) is also seconds in csim.
* 2D int (blur uint16, sobel2d, erosion, xcorr): W=H=1024 for blur/sobel; erosion/xcorr W=H=512..1024 (19x19 window needs a 2*19*W buffer).
* contrast (17x17 float): W=H=512 (build time 48 s at defaults due to the huge expression).
* 3D (heat3d, jacobi3d, denoise3d): 128x128x128 (8 MB float; csim 7 s); stencil distance 131588 (2*T0*T1+...), reuse buffer ~ 0.5 MB per iteration, fits BRAM/URAM.
  64^3 is the safe lower bound; 256^3 is paper scale (tile 256x257) but csim and csynth buffers get large.
* The LLM kernels should accept natural `float in[H][W]` (or coalesced 512-bit) and a testbench should compare only the valid interior
  (section 2.5) with a tolerance of 1e-3 relative for float and exact for ints. A shift/adapter layer is needed for SODA's stream-shifted output
  (output pixel L at stream index L + D - c, section 2.4); if the adapter is not wanted, compare against the SODA golden numbers only.
* Iterations: compare equal q for both tools (the LLM kernel must internally iterate; SODA gets q chained stages). An LLM kernel with a
  ping-pong DRAM loop will be DRAM-bound by q; SODA fuses q stages at ~1 DRAM pass, which is the paper's temporal-parallelism advantage.

---------------------------------------------------------------------------------------------------------------------------

## 5. Not accessed / not verified

* Vitis HLS csynth, cosim, hw_emu, TAPA/tapacc and FRT were not run (forbidden/not installed); every statement about csynth latency being
  `?`, about "sub-optimal" causes, resources, and whether the sodac xo flow works with the installed tapa 0.0.20240825.1 is UNVERIFIED.
* ICCAD Table 1 "524302" tile for SOBEL and the DAC'20 marker for which kernels are 8-bit came out of pdftotext; figures (ICCAD Fig. 6-8
  and DAC'20 Fig. 6-9) were not read as images. DAC'20 gives no unroll/burst/tile/iterations; none are available to mirror.
* Paper clock/resource numbers for the specific U280 are not in the papers (KU060 at 250 MHz; U250 at 100-125 MHz).
* The multi-tile host bug was found by running; I did not isolate whether the kernel or the host indexing is at fault (only that the generated pair fails).
* `border: preserve` was confirmed unused by code reading (grep of src/), not by diffing generated kernels.
* xcorr at b512/u32/tile 1024 (csim) did not finish compiling before I wrote this; default xcorr passes.
* The reference formulas in 2.5 were derived from the generated goldens and sodac sources; I did not write independent C and diff it against the
  kernels, although the golden-vs-kernel csim PASS results above validate that the golden text itself is correct.

## Terminology (2026-10-17)
- **naive baseline generated from SODA**: `bench_soda/<k>/hls_baseline.cpp`, the c2hls flow starting point.
- **SODA target**: the sodac-generated kernels in `study/soda/dse/designs/` (csynth copies with tripcount in `designs_tc/`), the competitor.
