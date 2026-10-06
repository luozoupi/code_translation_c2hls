# Matrix-multiply HLS factors

Instruction for reading one AutoSA `sa_sizes` config and knowing which modules, streams, and buffers the HLS printer will emit. Source of the examples is the 1024×1024 U280 campaign under `artifacts/dse/campaigns/20260927_mm1024_u280_st{0,3,4}/`. The kernel is `autosa_tests/mm1024/kernel.c`:

```c
C[i][j] = C[i][j] + A[i][k] * B[j][k];
```

The printed statement in `latency_est/PE_loop_info.json` is `S_0(i, j, k)` with `I = J = K = 1024`. This note covers AutoSA’s generator (`src/autosa_trans.cpp`, `src/autosa_comm.cpp`, `src/autosa_codegen.cpp`, `src/autosa_print.cpp`, `src/autosa_xilinx_hls_c.cpp`). It does not cover a later manual rewrite of the kernel.

## 1. What a config is

A config is one `space_time` plus three integer lists in `--sa-sizes`:

```text
kernel[]->space_time[s]; kernel[]->array_part[...]; kernel[]->latency[...]; kernel[]->simd[...]
```

`space_time` chooses which original loops are space loops. AutoSA enumerates them in `sa_space_time_transform_at_dim_async`: every legal 1-D choice, then every legal pair, in loop order `i`, `j`, `k`. `docs/examples/mm.rst` names the six results. This 1024 campaign only builds 0, 3, and 4. Space times 1, 2, and 5 exist in `mm.rst` and are not in this campaign.

| id | space loops | time loops | `array_part` axis order | in this campaign |
|----|-------------|------------|-------------------------|------------------|
| 0 | `i` | `j`, `k` | `i, j, k` | yes |
| 1 | `j` | `i`, `k` | (see `mm.rst`, array 2) | no |
| 2 | `k` | `i`, `j` | (see `mm.rst`, array 3) | no |
| 3 | `i`, `j` | `k` | `i, j, k` | yes |
| 4 | `i`, `k` | `j` | `i, k, j` | yes |
| 5 | `j`, `k` | `i` | (see `mm.rst`, array 6) | no |

The permutation that makes the chosen space loops outermost is why ST4’s `array_part` is not `i, j, k`. For the pair `(i, k)` the transform bubbles those two loops to the front and leaves `j` last, so the tiled band is `i`, `k`, `j`. ST0 and ST3 leave the band as `i`, `j`, `k`. The statement arguments stay `(i, j, k)` either way. `array_part[t]` is the tile size of axis `t` in that permuted order. The loops outside the tile run `1024 / array_part[t]` times.

`latency` has one factor per coincident loop. For this GEMM those loops are `i` and `j`. `k` carries the reduction on `C` and is not a latency factor, so the list length is 2 for space times 0, 3, and 4. A factor greater than 1 is tiled and its point loop is sunk under a `latency` mark (`autosa_latency_tile_band_loop`). A factor of 1 is not tiled and emits no extra loop. If the loop being tiled is a space loop, AutoSA divides that PE-array dimension by the factor even when the factor is 1.

`simd` is the list of SIMD-candidate loops. Without `--simd-touch-space`, only time loops that are not already under a `latency` mark are candidates, which for ST0 and ST3 is the reduction `k`, so the list has one entry. With `--simd-touch-space`, space loops are candidates too. A factor of 1 is skipped and emits no unroll. A factor greater than 1 is unrolled, and if that loop is a space loop the PE dimension is divided by it.

PE-array shape for the three campaign space times:

| space_time | PE array | SIMD site |
|------------|----------|-----------|
| 0 | 1-D, length `array_part[0] / latency[0]` (`p0` only) | time loop `k` |
| 3 | 2-D, `(array_part[0] / latency[0]) × (array_part[1] / latency[1])` (`p0`, `p1`) | time loop `k` |
| 4 | 2-D, `(array_part[0] / latency[0]) × (array_part[1] / simd[0])` (`p0`, `p1`) | space loop `k` |

`p0` is the first space axis (`i` for 0, 3, and 4). `p1` is the second (`j` for ST3, `k` for ST4). `tuning.json` in a finished ST4 build stores the resulting `sa_dims`.

The campaign command also fixes flags that are not inside `sa_sizes`. ST0 and ST3 are compiled with `--host-serialize --hls --tuning-method=0` and without `--local-reduce` or `--simd-touch-space`. Every ST4 command adds `--local-reduce --reduce-op=+ --simd-touch-space --array-contraction`. The campaign `autosa_config.json` sets `array_part_L2.enable` to 0 and `hbm.enable` to 0.

## 2. Code elements that accompany a config

`kernel0` is the top function. AutoSA prints `#pragma HLS DATAFLOW` and then one call per hardware instance (`autosa_xilinx_hls_c.cpp`). There is no C `for` over PEs in these kernels. Each instance is a separate dataflow process. `PE_wrapper` and the `*_wrapper` IO functions are thin callers around the real module. Module bodies are marked `INLINE OFF`.

### PE array

| space_time | shape in the studied kernels | ids |
|------------|------------------------------|-----|
| 0 | 1-D chain, `p0 = 0 .. array_part[0]/latency[0] - 1` | `PE(int idx)` |
| 3 | 2-D, `p0` along `i`, `p1` along `j` | `PE(int idx, int idy)` |
| 4 | 2-D, `p0` along `i`, `p1` along `k` | `PE(int idx, int idy)` |

The last index on a forwarding chain is a dummy that only drains or sources the FIFO: `B_PE_dummy_in` on ST0/ST3/ST4, `A_PE_dummy_in` on ST3, `C_PE_dummy_out` on ST4. `C_PE_dummy_out` writes a zero float so the first PE on the reduction chain has a source.

### Load and store modules

Names are the ones printed in `kernel_kernel.cpp`. `L1` sits on the PE boundary, `L2` is the next level out, `L3` is the outer level of a 2-D array. A 1-D array (ST0) stops at `L2`. A level is emitted only for an array that actually crosses that boundary. `*_boundary` is the last module of a chain. `*_serialize` is the DRAM port, present because the campaign passes `--host-serialize`.

| space_time | A (load) | B (load) | C (store) |
|------------|----------|----------|-----------|
| 0 | `A_IO_L2_in`, `A_IO_L2_in_serialize`, `A_IO_L1_in` chain | `B_IO_L2_in_boundary`, `B_IO_L2_in_boundary_serialize` | `C_drain_IO_L1_out` chain, `C_drain_IO_L2_out`, `C_drain_IO_L2_out_serialize` |
| 3 | `A_IO_L3_in`, `A_IO_L3_in_serialize`, `A_IO_L2_in` chain | `B_IO_L3_in`, `B_IO_L3_in_serialize`, `B_IO_L2_in` chain | `C_drain_IO_L1_out`, `C_drain_IO_L2_out`, `C_drain_IO_L3_out`, `C_drain_IO_L3_out_serialize` |
| 4 | `A_IO_L3_in`, `A_IO_L3_in_serialize`, `A_IO_L2_in`, `A_IO_L1_in` | `B_IO_L3_in`, `B_IO_L3_in_serialize`, `B_IO_L2_in` | `C_IO_L2_out`, `C_IO_L3_out`, `C_IO_L3_out_serialize` |

ST0 and ST3 store modules are named `C_drain_*` because `C` is finished inside the PE and then drained. ST4 store modules are named `C_IO_*` because `C` is a partial sum that arrives from the PE chain. The numeric lists change trip counts and buffer extents inside these modules. They do not rename them or add a level. The level count follows `space_time` (1-D vs 2-D) and whether that array is forwarded by the PE or injected at the PE.

Inside an IO module that owns a local buffer, AutoSA prints `intra_trans` and `inter_trans` (plus `*_inter_trans_boundary`). `inter_trans` forwards along the chain and keeps the element whose `module id` matches the loop index. `intra_trans` moves that element to the downstream FIFO. Those two functions alternate on `local_*_ping` and `local_*_pong`. That pair is the default `--double-buffer` (default on). It is not the `--two-level-buffer` flag. Two-level buffering is off in this campaign: the flag defaults off, and `autosa_comm.cpp` forces it off whenever `--host-serialize` is on.

### Streams and which array moves which way

Links are `hls::stream<...>`. Depth is the `--fifo-depth` default, 2, printed as `#pragma HLS STREAM variable=... depth=2`. PE and IO-chain streams also get `#pragma HLS RESOURCE variable=... core=FIFO_SRL` (`autosa_print.cpp` inserts SRL on these FIFOs). The three `*_serialize` streams at the DRAM edge in the ST0 and ST4 kernels are depth 2 and do not carry the `FIFO_SRL` pragma.

Directions, from the PE port list and from the `fifo_*_PE_<p0>_<p1>` wiring:

| array | ST0 | ST3 | ST4 |
|-------|-----|-----|-----|
| A | IO chain feeds each PE. The PE has `fifo_A_in` and no `fifo_A_out`. | PE forwards along `p1` (`j`): `fifo_A_PE_0_0` into `PE(0,0)`, `fifo_A_PE_0_1` into `PE(0,1)`. | IO `L1` feeds each PE. The PE has `fifo_A_in` only. `PE(0,0)` reads `fifo_A_PE_0_0`, `PE(0,1)` reads `fifo_A_PE_0_1`. |
| B | PE forwards along `p0`. `B_PE_dummy_in` consumes the tail. | PE forwards along `p0` (`i`): `fifo_B_PE_0_0` into `PE(0,0)`, `fifo_B_PE_1_0` out toward the next `i`. | PE forwards along `p0` (`i`): `PE(0,0)` reads `fifo_B_PE_0_0` and writes `fifo_B_PE_1_0`. |
| C | PE writes `fifo_C_drain_out` once the `k` tile loop finishes (`c2 == 127` on the ST0 point). Drain IO modules collect it. No PE-to-PE `C` link. | Same drain, fired when the `k` loops finish (`c2 == 63 && c5 == 1` on the ST3 point below). | PE reads `fifo_C_in` and writes `fifo_C_out` every cycle. `PE(0,0)` reads `fifo_C_PE_0_0` and writes `fifo_C_PE_0_1`, so the partial moves along `p1` (`k`). |

That matches `mm.rst`: ST0 feeds A and reuses B, with C stationary; ST3 reuses A along `j` and B along `i`, with C stationary; ST4 feeds A, reuses B along `i`, and reduces C along `k`.

### On-chip buffers

PE-local arrays are `local_A`, `local_B`, `local_C` of the scalar typedefs `A_t1`, `B_t1`, `C_t1` (`float`). `local_A` and `local_B` are the SIMD lanes of one beat and are fully partitioned (`ARRAY_PARTITION dim=0 complete`). Their second dimension equals the SIMD factor that was actually unrolled.

`local_C` is the set of `C` elements that PE owns:

- On an output-stationary array (ST0, ST3) the contraction check does not run. `compute_group_bounds_core_pe` runs it only when `local_reduce && array_contraction`, or when `tuning_method == 1`. This campaign uses `tuning_method=0` and passes `--local-reduce` only for ST4. The default `--array-contraction` (default on) therefore does not shrink ST0 or ST3.
- Each space axis contributes `latency` elements, because that is the point loop left on the PE after the space loop is divided across PEs. A time axis that is an output index contributes the whole `array_part` tile: the latency split only reorders the loop, and the values stay live across `k`.
- ST0 point `array_part[128,128,8]`, `latency[1,32]`: `local_C[1][128]`. The `1` is `latency[0]` on `i`. The `128` is the full `j` tile. It is mapped with `#pragma HLS RESOURCE variable=local_C core=RAM_2P_BRAM`.
- ST3 point `array_part[64,64,16]`, `latency[4,8]`, `simd[8]` (`candidate_1`): `local_C[4][8]`, indexed `local_C[c7][c6]`, also `RAM_2P_BRAM`. Both extents are latency factors because both `i` and `j` are space loops.
- ST4, with `--local-reduce` and `--array-contraction`: `local_C[1][1]`, fully partitioned. The PE adds one SIMD beat into the partial it just read and writes it back out. It does not keep a `j` tile.

IO-module buffers are packed words, not scalars, and they are ping-pong BRAM (`RAM_1P_BRAM`) when double buffering is on. Extents follow the tile the module holds. On the ST4 point, `A_IO_L1` holds `A_t4 local_A[32][1]` (the `i` extent of one PE, with `k` packed into `A_t4`), `B_IO_L2` holds `B_t4 local_B[128][1]` (the `j` tile), and `C_IO_L2` holds `C_t16 local_C[32][8]` (32 `i` values by 128 `j` values, packed 16-wide). On the ST0 point, `A_IO_L1` holds `A_t8 local_A[1][1]` and `B_IO_L2` holds `B_t8 local_B[128][1]`.

### 512-bit accesses

AutoSA writes the bit width into the typedef. `print_data_types_xilinx` emits `typedef ap_uint<N> A_tF` with `N = element_bytes * 8 * F`. The outermost lane count comes from `compute_io_group_data_pack`: the default outer data-pack bound is 64 bytes (`data_pack_ubs[2] = 64`), and `float` is 4 bytes, so the outer factor is 16. The studied kernels all contain:

```c
typedef ap_uint<512> A_t16;
typedef ap_uint<512> B_t16;
typedef ap_uint<512> C_t16;
```

The `m_axi` pragma AutoSA prints has no width:

```c
#pragma HLS INTERFACE m_axi port=A offset=slave bundle=gmem_A
#pragma HLS INTERFACE m_axi port=B offset=slave bundle=gmem_B
#pragma HLS INTERFACE m_axi port=C offset=slave bundle=gmem_C
```

`kernel0`’s argument types are `A_t16 *`, `B_t16 *`, `C_t16 *`, so the port is 512 bits because the C type is 512 bits. `sa_sizes` does not contain a 512 knob.

Vitis HLS, after synthesis, records `config_interface -m_axi_max_widen_bitwidth=512` in `hls_prj/sol1/sol1_data.json`. The campaign `hls_config.cfg` does not set that option. It is the tool’s recorded default, not a line AutoSA printed. The kernel is already 512 bits wide, so that default is not what chooses `A_t16`.

Interior FIFO widths follow the SIMD lane of that array, and the serialize module splits the 512-bit DRAM word down to the FIFO word:

| point | DRAM type | FIFO type at the PE | split in `*_serialize` |
|-------|-----------|---------------------|------------------------|
| ST0 `simd[8]` | `A_t16` / `B_t16` (512) | `A_t8` / `B_t8` (`ap_uint<256>`) | one 512-bit read, two 256-bit writes (`mem_data >> 256`) |
| ST3 `simd[8]` | `A_t16` / `B_t16` | PE FIFOs are `A_t8` / `B_t8`; the L3 FIFO stays `A_t16` and L2 unpacks it | L3 serialize copies `A_t16` through; the 256-bit split is in the L2 transfer |
| ST4 `simd[4,1]` | `A_t16` / `B_t16` | `A_t4` / `B_t4` (`ap_uint<128>`) | one 512-bit read, four 128-bit writes (`mem_data >> 128`) |

`C` leaves the PE as `hls::stream<float>` on all three. The drain or store IO packs it back up (`C_t4` / `ap_uint<128>` on the ST0 and ST3 drain chains; `C_t16` on the ST4 `C_IO_L2` / `C_IO_L3` path, because that buffer’s `j` extent is a multiple of the outer pack factor 16).

`__burst_coalesced_load` is printed only on the Intel OpenCL path in `autosa_print.cpp`. These Xilinx kernels do not call it. The serialize loop is a pipelined `II=1` read of the packed array (`mem_data = A[i]`). The 64-byte burst test in `autosa_comm.cpp` belongs to two-level buffering, which this campaign turns off.

The header also prints `host_serialize_A`, `host_serialize_B`, and `host_deserialize_C`. Those reorder the host arrays into the transfer order. They are host code, generated with the kernel because `--host-serialize` is on.

### Pipeline and unroll

`#pragma HLS PIPELINE II=1` is placed on the innermost loop that sits under the `latency` / `hls_pipeline` mark. In `PE_loop_info.json` that mark is named `hls_pipeline`. The SIMD loop inside it is marked `simd` / `hls_unroll` and printed as `#pragma HLS UNROLL`. The loops that shift a packed FIFO word into `local_A` / `local_B` are also unrolled, with trip count equal to the FIFO lane count.

A latency factor of 1 does not create a pipelined point loop. ST4’s `latency[1] = 1` leaves `j` as one loop of trip count 128 (`c5`). The pipeline pragma sits on the `i` point loop `c6` (trip count 32), and the unroll sits on `c7` (trip count 4).

### Local reduce (ST4 only)

ST4 is the only campaign space time compiled with `--local-reduce --reduce-op=+`. The PE body is the reduction across `p1`:

```c
if (p1 + c2 >= 1)
  local_C[0][0] = fifo_C_in.read();
local_C[0][0] = (local_C[0][0] + (local_A[0][c7] * local_B[0][c7]));
fifo_C_out.write(local_C[0][0]);
```

`C_PE_dummy_out` supplies the zeros at the start of that chain. ST0 and ST3 accumulate into `local_C` and write the drain only on the last `k` iteration. They do not read a partial from a neighbor.

### Present in these kernels, and not

Also always present for this campaign: `s_axilite` control ports, `gmem_A` / `gmem_B` / `gmem_C` bundles, ping-pong IO buffers, boundary modules, and the serialize split. Not present in these kernels: a credit FIFO (`module->credit` stays 0 unless `--credit-control` is passed, and that path is marked unsupported in `autosa_codegen.cpp`), `__burst_coalesced_load`, HBM multi-ports, `array_part_L2`, and two-level IO buffers.

## 3. Two points

### ST0 candidate 1

`artifacts/dse/campaigns/20260927_mm1024_u280_st0/u280_paper/validation/mm1024/candidate_1/`

```text
space_time[0]; array_part[128,128,8]; latency[1,32]; simd[8]
```

Axis order is `i, j, k`. PE count is `128/1 = 128`. The printed calls use module id `0` through `127`. SIMD 8 unrolls `k`. Latency 32 tiles `j` inside the PE and does not change the PE count.

`S_0(p0 + 128 * c0, 128 * c1 + 32 * c4 + c6, 8 * c2 + c7)` with `c0,c1` in `0..7` (`1024/128`), `c2` in `0..127` (`1024/8`), `c4` in `0..3` (`128/32`), `c6` in `0..31`, `c7` in `0..7`.

The kernel contains: a 1-D `PE` with `fifo_A_in`, `fifo_B_in`, `fifo_B_out`, `fifo_C_drain_out`; `local_A[1][8]`, `local_B[1][8]`, `local_C[1][128]` in BRAM; `A_IO_L2_in_serialize` splitting `A_t16` into `A_t8`; an `A_IO_L1_in` module per PE; `B_IO_L2_in_boundary_serialize` and `B_PE_dummy_in`; `C_drain_IO_L1_out` / `C_drain_IO_L2_out_serialize` packing the drained floats. `kernel0` is `DATAFLOW` with one call per id. The MAC loop is pipelined `II=1` on `c6` and unrolled on `c7`.

### ST4 candidate 9

`artifacts/dse/campaigns/20260927_mm1024_u280_st4/u280_paper/validation/mm1024/candidate_9/`

```text
space_time[4]; array_part[512,32,128]; latency[32,1]; simd[4,1]
```

Axis order is `i, k, j`, so the tiles are `i=512`, `k=32`, `j=128`. PE array is `(512/32) × (32/4) = 16 × 8`. `tuning.json` records `"sa_dims": [16, 8]`. `simd[1]` is the skipped candidate and does not appear as a loop. `latency[1] = 1` does not tile `j`.

`S_0(32 * p0 + 512 * c0 + c6, 128 * c1 + c5, 4 * p1 + 32 * c2 + c7)` with `c0` in `0..1` (`1024/512`), `c1` in `0..7` (`1024/128`), `c2` in `0..31` (`1024/32`), `c5` in `0..127`, `c6` in `0..31`, `c7` in `0..3`.

The kernel contains: `PE(p0, p1)` with `fifo_A_in`, `fifo_B_in`, `fifo_B_out`, `fifo_C_in`, `fifo_C_out`; `local_A[1][4]`, `local_B[1][4]`, `local_C[1][1]`; A loaded through `A_IO_L3_in_serialize` (512-bit to four `A_t4` beats) then `A_IO_L2_in` and `A_IO_L1_in`; B loaded through `B_IO_L3_in_serialize` and `B_IO_L2_in` and forwarded on `p0`; C reduced on `p1`, with `C_PE_dummy_out` at the head and `C_IO_L2_out` / `C_IO_L3_out_serialize` at the tail. The MAC loop is pipelined `II=1` on `c6` and unrolled on `c7`.

The same ST4 command flags are on the other ST4 candidates. Changing `array_part`, `latency`, or `simd` changes the trip counts, the `16×8` shape, and the buffer extents. It does not remove the C partial path or the `local_C[1][1]` register.

## 4. What is not a free knob

Fixed once `space_time` is chosen, for this campaign’s flags:

- Which loops are space and which are time, and therefore the `array_part` axis order (`i,j,k` for 0 and 3, `i,k,j` for 4).
- 1-D versus 2-D PE array, and whether the second dimension is divided by a latency factor (ST3) or by the SIMD factor (ST4).
- Which array is fed, which is forwarded PE to PE, and which is drained versus reduced. ST4 always forwards a C partial along `k` and always keeps `local_C[1][1]`, because every ST4 compile passes `--local-reduce --array-contraction`. ST0 and ST3 always drain C from inside the PE and keep a `local_C` of the per-PE `i×j` footprint.
- IO level names (`L1`/`L2` on ST0, `L3` as well on ST3 and ST4) and the `C_drain_*` versus `C_IO_*` names.
- Dummy modules on the tail of a forwarded array, and `C_PE_dummy_out` on ST4.
- `DATAFLOW` with one call per PE, not a C loop of PEs.
- Outermost DRAM type `ap_uint<512>` (`*_t16`), from the 64-byte data-pack default plus `--data-pack`. Not from `array_part`, `latency`, or `simd`, and not from the Vitis `m_axi_max_widen_bitwidth=512` default.

Changed by the numeric lists:

- Tile trip counts `1024 / array_part[t]`, and the inner extents of `local_A` / `local_B` / `local_C` and of the IO ping-pong buffers.
- PE counts, by the formulas in section 1.
- The pipelined latency-point trip count, when the factor is greater than 1.
- The unrolled SIMD trip count, and the interior FIFO width (`A_t8` at SIMD 8, `A_t4` at SIMD 4). The serialize module splits 512 bits down to that width.

Left off for the whole campaign, so they do not vary per candidate: `array_part_L2`, `--two-level-buffer`, `--credit-control`, `--hbm`, and the Intel burst intrinsic.
