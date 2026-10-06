# How the modules are stitched with streams

This page is the wiring shared by the thirty 1024³ U280 designs. A per-config file lists the FIFOs, types, and trip counts of one candidate. Read that file for the schedule, then come back here for the directions.

The examples below are copied from generated `kernel0` calls:

- Space-time 0, candidate 1: `.../20260927_mm1024_u280_st0/.../candidate_1/.../kernel_kernel.cpp` (128 PEs).
- Space-time 3, candidate 1: `.../20260927_mm1024_u280_st3/.../candidate_1/...` (16×8).
- Space-time 4, candidate 9: `.../20260927_mm1024_u280_st4/.../candidate_9/...` (16×8). This is the `array_part` `(512, 32, 128)`, latency `(32, 1)`, SIMD `(4, 1)` design.

The same module names and the same directions appear on every sibling of that space-time. Counts scale with the PE array. Element types scale with the SIMD and with where that candidate splits the 512-bit DRAM word. Both are in the per-config file.

## What `kernel0` is

`kernel0(A_t16 *A, B_t16 *B, C_t16 *C)` is one `#pragma HLS DATAFLOW` region. Each module call is its own process (`#pragma HLS INLINE OFF` on the module bodies). There is no C `for` over PEs. The array is the set of calls.

A stream is an `hls::stream<T>` declared in `kernel0`. Every one of them carries:

```c
#pragma HLS STREAM variable=fifo_... depth=2
```

Depth 2 is the generator default (`--fifo-depth`). The tile size changes how many beats travel through the FIFO. It does not change the depth. A producer stalls when two beats are already sitting unread. A consumer stalls when the FIFO is empty. That is the only handshake in these kernels: the generated files declare no credit FIFO.

Interior FIFOs also carry `#pragma HLS RESOURCE variable=fifo_... core=FIFO_SRL`. The three DRAM-edge FIFOs (one A serialize, one B serialize, one C serialize) are depth 2 and do not carry `FIFO_SRL`. On the 128-PE space-time 0 candidate that is 643 SRL streams and 3 edge streams (646 declarations). The 16×8 space-time 3 and space-time 4 candidates each declare 582 streams, 579 of them SRL.

`A_t16`, `B_t16`, and `C_t16` are `ap_uint<512>`, sixteen floats. That is the outer data-pack on the kernel arguments. The PE port is narrower when the SIMD lane count is narrower. SIMD 8 uses `A_t8` / `B_t8` (`ap_uint<256>`). SIMD 4 uses `A_t4` / `B_t4` (`ap_uint<128>`). SIMD 1 uses `hls::stream<float>`. C leaves every PE as `hls::stream<float>`.

## The three directions

| | Space-time 0 | Space-time 3 | Space-time 4 |
|---|---|---|---|
| Space axes | `p0` = `i` | `p0` = `i`, `p1` = `j` | `p0` = `i`, `p1` = `k` |
| A | IO chain along `i`, one tap per PE. The PE has `fifo_A_in` only. | L2 chain along `i`, injected at `p1 = 0`. The PE forwards A along `j`. | L2 chain along `k`, then an L1 chain along `i` for each `k`. The PE has `fifo_A_in` only. |
| B | `B_IO_L2` feeds `PE(0)`. Each PE forwards B to `p0+1`. | L2 chain along `j`, injected at `p0 = 0`. The PE forwards B along `i`. | L2 chain along `k`, injected at `p0 = 0`. The PE forwards B along `i`. |
| C | Finished inside the PE. `C_drain_*` collects it, high `p0` down to 0. | Same drain, one chain per `j`, high `p0` down to 0, then an L2 chain along `j`. | A partial `float` moves along `k`. `C_IO_*` adds the chain into a `C_t16` tile. |
| Tail modules | `B_PE_dummy_in` | `A_PE_dummy_in` and `B_PE_dummy_in` | `C_PE_dummy_out` and `B_PE_dummy_in` |

A feed always increases the chain index. A drain always decreases it. B reuse follows the `i` axis from `p0 = 0` toward the last `p0`.

## Space-time 0

Candidate 1 wires a line of 128 PEs. `kernel0` calls, with the comments stripped:

```text
A_IO_L2_in_serialize(A, fifo_A_A_IO_L2_in_serialize)
A_IO_L2_in(fifo_A_A_IO_L2_in_serialize, fifo_A_A_IO_L1_in_0)
A_IO_L1_in_wrapper(0, fifo_A_A_IO_L1_in_0, fifo_A_A_IO_L1_in_1, fifo_A_PE_0)
A_IO_L1_in_boundary_wrapper(127, fifo_A_A_IO_L1_in_127, fifo_A_PE_127)
B_IO_L2_in_boundary_serialize(B, fifo_B_B_IO_L2_in_serialize)
B_IO_L2_in_boundary(fifo_B_B_IO_L2_in_serialize, fifo_B_PE_0)
PE_wrapper(0, fifo_A_PE_0, fifo_B_PE_0, fifo_B_PE_1, fifo_C_drain_PE_0)
PE_wrapper(127, fifo_A_PE_127, fifo_B_PE_127, fifo_B_PE_128, fifo_C_drain_PE_127)
B_PE_dummy_in(127, fifo_B_PE_128)
C_drain_IO_L1_out_boundary_wrapper(127, fifo_C_drain_C_drain_IO_L1_out_127, fifo_C_drain_PE_127)
C_drain_IO_L1_out_wrapper(126, fifo_C_drain_C_drain_IO_L1_out_127, fifo_C_drain_C_drain_IO_L1_out_126, fifo_C_drain_PE_126)
C_drain_IO_L1_out_wrapper(0, fifo_C_drain_C_drain_IO_L1_out_1, fifo_C_drain_C_drain_IO_L1_out_0, fifo_C_drain_PE_0)
C_drain_IO_L2_out(fifo_C_drain_C_drain_IO_L2_out_serialize, fifo_C_drain_C_drain_IO_L1_out_0)
C_drain_IO_L2_out_serialize(C, fifo_C_drain_C_drain_IO_L2_out_serialize)
```

There are 127 `A_IO_L1_in_wrapper` calls (ids 0..126) and one boundary at 127. The same split exists on the C drain: the boundary is the highest `p0`, and the wrappers run from `N-2` down to 0.

### A feed

`A_IO_L2_in_serialize` reads `A_t16 *A` and writes the edge FIFO. On candidate 1 that stream is already `A_t8` (the serialize loop shifts by 256, `p < 2`). `A_IO_L2_in` reads that edge, holds a ping/pong buffer, and writes `fifo_A_A_IO_L1_in_0`.

Each `A_IO_L1_in` reads one chain FIFO and writes the next, and also writes `fifo_A_PE_{id}`. The filter is in `A_IO_L1_in_inter_trans`. For module id `p0` the loop starts at that id and runs to the last PE:

```c
for (ap_uint<8> c3 = p0; c3 <= 127; c3 += 1) {
  if (c3 == p0)
    local_A[0][0] = fifo_A_in.read();
  else
    fifo_A_out.write(fifo_A_in.read());
}
```

The beat whose index equals `p0` stays. Every later beat is forwarded. The boundary module has no `fifo_A_out`; its loop only keeps `c3 == p0`. `intra_trans` then writes the kept word onto `fifo_A_PE_p0`. The PE reads `fifo_A_in` and has no A output.

So one burst from DRAM is a sequence of beats for `p0 = 0, 1, …, 127`. Module 0 peels the first beat and forwards the rest. Module 1 peels the next, and so on. The PE array therefore sees its own A in the same order the chain was written.

### B reuse

`B_IO_L2_in_boundary_serialize` reads `B_t16 *B`. `B_IO_L2_in_boundary` is the only B loader, and it writes `fifo_B_PE_0`. `PE(p0)` reads `fifo_B_PE_p0` and writes the same beat to `fifo_B_PE_{p0+1}`. `B_PE_dummy_in(127, fifo_B_PE_128)` consumes the tail so the last PE's write has a reader.

B is broadcast along `i`: every PE in the line sees the same `B[j][k]` beat, one FIFO hop later than its neighbor. DRAM is read once per `i` tile, which is `1024 / array_part[0]` full passes of B.

On some candidates the serialize stream is still `B_t16` while `fifo_B_PE` is `B_t8` or `B_t4`. Candidate 5 (`k` tile 16, SIMD 8) is that case: `B_IO_L2_in_boundary` holds `B_t16 local_B[1024][1]` and the PE port is `B_t8`. The split is inside that IO module. The per-config file names the two types.

### C drain

The PE accumulates `local_C` and writes `fifo_C_drain_PE_p0` only on the last `k` iteration (the `if` is in the per-config file; on candidate 1 it is `if (c2 == 127)`). The element is `float`.

`C_drain_IO_L1_out_boundary(127)` reads `fifo_C_drain_PE_127` and writes `fifo_C_drain_C_drain_IO_L1_out_127`. Each lower wrapper reads the chain FIFO from `p0+1` and the PE FIFO of its own `p0`, packs them into `C_t4`, and writes the chain FIFO of its own `p0`. `C_drain_IO_L2_out` reads index 0. `C_drain_IO_L2_out_serialize` concatenates four `C_t4` words into one `C_t16` and writes DRAM.

The drain therefore runs from the high PE index down to 0, after the reduction inside the PE is finished. Names stay `C_drain_*` on space-time 0 and 3 because of that.

### How many FIFOs, for N PEs

| Family | Count | On candidate 1 |
|---|---:|---|
| `fifo_A_A_IO_L1_in` | N+1 | 129 |
| `fifo_A_PE` | N | 128 |
| `fifo_B_PE` | N+1 | 129 |
| `fifo_C_drain_PE` | N | 128 |
| `fifo_C_drain_C_drain_IO_L1_out` | N+1 | 129 |
| three serialize edges | 3 | 3 |

Candidate 7 and candidate 10 use N = 256. The same formulas hold.

### Order to follow in the file

1. `A_IO_L2_in_serialize`, then `A_IO_L2_in`, then one `A_IO_L1_in` and the boundary.
2. `B_IO_L2_in_boundary_serialize`, then `B_IO_L2_in_boundary`.
3. `PE`. Match `fifo_A_PE_p0`, `fifo_B_PE_p0` in, `fifo_B_PE_{p0+1}` out, `fifo_C_drain_PE_p0` out.
4. `B_PE_dummy_in`.
5. `C_drain_IO_L1_out` from the boundary at `N-1` down to wrapper 0, then `C_drain_IO_L2_out`, then `C_drain_IO_L2_out_serialize`.

Inside a buffered loader, `inter_trans` fills one local array from the upstream FIFO and `intra_trans` writes the downstream FIFO from the other. They swap when the module is double-buffered (`local_*_ping` / `local_*_pong`). `C_drain_IO_L1_out` declares a single `local_C` and runs intra then inter on that buffer.

## Space-time 3

Candidate 1 is a 16×8 array: `p0` is `i` (0..15), `p1` is `j` (0..7). A is forwarded along `j`. B is forwarded along `i`. C is still a drain, so there is no PE-to-PE C stream.

```text
A_IO_L3_in_serialize(A, fifo_A_A_IO_L3_in_serialize)
A_IO_L3_in(fifo_A_A_IO_L3_in_serialize, fifo_A_A_IO_L2_in_0)
A_IO_L2_in(0, fifo_A_A_IO_L2_in_0, fifo_A_A_IO_L2_in_1, fifo_A_PE_0_0)
A_IO_L2_in_boundary(15, fifo_A_A_IO_L2_in_15, fifo_A_PE_15_0)
B_IO_L3_in_serialize(B, fifo_B_B_IO_L3_in_serialize)
B_IO_L3_in(fifo_B_B_IO_L3_in_serialize, fifo_B_B_IO_L2_in_0)
B_IO_L2_in(0, fifo_B_B_IO_L2_in_0, fifo_B_B_IO_L2_in_1, fifo_B_PE_0_0)
B_IO_L2_in_boundary(7, fifo_B_B_IO_L2_in_7, fifo_B_PE_0_7)
PE_wrapper(0, 0, fifo_A_PE_0_0, fifo_A_PE_0_1, fifo_B_PE_0_0, fifo_B_PE_1_0, fifo_C_drain_PE_0_0)
PE_wrapper(15, 7, fifo_A_PE_15_7, fifo_A_PE_15_8, fifo_B_PE_15_7, fifo_B_PE_16_7, fifo_C_drain_PE_15_7)
A_PE_dummy_in(0, 7, fifo_A_PE_0_8)
A_PE_dummy_in(15, 7, fifo_A_PE_15_8)
B_PE_dummy_in(15, 0, fifo_B_PE_16_0)
B_PE_dummy_in(15, 7, fifo_B_PE_16_7)
C_drain_IO_L1_out_boundary_wrapper(0, 15, fifo_C_drain_C_drain_IO_L1_out_0_15, fifo_C_drain_PE_15_0)
C_drain_IO_L1_out_wrapper(0, 14, fifo_C_drain_C_drain_IO_L1_out_0_15, fifo_C_drain_C_drain_IO_L1_out_0_14, fifo_C_drain_PE_14_0)
C_drain_IO_L2_out_boundary(7, fifo_C_drain_C_drain_IO_L2_out_7, fifo_C_drain_C_drain_IO_L1_out_7_0)
C_drain_IO_L2_out(0, fifo_C_drain_C_drain_IO_L2_out_1, fifo_C_drain_C_drain_IO_L2_out_0, fifo_C_drain_C_drain_IO_L1_out_0_0)
C_drain_IO_L3_out(fifo_C_drain_C_drain_IO_L3_out_serialize, fifo_C_drain_C_drain_IO_L2_out_0)
C_drain_IO_L3_out_serialize(C, fifo_C_drain_C_drain_IO_L3_out_serialize)
```

`PE(idx, idy)` uses `p0 = idx` and `p1 = idy`. The FIFO suffix `fifo_*_PE_{p0}_{p1}` follows that.

### What each PE reads and writes

`PE(p0, p1)` reads `fifo_A_PE_{p0}_{p1}` and writes `fifo_A_PE_{p0}_{p1+1}`. A enters at `p1 = 0` from `A_IO_L2` and walks along `j`. `A_PE_dummy_in(p0, P1-1, fifo_A_PE_{p0}_{P1})` consumes the tail. There is one dummy per `p0` (16 on this candidate).

`PE(p0, p1)` reads `fifo_B_PE_{p0}_{p1}` and writes `fifo_B_PE_{p0+1}_{p1}`. B enters at `p0 = 0` from `B_IO_L2` and walks along `i`. `B_PE_dummy_in(P0-1, p1, fifo_B_PE_{P0}_{p1})` consumes the tail. There is one dummy per `p1` (8 on this candidate).

`fifo_C_drain_PE_{p0}_{p1}` is written by that PE and read by the drain. The PE has no C input.

### Where the 512-bit word is split

On every space-time 3 candidate in this set, `A_IO_L3_in_serialize` copies `A_t16` through (`hls::stream<A_t16>`, no shift). `A_IO_L2` holds `A_t16` and its local stream toward the PE is the SIMD width (`A_t8` on candidate 1, `A_t4` when SIMD is 4). The same pattern holds for B. The narrowing is `A_IO_L2_in_intra_trans` / `B_IO_L2_in_intra_trans`, which unpack the buffered `*_t16` word into the PE stream.

### C drain, and the argument order

The L1 chain is one column of `i` for each fixed `j`. It flows from `p0 = P0-1` down to `p0 = 0`.

The call ids are `(p1, p0)` while the PE FIFO is `fifo_C_drain_PE_{p0}_{p1}`:

```text
C_drain_IO_L1_out_boundary_wrapper(0, 15, fifo_C_drain_C_drain_IO_L1_out_0_15, fifo_C_drain_PE_15_0)
```

That call drains `PE(15, 0)`. The first argument is the `j` index, the second is the `i` index. The chain FIFO `fifo_C_drain_C_drain_IO_L1_out_{p1}_{p0}` uses the same order as the call, which is the opposite of the PE FIFO suffix. Follow the `fifo_C_drain_PE_*` argument when you want the PE.

`C_drain_IO_L2` then chains those column results along `j`, from `p1 = P1-1` down to 0, and `C_drain_IO_L3_out_serialize` writes `C_t16`.

### How many FIFOs, for a P0×P1 array

| Family | Count | 16×8 |
|---|---:|---:|
| `fifo_A_A_IO_L2_in` | P0+1 | 17 |
| `fifo_A_PE` | P0×(P1+1) | 144 |
| `fifo_B_B_IO_L2_in` | P1+1 | 9 |
| `fifo_B_PE` | (P0+1)×P1 | 136 |
| `fifo_C_drain_PE` | P0×P1 | 128 |
| `fifo_C_drain_C_drain_IO_L1_out` | (P0+1)×P1 | 136 |
| `fifo_C_drain_C_drain_IO_L2_out` | P1+1 | 9 |
| three serialize edges | 3 | 3 |

A 16×16 array (SIMD 4 candidates) uses the same formulas with P1 = 16.

### Order to follow

1. A: serialize, `A_IO_L3_in`, one `A_IO_L2_in`, the A L2 boundary.
2. B: serialize, `B_IO_L3_in`, one `B_IO_L2_in`, the B L2 boundary.
3. One `PE(p0, p1)`. A steps in `p1`. B steps in `p0`.
4. One `A_PE_dummy_in` and one `B_PE_dummy_in`.
5. C drain L1 from the high `p0` boundary down the column, then L2 from the high `p1` down to 0, then L3, then serialize.

## Space-time 4

Candidate 9 is also 16×8, but `p1` is `k`, not `j`. `array_part` order is `(i, k, j) = (512, 32, 128)`. `PE(idx, idy)` is `p0 = idx` along `i` and `p1 = idy` along `k`. The reliable name is the FIFO suffix `fifo_*_PE_{p0}_{p1}`.

```text
A_IO_L3_in_serialize(A, fifo_A_A_IO_L3_in_serialize)
A_IO_L3_in(fifo_A_A_IO_L3_in_serialize, fifo_A_A_IO_L2_in_0)
A_IO_L2_in(0, fifo_A_A_IO_L2_in_0, fifo_A_A_IO_L2_in_1, fifo_A_A_IO_L1_in_0_0)
A_IO_L2_in_boundary(7, fifo_A_A_IO_L2_in_7, fifo_A_A_IO_L1_in_7_0)
A_IO_L1_in_wrapper(0, 0, fifo_A_A_IO_L1_in_0_0, fifo_A_A_IO_L1_in_0_1, fifo_A_PE_0_0)
A_IO_L1_in_boundary_wrapper(0, 15, fifo_A_A_IO_L1_in_0_15, fifo_A_PE_15_0)
B_IO_L3_in_serialize(B, fifo_B_B_IO_L3_in_serialize)
B_IO_L3_in(fifo_B_B_IO_L3_in_serialize, fifo_B_B_IO_L2_in_0)
B_IO_L2_in(0, fifo_B_B_IO_L2_in_0, fifo_B_B_IO_L2_in_1, fifo_B_PE_0_0)
B_IO_L2_in_boundary(7, fifo_B_B_IO_L2_in_7, fifo_B_PE_0_7)
C_PE_dummy_out(0, 0, fifo_C_PE_0_0)
C_PE_dummy_out(15, 0, fifo_C_PE_15_0)
PE_wrapper(0, 0, fifo_A_PE_0_0, fifo_B_PE_0_0, fifo_B_PE_1_0, fifo_C_PE_0_0, fifo_C_PE_0_1)
PE_wrapper(15, 7, fifo_A_PE_15_7, fifo_B_PE_15_7, fifo_B_PE_16_7, fifo_C_PE_15_7, fifo_C_PE_15_8)
B_PE_dummy_in(15, 0, fifo_B_PE_16_0)
C_IO_L2_out_boundary(15, fifo_C_C_IO_L2_out_15, fifo_C_PE_15_8)
C_IO_L2_out(14, fifo_C_C_IO_L2_out_15, fifo_C_C_IO_L2_out_14, fifo_C_PE_14_8)
C_IO_L2_out(0, fifo_C_C_IO_L2_out_1, fifo_C_C_IO_L2_out_0, fifo_C_PE_0_8)
C_IO_L3_out(fifo_C_C_IO_L3_out_serialize, fifo_C_C_IO_L2_out_0)
C_IO_L3_out_serialize(C, fifo_C_C_IO_L3_out_serialize)
```

There is no `C_drain_*` module. C is `C_IO_*` plus `C_PE_dummy_out`.

### A feed, and the L1 argument names

`A_IO_L2` is a chain along `k` (`p1` = 0..7). Each stage taps the head of an L1 chain, `fifo_A_A_IO_L1_in_{p1}_0`.

`A_IO_L1` is a chain along `i` for that fixed `k`. The call

```text
A_IO_L1_in_boundary_wrapper(0, 15, fifo_A_A_IO_L1_in_0_15, fifo_A_PE_15_0)
```

feeds `PE(p0=15, p1=0)`. The first argument is the `k` index. The second argument is the `i` index. The chain FIFO is `fifo_A_A_IO_L1_in_{k}_{i}`.

Inside `A_IO_L1_in_inter_trans` the source writes `int p0 = idx, p1 = idy` and then loops `for (c4 = p1; c4 <= 15; c4++)`, keeping the beat when `c4 == p1`. Those local names do not match the PE axes: `idx` is `k` and `idy` is `i`. The loop bound 15 is the last `i`, which is the PE's `p0`. Use the `fifo_A_PE_{p0}_{p1}` argument as the PE address.

The keep-or-forward rule is the same idea as space-time 0. On candidate 9 the kept beat is a whole `i` extent (`c5 = 0..31` into `local_A[c5][0]`), and the other `i` positions are written to `fifo_A_out`.

The PE reads `fifo_A_in` only. A is not forwarded across PEs.

### B reuse

`B_IO_L2` chains along `k` and injects `fifo_B_PE_0_{p1}`. `PE(p0, p1)` reads `fifo_B_PE_{p0}_{p1}` and writes `fifo_B_PE_{p0+1}_{p1}`. `B_PE_dummy_in(15, p1, fifo_B_PE_16_{p1})` consumes the tail. B still walks along `i`, one chain per `k` lane.

### C partial along k

`C_PE_dummy_out(p0, 0)` writes `fifo_C_PE_{p0}_0`. `PE(p0, p1)` reads `fifo_C_PE_{p0}_{p1}` and writes `fifo_C_PE_{p0}_{p1+1}`. The last link `fifo_C_PE_{p0}_{P1}` is read by `C_IO_L2`. On candidate 9 that is `fifo_C_PE_15_8` into `C_IO_L2_out_boundary(15, ...)`.

The element on this path is `float`. Each PE holds `local_C[1][1]`, adds its SIMD lanes for the current `j`, and forwards the sum. The `i×j` tile lives in `C_IO_L2` (`C_t16 local_C[32][8]` on candidate 9), which is where array contraction puts it.

`C_IO_L2` then chains along `i` from `p0 = P0-1` down to 0, in `C_t16`. `C_IO_L3_out_serialize` copies `C_t16` to DRAM (trip 65536, no shift). The 512-bit pack of C is assembled in `C_IO_L2`, not by concatenating `C_t4` the way the drain does.

### When a partial is read

`C_PE_dummy_out` on candidate 9 writes a zero only under the guard, and it writes one zero per pipelined `j` point:

```c
for (ap_uint<6> c2 = 0; c2 <= 31; c2 += 1)
  if (p1 + c2 >= 1) {
    for (ap_uint<8> c5 = 0; c5 <= 127; c5 += 1)
      for (ap_uint<6> c6 = 0; c6 <= 31; c6 += 1) {
        C_t1 fifo_data = 0;
        fifo_C_out.write(fifo_data);
      }
  }
```

`p1` is 0 on every dummy call. The guard is false only for `c2 == 0`. The first `k` tile therefore produces no token, and the PE at `p1 = 0` does not read `fifo_C_in` on that same guard (`p1 + c2 >= 1` in the PE). Later `k` tiles do: the dummy injects zeros, `PE(p0, 0)` adds its lanes, and each later `p1` adds its lanes onto that partial.

`C_IO_L2_out_intra_trans` zeros its own buffer, then for every `k` tile reads one float and adds it. The source marks the add `/* Local Reduction */`. The sum across `k` tiles is that add in `C_IO`. The PE's `local_C[1][1]` is the partial for the current `j` point inside one `k` position; the IO buffer is the place that is explicitly set to 0.

Other space-time 4 candidates write the guard as `p1 + 8*c2 >= 1` or `p1 + 16*c2 >= 1`. For these nonnegative iterators the false case is still only `p1 == 0` and `c2 == 0`. The per-config file quotes the guard that candidate prints.

### How many FIFOs, for a P0×P1 array

| Family | Count | 16×8 | 8×16 (candidate 8) | 32×16 (candidate 10) |
|---|---:|---:|---:|---:|
| `fifo_A_A_IO_L2_in` | P1+1 | 9 | 17 | 17 |
| `fifo_A_A_IO_L1_in` | (P0+1)×P1 | 136 | 144 | 528 |
| `fifo_A_PE` | P0×P1 | 128 | 128 | 512 |
| `fifo_B_B_IO_L2_in` | P1+1 | 9 | 17 | 17 |
| `fifo_B_PE` | (P0+1)×P1 | 136 | 144 | 528 |
| `fifo_C_PE` | P0×(P1+1) | 144 | 136 | 544 |
| `fifo_C_C_IO_L2_out` | P0+1 | 17 | 9 | 33 |
| three serialize edges | 3 | 3 | 3 | 3 |

Candidate 10 (SIMD 1) uses the same directions. Its A, B, and C-partial streams are `hls::stream<float>`. `A_IO_L3_in_serialize` splits `A_t16` into 16 floats (`p < 16`, shift 32). `C_IO_L2` is still `C_t16`.

### Order to follow

1. A: serialize, `A_IO_L3_in`, one `A_IO_L2_in` (this index is `k`), one `A_IO_L1_in` (first argument `k`, second argument `i`).
2. B: serialize, `B_IO_L3_in`, one `B_IO_L2_in`.
3. `C_PE_dummy_out(p0, 0)`, then `PE(p0, p1)`, then `B_PE_dummy_in`.
4. `C_IO_L2_out` from the boundary at the last `p0` down to 0, then `C_IO_L3_out`, then serialize.
5. In `C_IO_L2_out_intra_trans`, read the zeroing loop and the `/* Local Reduction */` add before the PE's MAC. That is where the `k` tiles become one `C` word.

## Produce, consume, and the depth

Walk one beat as a reader of `kernel0`:

1. A serialize `read`s one DRAM word and `write`s one or more interior beats (the `p` loop) onto the edge FIFO. The consumer is the next IO module. Depth is 2, so serialize stalls after two unread beats.
2. A chain module `read`s every beat that is still upstream of it. It `write`s the beats that belong to later PEs, and it parks its own beat in `local_A`. `intra_trans` later `write`s that parked beat toward the PE. Ping and pong let one buffer fill while the other drains.
3. The PE `read`s one A beat and one B beat per pipelined step, unpacks the lanes (the unpack is a full unroll of the SIMD factor), and updates `local_C`.
4. B's producer is either the IO module (`p0 = 0`) or the previous PE. The consumer is the next PE, or `B_PE_dummy_in` after the last `p0`. The dummy's job is to `read` those tail beats. Without it the last PE would stall on a full FIFO.
5. On space-time 0 and 3 the PE `write`s C once, when the `k` loop hits its last iteration. The drain chain `read`s those floats from high index to low index and the serialize module `write`s `C_t16`.
6. On space-time 4 the PE `write`s one float partial per pipelined `j` point, every time the guard passes. The next `p1` `read`s it. `C_IO_L2` `read`s the end of that chain and adds it into the packed tile, then the `i` chain carries `C_t16` down to serialize.

The FIFO depth stays 2 on every family. A longer tile means more beats and a longer stall if the consumer is late. It does not allocate a deeper FIFO.

## What these kernels leave out of the wiring

`kernel0` takes three pointers, `A`, `B`, and `C`. The campaign flags that produce this shape are recorded in [mm_codegen_factors.md](../mm_codegen_factors.md): `--host-serialize`, `hbm.enable=0`, `array_part_L2.enable=0`, and `--two-level-buffer` left off. The double buffer you see as `local_*_ping` / `local_*_pong` is the default `--double-buffer`. Space-time 4 is the build that passes `--local-reduce --array-contraction`, which is why its C path is `C_IO_*` and `local_C[1][1]` in the PE.
