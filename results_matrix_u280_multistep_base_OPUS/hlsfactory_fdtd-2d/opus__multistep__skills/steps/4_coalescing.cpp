#include "fdtd-2d.h"
#include <cstring>

// ---- Wide bus definitions (inlined; no Xilinx headers required) ----
#ifndef LARGE_BUS
#define LARGE_BUS 512
#endif

// Number of 64-bit double elements per wide bus word
#define WIDE_BUS_DOUBLES (LARGE_BUS / 64)

// A 512-bit wide bus word represented as an array of doubles.
typedef struct {
    double data[WIDE_BUS_DOUBLES];
} MARS_WIDE_BUS_TYPE;

// Read `byte_len` bytes (as doubles) from wide bus `bus` at `byte_offset`
// into the local linear buffer `local`.
static void memcpy_wide_bus_read_float(double *local,
                                       MARS_WIDE_BUS_TYPE *bus,
                                       long byte_offset,
                                       long byte_len)
{
    long num_doubles = byte_len / (long)sizeof(double);
    long word_offset = byte_offset / (long)sizeof(double); // offset in doubles
    long base_word = word_offset / WIDE_BUS_DOUBLES;
    long lane = word_offset % WIDE_BUS_DOUBLES;

    long produced = 0;
    long w = base_word;
read_outer:
    while (produced < num_doubles) {
#pragma HLS LOOP_TRIPCOUNT min=1 max=16
        MARS_WIDE_BUS_TYPE val = bus[w];
    read_inner:
        for (long k = lane; k < WIDE_BUS_DOUBLES && produced < num_doubles; k++) {
#pragma HLS LOOP_TRIPCOUNT min=1 max=8
#pragma HLS PIPELINE II=1
            local[produced++] = val.data[k];
        }
        lane = 0;
        w++;
    }
}

// Write `byte_len` bytes (as doubles) from local linear buffer `local`
// into wide bus `bus` at `byte_offset`.
static void memcpy_wide_bus_write_float(MARS_WIDE_BUS_TYPE *bus,
                                        double *local,
                                        long byte_offset,
                                        long byte_len)
{
    long num_doubles = byte_len / (long)sizeof(double);
    long word_offset = byte_offset / (long)sizeof(double);
    long base_word = word_offset / WIDE_BUS_DOUBLES;
    long lane = word_offset % WIDE_BUS_DOUBLES;

    long consumed = 0;
    long w = base_word;
write_outer:
    while (consumed < num_doubles) {
#pragma HLS LOOP_TRIPCOUNT min=1 max=16
        MARS_WIDE_BUS_TYPE val = bus[w];
    write_inner:
        for (long k = lane; k < WIDE_BUS_DOUBLES && consumed < num_doubles; k++) {
#pragma HLS LOOP_TRIPCOUNT min=1 max=8
#pragma HLS PIPELINE II=1
            val.data[k] = local[consumed++];
        }
        bus[w] = val;
        lane = 0;
        w++;
    }
}

// Load the fict value for one time step into the selected ping-pong buffer
static void load_fict(double fict_buf[2], const double _fict_[TMAX], int t, int flag)
{
    if (t < TMAX) {
        fict_buf[flag] = _fict_[t];
    }
}

// Compute one full time step using the fict value in the selected buffer
static void compute_step(
    double ex_buf[NX][NY],
    double ey_buf[NX][NY],
    double hz_buf[NX][NY],
    double fict_buf[2],
    int flag)
{
    int i, j;
    double fict_val = fict_buf[flag];

    for (j = 0; j < NY; j++) {
#pragma HLS LOOP_TRIPCOUNT min=NY max=NY
#pragma HLS PIPELINE II=1
#pragma HLS UNROLL factor=4
        ey_buf[0][j] = fict_val;
    }
    for (i = 1; i < NX; i++)
        for (j = 0; j < NY; j++) {
#pragma HLS LOOP_TRIPCOUNT min=NY max=NY
#pragma HLS PIPELINE II=1
#pragma HLS UNROLL factor=4
            ey_buf[i][j] = ey_buf[i][j] - 0.5*(hz_buf[i][j]-hz_buf[i-1][j]);
        }
    for (i = 0; i < NX; i++)
        for (j = 1; j < NY; j++) {
#pragma HLS LOOP_TRIPCOUNT min=NY max=NY
#pragma HLS PIPELINE II=1
#pragma HLS UNROLL factor=4
            ex_buf[i][j] = ex_buf[i][j] - 0.5*(hz_buf[i][j]-hz_buf[i][j-1]);
        }
    for (i = 0; i < NX - 1; i++)
        for (j = 0; j < NY - 1; j++) {
#pragma HLS LOOP_TRIPCOUNT min=NY max=NY
#pragma HLS PIPELINE II=1
#pragma HLS UNROLL factor=4
            hz_buf[i][j] = hz_buf[i][j] - 0.7*  (ex_buf[i][j+1] - ex_buf[i][j] +
                                                 ey_buf[i+1][j] - ey_buf[i][j]);
        }
}

// Internal coalesced implementation operating on wide-bus pointers.
static void kernel_fdtd_2d_wide(
                    MARS_WIDE_BUS_TYPE *ex,
                    MARS_WIDE_BUS_TYPE *ey,
                    MARS_WIDE_BUS_TYPE *hz,
                    MARS_WIDE_BUS_TYPE *_fict_)
{
    const int tmax = TMAX;
    const int nx = NX;
    const int ny = NY;

    int t, i, j;

    // Local tile buffers for the full 2D grid (staged once per time step)
    double ex_buf[NX][NY];
    double ey_buf[NX][NY];
    double hz_buf[NX][NY];

    // Ping-pong (double) buffer for the per-time-step fict value, so the load
    // of fict for step t+1 can overlap the compute of step t.
    double fict_buf[2];
#pragma HLS ARRAY_PARTITION variable=fict_buf complete dim=1

    // Local linear staging buffer for fict (read via wide bus).
    double fict_local[TMAX];

    // Partition along the column dimension (dim=2) with factor matching the
    // unroll factor (4) so that unrolled iterations can access their data in
    // parallel. cyclic banking provides simultaneous access to consecutive
    // columns [j], [j+1], [j+2], [j+3].
#pragma HLS ARRAY_PARTITION variable=ex_buf cyclic factor=4 dim=2
#pragma HLS ARRAY_PARTITION variable=ey_buf cyclic factor=4 dim=2
#pragma HLS ARRAY_PARTITION variable=hz_buf cyclic factor=4 dim=2

    // ---- Load entire grids into local buffers via wide bus ----
load_ex:
    for (i = 0; i < nx; i++) {
#pragma HLS LOOP_TRIPCOUNT min=NX max=NX
        memcpy_wide_bus_read_float(&ex_buf[i][0], ex, (long)i * ny * sizeof(double), ny * sizeof(double));
    }
load_ey:
    for (i = 0; i < nx; i++) {
#pragma HLS LOOP_TRIPCOUNT min=NX max=NX
        memcpy_wide_bus_read_float(&ey_buf[i][0], ey, (long)i * ny * sizeof(double), ny * sizeof(double));
    }
load_hz:
    for (i = 0; i < nx; i++) {
#pragma HLS LOOP_TRIPCOUNT min=NX max=NX
        memcpy_wide_bus_read_float(&hz_buf[i][0], hz, (long)i * ny * sizeof(double), ny * sizeof(double));
    }

    // ---- Load fict into local buffer via wide bus ----
    memcpy_wide_bus_read_float(&fict_local[0], _fict_, 0, tmax * sizeof(double));

    // ---- Compute phase with double buffering on the fict value ----
    // Prologue: load fict for the first time step into buffer 0.
    load_fict(fict_buf, fict_local, 0, 0);

compute_time:
    for (t = 0; t < tmax; t++)
    {
#pragma HLS LOOP_TRIPCOUNT min=TMAX max=TMAX
        int flag = t & 1;          // buffer currently holding fict for step t
        int next_flag = flag ^ 1;  // buffer to prefetch fict for step t+1

        // Overlap: load fict for the next step while computing the current one.
        load_fict(fict_buf, fict_local, t + 1, next_flag);
        compute_step(ex_buf, ey_buf, hz_buf, fict_buf, flag);
    }

    // ---- Store results back to global memory via wide bus ----
store_ex:
    for (i = 0; i < nx; i++) {
#pragma HLS LOOP_TRIPCOUNT min=NX max=NX
        memcpy_wide_bus_write_float(ex, &ex_buf[i][0], (long)i * ny * sizeof(double), ny * sizeof(double));
    }
store_ey:
    for (i = 0; i < nx; i++) {
#pragma HLS LOOP_TRIPCOUNT min=NX max=NX
        memcpy_wide_bus_write_float(ey, &ey_buf[i][0], (long)i * ny * sizeof(double), ny * sizeof(double));
    }
store_hz:
    for (i = 0; i < nx; i++) {
#pragma HLS LOOP_TRIPCOUNT min=NX max=NX
        memcpy_wide_bus_write_float(hz, &hz_buf[i][0], (long)i * ny * sizeof(double), ny * sizeof(double));
    }
}

// Top-level wrapper matching the header-declared signature. The arrays are
// reinterpreted as wide-bus words for coalesced burst transfers.
void kernel_fdtd_2d(
		    double ex[ NX + 0][NY + 0],
		    double ey[ NX + 0][NY + 0],
		    double hz[ NX + 0][NY + 0],
		    double _fict_[ TMAX + 0])
{
#pragma HLS INTERFACE m_axi port=ex     offset=slave bundle=gmem0 max_read_burst_length=256 max_write_burst_length=256
#pragma HLS INTERFACE m_axi port=ey     offset=slave bundle=gmem1 max_read_burst_length=256 max_write_burst_length=256
#pragma HLS INTERFACE m_axi port=hz     offset=slave bundle=gmem2 max_read_burst_length=256 max_write_burst_length=256
#pragma HLS INTERFACE m_axi port=_fict_ offset=slave bundle=gmem3 max_read_burst_length=256 max_write_burst_length=256
#pragma HLS INTERFACE s_axilite port=ex     bundle=control
#pragma HLS INTERFACE s_axilite port=ey     bundle=control
#pragma HLS INTERFACE s_axilite port=hz     bundle=control
#pragma HLS INTERFACE s_axilite port=_fict_ bundle=control
#pragma HLS INTERFACE s_axilite port=return bundle=control

    kernel_fdtd_2d_wide(
        reinterpret_cast<MARS_WIDE_BUS_TYPE *>(ex),
        reinterpret_cast<MARS_WIDE_BUS_TYPE *>(ey),
        reinterpret_cast<MARS_WIDE_BUS_TYPE *>(hz),
        reinterpret_cast<MARS_WIDE_BUS_TYPE *>(_fict_));
}