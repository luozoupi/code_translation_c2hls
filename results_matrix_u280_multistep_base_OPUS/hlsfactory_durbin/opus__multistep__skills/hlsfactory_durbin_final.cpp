#include "durbin.h"
#include <cstring>
#include <cstdint>

#define LARGE_BUS 512

// Wide bus type: 512-bit beat represented as 8 x 64-bit lanes (portable)
typedef struct {
    uint64_t lane[LARGE_BUS / 64];
} MARS_WIDE_BUS_TYPE;

#define TILE 256

// ---- Wide-bus helper: read doubles from a wide AXI bus into a local buffer ----
static void memcpy_wide_bus_read_double(double *local, MARS_WIDE_BUS_TYPE *bus,
                                        size_t byte_offset, size_t num_bytes)
{
    const int DOUBLES_PER_BEAT = LARGE_BUS / 64; // 8 doubles per 512-bit beat
    size_t num_doubles = num_bytes / sizeof(double);
    size_t beat_base = byte_offset / (LARGE_BUS / 8);
    size_t num_beats = (num_doubles + DOUBLES_PER_BEAT - 1) / DOUBLES_PER_BEAT;

RD_BEATS:
    for (size_t b = 0; b < num_beats; b++) {
#pragma HLS LOOP_TRIPCOUNT min=1 max=16
#pragma HLS PIPELINE II=1
        MARS_WIDE_BUS_TYPE beat = bus[beat_base + b];
    RD_LANES:
        for (int l = 0; l < DOUBLES_PER_BEAT; l++) {
#pragma HLS UNROLL
            size_t idx = b * DOUBLES_PER_BEAT + l;
            if (idx < num_doubles) {
                uint64_t bits = beat.lane[l];
                double v;
                std::memcpy(&v, &bits, sizeof(double));
                local[idx] = v;
            }
        }
    }
}

// ---- Wide-bus helper: write doubles from a local buffer to a wide AXI bus ----
static void memcpy_wide_bus_write_double(MARS_WIDE_BUS_TYPE *bus, double *local,
                                         size_t byte_offset, size_t num_bytes)
{
    const int DOUBLES_PER_BEAT = LARGE_BUS / 64;
    size_t num_doubles = num_bytes / sizeof(double);
    size_t beat_base = byte_offset / (LARGE_BUS / 8);
    size_t num_beats = (num_doubles + DOUBLES_PER_BEAT - 1) / DOUBLES_PER_BEAT;

WR_BEATS:
    for (size_t b = 0; b < num_beats; b++) {
#pragma HLS LOOP_TRIPCOUNT min=1 max=16
#pragma HLS PIPELINE II=1
        MARS_WIDE_BUS_TYPE beat;
    WR_LANES:
        for (int l = 0; l < DOUBLES_PER_BEAT; l++) {
#pragma HLS UNROLL
            size_t idx = b * DOUBLES_PER_BEAT + l;
            double v = (idx < num_doubles) ? local[idx] : 0.0;
            uint64_t bits;
            std::memcpy(&bits, &v, sizeof(double));
            beat.lane[l] = bits;
        }
        bus[beat_base + b] = beat;
    }
}

// Load a tile of r[] (already in local buffer) into one of two ping-pong buffers selected by flag
static void load_tile(double r_src[N], double r_buf0[N], double r_buf1[N],
                      int t, int chunk, int flag)
{
LOAD_INNER:
    for (int i = 0; i < chunk; i++) {
#pragma HLS LOOP_TRIPCOUNT min=1 max=256
#pragma HLS PIPELINE II=1
        double v = r_src[t + i];
        if (flag == 0)
            r_buf0[t + i] = v;
        else
            r_buf1[t + i] = v;
    }
}

// Signature matches durbin.h declaration: double* pointers, reinterpreted to wide bus internally
void kernel_durbin(
		   double r[ N + 0],
		   double y[ N + 0])
{
#pragma HLS INTERFACE m_axi port=r offset=slave bundle=gmem0 max_read_burst_length=256 max_write_burst_length=256
#pragma HLS INTERFACE m_axi port=y offset=slave bundle=gmem1 max_read_burst_length=256 max_write_burst_length=256
#pragma HLS INTERFACE s_axilite port=r bundle=control
#pragma HLS INTERFACE s_axilite port=y bundle=control
#pragma HLS INTERFACE s_axilite port=return bundle=control

    // Reinterpret the global double pointers as wide-bus pointers for coalesced bursts
    MARS_WIDE_BUS_TYPE *r_wide = reinterpret_cast<MARS_WIDE_BUS_TYPE *>(r);
    MARS_WIDE_BUS_TYPE *y_wide = reinterpret_cast<MARS_WIDE_BUS_TYPE *>(y);

    const int n = N;

    // ---- DOUBLE-BUFFERED input staging buffers (ping-pong) ----
    double r_dma[N];
    double r_local_0[N];
    double r_local_1[N];
    double r_local[N];
    double y_local[N];
    double z[N];
#pragma HLS ARRAY_PARTITION variable=r_dma cyclic factor=8 dim=1
#pragma HLS ARRAY_PARTITION variable=r_local_0 cyclic factor=8 dim=1
#pragma HLS ARRAY_PARTITION variable=r_local_1 cyclic factor=8 dim=1
#pragma HLS ARRAY_PARTITION variable=r_local cyclic factor=8 dim=1
#pragma HLS ARRAY_PARTITION variable=y_local cyclic factor=8 dim=1
#pragma HLS ARRAY_PARTITION variable=z cyclic factor=8 dim=1

    double alpha;
    double beta;
    double sum;

    int i, k, t;

    // ---- WIDE BUS READ: bring all of r[] into local memory via coalesced burst ----
    memcpy_wide_bus_read_double(r_dma, r_wide, 0, (size_t)n * sizeof(double));

    // ---- LOAD PHASE: stage input r into local memory with double buffering ----
    // Issue loads into alternating buffers so successive tile loads can overlap,
    // then merge the staged tiles into the working buffer.
    int tile_count = 0;
LOAD_TILE:
    for (t = 0; t < n; t += TILE) {
#pragma HLS LOOP_TRIPCOUNT min=1 max=1
        int chunk = (n - t < TILE) ? (n - t) : TILE;
        int flag = (t / TILE) % 2;   // alternate ping-pong buffer per tile

        // Load current tile into the selected ping-pong buffer
        load_tile(r_dma, r_local_0, r_local_1, t, chunk, flag);

        // Consume the just-loaded tile into the working buffer
    MERGE_INNER:
        for (i = 0; i < chunk; i++) {
#pragma HLS LOOP_TRIPCOUNT min=1 max=256
#pragma HLS PIPELINE II=1
            r_local[t + i] = (flag == 0) ? r_local_0[t + i] : r_local_1[t + i];
        }
        tile_count++;
    }

    // ---- COMPUTE PHASE: operate entirely on local buffers ----
    y_local[0] = -r_local[0];
    beta = 1.0;
    alpha = -r_local[0];

COMPUTE:
    for (k = 1; k < n; k++) {
#pragma HLS LOOP_TRIPCOUNT min=1 max=119
        beta = (1 - alpha * alpha) * beta;

        sum = 0.0;
    SUM_LOOP:
        for (i = 0; i < k; i++) {
#pragma HLS LOOP_TRIPCOUNT min=1 max=119
#pragma HLS PIPELINE II=1
            sum += r_local[k - i - 1] * y_local[i];
        }

        alpha = -(r_local[k] + sum) / beta;

    UPDATE_Z:
        for (i = 0; i < k; i++) {
#pragma HLS LOOP_TRIPCOUNT min=1 max=119
#pragma HLS PIPELINE II=1
#pragma HLS UNROLL factor=8
#pragma HLS DEPENDENCE variable=z inter false
#pragma HLS DEPENDENCE variable=y_local inter false
            z[i] = y_local[i] + alpha * y_local[k - i - 1];
        }

    COPY_Y:
        for (i = 0; i < k; i++) {
#pragma HLS LOOP_TRIPCOUNT min=1 max=119
#pragma HLS PIPELINE II=1
#pragma HLS UNROLL factor=8
#pragma HLS DEPENDENCE variable=y_local inter false
#pragma HLS DEPENDENCE variable=z inter false
            y_local[i] = z[i];
        }

        y_local[k] = alpha;
    }

    // ---- STORE PHASE: write local results back to global memory via coalesced burst ----
    memcpy_wide_bus_write_double(y_wide, y_local, 0, (size_t)n * sizeof(double));
}