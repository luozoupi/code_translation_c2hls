#include "syr2k.h"
#include <string.h>

#define TILE 16

// ---- Wide-bus definitions (plain C++ since Xilinx headers are unavailable) ----
#define LARGE_BUS 512
// Number of doubles per wide-bus word (512 / 64 = 8).
#define DOUBLES_PER_BUS (LARGE_BUS / 64)

// 512-bit memory word represented as 8 packed 64-bit lanes.
struct MARS_WIDE_BUS_TYPE {
    unsigned long long lane[DOUBLES_PER_BUS];
};

// Read `num` doubles starting at byte-offset from the wide bus into buf.
static void memcpy_wide_bus_read_double(double *buf,
                                        MARS_WIDE_BUS_TYPE *bus,
                                        long offset_bytes, int num)
{
    long base_word = offset_bytes / (long)sizeof(double) / DOUBLES_PER_BUS;
    int idx = 0;
    int words = (num + DOUBLES_PER_BUS - 1) / DOUBLES_PER_BUS;
    for (int w = 0; w < words; w++) {
#pragma HLS PIPELINE II=1
        MARS_WIDE_BUS_TYPE val = bus[base_word + w];
        for (int e = 0; e < DOUBLES_PER_BUS; e++) {
#pragma HLS UNROLL
            if (idx < num) {
                unsigned long long u = val.lane[e];
                double d;
                memcpy(&d, &u, sizeof(double));
                buf[idx] = d;
                idx++;
            }
        }
    }
}

// Write `num` doubles from buf to the wide bus starting at byte-offset.
static void memcpy_wide_bus_write_double(MARS_WIDE_BUS_TYPE *bus,
                                         double *buf,
                                         long offset_bytes, int num)
{
    long base_word = offset_bytes / (long)sizeof(double) / DOUBLES_PER_BUS;
    int idx = 0;
    int words = (num + DOUBLES_PER_BUS - 1) / DOUBLES_PER_BUS;
    for (int w = 0; w < words; w++) {
#pragma HLS PIPELINE II=1
        MARS_WIDE_BUS_TYPE val;
        for (int e = 0; e < DOUBLES_PER_BUS; e++) {
#pragma HLS UNROLL
            unsigned long long u = 0;
            if (idx < num) {
                double d = buf[idx];
                memcpy(&u, &d, sizeof(double));
                idx++;
            }
            val.lane[e] = u;
        }
        bus[base_word + w] = val;
    }
}

// Wide-bus loader for one row segment of C into a tile buffer.
static void load_C(MARS_WIDE_BUS_TYPE *C, double C_tile[TILE][N],
                   int it, int trows)
{
    for (int ti = 0; ti < trows; ti++) {
#pragma HLS LOOP_TRIPCOUNT min=1 max=TILE
        int i = it + ti;
        // Burst read the entire row (N doubles) from global memory.
        memcpy_wide_bus_read_double(C_tile[ti], C,
                                    (long)i * N * sizeof(double), N);
    }
}

static void compute_C(double C_tile[TILE][N],
                      double A_buf[N][M], double B_buf[N][M],
                      double alpha, double beta,
                      int it, int trows)
{
    const int m = M;
    for (int ti = 0; ti < trows; ti++) {
#pragma HLS LOOP_TRIPCOUNT min=1 max=TILE
        int i = it + ti;

        // Scale lower-triangular part by beta.
        for (int j = 0; j <= i; j++) {
#pragma HLS LOOP_TRIPCOUNT min=1 max=N
#pragma HLS PIPELINE II=1
            C_tile[ti][j] *= beta;
        }

        // Accumulate the syr2k update.
        for (int j = 0; j <= i; j++) {
#pragma HLS LOOP_TRIPCOUNT min=1 max=N
#pragma HLS PIPELINE II=1
#pragma HLS DEPENDENCE variable=C_tile inter false
            double acc = C_tile[ti][j];
            for (int k = 0; k < m; k++) {
#pragma HLS UNROLL
                acc += A_buf[j][k] * alpha * B_buf[i][k]
                     + B_buf[j][k] * alpha * A_buf[i][k];
            }
            C_tile[ti][j] = acc;
        }
    }
}

static void store_C(MARS_WIDE_BUS_TYPE *C, double C_tile[TILE][N],
                    int it, int trows)
{
    for (int ti = 0; ti < trows; ti++) {
#pragma HLS LOOP_TRIPCOUNT min=1 max=TILE
        int i = it + ti;
        // Burst write the entire row (N doubles) back to global memory.
        memcpy_wide_bus_write_double(C, C_tile[ti],
                                     (long)i * N * sizeof(double), N);
    }
}

void kernel_syr2k(
		  double alpha,
		  double beta,
		  MARS_WIDE_BUS_TYPE *C,
		  MARS_WIDE_BUS_TYPE *A,
		  MARS_WIDE_BUS_TYPE *B)
{
#pragma HLS INTERFACE m_axi port=C offset=slave bundle=gmem0 max_read_burst_length=256 max_write_burst_length=256
#pragma HLS INTERFACE m_axi port=A offset=slave bundle=gmem1 max_read_burst_length=256 max_write_burst_length=256
#pragma HLS INTERFACE m_axi port=B offset=slave bundle=gmem2 max_read_burst_length=256 max_write_burst_length=256
#pragma HLS INTERFACE s_axilite port=alpha bundle=control
#pragma HLS INTERFACE s_axilite port=beta bundle=control
#pragma HLS INTERFACE s_axilite port=C bundle=control
#pragma HLS INTERFACE s_axilite port=A bundle=control
#pragma HLS INTERFACE s_axilite port=B bundle=control
#pragma HLS INTERFACE s_axilite port=return bundle=control

    const int n = N;
    const int m = M;

    // Stage the full A and B matrices once: every output row needs rows 0..i
    // of A and B, so the entire A/B working set is reused across tiles.
    double A_buf[N][M];
    double B_buf[N][M];
#pragma HLS ARRAY_PARTITION variable=A_buf cyclic factor=8 dim=2
#pragma HLS ARRAY_PARTITION variable=B_buf cyclic factor=8 dim=2
#pragma HLS ARRAY_PARTITION variable=A_buf cyclic factor=4 dim=1
#pragma HLS ARRAY_PARTITION variable=B_buf cyclic factor=4 dim=1

    // ---- Load A and B (full reuse working set) via wide-bus bursts ----
    for (int i = 0; i < n; i++) {
        memcpy_wide_bus_read_double(A_buf[i], A,
                                    (long)i * M * sizeof(double), M);
        memcpy_wide_bus_read_double(B_buf[i], B,
                                    (long)i * M * sizeof(double), M);
    }

    // ---- Double-buffered C tile processing ----
    // Two ping-pong buffers for the C tile.
    double C_tile_0[TILE][N];
    double C_tile_1[TILE][N];
#pragma HLS ARRAY_PARTITION variable=C_tile_0 cyclic factor=2 dim=1
#pragma HLS ARRAY_PARTITION variable=C_tile_0 cyclic factor=8 dim=2
#pragma HLS ARRAY_PARTITION variable=C_tile_1 cyclic factor=2 dim=1
#pragma HLS ARRAY_PARTITION variable=C_tile_1 cyclic factor=8 dim=2

    // Number of tiles
    const int num_tiles = (n + TILE - 1) / TILE;

    // Software-pipelined loop:
    //   stage 0: load tile t
    //   stage 1: compute tile t-1
    //   stage 2: store tile t-2
    // Run with an extra +2 iterations to drain the pipeline.
    for (int t = 0; t < num_tiles + 2; t++) {

        // ---- LOAD (tile t) into buffer selected by t%2 ----
        if (t < num_tiles) {
            int it_l = t * TILE;
            int i_end_l = (it_l + TILE < n) ? (it_l + TILE) : n;
            int trows_l = i_end_l - it_l;
            if ((t & 1) == 0)
                load_C(C, C_tile_0, it_l, trows_l);
            else
                load_C(C, C_tile_1, it_l, trows_l);
        }

        // ---- COMPUTE (tile t-1) on buffer selected by (t-1)%2 ----
        if (t >= 1 && (t - 1) < num_tiles) {
            int tc = t - 1;
            int it_c = tc * TILE;
            int i_end_c = (it_c + TILE < n) ? (it_c + TILE) : n;
            int trows_c = i_end_c - it_c;
            if (((tc) & 1) == 0)
                compute_C(C_tile_0, A_buf, B_buf, alpha, beta, it_c, trows_c);
            else
                compute_C(C_tile_1, A_buf, B_buf, alpha, beta, it_c, trows_c);
        }

        // ---- STORE (tile t-2) from buffer selected by (t-2)%2 ----
        if (t >= 2 && (t - 2) < num_tiles) {
            int ts = t - 2;
            int it_s = ts * TILE;
            int i_end_s = (it_s + TILE < n) ? (it_s + TILE) : n;
            int trows_s = i_end_s - it_s;
            if (((ts) & 1) == 0)
                store_C(C, C_tile_0, it_s, trows_s);
            else
                store_C(C, C_tile_1, it_s, trows_s);
        }
    }
}