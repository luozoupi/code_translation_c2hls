#include "syr2k.h"

#define TILE 16

// Number of tiles
static void load_C(double C[N][N], double C_tile[TILE][N],
                   int it, int trows)
{
    for (int ti = 0; ti < trows; ti++) {
#pragma HLS LOOP_TRIPCOUNT min=1 max=TILE
        int i = it + ti;
        for (int j = 0; j < N; j++) {
#pragma HLS PIPELINE II=1
            C_tile[ti][j] = C[i][j];
        }
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

static void store_C(double C[N][N], double C_tile[TILE][N],
                    int it, int trows)
{
    for (int ti = 0; ti < trows; ti++) {
#pragma HLS LOOP_TRIPCOUNT min=1 max=TILE
        int i = it + ti;
        for (int j = 0; j < N; j++) {
#pragma HLS PIPELINE II=1
            C[i][j] = C_tile[ti][j];
        }
    }
}

void kernel_syr2k( 
		  double alpha,
		  double beta,
		  double C[ N + 0][N + 0],
		  double A[ N + 0][M + 0],
		  double B[ N + 0][M + 0])
{
#pragma HLS INTERFACE m_axi port=C offset=slave bundle=gmem0
#pragma HLS INTERFACE m_axi port=A offset=slave bundle=gmem1
#pragma HLS INTERFACE m_axi port=B offset=slave bundle=gmem2
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

    // ---- Load A and B (full reuse working set) ----
    for (int i = 0; i < n; i++) {
        for (int k = 0; k < m; k++) {
#pragma HLS PIPELINE II=1
            A_buf[i][k] = A[i][k];
            B_buf[i][k] = B[i][k];
        }
    }

    // ---- Double-buffered C tile processing ----
    // Two ping-pong buffers for the C tile.
    double C_tile_0[TILE][N];
    double C_tile_1[TILE][N];
#pragma HLS ARRAY_PARTITION variable=C_tile_0 cyclic factor=2 dim=1
#pragma HLS ARRAY_PARTITION variable=C_tile_0 cyclic factor=4 dim=2
#pragma HLS ARRAY_PARTITION variable=C_tile_1 cyclic factor=2 dim=1
#pragma HLS ARRAY_PARTITION variable=C_tile_1 cyclic factor=4 dim=2

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