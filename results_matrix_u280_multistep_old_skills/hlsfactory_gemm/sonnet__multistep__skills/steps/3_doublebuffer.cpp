#include "gemm.h"

// Tile sizes for tiling optimization
#define TILE_I 16
#define TILE_J 16

extern "C" {

static void load_tile_A(double l_A_0[TILE_I][NK],
                        double l_A_1[TILE_I][NK],
                        double A[NI][NK],
                        int i0, int ping)
{
#pragma HLS ARRAY_PARTITION variable=l_A_0 cyclic factor=8 dim=2
#pragma HLS ARRAY_PARTITION variable=l_A_1 cyclic factor=8 dim=2
    for (int i = 0; i < TILE_I; i++) {
        int gi = i0 + i;
        if (gi < NI) {
            for (int k = 0; k < NK; k++) {
#pragma HLS PIPELINE II=1
#pragma HLS LOOP_TRIPCOUNT min=80 max=80
                if (ping == 0)
                    l_A_0[i][k] = A[gi][k];
                else
                    l_A_1[i][k] = A[gi][k];
            }
        }
    }
}

static void load_tile_B(double l_B_0[TILE_J][NK],
                        double l_B_1[TILE_J][NK],
                        double B[NK][NJ],
                        int j0, int ping)
{
#pragma HLS ARRAY_PARTITION variable=l_B_0 cyclic factor=8 dim=2
#pragma HLS ARRAY_PARTITION variable=l_B_1 cyclic factor=8 dim=2
    for (int j = 0; j < TILE_J; j++) {
        int gj = j0 + j;
        if (gj < NJ) {
            for (int k = 0; k < NK; k++) {
#pragma HLS PIPELINE II=1
#pragma HLS LOOP_TRIPCOUNT min=80 max=80
                if (ping == 0)
                    l_B_0[j][k] = B[k][gj];
                else
                    l_B_1[j][k] = B[k][gj];
            }
        }
    }
}

static void load_tile_C(double l_C_0[TILE_I][TILE_J],
                        double l_C_1[TILE_I][TILE_J],
                        double C[NI][NJ],
                        int i0, int j0, int ping)
{
#pragma HLS ARRAY_PARTITION variable=l_C_0 complete dim=2
#pragma HLS ARRAY_PARTITION variable=l_C_1 complete dim=2
    for (int i = 0; i < TILE_I; i++) {
        int gi = i0 + i;
        if (gi < NI) {
            for (int j = 0; j < TILE_J; j++) {
                int gj = j0 + j;
#pragma HLS PIPELINE II=1
#pragma HLS LOOP_TRIPCOUNT min=16 max=16
                if (gj < NJ) {
                    if (ping == 0)
                        l_C_0[i][j] = C[gi][gj];
                    else
                        l_C_1[i][j] = C[gi][gj];
                }
            }
        }
    }
}

static void compute_tile(double l_C_0[TILE_I][TILE_J],
                         double l_C_1[TILE_I][TILE_J],
                         double l_A_0[TILE_I][NK],
                         double l_A_1[TILE_I][NK],
                         double l_B_0[TILE_J][NK],
                         double l_B_1[TILE_J][NK],
                         double alpha, double beta, int ping)
{
#pragma HLS ARRAY_PARTITION variable=l_C_0 complete dim=2
#pragma HLS ARRAY_PARTITION variable=l_C_1 complete dim=2
#pragma HLS ARRAY_PARTITION variable=l_A_0 cyclic factor=8 dim=2
#pragma HLS ARRAY_PARTITION variable=l_A_1 cyclic factor=8 dim=2
#pragma HLS ARRAY_PARTITION variable=l_B_0 cyclic factor=8 dim=2
#pragma HLS ARRAY_PARTITION variable=l_B_1 cyclic factor=8 dim=2

    // Scale C tile by beta
    for (int i = 0; i < TILE_I; i++) {
#pragma HLS LOOP_TRIPCOUNT min=16 max=16
        for (int j = 0; j < TILE_J; j++) {
#pragma HLS PIPELINE II=1
#pragma HLS UNROLL factor=8
#pragma HLS LOOP_TRIPCOUNT min=16 max=16
#pragma HLS DEPENDENCE variable=l_C_0 inter false
#pragma HLS DEPENDENCE variable=l_C_1 inter false
            if (ping == 0)
                l_C_0[i][j] *= beta;
            else
                l_C_1[i][j] *= beta;
        }
    }

    // Accumulate alpha * A[i][k] * B[k][j] into C tile
    for (int i = 0; i < TILE_I; i++) {
#pragma HLS LOOP_TRIPCOUNT min=16 max=16
        for (int k = 0; k < NK; k++) {
#pragma HLS PIPELINE II=1
#pragma HLS LOOP_TRIPCOUNT min=80 max=80
            for (int j = 0; j < TILE_J; j++) {
#pragma HLS UNROLL factor=8
#pragma HLS LOOP_TRIPCOUNT min=16 max=16
#pragma HLS DEPENDENCE variable=l_C_0 inter false
#pragma HLS DEPENDENCE variable=l_C_1 inter false
                if (ping == 0)
                    l_C_0[i][j] += alpha * l_A_0[i][k] * l_B_0[j][k];
                else
                    l_C_1[i][j] += alpha * l_A_1[i][k] * l_B_1[j][k];
            }
        }
    }
}

static void store_tile_C(double l_C_0[TILE_I][TILE_J],
                         double l_C_1[TILE_I][TILE_J],
                         double C[NI][NJ],
                         int i0, int j0, int ping)
{
#pragma HLS ARRAY_PARTITION variable=l_C_0 complete dim=2
#pragma HLS ARRAY_PARTITION variable=l_C_1 complete dim=2
    for (int i = 0; i < TILE_I; i++) {
        int gi = i0 + i;
        if (gi < NI) {
            for (int j = 0; j < TILE_J; j++) {
                int gj = j0 + j;
#pragma HLS PIPELINE II=1
#pragma HLS LOOP_TRIPCOUNT min=16 max=16
                if (gj < NJ) {
                    if (ping == 0)
                        C[gi][gj] = l_C_0[i][j];
                    else
                        C[gi][gj] = l_C_1[i][j];
                }
            }
        }
    }
}

void kernel_gemm(  
		 double alpha,
		 double beta,
		 double C[ NI + 0][NJ + 0],
		 double A[ NI + 0][NK + 0],
		 double B[ NK + 0][NJ + 0])
{
#pragma HLS INTERFACE m_axi port=C offset=slave bundle=gmem0
#pragma HLS INTERFACE m_axi port=A offset=slave bundle=gmem1
#pragma HLS INTERFACE m_axi port=B offset=slave bundle=gmem2
#pragma HLS INTERFACE s_axilite port=alpha bundle=control
#pragma HLS INTERFACE s_axilite port=beta  bundle=control
#pragma HLS INTERFACE s_axilite port=C     bundle=control
#pragma HLS INTERFACE s_axilite port=A     bundle=control
#pragma HLS INTERFACE s_axilite port=B     bundle=control
#pragma HLS INTERFACE s_axilite port=return bundle=control

    // Double-buffered local tile buffers (ping-pong buffers)
    double l_A_0[TILE_I][NK];
    double l_A_1[TILE_I][NK];
    double l_B_0[TILE_J][NK];
    double l_B_1[TILE_J][NK];
    double l_C_0[TILE_I][TILE_J];
    double l_C_1[TILE_I][TILE_J];

#pragma HLS ARRAY_PARTITION variable=l_A_0 cyclic factor=8 dim=2
#pragma HLS ARRAY_PARTITION variable=l_A_1 cyclic factor=8 dim=2
#pragma HLS ARRAY_PARTITION variable=l_B_0 cyclic factor=8 dim=2
#pragma HLS ARRAY_PARTITION variable=l_B_1 cyclic factor=8 dim=2
#pragma HLS ARRAY_PARTITION variable=l_C_0 complete dim=2
#pragma HLS ARRAY_PARTITION variable=l_C_1 complete dim=2

    // Total number of tiles
    const int NUM_TILES_I = (NI + TILE_I - 1) / TILE_I;  // 4
    const int NUM_TILES_J = (NJ + TILE_J - 1) / TILE_J;  // 5
    const int TOTAL_TILES = NUM_TILES_I * NUM_TILES_J;    // 20

    // Double-buffering loop:
    // Iteration t: load tile t into buffer[t%2], compute tile t-1 from buffer[(t-1)%2], store tile t-1
    // We run TOTAL_TILES+1 iterations:
    //   iter 0:         load tile 0 into buf[0]
    //   iter 1..T-1:    load tile t into buf[t%2], compute+store tile t-1 from buf[(t-1)%2]
    //   iter T:         compute+store tile T-1 from buf[(T-1)%2] (no load)

    int prev_i0 = 0, prev_j0 = 0;

    // Flatten the 2D tile loop into 1D for clean double-buffering
    tile_loop: for (int t = 0; t < TOTAL_TILES + 1; t++) {
#pragma HLS LOOP_TRIPCOUNT min=21 max=21

        // Current tile indices (for loading)
        int cur_i0 = -1, cur_j0 = -1;
        if (t < TOTAL_TILES) {
            int ti = t / NUM_TILES_J;
            int tj = t % NUM_TILES_J;
            cur_i0 = ti * TILE_I;
            cur_j0 = tj * TILE_J;
        }

        // ping selects which buffer the CURRENT tile loads into
        // (t % 2): buffer set for tile t
        int load_ping = t % 2;
        // compute_ping: buffer set for the PREVIOUS tile (t-1)
        int compute_ping = 1 - load_ping;

        // LOAD phase: load current tile t into buffer[load_ping]
        if (t < TOTAL_TILES) {
            load_tile_A(l_A_0, l_A_1, A, cur_i0, load_ping);
            load_tile_B(l_B_0, l_B_1, B, cur_j0, load_ping);
            load_tile_C(l_C_0, l_C_1, C, cur_i0, cur_j0, load_ping);
        }

        // COMPUTE + STORE phase: process previous tile t-1 from buffer[compute_ping]
        if (t > 0) {
            compute_tile(l_C_0, l_C_1, l_A_0, l_A_1, l_B_0, l_B_1,
                         alpha, beta, compute_ping);
            store_tile_C(l_C_0, l_C_1, C, prev_i0, prev_j0, compute_ping);
        }

        // Save current tile coordinates as previous for next iteration
        prev_i0 = cur_i0;
        prev_j0 = cur_j0;
    }
}

} // extern "C"