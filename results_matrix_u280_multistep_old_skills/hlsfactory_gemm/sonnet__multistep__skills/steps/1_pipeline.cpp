#include "gemm.h"

// Tile sizes for tiling optimization
#define TILE_I 16
#define TILE_J 16

extern "C" {

static void load_tile_A(double l_A[TILE_I][NK],
                        double A[NI][NK],
                        int i0)
{
#pragma HLS ARRAY_PARTITION variable=l_A cyclic factor=8 dim=2
    for (int i = 0; i < TILE_I; i++) {
        int gi = i0 + i;
        if (gi < NI) {
            for (int k = 0; k < NK; k++) {
#pragma HLS PIPELINE II=1
#pragma HLS LOOP_TRIPCOUNT min=80 max=80
                l_A[i][k] = A[gi][k];
            }
        }
    }
}

static void load_tile_B(double l_B[TILE_J][NK],
                        double B[NK][NJ],
                        int j0)
{
#pragma HLS ARRAY_PARTITION variable=l_B cyclic factor=8 dim=2
    for (int j = 0; j < TILE_J; j++) {
        int gj = j0 + j;
        if (gj < NJ) {
            for (int k = 0; k < NK; k++) {
#pragma HLS PIPELINE II=1
#pragma HLS LOOP_TRIPCOUNT min=80 max=80
                l_B[j][k] = B[k][gj];
            }
        }
    }
}

static void load_tile_C(double l_C[TILE_I][TILE_J],
                        double C[NI][NJ],
                        int i0, int j0)
{
#pragma HLS ARRAY_PARTITION variable=l_C complete dim=2
    for (int i = 0; i < TILE_I; i++) {
        int gi = i0 + i;
        if (gi < NI) {
            for (int j = 0; j < TILE_J; j++) {
                int gj = j0 + j;
#pragma HLS PIPELINE II=1
#pragma HLS LOOP_TRIPCOUNT min=16 max=16
                if (gj < NJ)
                    l_C[i][j] = C[gi][gj];
            }
        }
    }
}

static void compute_tile(double l_C[TILE_I][TILE_J],
                         double l_A[TILE_I][NK],
                         double l_B[TILE_J][NK],
                         double alpha, double beta)
{
#pragma HLS ARRAY_PARTITION variable=l_C complete dim=2
#pragma HLS ARRAY_PARTITION variable=l_A cyclic factor=8 dim=2
#pragma HLS ARRAY_PARTITION variable=l_B cyclic factor=8 dim=2

    // Scale C tile by beta
    for (int i = 0; i < TILE_I; i++) {
#pragma HLS LOOP_TRIPCOUNT min=16 max=16
        for (int j = 0; j < TILE_J; j++) {
#pragma HLS PIPELINE II=1
#pragma HLS LOOP_TRIPCOUNT min=16 max=16
#pragma HLS DEPENDENCE variable=l_C inter false
            l_C[i][j] *= beta;
        }
    }

    // Accumulate alpha * A[i][k] * B[k][j] into C tile
    // Note: l_B is stored as [j][k] for better access pattern
    for (int i = 0; i < TILE_I; i++) {
#pragma HLS LOOP_TRIPCOUNT min=16 max=16
        for (int k = 0; k < NK; k++) {
#pragma HLS LOOP_TRIPCOUNT min=80 max=80
            for (int j = 0; j < TILE_J; j++) {
#pragma HLS PIPELINE II=1
#pragma HLS LOOP_TRIPCOUNT min=16 max=16
#pragma HLS DEPENDENCE variable=l_C inter false
                l_C[i][j] += alpha * l_A[i][k] * l_B[j][k];
            }
        }
    }
}

static void store_tile_C(double l_C[TILE_I][TILE_J],
                         double C[NI][NJ],
                         int i0, int j0)
{
#pragma HLS ARRAY_PARTITION variable=l_C complete dim=2
    for (int i = 0; i < TILE_I; i++) {
        int gi = i0 + i;
        if (gi < NI) {
            for (int j = 0; j < TILE_J; j++) {
                int gj = j0 + j;
#pragma HLS PIPELINE II=1
#pragma HLS LOOP_TRIPCOUNT min=16 max=16
                if (gj < NJ)
                    C[gi][gj] = l_C[i][j];
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

    // Local tile buffers - much smaller than full matrices
    double l_A[TILE_I][NK];
    double l_B[TILE_J][NK];
    double l_C[TILE_I][TILE_J];

#pragma HLS ARRAY_PARTITION variable=l_A cyclic factor=8 dim=2
#pragma HLS ARRAY_PARTITION variable=l_B cyclic factor=8 dim=2
#pragma HLS ARRAY_PARTITION variable=l_C complete dim=2

    // Tile over i and j dimensions
    tile_i: for (int i0 = 0; i0 < NI; i0 += TILE_I) {
#pragma HLS LOOP_TRIPCOUNT min=4 max=4
        tile_j: for (int j0 = 0; j0 < NJ; j0 += TILE_J) {
#pragma HLS LOOP_TRIPCOUNT min=5 max=5

            // --- LOAD PHASE ---
            // Load A tile: rows [i0..i0+TILE_I), all K columns
            load_tile_A(l_A, A, i0);

            // Load B tile: cols [j0..j0+TILE_J), all K rows
            // Stored transposed as [j][k] for sequential k access in compute
            load_tile_B(l_B, B, j0);

            // Load C tile: rows [i0..i0+TILE_I), cols [j0..j0+TILE_J)
            load_tile_C(l_C, C, i0, j0);

            // --- COMPUTE PHASE ---
            // Operates entirely on local tile buffers
            compute_tile(l_C, l_A, l_B, alpha, beta);

            // --- STORE PHASE ---
            // Write computed C tile back to global memory
            store_tile_C(l_C, C, i0, j0);
        }
    }
}

} // extern "C"