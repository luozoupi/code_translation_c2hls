#include "kernel.h"

#define TI 32
#define TJ 64

extern "C" void autosa_mm(data_t A[I][K], data_t B[J][K], data_t C[I][J]) {
#pragma HLS INTERFACE m_axi port=A offset=slave bundle=gmem0 depth=1048576 max_read_burst_length=256
#pragma HLS INTERFACE m_axi port=B offset=slave bundle=gmem1 depth=1048576 max_read_burst_length=256
#pragma HLS INTERFACE m_axi port=C offset=slave bundle=gmem2 depth=1048576 max_read_burst_length=256 max_write_burst_length=256
#pragma HLS INTERFACE s_axilite port=A bundle=control
#pragma HLS INTERFACE s_axilite port=B bundle=control
#pragma HLS INTERFACE s_axilite port=C bundle=control
#pragma HLS INTERFACE s_axilite port=return bundle=control

    static data_t A_buf[TI][K];
#pragma HLS ARRAY_PARTITION variable=A_buf complete dim=1
    static data_t B_buf[TJ][K];
    static data_t C_loc[TI][TJ];
#pragma HLS ARRAY_PARTITION variable=C_loc complete dim=1

ILOOP:
    for (int i0 = 0; i0 < I; i0 += TI) {
        // Load block of A rows
    LOAD_A:
        for (int ti = 0; ti < TI; ti++) {
            for (int k = 0; k < K; k++) {
#pragma HLS PIPELINE II=1
                A_buf[ti][k] = A[i0 + ti][k];
            }
        }

    JLOOP:
        for (int j0 = 0; j0 < J; j0 += TJ) {
            // Load block of B rows
        LOAD_B:
            for (int tj = 0; tj < TJ; tj++) {
                for (int k = 0; k < K; k++) {
#pragma HLS PIPELINE II=1
                    B_buf[tj][k] = B[j0 + tj][k];
                }
            }

            // Load C tile
        LOAD_C:
            for (int ti = 0; ti < TI; ti++) {
                for (int tj = 0; tj < TJ; tj++) {
#pragma HLS PIPELINE II=1
                    C_loc[ti][tj] = C[i0 + ti][j0 + tj];
                }
            }

            // Compute: sequential k accumulation per (i,j), matching original order
        COMP_K:
            for (int k = 0; k < K; k++) {
            COMP_J:
                for (int tj = 0; tj < TJ; tj++) {
#pragma HLS PIPELINE II=1
#pragma HLS DEPENDENCE variable=C_loc inter false
                    data_t b = B_buf[tj][k];
                COMP_I:
                    for (int ti = 0; ti < TI; ti++) {
#pragma HLS UNROLL
                        C_loc[ti][tj] = C_loc[ti][tj] + A_buf[ti][k] * b;
                    }
                }
            }

            // Store C tile
        STORE_C:
            for (int ti = 0; ti < TI; ti++) {
                for (int tj = 0; tj < TJ; tj++) {
#pragma HLS PIPELINE II=1
                    C[i0 + ti][j0 + tj] = C_loc[ti][tj];
                }
            }
        }
    }
}