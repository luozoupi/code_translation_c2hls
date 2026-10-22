#include "kernel.h"

#define TI 16   // rows of A buffered on chip per block
#define KU 16   // k-elements reduced per pipelined iteration

static float dot_ku(const data_t a[KU], const data_t b[KU]) {
#pragma HLS INLINE
    data_t p[KU];
#pragma HLS ARRAY_PARTITION variable=p complete
    for (int u = 0; u < KU; u++) {
#pragma HLS UNROLL
        p[u] = a[u] * b[u];
    }
    // balanced adder tree
    for (int s = KU / 2; s > 0; s >>= 1) {
#pragma HLS UNROLL
        for (int u = 0; u < s; u++) {
#pragma HLS UNROLL
            p[u] = p[u] + p[u + s];
        }
    }
    return p[0];
}

extern "C" void autosa_mm(data_t A[I][K], data_t B[J][K], data_t C[I][J]) {
#pragma HLS INTERFACE m_axi port=A offset=slave bundle=gmem0 depth=1048576 max_read_burst_length=256
#pragma HLS INTERFACE m_axi port=B offset=slave bundle=gmem1 depth=1048576 max_read_burst_length=256
#pragma HLS INTERFACE m_axi port=C offset=slave bundle=gmem2 depth=1048576
#pragma HLS INTERFACE s_axilite port=A bundle=control
#pragma HLS INTERFACE s_axilite port=B bundle=control
#pragma HLS INTERFACE s_axilite port=C bundle=control
#pragma HLS INTERFACE s_axilite port=return bundle=control

    data_t a_buf[TI][K];
#pragma HLS ARRAY_PARTITION variable=a_buf cyclic factor=KU dim=2
    data_t b_buf[K];
#pragma HLS ARRAY_PARTITION variable=b_buf cyclic factor=KU dim=1
    data_t acc[TI];

I_BLK:
    for (int ib = 0; ib < I; ib += TI) {
        // Load a block of TI rows of A
    LOAD_A:
        for (int ti = 0; ti < TI; ti++) {
            for (int k = 0; k < K; k++) {
#pragma HLS PIPELINE II=1
                a_buf[ti][k] = A[ib + ti][k];
            }
        }

    J_LOOP:
        for (int j = 0; j < J; j++) {
            // Load row j of B
        LOAD_B:
            for (int k = 0; k < K; k++) {
#pragma HLS PIPELINE II=1
                b_buf[k] = B[j][k];
            }

            // Initialize accumulators with existing C values
        INIT_ACC:
            for (int ti = 0; ti < TI; ti++) {
#pragma HLS PIPELINE II=1
                acc[ti] = C[ib + ti][j];
            }

            // Compute: each iteration does a KU-wide dot product for one row.
            // The accumulator for a row is revisited every TI iterations,
            // which covers the floating-point adder latency.
        COMPUTE_K:
            for (int kk = 0; kk < K / KU; kk++) {
            COMPUTE_TI:
                for (int ti = 0; ti < TI; ti++) {
#pragma HLS PIPELINE II=1
#pragma HLS DEPENDENCE variable=acc inter false
                    data_t av[KU], bv[KU];
#pragma HLS ARRAY_PARTITION variable=av complete
#pragma HLS ARRAY_PARTITION variable=bv complete
                    for (int u = 0; u < KU; u++) {
#pragma HLS UNROLL
                        av[u] = a_buf[ti][kk * KU + u];
                        bv[u] = b_buf[kk * KU + u];
                    }
                    acc[ti] += dot_ku(av, bv);
                }
            }

            // Write results back
        STORE_C:
            for (int ti = 0; ti < TI; ti++) {
#pragma HLS PIPELINE II=1
                C[ib + ti][j] = acc[ti];
            }
        }
    }
}