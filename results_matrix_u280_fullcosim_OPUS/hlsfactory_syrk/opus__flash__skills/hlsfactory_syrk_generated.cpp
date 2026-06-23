#include "syrk.h"


void kernel_syrk( 
		 double alpha,
		 double beta,
		 double C[ N + 0][N + 0],
		 double A[ N + 0][M + 0])
{
#pragma HLS INTERFACE m_axi port=C offset=slave bundle=gmem0
#pragma HLS INTERFACE m_axi port=A offset=slave bundle=gmem1
#pragma HLS INTERFACE s_axilite port=alpha bundle=control
#pragma HLS INTERFACE s_axilite port=beta bundle=control
#pragma HLS INTERFACE s_axilite port=C bundle=control
#pragma HLS INTERFACE s_axilite port=A bundle=control
#pragma HLS INTERFACE s_axilite port=return bundle=control

    const int n = N;
    const int m = M;

    // Local buffers to enable reuse and parallel access
    double A_local[N][M];
#pragma HLS ARRAY_PARTITION variable=A_local cyclic factor=8 dim=2
    double C_local[N][N];
#pragma HLS ARRAY_PARTITION variable=C_local cyclic factor=8 dim=2

    // Stage A into local memory
LOAD_A_I:
    for (int i = 0; i < n; i++) {
    LOAD_A_J:
        for (int j = 0; j < m; j++) {
#pragma HLS PIPELINE II=1
            A_local[i][j] = A[i][j];
        }
    }

    // Stage C into local memory
LOAD_C_I:
    for (int i = 0; i < n; i++) {
    LOAD_C_J:
        for (int j = 0; j < n; j++) {
#pragma HLS PIPELINE II=1
            C_local[i][j] = C[i][j];
        }
    }

    int i, j, k;
COMP_I:
    for (i = 0; i < n; i++) {
    BETA_J:
        for (j = 0; j <= i; j++) {
#pragma HLS PIPELINE II=1
            C_local[i][j] *= beta;
        }
    COMP_K:
        for (k = 0; k < m; k++) {
            double a_ik = A_local[i][k];
        COMP_J:
            for (j = 0; j <= i; j++) {
#pragma HLS PIPELINE II=1
                C_local[i][j] += alpha * a_ik * A_local[j][k];
            }
        }
    }

    // Write back C
STORE_C_I:
    for (int ii = 0; ii < n; ii++) {
    STORE_C_J:
        for (int jj = 0; jj < n; jj++) {
#pragma HLS PIPELINE II=1
            C[ii][jj] = C_local[ii][jj];
        }
    }
}