#include "gesummv.h"
#include <cstring>

#define TILE 256

void kernel_gesummv(
		    double alpha,
		    double beta,
		    double A[ N + 0][N + 0],
		    double B[ N + 0][N + 0],
		    double tmp[ N + 0],
		    double x[ N + 0],
		    double y[ N + 0])
{
#pragma HLS INTERFACE m_axi port=A offset=slave bundle=gmem0
#pragma HLS INTERFACE m_axi port=B offset=slave bundle=gmem1
#pragma HLS INTERFACE m_axi port=tmp offset=slave bundle=gmem2
#pragma HLS INTERFACE m_axi port=x offset=slave bundle=gmem3
#pragma HLS INTERFACE m_axi port=y offset=slave bundle=gmem2

#pragma HLS INTERFACE s_axilite port=alpha bundle=control
#pragma HLS INTERFACE s_axilite port=beta bundle=control
#pragma HLS INTERFACE s_axilite port=A bundle=control
#pragma HLS INTERFACE s_axilite port=B bundle=control
#pragma HLS INTERFACE s_axilite port=tmp bundle=control
#pragma HLS INTERFACE s_axilite port=x bundle=control
#pragma HLS INTERFACE s_axilite port=y bundle=control
#pragma HLS INTERFACE s_axilite port=return bundle=control

    const int n = N;

    int i, j;

    // Stage the reused vector x[] into a local buffer to enable
    // partitioned parallel access in the unrolled inner loop.
    double x_local[N];
#pragma HLS ARRAY_PARTITION variable=x_local cyclic factor=10 dim=1

    // --- LOAD x phase ---
    for (int k = 0; k < n; k++) {
#pragma HLS PIPELINE II=1
        x_local[k] = x[k];
    }

    // Local tile buffers for one row of A and B.
    double A_local[N];
    double B_local[N];

    for (i = 0; i < n; i++)
    {
        // --- LOAD phase: stage row i of A and B into local buffers ---
        memcpy(A_local, A[i], n * sizeof(double));
        memcpy(B_local, B[i], n * sizeof(double));

        // --- COMPUTE phase: operate only on local buffers ---
        double tmp_acc = 0.0;
        double y_acc = 0.0;

        for (j = 0; j < n; j++)
        {
#pragma HLS PIPELINE II=1
            tmp_acc = A_local[j] * x_local[j] + tmp_acc;
            y_acc = B_local[j] * x_local[j] + y_acc;
        }

        // --- STORE phase: write results back to global memory ---
        tmp[i] = tmp_acc;
        y[i] = alpha * tmp_acc + beta * y_acc;
    }

}