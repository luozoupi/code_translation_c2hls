#include "gesummv.h"
#include <cstring>

#define TILE 256

static void load_row(double A[N + 0][N + 0], double B[N + 0][N + 0],
                     double A_local_1[N], double B_local_1[N],
                     double A_local_2[N], double B_local_2[N],
                     int i, int n, int flag)
{
    if (flag == 0) {
        memcpy(A_local_1, A[i], n * sizeof(double));
        memcpy(B_local_1, B[i], n * sizeof(double));
    } else {
        memcpy(A_local_2, A[i], n * sizeof(double));
        memcpy(B_local_2, B[i], n * sizeof(double));
    }
}

static void compute_row(double A_local_1[N], double B_local_1[N],
                        double A_local_2[N], double B_local_2[N],
                        double x_local[N],
                        double alpha, double beta,
                        double tmp[N + 0], double y[N + 0],
                        int i, int n, int flag)
{
    double tmp_acc = 0.0;
    double y_acc = 0.0;

    for (int j = 0; j < n; j++)
    {
#pragma HLS PIPELINE II=1
#pragma HLS UNROLL factor=2
#pragma HLS LOOP_TRIPCOUNT min=N max=N
#pragma HLS DEPENDENCE variable=tmp_acc inter false
#pragma HLS DEPENDENCE variable=y_acc inter false
        double a_val = (flag == 0) ? A_local_1[j] : A_local_2[j];
        double b_val = (flag == 0) ? B_local_1[j] : B_local_2[j];
        tmp_acc = a_val * x_local[j] + tmp_acc;
        y_acc = b_val * x_local[j] + y_acc;
    }

    tmp[i] = tmp_acc;
    y[i] = alpha * tmp_acc + beta * y_acc;
}

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

    int i;

    // Stage the reused vector x[] into a local buffer to enable
    // partitioned parallel access in the unrolled inner loop.
    double x_local[N];
#pragma HLS ARRAY_PARTITION variable=x_local cyclic factor=20 dim=1

    // --- LOAD x phase ---
    for (int k = 0; k < n; k++) {
#pragma HLS PIPELINE II=1
#pragma HLS LOOP_TRIPCOUNT min=N max=N
        x_local[k] = x[k];
    }

    // Double-buffered tile buffers for one row of A and B.
    double A_local_1[N];
    double B_local_1[N];
    double A_local_2[N];
    double B_local_2[N];
#pragma HLS ARRAY_PARTITION variable=A_local_1 cyclic factor=20 dim=1
#pragma HLS ARRAY_PARTITION variable=B_local_1 cyclic factor=20 dim=1
#pragma HLS ARRAY_PARTITION variable=A_local_2 cyclic factor=20 dim=1
#pragma HLS ARRAY_PARTITION variable=B_local_2 cyclic factor=20 dim=1

    // --- Prologue: load row 0 into buffer set 0 ---
    load_row(A, B, A_local_1, B_local_1, A_local_2, B_local_2, 0, n, 0);

    // --- Main loop: overlap load of row i+1 with compute of row i ---
    for (i = 0; i < n; i++)
    {
#pragma HLS LOOP_TRIPCOUNT min=N max=N
        int flag = i % 2;

        // Load next row (i+1) into the opposite buffer set while
        // computing the current row from the current buffer set.
        if (i + 1 < n) {
            load_row(A, B, A_local_1, B_local_1, A_local_2, B_local_2,
                     i + 1, n, (i + 1) % 2);
        }

        compute_row(A_local_1, B_local_1, A_local_2, B_local_2,
                    x_local, alpha, beta, tmp, y, i, n, flag);
    }

}