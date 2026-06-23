#include "jacobi-1d.h"
#include <cstring>

void kernel_jacobi_1d(

			    double A[ N + 0],
			    double B[ N + 0])
{


    const int n = N;
    const int tsteps = TSTEPS;

    int t, i;

    // Local tile buffers (entire array fits in local memory since N is bounded)
    double A_local[N];
    double B_local[N];
#pragma HLS ARRAY_PARTITION variable=A_local cyclic factor=4 dim=1
#pragma HLS ARRAY_PARTITION variable=B_local cyclic factor=4 dim=1

    // ---- LOAD phase ----
load_A:
    for (i = 0; i < n; i++) {
#pragma HLS PIPELINE II=1
        A_local[i] = A[i];
    }
load_B:
    for (i = 0; i < n; i++) {
#pragma HLS PIPELINE II=1
        B_local[i] = B[i];
    }

    // ---- COMPUTE phase (operates on local buffers) ----
compute_t:
    for (t = 0; t < tsteps; t++)
    {
#pragma HLS LOOP_TRIPCOUNT min=TSTEPS max=TSTEPS
    update_B:
        for (i = 1; i < n - 1; i++) {
#pragma HLS PIPELINE II=1
#pragma HLS UNROLL factor=4
#pragma HLS DEPENDENCE variable=B_local inter false
            B_local[i] = 0.33333 * (A_local[i-1] + A_local[i] + A_local[i + 1]);
        }
    update_A:
        for (i = 1; i < n - 1; i++) {
#pragma HLS PIPELINE II=1
#pragma HLS UNROLL factor=4
#pragma HLS DEPENDENCE variable=A_local inter false
            A_local[i] = 0.33333 * (B_local[i-1] + B_local[i] + B_local[i + 1]);
        }
    }

    // ---- STORE phase ----
store_A:
    for (i = 0; i < n; i++) {
#pragma HLS PIPELINE II=1
        A[i] = A_local[i];
    }
store_B:
    for (i = 0; i < n; i++) {
#pragma HLS PIPELINE II=1
        B[i] = B_local[i];
    }

}

extern "C" {
void workload(
			    double A[ N + 0],
			    double B[ N + 0])
{
#pragma HLS INTERFACE m_axi port=A offset=slave bundle=gmem0
#pragma HLS INTERFACE m_axi port=B offset=slave bundle=gmem1
#pragma HLS INTERFACE s_axilite port=A bundle=control
#pragma HLS INTERFACE s_axilite port=B bundle=control
#pragma HLS INTERFACE s_axilite port=return bundle=control

    kernel_jacobi_1d(A, B);
}
}