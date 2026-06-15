#include "gemver.h"

extern "C" {

void kernel_gemver(
		   double alpha,
		   double beta,
		   double A[ N + 0][N + 0],
		   double u1[ N + 0],
		   double v1[ N + 0],
		   double u2[ N + 0],
		   double v2[ N + 0],
		   double w[ N + 0],
		   double x[ N + 0],
		   double y[ N + 0],
		   double z[ N + 0])
{
#pragma HLS INTERFACE m_axi port=A    offset=slave bundle=gmem0
#pragma HLS INTERFACE m_axi port=u1   offset=slave bundle=gmem1
#pragma HLS INTERFACE m_axi port=v1   offset=slave bundle=gmem2
#pragma HLS INTERFACE m_axi port=u2   offset=slave bundle=gmem3
#pragma HLS INTERFACE m_axi port=v2   offset=slave bundle=gmem4
#pragma HLS INTERFACE m_axi port=w    offset=slave bundle=gmem5
#pragma HLS INTERFACE m_axi port=x    offset=slave bundle=gmem6
#pragma HLS INTERFACE m_axi port=y    offset=slave bundle=gmem7
#pragma HLS INTERFACE m_axi port=z    offset=slave bundle=gmem8

#pragma HLS INTERFACE s_axilite port=alpha  bundle=control
#pragma HLS INTERFACE s_axilite port=beta   bundle=control
#pragma HLS INTERFACE s_axilite port=A      bundle=control
#pragma HLS INTERFACE s_axilite port=u1     bundle=control
#pragma HLS INTERFACE s_axilite port=v1     bundle=control
#pragma HLS INTERFACE s_axilite port=u2     bundle=control
#pragma HLS INTERFACE s_axilite port=v2     bundle=control
#pragma HLS INTERFACE s_axilite port=w      bundle=control
#pragma HLS INTERFACE s_axilite port=x      bundle=control
#pragma HLS INTERFACE s_axilite port=y      bundle=control
#pragma HLS INTERFACE s_axilite port=z      bundle=control
#pragma HLS INTERFACE s_axilite port=return bundle=control

    const int n = N;

    // Local buffers for frequently reused 1-D vectors
    double l_u1[N], l_u2[N];
    double l_v1[N], l_v2[N];
    double l_y[N],  l_z[N];
    double l_x[N],  l_w[N];

#pragma HLS ARRAY_PARTITION variable=l_u1 complete dim=1
#pragma HLS ARRAY_PARTITION variable=l_u2 complete dim=1
#pragma HLS ARRAY_PARTITION variable=l_v1 complete dim=1
#pragma HLS ARRAY_PARTITION variable=l_v2 complete dim=1
#pragma HLS ARRAY_PARTITION variable=l_y  complete dim=1
#pragma HLS ARRAY_PARTITION variable=l_z  complete dim=1
#pragma HLS ARRAY_PARTITION variable=l_x  complete dim=1
#pragma HLS ARRAY_PARTITION variable=l_w  complete dim=1

    // Load 1-D vectors from global memory
    for (int i = 0; i < n; i++) {
#pragma HLS PIPELINE II=1
        l_u1[i] = u1[i];
        l_u2[i] = u2[i];
        l_v1[i] = v1[i];
        l_v2[i] = v2[i];
        l_y[i]  = y[i];
        l_z[i]  = z[i];
        l_x[i]  = x[i];
        l_w[i]  = w[i];
    }

    // -------------------------------------------------------
    // Loop 1: A[i][j] += u1[i]*v1[j] + u2[i]*v2[j]
    // -------------------------------------------------------
    for (int i = 0; i < n; i++) {
        double u1i = l_u1[i];
        double u2i = l_u2[i];
        for (int j = 0; j < n; j++) {
#pragma HLS PIPELINE II=1
            A[i][j] = A[i][j] + u1i * l_v1[j] + u2i * l_v2[j];
        }
    }

    // -------------------------------------------------------
    // Loop 2: x[i] += beta * A[j][i] * y[j]
    // Reads A column-wise; iterate row-major and accumulate
    // into a local x buffer to avoid repeated RMW on global mem.
    // -------------------------------------------------------
    for (int i = 0; i < n; i++) {
        double acc = 0.0;
        for (int j = 0; j < n; j++) {
#pragma HLS PIPELINE II=1
            acc += beta * A[j][i] * l_y[j];
        }
        l_x[i] += acc;
    }

    // -------------------------------------------------------
    // Loop 3: x[i] += z[i]
    // -------------------------------------------------------
    for (int i = 0; i < n; i++) {
#pragma HLS PIPELINE II=1
        l_x[i] = l_x[i] + l_z[i];
    }

    // -------------------------------------------------------
    // Loop 4: w[i] += alpha * A[i][j] * x[j]
    // -------------------------------------------------------
    for (int i = 0; i < n; i++) {
        double acc = 0.0;
        for (int j = 0; j < n; j++) {
#pragma HLS PIPELINE II=1
            acc += alpha * A[i][j] * l_x[j];
        }
        l_w[i] += acc;
    }

    // Write results back to global memory
    for (int i = 0; i < n; i++) {
#pragma HLS PIPELINE II=1
        x[i] = l_x[i];
        w[i] = l_w[i];
    }
}

} // extern "C"