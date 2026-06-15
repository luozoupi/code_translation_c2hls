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
#pragma HLS INTERFACE m_axi port=A   offset=slave bundle=gmem0
#pragma HLS INTERFACE m_axi port=u1  offset=slave bundle=gmem1
#pragma HLS INTERFACE m_axi port=v1  offset=slave bundle=gmem2
#pragma HLS INTERFACE m_axi port=u2  offset=slave bundle=gmem3
#pragma HLS INTERFACE m_axi port=v2  offset=slave bundle=gmem4
#pragma HLS INTERFACE m_axi port=w   offset=slave bundle=gmem5
#pragma HLS INTERFACE m_axi port=x   offset=slave bundle=gmem6
#pragma HLS INTERFACE m_axi port=y   offset=slave bundle=gmem7
#pragma HLS INTERFACE m_axi port=z   offset=slave bundle=gmem8

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

    // Local caches for arrays accessed repeatedly
    double l_A[N][N];
    double l_u1[N], l_v1[N], l_u2[N], l_v2[N];
    double l_w[N], l_x[N], l_y[N], l_z[N];

#pragma HLS ARRAY_PARTITION variable=l_A   cyclic factor=8 dim=2
#pragma HLS ARRAY_PARTITION variable=l_u1  cyclic factor=8 dim=1
#pragma HLS ARRAY_PARTITION variable=l_v1  cyclic factor=8 dim=1
#pragma HLS ARRAY_PARTITION variable=l_u2  cyclic factor=8 dim=1
#pragma HLS ARRAY_PARTITION variable=l_v2  cyclic factor=8 dim=1
#pragma HLS ARRAY_PARTITION variable=l_w   cyclic factor=8 dim=1
#pragma HLS ARRAY_PARTITION variable=l_x   cyclic factor=8 dim=1
#pragma HLS ARRAY_PARTITION variable=l_y   cyclic factor=8 dim=1
#pragma HLS ARRAY_PARTITION variable=l_z   cyclic factor=8 dim=1

    const int n = N;
    int i, j;

    // Load inputs into local arrays
    load_u1: for (i = 0; i < n; i++) {
#pragma HLS PIPELINE II=1
        l_u1[i] = u1[i];
        l_v1[i] = v1[i];
        l_u2[i] = u2[i];
        l_v2[i] = v2[i];
        l_x[i]  = x[i];
        l_y[i]  = y[i];
        l_z[i]  = z[i];
        l_w[i]  = w[i];
    }

    load_A: for (i = 0; i < n; i++) {
        for (j = 0; j < n; j++) {
#pragma HLS PIPELINE II=1
            l_A[i][j] = A[i][j];
        }
    }

    // Loop 1: A[i][j] += u1[i]*v1[j] + u2[i]*v2[j]
    loop1_i: for (i = 0; i < n; i++) {
        loop1_j: for (j = 0; j < n; j++) {
#pragma HLS PIPELINE II=1
            l_A[i][j] = l_A[i][j] + l_u1[i] * l_v1[j] + l_u2[i] * l_v2[j];
        }
    }

    // Loop 2: x[i] += beta * A[j][i] * y[j]
    loop2_i: for (i = 0; i < n; i++) {
        double sum = l_x[i];
        loop2_j: for (j = 0; j < n; j++) {
#pragma HLS PIPELINE II=1
            sum += beta * l_A[j][i] * l_y[j];
        }
        l_x[i] = sum;
    }

    // Loop 3: x[i] += z[i]
    loop3: for (i = 0; i < n; i++) {
#pragma HLS PIPELINE II=1
        l_x[i] = l_x[i] + l_z[i];
    }

    // Loop 4: w[i] += alpha * A[i][j] * x[j]
    loop4_i: for (i = 0; i < n; i++) {
        double sum = l_w[i];
        loop4_j: for (j = 0; j < n; j++) {
#pragma HLS PIPELINE II=1
            sum += alpha * l_A[i][j] * l_x[j];
        }
        l_w[i] = sum;
    }

    // Write back results
    store_A: for (i = 0; i < n; i++) {
        for (j = 0; j < n; j++) {
#pragma HLS PIPELINE II=1
            A[i][j] = l_A[i][j];
        }
    }

    store_out: for (i = 0; i < n; i++) {
#pragma HLS PIPELINE II=1
        x[i] = l_x[i];
        w[i] = l_w[i];
    }
}

} // extern "C"