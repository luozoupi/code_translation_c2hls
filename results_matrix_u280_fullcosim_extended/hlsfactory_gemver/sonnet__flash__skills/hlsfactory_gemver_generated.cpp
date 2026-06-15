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

    const int n = N;

    int i, j;

    // Local caches for 1D arrays to reduce global memory traffic
    double u1_local[N];
    double v1_local[N];
    double u2_local[N];
    double v2_local[N];
    double x_local[N];
    double y_local[N];
    double z_local[N];
    double w_local[N];

#pragma HLS ARRAY_PARTITION variable=u1_local complete dim=1
#pragma HLS ARRAY_PARTITION variable=v1_local complete dim=1
#pragma HLS ARRAY_PARTITION variable=u2_local complete dim=1
#pragma HLS ARRAY_PARTITION variable=v2_local complete dim=1
#pragma HLS ARRAY_PARTITION variable=x_local  complete dim=1
#pragma HLS ARRAY_PARTITION variable=y_local  complete dim=1
#pragma HLS ARRAY_PARTITION variable=z_local  complete dim=1
#pragma HLS ARRAY_PARTITION variable=w_local  complete dim=1

    // Load 1D arrays from global memory
    load_u1: for (i = 0; i < n; i++) {
#pragma HLS PIPELINE II=1
        u1_local[i] = u1[i];
    }
    load_v1: for (i = 0; i < n; i++) {
#pragma HLS PIPELINE II=1
        v1_local[i] = v1[i];
    }
    load_u2: for (i = 0; i < n; i++) {
#pragma HLS PIPELINE II=1
        u2_local[i] = u2[i];
    }
    load_v2: for (i = 0; i < n; i++) {
#pragma HLS PIPELINE II=1
        v2_local[i] = v2[i];
    }
    load_x: for (i = 0; i < n; i++) {
#pragma HLS PIPELINE II=1
        x_local[i] = x[i];
    }
    load_y: for (i = 0; i < n; i++) {
#pragma HLS PIPELINE II=1
        y_local[i] = y[i];
    }
    load_z: for (i = 0; i < n; i++) {
#pragma HLS PIPELINE II=1
        z_local[i] = z[i];
    }
    load_w: for (i = 0; i < n; i++) {
#pragma HLS PIPELINE II=1
        w_local[i] = w[i];
    }

    // Loop 1: A[i][j] += u1[i]*v1[j] + u2[i]*v2[j]
    // No reduction — each element independent, pipeline inner loop
    loop1_i: for (i = 0; i < n; i++) {
        loop1_j: for (j = 0; j < n; j++) {
#pragma HLS PIPELINE II=1
            A[i][j] = A[i][j] + u1_local[i] * v1_local[j] + u2_local[i] * v2_local[j];
        }
    }

    // Loop 2: x[i] += beta * A[j][i] * y[j]  (serial FP reduction over j)
    // Must keep reduction serial — no unroll on j
    loop2_i: for (i = 0; i < n; i++) {
        loop2_j: for (j = 0; j < n; j++) {
#pragma HLS PIPELINE II=1
            x_local[i] = x_local[i] + beta * A[j][i] * y_local[j];
        }
    }

    // Loop 3: x[i] += z[i]  (independent per i)
    loop3: for (i = 0; i < n; i++) {
#pragma HLS PIPELINE II=1
        x_local[i] = x_local[i] + z_local[i];
    }

    // Loop 4: w[i] += alpha * A[i][j] * x[j]  (serial FP reduction over j)
    // Must keep reduction serial — no unroll on j
    loop4_i: for (i = 0; i < n; i++) {
        loop4_j: for (j = 0; j < n; j++) {
#pragma HLS PIPELINE II=1
            w_local[i] = w_local[i] + alpha * A[i][j] * x_local[j];
        }
    }

    // Store results back to global memory
    store_x: for (i = 0; i < n; i++) {
#pragma HLS PIPELINE II=1
        x[i] = x_local[i];
    }
    store_w: for (i = 0; i < n; i++) {
#pragma HLS PIPELINE II=1
        w[i] = w_local[i];
    }
}

} // extern "C"