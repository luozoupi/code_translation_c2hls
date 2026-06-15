#include "bicg.h"

extern "C" {

void kernel_bicg( 
		 double A[ N + 0][M + 0],
		 double s[ M + 0],
		 double q[ N + 0],
		 double p[ M + 0],
		 double r[ N + 0])
{
    #pragma HLS INTERFACE m_axi port=A offset=slave bundle=gmem
    #pragma HLS INTERFACE m_axi port=s offset=slave bundle=gmem1
    #pragma HLS INTERFACE m_axi port=q offset=slave bundle=gmem2
    #pragma HLS INTERFACE m_axi port=p offset=slave bundle=gmem3
    #pragma HLS INTERFACE m_axi port=r offset=slave bundle=gmem4
    #pragma HLS INTERFACE s_axilite port=A bundle=control
    #pragma HLS INTERFACE s_axilite port=s bundle=control
    #pragma HLS INTERFACE s_axilite port=q bundle=control
    #pragma HLS INTERFACE s_axilite port=p bundle=control
    #pragma HLS INTERFACE s_axilite port=r bundle=control
    #pragma HLS INTERFACE s_axilite port=return bundle=control

    // Local buffers for parallel access
    double s_local[M];
    double q_local[N];
    double p_local[M];
    double r_local[N];
    double A_local[N][M];

    #pragma HLS ARRAY_PARTITION variable=s_local cyclic factor=8 dim=1
    #pragma HLS ARRAY_PARTITION variable=q_local cyclic factor=8 dim=1
    #pragma HLS ARRAY_PARTITION variable=p_local cyclic factor=8 dim=1
    #pragma HLS ARRAY_PARTITION variable=r_local cyclic factor=8 dim=1
    #pragma HLS ARRAY_PARTITION variable=A_local cyclic factor=8 dim=2

    const int n = N;
    const int m = M;

    int i, j;

    // Load inputs into local buffers
    load_r: for (i = 0; i < n; i++) {
        #pragma HLS PIPELINE II=1
        r_local[i] = r[i];
    }
    load_p: for (j = 0; j < m; j++) {
        #pragma HLS PIPELINE II=1
        p_local[j] = p[j];
    }
    load_A: for (i = 0; i < n; i++) {
        for (j = 0; j < m; j++) {
            #pragma HLS PIPELINE II=1
            A_local[i][j] = A[i][j];
        }
    }

    // Initialize s_local
    init_s: for (i = 0; i < m; i++) {
        #pragma HLS PIPELINE II=1
        s_local[i] = 0.0;
    }

    // Initialize q_local
    init_q: for (i = 0; i < n; i++) {
        #pragma HLS PIPELINE II=1
        q_local[i] = 0.0;
    }

    // Main computation
    for (i = 0; i < n; i++) {
        double q_acc = 0.0;
        double r_val = r_local[i];
        for (j = 0; j < m; j++) {
            #pragma HLS PIPELINE II=1
            double a_val = A_local[i][j];
            s_local[j] = s_local[j] + r_val * a_val;
            q_acc = q_acc + a_val * p_local[j];
        }
        q_local[i] = q_acc;
    }

    // Write outputs back
    write_s: for (j = 0; j < m; j++) {
        #pragma HLS PIPELINE II=1
        s[j] = s_local[j];
    }
    write_q: for (i = 0; i < n; i++) {
        #pragma HLS PIPELINE II=1
        q[i] = q_local[i];
    }
}

} // extern "C"