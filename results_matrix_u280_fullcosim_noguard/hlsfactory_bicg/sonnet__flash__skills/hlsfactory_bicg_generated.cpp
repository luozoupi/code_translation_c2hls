#include "bicg.h"

extern "C" {

void kernel_bicg( 
         double A[ N + 0][M + 0],
         double s[ M + 0],
         double q[ N + 0],
         double p[ M + 0],
         double r[ N + 0])
{
    #pragma HLS INTERFACE m_axi port=A offset=slave bundle=gmem0
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

    // Local buffers for arrays accessed repeatedly in the loop nest
    double local_s[M];
    double local_q[N];
    double local_p[M];
    double local_r[N];
    double local_A[N][M];

    #pragma HLS ARRAY_PARTITION variable=local_s  cyclic factor=8 dim=1
    #pragma HLS ARRAY_PARTITION variable=local_q  cyclic factor=8 dim=1
    #pragma HLS ARRAY_PARTITION variable=local_p  cyclic factor=8 dim=1
    #pragma HLS ARRAY_PARTITION variable=local_r  cyclic factor=8 dim=1
    #pragma HLS ARRAY_PARTITION variable=local_A  cyclic factor=8 dim=2

    // Load inputs into local buffers
    load_r: for (int i = 0; i < N; i++) {
        #pragma HLS PIPELINE II=1
        local_r[i] = r[i];
    }
    load_p: for (int j = 0; j < M; j++) {
        #pragma HLS PIPELINE II=1
        local_p[j] = p[j];
    }
    load_A: for (int i = 0; i < N; i++) {
        for (int j = 0; j < M; j++) {
            #pragma HLS PIPELINE II=1
            local_A[i][j] = A[i][j];
        }
    }

    // Initialize s
    init_s: for (int i = 0; i < M; i++) {
        #pragma HLS PIPELINE II=1
        local_s[i] = 0.0;
    }

    // Main computation
    outer_loop: for (int i = 0; i < N; i++) {
        double q_i = 0.0;
        inner_loop: for (int j = 0; j < M; j++) {
            #pragma HLS PIPELINE II=1
            double a_ij = local_A[i][j];
            local_s[j] = local_s[j] + local_r[i] * a_ij;
            q_i = q_i + a_ij * local_p[j];
        }
        local_q[i] = q_i;
    }

    // Store outputs
    store_s: for (int j = 0; j < M; j++) {
        #pragma HLS PIPELINE II=1
        s[j] = local_s[j];
    }
    store_q: for (int i = 0; i < N; i++) {
        #pragma HLS PIPELINE II=1
        q[i] = local_q[i];
    }
}

} // extern "C"