#include "correlation.h"

extern "C" {

void kernel_correlation(
            double float_n,
            double data[N + 0][M + 0],
            double corr[M + 0][M + 0],
            double mean[M + 0],
            double stddev[M + 0])
{
    #pragma HLS INTERFACE s_axilite port=return bundle=control
    #pragma HLS INTERFACE s_axilite port=float_n bundle=control
    #pragma HLS INTERFACE m_axi port=data   offset=slave bundle=gmem
    #pragma HLS INTERFACE m_axi port=corr   offset=slave bundle=gmem
    #pragma HLS INTERFACE m_axi port=mean   offset=slave bundle=gmem
    #pragma HLS INTERFACE m_axi port=stddev offset=slave bundle=gmem
    #pragma HLS INTERFACE s_axilite port=data   bundle=control
    #pragma HLS INTERFACE s_axilite port=corr   bundle=control
    #pragma HLS INTERFACE s_axilite port=mean   bundle=control
    #pragma HLS INTERFACE s_axilite port=stddev bundle=control

    // Local buffers for intermediate values to enable parallel access
    double local_mean[M];
    double local_stddev[M];
    double local_data[N][M];

    #pragma HLS ARRAY_PARTITION variable=local_mean   complete dim=1
    #pragma HLS ARRAY_PARTITION variable=local_stddev complete dim=1
    #pragma HLS ARRAY_PARTITION variable=local_data   cyclic factor=8 dim=2

    const int n = N;
    const int m = M;

    double eps = 0.1;

    // Load data into local buffer
    for (int i = 0; i < n; i++) {
        #pragma HLS PIPELINE II=1
        for (int j = 0; j < m; j++) {
            local_data[i][j] = data[i][j];
        }
    }

    // Compute mean for each column
    for (int j = 0; j < m; j++) {
        local_mean[j] = 0.0;
        for (int i = 0; i < n; i++) {
            #pragma HLS PIPELINE II=1
            local_mean[j] += local_data[i][j];
        }
        local_mean[j] /= float_n;
    }

    // Compute stddev for each column
    for (int j = 0; j < m; j++) {
        local_stddev[j] = 0.0;
        for (int i = 0; i < n; i++) {
            #pragma HLS PIPELINE II=1
            local_stddev[j] += (local_data[i][j] - local_mean[j]) *
                               (local_data[i][j] - local_mean[j]);
        }
        local_stddev[j] /= float_n;
        local_stddev[j] = sqrt(local_stddev[j]);
        local_stddev[j] = local_stddev[j] <= eps ? 1.0 : local_stddev[j];
    }

    // Normalize data
    double sqrt_float_n = sqrt(float_n);
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < m; j++) {
            #pragma HLS PIPELINE II=1
            local_data[i][j] -= local_mean[j];
            local_data[i][j] /= sqrt_float_n * local_stddev[j];
        }
    }

    // Compute correlation matrix
    for (int i = 0; i < m - 1; i++) {
        corr[i][i] = 1.0;
        for (int j = i + 1; j < m; j++) {
            double acc = 0.0;
            for (int k = 0; k < n; k++) {
                #pragma HLS PIPELINE II=1
                acc += local_data[k][i] * local_data[k][j];
            }
            corr[i][j] = acc;
            corr[j][i] = acc;
        }
    }
    corr[m - 1][m - 1] = 1.0;

    // Write back mean and stddev
    for (int j = 0; j < m; j++) {
        #pragma HLS PIPELINE II=1
        mean[j]   = local_mean[j];
        stddev[j] = local_stddev[j];
    }
}

} // extern "C"