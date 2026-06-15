#include "covariance.h"

extern "C" {

void kernel_covariance(
        double float_n,
        double data[N + 0][M + 0],
        double cov[M + 0][M + 0],
        double mean[M + 0])
{
#pragma HLS INTERFACE m_axi port=data   offset=slave bundle=gmem
#pragma HLS INTERFACE m_axi port=cov    offset=slave bundle=gmem
#pragma HLS INTERFACE m_axi port=mean   offset=slave bundle=gmem
#pragma HLS INTERFACE s_axilite port=float_n bundle=control
#pragma HLS INTERFACE s_axilite port=data    bundle=control
#pragma HLS INTERFACE s_axilite port=cov     bundle=control
#pragma HLS INTERFACE s_axilite port=mean    bundle=control
#pragma HLS INTERFACE s_axilite port=return  bundle=control

    const int n = N;
    const int m = M;

    // Local buffers for data and mean to enable partitioning
    double l_data[N][M];
    double l_mean[M];
    double l_cov[M][M];

#pragma HLS ARRAY_PARTITION variable=l_mean  complete  dim=1
#pragma HLS ARRAY_PARTITION variable=l_data  cyclic    factor=8 dim=2
#pragma HLS ARRAY_PARTITION variable=l_cov   cyclic    factor=8 dim=2

    // Load data from global memory
    for (int i = 0; i < n; i++) {
#pragma HLS PIPELINE II=1
        for (int j = 0; j < m; j++) {
            l_data[i][j] = data[i][j];
        }
    }

    // Compute mean for each column
    for (int j = 0; j < m; j++) {
        l_mean[j] = 0.0;
        for (int i = 0; i < n; i++) {
#pragma HLS PIPELINE II=1
            l_mean[j] += l_data[i][j];
        }
        l_mean[j] /= float_n;
    }

    // Store mean back to global memory
    for (int j = 0; j < m; j++) {
#pragma HLS PIPELINE II=1
        mean[j] = l_mean[j];
    }

    // Center the data: subtract mean from each element
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < m; j++) {
#pragma HLS PIPELINE II=1
            l_data[i][j] -= l_mean[j];
        }
    }

    // Compute covariance matrix
    for (int i = 0; i < m; i++) {
        for (int j = i; j < m; j++) {
            double sum = 0.0;
            for (int k = 0; k < n; k++) {
#pragma HLS PIPELINE II=1
                sum += l_data[k][i] * l_data[k][j];
            }
            l_cov[i][j] = sum / (float_n - 1.0);
            l_cov[j][i] = l_cov[i][j];
        }
    }

    // Store covariance matrix back to global memory
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < m; j++) {
#pragma HLS PIPELINE II=1
            cov[i][j] = l_cov[i][j];
        }
    }
}

} // extern "C"