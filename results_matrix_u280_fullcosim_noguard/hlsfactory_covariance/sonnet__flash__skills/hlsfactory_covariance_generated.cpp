#include "covariance.h"

extern "C" {

void kernel_covariance(
		       double float_n,
		       double data[ N + 0][M + 0],
		       double cov[ M + 0][M + 0],
		       double mean[ M + 0])
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

    // Local buffers for data and mean to allow partitioning and fast access
    double local_data[N][M];
    double local_mean[M];
    double local_cov[M][M];

#pragma HLS ARRAY_PARTITION variable=local_data cyclic factor=8 dim=2
#pragma HLS ARRAY_PARTITION variable=local_mean complete dim=1
#pragma HLS ARRAY_PARTITION variable=local_cov  cyclic factor=8 dim=2

    int i, j, k;

    // Load data from global memory
    for (i = 0; i < n; i++) {
        for (j = 0; j < m; j++) {
#pragma HLS PIPELINE II=1
            local_data[i][j] = data[i][j];
        }
    }

    // Compute mean for each column
    for (j = 0; j < m; j++)
    {
        local_mean[j] = 0.0;
        for (i = 0; i < n; i++) {
#pragma HLS PIPELINE II=1
            local_mean[j] += local_data[i][j];
        }
        local_mean[j] /= float_n;
    }

    // Subtract mean from each element
    for (i = 0; i < n; i++) {
        for (j = 0; j < m; j++) {
#pragma HLS PIPELINE II=1
            local_data[i][j] -= local_mean[j];
        }
    }

    // Compute covariance matrix
    for (i = 0; i < m; i++) {
        for (j = i; j < m; j++) {
            double cov_ij = 0.0;
            for (k = 0; k < n; k++) {
#pragma HLS PIPELINE II=1
                cov_ij += local_data[k][i] * local_data[k][j];
            }
            cov_ij /= (float_n - 1.0);
            local_cov[i][j] = cov_ij;
            local_cov[j][i] = cov_ij;
        }
    }

    // Write mean to global memory
    for (j = 0; j < m; j++) {
#pragma HLS PIPELINE II=1
        mean[j] = local_mean[j];
    }

    // Write cov to global memory
    for (i = 0; i < m; i++) {
        for (j = 0; j < m; j++) {
#pragma HLS PIPELINE II=1
            cov[i][j] = local_cov[i][j];
        }
    }

}

} // extern "C"