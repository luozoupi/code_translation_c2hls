#include "covariance.h"

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

    // Local copies for efficient on-chip access
    double local_data[N][M];
    double local_cov[M][M];
    double local_mean[M];

#pragma HLS ARRAY_PARTITION variable=local_data cyclic factor=8 dim=2
#pragma HLS ARRAY_PARTITION variable=local_cov  cyclic factor=8 dim=2
#pragma HLS ARRAY_PARTITION variable=local_mean complete dim=1

    // Load data from global memory
    load_data_i: for (int i = 0; i < n; i++) {
        load_data_j: for (int j = 0; j < m; j++) {
#pragma HLS PIPELINE II=1
            local_data[i][j] = data[i][j];
        }
    }

    int i, j, k;

    // Compute mean for each column
    mean_j: for (j = 0; j < m; j++) {
        local_mean[j] = 0.0;
        mean_i: for (i = 0; i < n; i++) {
#pragma HLS PIPELINE II=1
            local_mean[j] += local_data[i][j];
        }
        local_mean[j] /= float_n;
    }

    // Subtract mean from data
    sub_i: for (i = 0; i < n; i++) {
        sub_j: for (j = 0; j < m; j++) {
#pragma HLS PIPELINE II=1
            local_data[i][j] -= local_mean[j];
        }
    }

    // Compute covariance matrix
    cov_i: for (i = 0; i < m; i++) {
        cov_j: for (j = i; j < m; j++) {
            local_cov[i][j] = 0.0;
            cov_k: for (k = 0; k < n; k++) {
#pragma HLS PIPELINE II=1
                local_cov[i][j] += local_data[k][i] * local_data[k][j];
            }
            local_cov[i][j] /= (float_n - 1.0);
            local_cov[j][i] = local_cov[i][j];
        }
    }

    // Write mean back to global memory
    write_mean: for (j = 0; j < m; j++) {
#pragma HLS PIPELINE II=1
        mean[j] = local_mean[j];
    }

    // Write cov back to global memory
    write_cov_i: for (i = 0; i < m; i++) {
        write_cov_j: for (j = 0; j < m; j++) {
#pragma HLS PIPELINE II=1
            cov[i][j] = local_cov[i][j];
        }
    }

    // Write data back to global memory
    write_data_i: for (i = 0; i < n; i++) {
        write_data_j: for (j = 0; j < m; j++) {
#pragma HLS PIPELINE II=1
            data[i][j] = local_data[i][j];
        }
    }
}