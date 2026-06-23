#include "correlation.h"

extern "C" {

void kernel_correlation(
            double float_n,
            double data[ N + 0][M + 0],
            double corr[ M + 0][M + 0],
            double mean[ M + 0],
            double stddev[ M + 0])
{
#pragma HLS INTERFACE m_axi port=data   offset=slave bundle=gmem
#pragma HLS INTERFACE m_axi port=corr   offset=slave bundle=gmem1
#pragma HLS INTERFACE m_axi port=mean   offset=slave bundle=gmem2
#pragma HLS INTERFACE m_axi port=stddev offset=slave bundle=gmem3
#pragma HLS INTERFACE s_axilite port=float_n bundle=control
#pragma HLS INTERFACE s_axilite port=data    bundle=control
#pragma HLS INTERFACE s_axilite port=corr    bundle=control
#pragma HLS INTERFACE s_axilite port=mean    bundle=control
#pragma HLS INTERFACE s_axilite port=stddev  bundle=control
#pragma HLS INTERFACE s_axilite port=return  bundle=control

    // Local copies of data, mean, stddev, corr for efficient access
    double l_data[N][M];
    double l_mean[M];
    double l_stddev[M];
    double l_corr[M][M];

#pragma HLS ARRAY_PARTITION variable=l_data   cyclic factor=8 dim=2
#pragma HLS ARRAY_PARTITION variable=l_mean   cyclic factor=8 dim=1
#pragma HLS ARRAY_PARTITION variable=l_stddev cyclic factor=8 dim=1
#pragma HLS ARRAY_PARTITION variable=l_corr   cyclic factor=8 dim=2

    const int n = N;
    const int m = M;
    int i, j, k;
    double eps = 0.1;

    // Load data from global memory
    load_data_i: for (i = 0; i < n; i++) {
        load_data_j: for (j = 0; j < m; j++) {
#pragma HLS PIPELINE II=1
            l_data[i][j] = data[i][j];
        }
    }

    // Compute mean
    mean_j: for (j = 0; j < m; j++) {
        double mean_val = 0.0;
        mean_i: for (i = 0; i < n; i++) {
#pragma HLS PIPELINE II=1
            mean_val += l_data[i][j];
        }
        l_mean[j] = mean_val / float_n;
    }

    // Compute stddev
    stddev_j: for (j = 0; j < m; j++) {
        double stddev_val = 0.0;
        stddev_i: for (i = 0; i < n; i++) {
#pragma HLS PIPELINE II=1
            double diff = l_data[i][j] - l_mean[j];
            stddev_val += diff * diff;
        }
        stddev_val /= float_n;
        stddev_val = sqrt(stddev_val);
        l_stddev[j] = stddev_val <= eps ? 1.0 : stddev_val;
    }

    // Store mean and stddev to global memory
    store_mean_j: for (j = 0; j < m; j++) {
#pragma HLS PIPELINE II=1
        mean[j]   = l_mean[j];
        stddev[j] = l_stddev[j];
    }

    // Normalize data
    double sqrt_float_n = sqrt(float_n);
    norm_i: for (i = 0; i < n; i++) {
        norm_j: for (j = 0; j < m; j++) {
#pragma HLS PIPELINE II=1
            l_data[i][j] -= l_mean[j];
            l_data[i][j] /= sqrt_float_n * l_stddev[j];
        }
    }

    // Compute correlation matrix
    corr_i: for (i = 0; i < m - 1; i++) {
        l_corr[i][i] = 1.0;
        corr_j: for (j = i + 1; j < m; j++) {
            double corr_val = 0.0;
            corr_k: for (k = 0; k < n; k++) {
#pragma HLS PIPELINE II=1
                corr_val += l_data[k][i] * l_data[k][j];
            }
            l_corr[i][j] = corr_val;
            l_corr[j][i] = corr_val;
        }
    }
    l_corr[m-1][m-1] = 1.0;

    // Write back normalized data to global memory
    store_data_i: for (i = 0; i < n; i++) {
        store_data_j: for (j = 0; j < m; j++) {
#pragma HLS PIPELINE II=1
            data[i][j] = l_data[i][j];
        }
    }

    // Write back corr to global memory
    store_corr_i: for (i = 0; i < m; i++) {
        store_corr_j: for (j = 0; j < m; j++) {
#pragma HLS PIPELINE II=1
            corr[i][j] = l_corr[i][j];
        }
    }
}

} // extern "C"