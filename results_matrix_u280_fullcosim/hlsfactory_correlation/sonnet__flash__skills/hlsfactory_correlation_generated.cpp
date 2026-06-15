#include "correlation.h"

extern "C" {

void kernel_correlation(
    double float_n,
    double data[N + 0][M + 0],
    double corr[M + 0][M + 0],
    double mean[M + 0],
    double stddev[M + 0])
{
#pragma HLS INTERFACE m_axi port=data   offset=slave bundle=gmem
#pragma HLS INTERFACE m_axi port=corr   offset=slave bundle=gmem
#pragma HLS INTERFACE m_axi port=mean   offset=slave bundle=gmem
#pragma HLS INTERFACE m_axi port=stddev offset=slave bundle=gmem
#pragma HLS INTERFACE s_axilite port=float_n bundle=control
#pragma HLS INTERFACE s_axilite port=data    bundle=control
#pragma HLS INTERFACE s_axilite port=corr    bundle=control
#pragma HLS INTERFACE s_axilite port=mean    bundle=control
#pragma HLS INTERFACE s_axilite port=stddev  bundle=control
#pragma HLS INTERFACE s_axilite port=return  bundle=control

    // Local copies for parallel access
    double l_data  [N][M];
    double l_corr  [M][M];
    double l_mean  [M];
    double l_stddev[M];

#pragma HLS ARRAY_PARTITION variable=l_data   cyclic factor=8 dim=2
#pragma HLS ARRAY_PARTITION variable=l_corr   cyclic factor=8 dim=2
#pragma HLS ARRAY_PARTITION variable=l_mean   complete
#pragma HLS ARRAY_PARTITION variable=l_stddev complete

    const int n = N;
    const int m = M;
    double eps = 0.1;

    // Load data from global memory
    load_data_outer: for (int i = 0; i < n; i++) {
        load_data_inner: for (int j = 0; j < m; j++) {
#pragma HLS PIPELINE II=1
            l_data[i][j] = data[i][j];
        }
    }

    // Compute mean
    mean_outer: for (int j = 0; j < m; j++) {
        double sum = 0.0;
        mean_inner: for (int i = 0; i < n; i++) {
#pragma HLS PIPELINE II=1
            sum += l_data[i][j];
        }
        l_mean[j] = sum / float_n;
    }

    // Compute stddev
    stddev_outer: for (int j = 0; j < m; j++) {
        double sum = 0.0;
        stddev_inner: for (int i = 0; i < n; i++) {
#pragma HLS PIPELINE II=1
            double diff = l_data[i][j] - l_mean[j];
            sum += diff * diff;
        }
        sum /= float_n;
        sum = sqrt(sum);
        l_stddev[j] = (sum <= eps) ? 1.0 : sum;
    }

    // Normalize data
    double sqrt_float_n = sqrt(float_n);
    norm_outer: for (int i = 0; i < n; i++) {
        norm_inner: for (int j = 0; j < m; j++) {
#pragma HLS PIPELINE II=1
            l_data[i][j] -= l_mean[j];
            l_data[i][j] /= sqrt_float_n * l_stddev[j];
        }
    }

    // Compute correlation matrix
    corr_i: for (int i = 0; i < m - 1; i++) {
        l_corr[i][i] = 1.0;
        corr_j: for (int j = i + 1; j < m; j++) {
            double sum = 0.0;
            corr_k: for (int k = 0; k < n; k++) {
#pragma HLS PIPELINE II=1
                sum += l_data[k][i] * l_data[k][j];
            }
            l_corr[i][j] = sum;
            l_corr[j][i] = sum;
        }
    }
    l_corr[m-1][m-1] = 1.0;

    // Write mean back
    write_mean: for (int j = 0; j < m; j++) {
#pragma HLS PIPELINE II=1
        mean[j] = l_mean[j];
    }

    // Write stddev back
    write_stddev: for (int j = 0; j < m; j++) {
#pragma HLS PIPELINE II=1
        stddev[j] = l_stddev[j];
    }

    // Write data back
    write_data_outer: for (int i = 0; i < n; i++) {
        write_data_inner: for (int j = 0; j < m; j++) {
#pragma HLS PIPELINE II=1
            data[i][j] = l_data[i][j];
        }
    }

    // Write corr back
    write_corr_outer: for (int i = 0; i < m; i++) {
        write_corr_inner: for (int j = 0; j < m; j++) {
#pragma HLS PIPELINE II=1
            corr[i][j] = l_corr[i][j];
        }
    }
}

} // extern "C"