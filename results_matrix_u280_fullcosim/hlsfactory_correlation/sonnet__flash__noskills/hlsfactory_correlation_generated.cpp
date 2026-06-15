#include "correlation.h"

extern "C" {

void kernel_correlation(
            double float_n,
            double data[ N + 0][M + 0],
            double corr[ M + 0][M + 0],
            double mean[ M + 0],
            double stddev[ M + 0])
{
    #pragma HLS INTERFACE s_axilite port=float_n bundle=control
    #pragma HLS INTERFACE m_axi port=data offset=slave bundle=gmem
    #pragma HLS INTERFACE m_axi port=corr offset=slave bundle=gmem
    #pragma HLS INTERFACE m_axi port=mean offset=slave bundle=gmem
    #pragma HLS INTERFACE m_axi port=stddev offset=slave bundle=gmem
    #pragma HLS INTERFACE s_axilite port=data bundle=control
    #pragma HLS INTERFACE s_axilite port=corr bundle=control
    #pragma HLS INTERFACE s_axilite port=mean bundle=control
    #pragma HLS INTERFACE s_axilite port=stddev bundle=control
    #pragma HLS INTERFACE s_axilite port=return bundle=control

    // Local arrays for better access patterns
    double local_data[N][M];
    double local_mean[M];
    double local_stddev[M];
    double local_corr[M][M];

    #pragma HLS ARRAY_PARTITION variable=local_mean complete dim=1
    #pragma HLS ARRAY_PARTITION variable=local_stddev complete dim=1
    #pragma HLS ARRAY_PARTITION variable=local_data cyclic factor=8 dim=2
    #pragma HLS ARRAY_PARTITION variable=local_corr cyclic factor=8 dim=2

    const int n = N;
    const int m = M;

    int i, j, k;
    double eps = 0.1;

    // Load data into local array
    load_data:
    for (i = 0; i < n; i++) {
        for (j = 0; j < m; j++) {
            #pragma HLS PIPELINE II=1
            local_data[i][j] = data[i][j];
        }
    }

    // Compute mean
    mean_outer:
    for (j = 0; j < m; j++) {
        local_mean[j] = 0.0;
        mean_inner:
        for (i = 0; i < n; i++) {
            #pragma HLS PIPELINE II=1
            local_mean[j] += local_data[i][j];
        }
        local_mean[j] /= float_n;
    }

    // Compute stddev
    stddev_outer:
    for (j = 0; j < m; j++) {
        local_stddev[j] = 0.0;
        stddev_inner:
        for (i = 0; i < n; i++) {
            #pragma HLS PIPELINE II=1
            double diff = local_data[i][j] - local_mean[j];
            local_stddev[j] += diff * diff;
        }
        local_stddev[j] /= float_n;
        local_stddev[j] = sqrt(local_stddev[j]);
        local_stddev[j] = local_stddev[j] <= eps ? 1.0 : local_stddev[j];
    }

    // Center and normalize data
    double sqrt_float_n = sqrt(float_n);
    normalize_outer:
    for (i = 0; i < n; i++) {
        normalize_inner:
        for (j = 0; j < m; j++) {
            #pragma HLS PIPELINE II=1
            local_data[i][j] -= local_mean[j];
            local_data[i][j] /= sqrt_float_n * local_stddev[j];
        }
    }

    // Compute correlation matrix
    corr_outer:
    for (i = 0; i < m - 1; i++) {
        local_corr[i][i] = 1.0;
        corr_mid:
        for (j = i + 1; j < m; j++) {
            local_corr[i][j] = 0.0;
            corr_inner:
            for (k = 0; k < n; k++) {
                #pragma HLS PIPELINE II=1
                local_corr[i][j] += local_data[k][i] * local_data[k][j];
            }
            local_corr[j][i] = local_corr[i][j];
        }
    }
    local_corr[m-1][m-1] = 1.0;

    // Write mean back
    write_mean:
    for (j = 0; j < m; j++) {
        #pragma HLS PIPELINE II=1
        mean[j] = local_mean[j];
    }

    // Write stddev back
    write_stddev:
    for (j = 0; j < m; j++) {
        #pragma HLS PIPELINE II=1
        stddev[j] = local_stddev[j];
    }

    // Write data back
    write_data:
    for (i = 0; i < n; i++) {
        for (j = 0; j < m; j++) {
            #pragma HLS PIPELINE II=1
            data[i][j] = local_data[i][j];
        }
    }

    // Write corr back
    write_corr:
    for (i = 0; i < m; i++) {
        for (j = 0; j < m; j++) {
            #pragma HLS PIPELINE II=1
            corr[i][j] = local_corr[i][j];
        }
    }
}

} // extern "C"