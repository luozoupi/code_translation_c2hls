#include "params.h"

#include <iostream>
#define FLOAT_VECT float
float sub_sum(FLOAT_VECT tmp_Ax[UNROLL_F], FLOAT_VECT tmp_x[UNROLL_F]) {
#pragma HLS array_partition variable=tmp_Ax type=block factor=2
#pragma HLS array_partition variable=tmp_x type=block factor=2
    FLOAT_VECT tmp_sum = 0;
    // parallel mul
    for (int ii = 0; ii < UNROLL_F; ii++)
#pragma HLS unroll factor=2
    tmp_sum += tmp_Ax[ii] * tmp_x[ii];

    // reduction
    return tmp_sum;
}

void spmv(
    int Ap[num_rows],
    int Aj[num_rows],
    int Ax[num_rows],
    int x[num_rows],
    int y[num_rows]) {
#pragma HLS array_partition variable=Ap type=block factor=2
#pragma HLS array_partition variable=Aj type=block factor=2
#pragma HLS array_partition variable=Ax type=block factor=2
#pragma HLS array_partition variable=x type=block factor=2
#pragma HLS array_partition variable=y type=block factor=2
    for (int row = 0; row < num_rows; row++) {
#pragma HLS unroll factor=1
        float sum = y[row];
        unsigned int row_start = Ap[row];
        unsigned int row_end = Ap[row + 1];
        int jj, ii, kk;
        FLOAT_VECT p_Ax[UNROLL_F];
        FLOAT_VECT p_x[UNROLL_F];
#pragma HLS array_partition variable=p_Ax type=block factor=2
#pragma HLS array_partition variable=p_x type=block factor=2

        for (jj = row_start; jj < row_end; jj = jj + UNROLL_F) {
#pragma HLS loop_tripcount min=loop_tripcount1 max=loop_tripcount1
#pragma HLS unroll factor=2
            if (jj + UNROLL_F - 1 < row_end) {
                for (ii = 0; ii < UNROLL_F; ii++)
#pragma HLS unroll factor=2
                p_x[ii] = x[Aj[jj + ii]];

                for (ii = 0; ii < UNROLL_F; ii++)
#pragma HLS unroll factor=2
                p_Ax[ii] = Ax[jj + ii];
                sum += sub_sum(p_Ax, p_x);
            } else {
                for (kk = jj; kk < row_end; kk++) {
#pragma HLS loop_tripcount min=loop_tripcount2 max=loop_tripcount2
#pragma HLS unroll factor=2
                    sum += Ax[kk] * x[Aj[kk]];
                }
            }
        }
        y[row] = sum;
    }
}
