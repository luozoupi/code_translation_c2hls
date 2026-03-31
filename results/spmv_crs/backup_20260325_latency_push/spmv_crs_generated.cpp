#include "spmv.h"

void spmv(TYPE val[NNZ], int32_t cols[NNZ], int32_t rowDelimiters[N + 1],
          TYPE vec[N], TYPE out[N]) {
#pragma HLS INLINE off

    int i, j;
    TYPE sum, Si;

spmv_1:
    for (i = 0; i < N; i++) {
#pragma HLS PIPELINE II=1
        sum = 0;
        Si = 0;
        int tmp_begin = rowDelimiters[i];
        int tmp_end = rowDelimiters[i + 1];

    spmv_2:
        for (j = tmp_begin; j < tmp_end; j++) {
            Si = val[j] * vec[cols[j]];
            sum = sum + Si;
        }
        out[i] = sum;
    }
}

extern "C" {

void workload(TYPE* val, int32_t* cols, int32_t* rowDelimiters,
              TYPE* vec, TYPE* out) {
#pragma HLS INTERFACE m_axi port=val offset=slave bundle=gmem
#pragma HLS INTERFACE m_axi port=cols offset=slave bundle=gmem
#pragma HLS INTERFACE m_axi port=rowDelimiters offset=slave bundle=gmem
#pragma HLS INTERFACE m_axi port=vec offset=slave bundle=gmem
#pragma HLS INTERFACE m_axi port=out offset=slave bundle=gmem

#pragma HLS INTERFACE s_axilite port=val bundle=control
#pragma HLS INTERFACE s_axilite port=cols bundle=control
#pragma HLS INTERFACE s_axilite port=rowDelimiters bundle=control
#pragma HLS INTERFACE s_axilite port=vec bundle=control
#pragma HLS INTERFACE s_axilite port=out bundle=control
#pragma HLS INTERFACE s_axilite port=return bundle=control

    TYPE l_val[NNZ];
    int32_t l_cols[NNZ];
    int32_t l_rowDelimiters[N + 1];
    TYPE l_vec[N];
    TYPE l_out[N];
    int i;

copy_nnz:
    for (i = 0; i < NNZ; i++) {
        l_val[i] = val[i];
        l_cols[i] = cols[i];
    }

copy_row_delimiters:
    for (i = 0; i < N + 1; i++) {
        l_rowDelimiters[i] = rowDelimiters[i];
    }

copy_vec:
    for (i = 0; i < N; i++) {
        l_vec[i] = vec[i];
    }

    spmv(l_val, l_cols, l_rowDelimiters, l_vec, l_out);

copy_out:
    for (i = 0; i < N; i++) {
        out[i] = l_out[i];
    }
}

}