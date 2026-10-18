#include "stencil.h"

void stencil(TYPE orig[row_size * col_size],
             TYPE sol[row_size * col_size],
             TYPE filter[f_size]) {
    int r, c, k1, k2;
    TYPE temp, mul;

    stencil_label1: for (r = 0; r < row_size - 2; r++) {
        stencil_label2: for (c = 0; c < col_size - 2; c++) {
            temp = (TYPE)0;
            stencil_label3: for (k1 = 0; k1 < 3; k1++) {
                stencil_label4: for (k2 = 0; k2 < 3; k2++) {
                    mul = filter[k1 * 3 + k2] * orig[(r + k1) * col_size + c + k2];
                    temp += mul;
                }
            }
            sol[(r * col_size) + c] = temp;
        }
    }
}

extern "C" {
void workload(TYPE* orig, TYPE* sol, TYPE* filter) {
#pragma HLS INTERFACE m_axi port=orig offset=slave bundle=gmem
#pragma HLS INTERFACE m_axi port=sol offset=slave bundle=gmem
#pragma HLS INTERFACE m_axi port=filter offset=slave bundle=gmem
#pragma HLS INTERFACE s_axilite port=orig bundle=control
#pragma HLS INTERFACE s_axilite port=sol bundle=control
#pragma HLS INTERFACE s_axilite port=filter bundle=control
#pragma HLS INTERFACE s_axilite port=return bundle=control

    TYPE l_orig[row_size * col_size];
    TYPE l_sol[row_size * col_size];
    TYPE l_filter[f_size];
    int i;

    for (i = 0; i < row_size * col_size; i++) l_orig[i] = orig[i];
    for (i = 0; i < f_size; i++) l_filter[i] = filter[i];

    stencil(l_orig, l_sol, l_filter);

    for (i = 0; i < row_size * col_size; i++) sol[i] = l_sol[i];
}
}
