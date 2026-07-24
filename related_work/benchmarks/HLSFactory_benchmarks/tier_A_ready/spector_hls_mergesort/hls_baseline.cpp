#include "params.h"
#include <iostream>
void mergesort(int in[no_size], int out[no_size]) {
    for (int i = 2; i <= no_size; i = i * 2) {
#pragma HLS unroll factor=2
        for (int j = 0; j < no_size / i; j++) {
#pragma HLS loop_tripcount min=1 max=max_iter
#pragma HLS unroll factor=2
            int left = i * j;
            int right = (i * j) + i - 1;
            int mid = ((2 * i * j) + i - 1) / 2;

            const int size_left = mid - left + 1;
            const int size_right = right - mid;

            int ii = 0;   // counter for left elements
            int jj = 0;   // counter for right elements
            int k = left; // counter for output

            while (ii < size_left && jj < size_right) {
#pragma HLS loop_tripcount min=1 max=no_size_loop
#pragma HLS unroll factor=2
                int left_val = in[ii + left];
                int right_val = in[jj + mid + 1];

                if (left_val <= right_val) {
                    out[k] = left_val;
                    ii++;
                } else {
                    out[k] = right_val;
                    jj++;
                }
                k++;
            }

            while (ii < size_left) {
#pragma HLS loop_tripcount min=0 max=9

                out[k] = in[ii + left];
                ii++;
                k++;
            }

            while (jj < size_right) {
#pragma HLS loop_tripcount min=0 max=9

                out[k] = in[jj + mid + 1];
                jj++;
                k++;
            }
        }
        for (int k = 0; k < no_size; k++)
#pragma HLS unroll factor=2
            in[k] = out[k];
    }
}
