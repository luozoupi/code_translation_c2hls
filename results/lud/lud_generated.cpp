#include "lud.h"

void lud(float result[GRID_ROWS * GRID_COLS])
{
    int i, j, k;
    float sum;

    for (i = 0; i < SIZE; i++) {
        for (j = i; j < SIZE; j++) {
            #pragma HLS PIPELINE II=1
            sum = result[i * SIZE + j];
            for (k = 0; k < i; k++) {
                #pragma HLS UNROLL factor=4
                sum -= result[i * SIZE + k] * result[k * SIZE + j];
            }
            result[i * SIZE + j] = sum;
        }

        for (j = i + 1; j < SIZE; j++) {
            #pragma HLS PIPELINE II=1
            sum = result[j * SIZE + i];
            for (k = 0; k < i; k++) {
                #pragma HLS UNROLL factor=4
                sum -= result[j * SIZE + k] * result[k * SIZE + i];
            }
            result[j * SIZE + i] = sum / result[i * SIZE + i];
        }
    }

    return;
}

extern "C" {
void workload(float result[GRID_ROWS * GRID_COLS])
{
    #pragma HLS INTERFACE m_axi port=result offset=slave bundle=gmem max_write_burst_length=256 max_read_burst_length=256
    #pragma HLS INTERFACE s_axilite port=return bundle=control

    lud(result);

    return;
}
}