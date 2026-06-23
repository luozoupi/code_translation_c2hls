#include "jacobi-1d.h"
#include <cstring>

#define TILE 60
#define NTILES ((N + TILE - 1) / TILE)

// Load a tile of A and B into the selected buffer set
static void load(double A[N], double B[N],
                 double A_local_1[TILE], double B_local_1[TILE],
                 double A_local_2[TILE], double B_local_2[TILE],
                 int tile, int flag)
{
    int base = tile * TILE;
    int len = (base + TILE <= N) ? TILE : (N - base);
    if (flag == 0) {
    load0:
        for (int i = 0; i < len; i++) {
#pragma HLS PIPELINE II=1
            A_local_1[i] = A[base + i];
            B_local_1[i] = B[base + i];
        }
    } else {
    load1:
        for (int i = 0; i < len; i++) {
#pragma HLS PIPELINE II=1
            A_local_2[i] = A[base + i];
            B_local_2[i] = B[base + i];
        }
    }
}

// Store a tile of A and B from the selected buffer set
static void store(double A[N], double B[N],
                  double A_local_1[TILE], double B_local_1[TILE],
                  double A_local_2[TILE], double B_local_2[TILE],
                  int tile, int flag)
{
    int base = tile * TILE;
    int len = (base + TILE <= N) ? TILE : (N - base);
    if (flag == 0) {
    store0:
        for (int i = 0; i < len; i++) {
#pragma HLS PIPELINE II=1
            A[base + i] = A_local_1[i];
            B[base + i] = B_local_1[i];
        }
    } else {
    store1:
        for (int i = 0; i < len; i++) {
#pragma HLS PIPELINE II=1
            A[base + i] = A_local_2[i];
            B[base + i] = B_local_2[i];
        }
    }
}

void kernel_jacobi_1d(

			    double A[ N + 0],
			    double B[ N + 0])
{


    const int n = N;
    const int tsteps = TSTEPS;

    int t, i;

    // Local tile buffers (entire array fits in local memory since N is bounded)
    double A_local[N];
    double B_local[N];
#pragma HLS ARRAY_PARTITION variable=A_local cyclic factor=4 dim=1
#pragma HLS ARRAY_PARTITION variable=B_local cyclic factor=4 dim=1

    // ---- LOAD phase with double buffering ----
    // Ping-pong buffers to overlap successive load chunks
    double A_buf_1[TILE];
    double B_buf_1[TILE];
    double A_buf_2[TILE];
    double B_buf_2[TILE];
#pragma HLS ARRAY_PARTITION variable=A_buf_1 cyclic factor=4 dim=1
#pragma HLS ARRAY_PARTITION variable=B_buf_1 cyclic factor=4 dim=1
#pragma HLS ARRAY_PARTITION variable=A_buf_2 cyclic factor=4 dim=1
#pragma HLS ARRAY_PARTITION variable=B_buf_2 cyclic factor=4 dim=1

    // Load all tiles into A_local/B_local using double-buffered staging
load_tiles:
    for (int tile = 0; tile < NTILES; tile++) {
        int flag = tile % 2;
        load(A, B, A_buf_1, B_buf_1, A_buf_2, B_buf_2, tile, flag);

        int base = tile * TILE;
        int len = (base + TILE <= N) ? TILE : (N - base);
        if (flag == 0) {
        copy_in0:
            for (i = 0; i < len; i++) {
#pragma HLS PIPELINE II=1
                A_local[base + i] = A_buf_1[i];
                B_local[base + i] = B_buf_1[i];
            }
        } else {
        copy_in1:
            for (i = 0; i < len; i++) {
#pragma HLS PIPELINE II=1
                A_local[base + i] = A_buf_2[i];
                B_local[base + i] = B_buf_2[i];
            }
        }
    }

    // ---- COMPUTE phase (operates on local buffers) ----
compute_t:
    for (t = 0; t < tsteps; t++)
    {
#pragma HLS LOOP_TRIPCOUNT min=TSTEPS max=TSTEPS
    update_B:
        for (i = 1; i < n - 1; i++) {
#pragma HLS PIPELINE II=1
#pragma HLS UNROLL factor=4
#pragma HLS DEPENDENCE variable=B_local inter false
            B_local[i] = 0.33333 * (A_local[i-1] + A_local[i] + A_local[i + 1]);
        }
    update_A:
        for (i = 1; i < n - 1; i++) {
#pragma HLS PIPELINE II=1
#pragma HLS UNROLL factor=4
#pragma HLS DEPENDENCE variable=A_local inter false
            A_local[i] = 0.33333 * (B_local[i-1] + B_local[i] + B_local[i + 1]);
        }
    }

    // ---- STORE phase with double buffering ----
store_tiles:
    for (int tile = 0; tile < NTILES; tile++) {
        int flag = tile % 2;
        int base = tile * TILE;
        int len = (base + TILE <= N) ? TILE : (N - base);
        if (flag == 0) {
        copy_out0:
            for (i = 0; i < len; i++) {
#pragma HLS PIPELINE II=1
                A_buf_1[i] = A_local[base + i];
                B_buf_1[i] = B_local[base + i];
            }
        } else {
        copy_out1:
            for (i = 0; i < len; i++) {
#pragma HLS PIPELINE II=1
                A_buf_2[i] = A_local[base + i];
                B_buf_2[i] = B_local[base + i];
            }
        }
        store(A, B, A_buf_1, B_buf_1, A_buf_2, B_buf_2, tile, flag);
    }

}

extern "C" {
void workload(
			    double A[ N + 0],
			    double B[ N + 0])
{
#pragma HLS INTERFACE m_axi port=A offset=slave bundle=gmem0
#pragma HLS INTERFACE m_axi port=B offset=slave bundle=gmem1
#pragma HLS INTERFACE s_axilite port=A bundle=control
#pragma HLS INTERFACE s_axilite port=B bundle=control
#pragma HLS INTERFACE s_axilite port=return bundle=control

    kernel_jacobi_1d(A, B);
}
}