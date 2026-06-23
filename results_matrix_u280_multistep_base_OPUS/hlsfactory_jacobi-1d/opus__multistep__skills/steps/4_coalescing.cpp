#include "jacobi-1d.h"
#include <cstring>

#ifndef LARGE_BUS
#define LARGE_BUS 512
#endif

// Number of doubles per wide bus word (512 bits / 64 bits)
#define DOUBLES_PER_BUS (LARGE_BUS / 64)

// Wide bus word: holds DOUBLES_PER_BUS doubles contiguously.
typedef struct {
    double data[DOUBLES_PER_BUS];
} MARS_WIDE_BUS_TYPE;

// Wide bus read: copy 'bytes' bytes from global wide-bus array 'src'
// (starting at byte offset 'offset') into local double buffer 'dst'.
static void memcpy_wide_bus_read_double(double *dst, MARS_WIDE_BUS_TYPE *src,
                                        long offset, long bytes)
{
    long num = bytes / (long)sizeof(double);
    long elem_off = offset / (long)sizeof(double);
read_outer:
    for (long i = 0; i < num; i += DOUBLES_PER_BUS) {
#pragma HLS PIPELINE II=1
        long widx = (elem_off + i) / DOUBLES_PER_BUS;
        MARS_WIDE_BUS_TYPE word = src[widx];
    read_inner:
        for (int j = 0; j < DOUBLES_PER_BUS; j++) {
#pragma HLS UNROLL
            if (i + j < num) {
                dst[i + j] = word.data[j];
            }
        }
    }
}

// Wide bus write: copy 'bytes' bytes from local double buffer 'src'
// into global wide-bus array 'dst' (starting at byte offset 'offset').
static void memcpy_wide_bus_write_double(MARS_WIDE_BUS_TYPE *dst, double *src,
                                         long offset, long bytes)
{
    long num = bytes / (long)sizeof(double);
    long elem_off = offset / (long)sizeof(double);
write_outer:
    for (long i = 0; i < num; i += DOUBLES_PER_BUS) {
#pragma HLS PIPELINE II=1
        long widx = (elem_off + i) / DOUBLES_PER_BUS;
        MARS_WIDE_BUS_TYPE word = dst[widx];
    write_inner:
        for (int j = 0; j < DOUBLES_PER_BUS; j++) {
#pragma HLS UNROLL
            if (i + j < num) {
                word.data[j] = src[i + j];
            }
        }
        dst[widx] = word;
    }
}

#define TILE 60
#define NTILES ((N + TILE - 1) / TILE)

// Load a tile of A and B into the selected buffer set (wide bus)
static void load(MARS_WIDE_BUS_TYPE *A, MARS_WIDE_BUS_TYPE *B,
                 double A_local_1[TILE], double B_local_1[TILE],
                 double A_local_2[TILE], double B_local_2[TILE],
                 int tile, int flag)
{
    int base = tile * TILE;
    int len = (base + TILE <= N) ? TILE : (N - base);
    if (flag == 0) {
        memcpy_wide_bus_read_double(A_local_1, A, (long)base * sizeof(double), (long)len * sizeof(double));
        memcpy_wide_bus_read_double(B_local_1, B, (long)base * sizeof(double), (long)len * sizeof(double));
    } else {
        memcpy_wide_bus_read_double(A_local_2, A, (long)base * sizeof(double), (long)len * sizeof(double));
        memcpy_wide_bus_read_double(B_local_2, B, (long)base * sizeof(double), (long)len * sizeof(double));
    }
}

// Store a tile of A and B from the selected buffer set (wide bus)
static void store(MARS_WIDE_BUS_TYPE *A, MARS_WIDE_BUS_TYPE *B,
                  double A_local_1[TILE], double B_local_1[TILE],
                  double A_local_2[TILE], double B_local_2[TILE],
                  int tile, int flag)
{
    int base = tile * TILE;
    int len = (base + TILE <= N) ? TILE : (N - base);
    if (flag == 0) {
        memcpy_wide_bus_write_double(A, A_local_1, (long)base * sizeof(double), (long)len * sizeof(double));
        memcpy_wide_bus_write_double(B, B_local_1, (long)base * sizeof(double), (long)len * sizeof(double));
    } else {
        memcpy_wide_bus_write_double(A, A_local_2, (long)base * sizeof(double), (long)len * sizeof(double));
        memcpy_wide_bus_write_double(B, B_local_2, (long)base * sizeof(double), (long)len * sizeof(double));
    }
}

void kernel_jacobi_1d(

			    MARS_WIDE_BUS_TYPE *A,
			    MARS_WIDE_BUS_TYPE *B)
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
			    MARS_WIDE_BUS_TYPE *A,
			    MARS_WIDE_BUS_TYPE *B)
{
#pragma HLS INTERFACE m_axi port=A offset=slave bundle=gmem0 max_read_burst_length=256 max_write_burst_length=256
#pragma HLS INTERFACE m_axi port=B offset=slave bundle=gmem1 max_read_burst_length=256 max_write_burst_length=256
#pragma HLS INTERFACE s_axilite port=A bundle=control
#pragma HLS INTERFACE s_axilite port=B bundle=control
#pragma HLS INTERFACE s_axilite port=return bundle=control

    kernel_jacobi_1d(A, B);
}
}