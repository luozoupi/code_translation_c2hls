#include "floyd-warshall.h"
#include <cstring>

// ---- Self-contained wide-bus definitions (no Vitis headers required) ----
#define LARGE_BUS 512
// Number of ints per wide bus word (512 bits / 32 bits)
#define INTS_PER_BUS (LARGE_BUS / 32)

// Wide bus word: 16 packed ints (512 bits total).
struct MARS_WIDE_BUS_TYPE {
    int data[INTS_PER_BUS];
};

// Burst read 'num' ints from wide-bus global memory into local int buffer.
// 'offset' is in units of ints.
static void memcpy_wide_bus_read_int(int *local, MARS_WIDE_BUS_TYPE *bus,
                                     long offset, int num)
{
#pragma HLS INLINE off
    long base_word = offset / INTS_PER_BUS;
    int  base_off  = (int)(offset % INTS_PER_BUS);
    int idx = 0;
    long w = base_word;
    int o = base_off;

    read_loop:
    while (idx < num)
    {
#pragma HLS LOOP_TRIPCOUNT min=N max=N
#pragma HLS PIPELINE II=1
        MARS_WIDE_BUS_TYPE word = bus[w];
        inner_read:
        for (; o < INTS_PER_BUS && idx < num; o++)
        {
#pragma HLS LOOP_TRIPCOUNT min=1 max=16
            local[idx] = word.data[o];
            idx++;
        }
        o = 0;
        w++;
    }
}

// Burst write 'num' ints from local int buffer to wide-bus global memory.
// 'offset' is in units of ints.
static void memcpy_wide_bus_write_int(MARS_WIDE_BUS_TYPE *bus, int *local,
                                      long offset, int num)
{
#pragma HLS INLINE off
    long base_word = offset / INTS_PER_BUS;
    int  base_off  = (int)(offset % INTS_PER_BUS);
    int idx = 0;
    long w = base_word;
    int o = base_off;

    write_loop:
    while (idx < num)
    {
#pragma HLS LOOP_TRIPCOUNT min=N max=N
#pragma HLS PIPELINE II=1
        MARS_WIDE_BUS_TYPE word = bus[w];
        inner_write:
        for (; o < INTS_PER_BUS && idx < num; o++)
        {
#pragma HLS LOOP_TRIPCOUNT min=1 max=16
            word.data[o] = local[idx];
            idx++;
        }
        bus[w] = word;
        o = 0;
        w++;
    }
}


// Load row i into the selected buffer
static void load_row(int row_buf[N], int row_i_1[N], int row_i_2[N],
                     int n, int flag)
{
#pragma HLS INLINE off
    load_row_i:
    for (int j = 0; j < n; j++)
    {
#pragma HLS LOOP_TRIPCOUNT min=N max=N
#pragma HLS PIPELINE II=1
        if (flag == 0)
            row_i_1[j] = row_buf[j];
        else
            row_i_2[j] = row_buf[j];
    }
}

// Compute on the selected buffer and store result back to local out buffer
static void compute_store(int row_k[N], int row_i_1[N], int row_i_2[N],
                          int out_buf[N],
                          int k, int n, int flag)
{
#pragma HLS INLINE off
    int path_ik = (flag == 0) ? row_i_1[k] : row_i_2[k];

    compute_row:
    for (int j = 0; j < n; j++)
    {
#pragma HLS LOOP_TRIPCOUNT min=N max=N
#pragma HLS PIPELINE II=1
#pragma HLS UNROLL factor=4
#pragma HLS DEPENDENCE variable=row_i_1 inter false
#pragma HLS DEPENDENCE variable=row_i_2 inter false
        if (flag == 0)
        {
            int candidate = path_ik + row_k[j];
            int v = row_i_1[j];
            v = v < candidate ? v : candidate;
            row_i_1[j] = v;
            out_buf[j] = v;
        }
        else
        {
            int candidate = path_ik + row_k[j];
            int v = row_i_2[j];
            v = v < candidate ? v : candidate;
            row_i_2[j] = v;
            out_buf[j] = v;
        }
    }
}


void kernel_floyd_warshall(MARS_WIDE_BUS_TYPE *path)
{
#pragma HLS INLINE off

    const int n = N;

    int i, j, k;

    // Local buffer for the k-th row (reused across all i)
    int row_k[N];
#pragma HLS ARRAY_PARTITION variable=row_k cyclic factor=4 dim=1

    // Two local buffers for the row i being processed (ping-pong)
    int row_i_1[N];
#pragma HLS ARRAY_PARTITION variable=row_i_1 cyclic factor=4 dim=1
    int row_i_2[N];
#pragma HLS ARRAY_PARTITION variable=row_i_2 cyclic factor=4 dim=1

    // Staging buffers for wide-bus burst transfers
    int load_buf[N];
#pragma HLS ARRAY_PARTITION variable=load_buf cyclic factor=4 dim=1
    int out_buf[N];
#pragma HLS ARRAY_PARTITION variable=out_buf cyclic factor=4 dim=1

    for (k = 0; k < n; k++)
    {
        // ---- LOAD phase: stage the k-th row into local memory ----
        // Burst read row k from global memory (offset in ints = k*n)
        memcpy_wide_bus_read_int(row_k, path, (long)k * n, n);

        // ---- Double-buffered pipeline over rows i ----
        // Prologue: load row 0 into buffer 1 (flag 0)
        memcpy_wide_bus_read_int(load_buf, path, 0L, n);
        load_row(load_buf, row_i_1, row_i_2, n, 0);

        row_loop:
        for (i = 0; i < n; i++)
        {
#pragma HLS LOOP_TRIPCOUNT min=N max=N
            int flag = i % 2;

            // Load next row (i+1) into the opposite buffer while we
            // compute/store the current row i. These two operations
            // touch different ping-pong buffers, allowing overlap.
            if (i + 1 < n)
            {
                memcpy_wide_bus_read_int(load_buf, path, (long)(i + 1) * n, n);
                load_row(load_buf, row_i_1, row_i_2, n, (i + 1) % 2);
            }

            compute_store(row_k, row_i_1, row_i_2, out_buf, k, n, flag);

            // Burst write the computed row i back to global memory
            memcpy_wide_bus_write_int(path, out_buf, (long)i * n, n);
        }
    }
}

extern "C" {
void workload(MARS_WIDE_BUS_TYPE *path)
{
#pragma HLS INTERFACE m_axi port=path offset=slave bundle=gmem \
    max_read_burst_length=256 max_write_burst_length=256
#pragma HLS INTERFACE s_axilite port=path bundle=control
#pragma HLS INTERFACE s_axilite port=return bundle=control

    kernel_floyd_warshall(path);
}
}