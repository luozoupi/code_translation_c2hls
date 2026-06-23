#include "symm.h"
#include <cstring>
#include <cstdint>

#ifndef LARGE_BUS
#define LARGE_BUS 512
#endif

// Plain-C++ wide-bus word: 512 bits = 8 x 64-bit lanes.
// Acts as the wide memory bus type for memory coalescing.
struct wide_bus_t {
    uint64_t lane[LARGE_BUS / 64];
};

// Wide-bus helper: read `count` double elements from a 512-bit
// wide bus into a local array, starting at byte offset `offset_bytes`.
static void memcpy_wide_bus_read_float(double* local, wide_bus_t* bus,
                                       long offset_bytes, int count) {
    const int lanes_per_word = LARGE_BUS / 64; // 8 doubles per 512-bit word
    long start_elem = offset_bytes / (long)sizeof(double);
    int idx = 0;
RD_OUTER:
    while (idx < count) {
        long global_elem = start_elem + idx;
        long word_idx = global_elem / lanes_per_word;
        int lane_off = (int)(global_elem % lanes_per_word);
        wide_bus_t word = bus[word_idx];
    RD_INNER:
        for (int e = lane_off; e < lanes_per_word && idx < count; e++) {
#pragma HLS PIPELINE II=1
            double val;
            std::memcpy(&val, &word.lane[e], sizeof(double));
            local[idx] = val;
            idx++;
        }
    }
}

// Wide-bus helper: write `count` double elements from a local array
// to a 512-bit wide bus, starting at byte offset `offset_bytes`.
static void memcpy_wide_bus_write_float(double* local, wide_bus_t* bus,
                                        long offset_bytes, int count) {
    const int lanes_per_word = LARGE_BUS / 64;
    long start_elem = offset_bytes / (long)sizeof(double);
    int idx = 0;
WR_OUTER:
    while (idx < count) {
        long global_elem = start_elem + idx;
        long word_idx = global_elem / lanes_per_word;
        int lane_off = (int)(global_elem % lanes_per_word);
        wide_bus_t word = bus[word_idx];
    WR_INNER:
        for (int e = lane_off; e < lanes_per_word && idx < count; e++) {
#pragma HLS PIPELINE II=1
            double val = local[idx];
            std::memcpy(&word.lane[e], &val, sizeof(double));
            idx++;
        }
        bus[word_idx] = word;
    }
}

extern "C" {

// Signature matches symm.h exactly. The double-array pointers are
// reinterpreted as wide (512-bit) bus words for memory-coalesced access.
void kernel_symm(
		 double alpha,
		 double beta,
		 double C[ M + 0][N + 0],
		 double A[ M + 0][M + 0],
		 double B[ M + 0][N + 0])
{
#pragma HLS INTERFACE m_axi port=C offset=slave bundle=gmem0 max_read_burst_length=256 max_write_burst_length=256
#pragma HLS INTERFACE m_axi port=A offset=slave bundle=gmem1 max_read_burst_length=256 max_write_burst_length=256
#pragma HLS INTERFACE m_axi port=B offset=slave bundle=gmem2 max_read_burst_length=256 max_write_burst_length=256
#pragma HLS INTERFACE s_axilite port=alpha bundle=control
#pragma HLS INTERFACE s_axilite port=beta  bundle=control
#pragma HLS INTERFACE s_axilite port=C     bundle=control
#pragma HLS INTERFACE s_axilite port=A     bundle=control
#pragma HLS INTERFACE s_axilite port=B     bundle=control
#pragma HLS INTERFACE s_axilite port=return bundle=control

    const int m = M;
    const int n = N;

    int i, j, k;

    // Reinterpret the global double arrays as wide-bus word streams.
    wide_bus_t* C_bus = reinterpret_cast<wide_bus_t*>(C);
    wide_bus_t* A_bus = reinterpret_cast<wide_bus_t*>(A);
    wide_bus_t* B_bus = reinterpret_cast<wide_bus_t*>(B);

    // Full buffers required: B_local is read across all rows (column-wise)
    // in the temp2 reduction, and C_local accumulates across all rows.
    static double C_local[M][N];
    static double B_local[M][N];
#pragma HLS ARRAY_PARTITION variable=C_local cyclic factor=8 dim=2
#pragma HLS ARRAY_PARTITION variable=B_local cyclic factor=8 dim=2

    // Double-buffered per-row staging for A's row i: A[i][*] is only used
    // for the current row's compute, so it can be ping-ponged. Loading
    // A row (i+1) overlaps with compute of row i.
    static double A_row_1[M];
    static double A_row_2[M];
#pragma HLS ARRAY_PARTITION variable=A_row_1 cyclic factor=8 dim=1
#pragma HLS ARRAY_PARTITION variable=A_row_2 cyclic factor=8 dim=1

    // ---------------- LOAD C PHASE (wide bus) ----------------
    LOAD_C_I:for (i = 0; i < m; i++) {
        // C is row-major: row i starts at element offset i*n
        memcpy_wide_bus_read_float(
            &C_local[i][0], C_bus, (long)i * n * sizeof(double), n);
    }

    // ---------------- LOAD B PHASE (wide bus) ----------------
    LOAD_B_I:for (i = 0; i < m; i++) {
        memcpy_wide_bus_read_float(
            &B_local[i][0], B_bus, (long)i * n * sizeof(double), n);
    }

    // ---------------- COMPUTE PHASE WITH DOUBLE-BUFFERED A ROW ----------
    // Prologue: load A row 0 into buffer set 1.
    memcpy_wide_bus_read_float(
        &A_row_1[0], A_bus, (long)0 * m * sizeof(double), m);

    COMP_I:for (i = 0; i < m; i++) {
        // flag selects which buffer holds the CURRENT row i.
        // i even -> current = A_row_1, next loads into A_row_2
        // i odd  -> current = A_row_2, next loads into A_row_1
        int flag = i % 2;

        // Load NEXT A row (i+1) into the opposite buffer, overlapping
        // with the compute below.
        if (i + 1 < m) {
            if (flag == 0) {
                memcpy_wide_bus_read_float(
                    &A_row_2[0], A_bus, (long)(i + 1) * m * sizeof(double), m);
            } else {
                memcpy_wide_bus_read_float(
                    &A_row_1[0], A_bus, (long)(i + 1) * m * sizeof(double), m);
            }
        }

        // Compute current row i using the current A-row buffer.
        if (flag == 0) {
            COMP_J0:for (j = 0; j < n; j++) {
                double temp2 = 0;
                COMP_K0:for (k = 0; k < i; k++) {
#pragma HLS LOOP_TRIPCOUNT min=0 max=M
#pragma HLS PIPELINE II=1
#pragma HLS UNROLL factor=8
#pragma HLS DEPENDENCE variable=C_local inter false
                    C_local[k][j] += alpha * B_local[i][j] * A_row_1[k];
                    temp2 += B_local[k][j] * A_row_1[k];
                }
                C_local[i][j] = beta * C_local[i][j]
                                + alpha * B_local[i][j] * A_row_1[i]
                                + alpha * temp2;
            }
        } else {
            COMP_J1:for (j = 0; j < n; j++) {
                double temp2 = 0;
                COMP_K1:for (k = 0; k < i; k++) {
#pragma HLS LOOP_TRIPCOUNT min=0 max=M
#pragma HLS PIPELINE II=1
#pragma HLS UNROLL factor=8
#pragma HLS DEPENDENCE variable=C_local inter false
                    C_local[k][j] += alpha * B_local[i][j] * A_row_2[k];
                    temp2 += B_local[k][j] * A_row_2[k];
                }
                C_local[i][j] = beta * C_local[i][j]
                                + alpha * B_local[i][j] * A_row_2[i]
                                + alpha * temp2;
            }
        }
    }

    // ---------------- STORE PHASE (wide bus) ----------------
    STORE_C_I:for (i = 0; i < m; i++) {
        memcpy_wide_bus_write_float(
            &C_local[i][0], C_bus, (long)i * n * sizeof(double), n);
    }
}

}