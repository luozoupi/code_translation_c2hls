#include "gesummv.h"
#include <cstring>

#define LARGE_BUS 512
// Number of double elements packed into one wide bus word (512/64 = 8)
#define DOUBLES_PER_BUS (LARGE_BUS / 64)

// Wide bus word: 512 bits = 8 doubles. Plain struct so it compiles without
// ap_int.h while still modeling a single wide coalesced memory transaction.
typedef struct {
    double data[DOUBLES_PER_BUS];
} MARS_WIDE_BUS_TYPE;

// --- Wide-bus helper functions (inline, self-contained) ---
static void memcpy_wide_bus_read_double(double *local, MARS_WIDE_BUS_TYPE *bus,
                                        long offset, int num)
{
    long base = offset;
    for (int i = 0; i < num; i++) {
#pragma HLS PIPELINE II=1
        long idx = base + i;
        long word = idx / DOUBLES_PER_BUS;
        int lane = idx % DOUBLES_PER_BUS;
        local[i] = bus[word].data[lane];
    }
}

static void memcpy_wide_bus_write_double(MARS_WIDE_BUS_TYPE *bus, double *local,
                                         long offset, int num)
{
    long base = offset;
    for (int i = 0; i < num; i++) {
#pragma HLS PIPELINE II=1
        long idx = base + i;
        long word = idx / DOUBLES_PER_BUS;
        int lane = idx % DOUBLES_PER_BUS;
        bus[word].data[lane] = local[i];
    }
}

#define TILE 256

static void load_row(MARS_WIDE_BUS_TYPE *A, MARS_WIDE_BUS_TYPE *B,
                     double A_local_1[N], double B_local_1[N],
                     double A_local_2[N], double B_local_2[N],
                     int i, int n, int flag)
{
    if (flag == 0) {
        memcpy_wide_bus_read_double(A_local_1, A, (long)i * N, n);
        memcpy_wide_bus_read_double(B_local_1, B, (long)i * N, n);
    } else {
        memcpy_wide_bus_read_double(A_local_2, A, (long)i * N, n);
        memcpy_wide_bus_read_double(B_local_2, B, (long)i * N, n);
    }
}

static void compute_row(double A_local_1[N], double B_local_1[N],
                        double A_local_2[N], double B_local_2[N],
                        double x_local[N],
                        double alpha, double beta,
                        double tmp_local[N + 0], double y_local[N + 0],
                        int i, int n, int flag)
{
    double tmp_acc = 0.0;
    double y_acc = 0.0;

    for (int j = 0; j < n; j++)
    {
#pragma HLS PIPELINE II=1
#pragma HLS UNROLL factor=2
#pragma HLS LOOP_TRIPCOUNT min=N max=N
#pragma HLS DEPENDENCE variable=tmp_acc inter false
#pragma HLS DEPENDENCE variable=y_acc inter false
        double a_val = (flag == 0) ? A_local_1[j] : A_local_2[j];
        double b_val = (flag == 0) ? B_local_1[j] : B_local_2[j];
        tmp_acc = a_val * x_local[j] + tmp_acc;
        y_acc = b_val * x_local[j] + y_acc;
    }

    tmp_local[i] = tmp_acc;
    y_local[i] = alpha * tmp_acc + beta * y_acc;
}

// Wide-bus coalesced implementation (distinct name to avoid clashing with the
// original signature declared in gesummv.h).
static void kernel_gesummv_wide(
		    double alpha,
		    double beta,
		    MARS_WIDE_BUS_TYPE *A,
		    MARS_WIDE_BUS_TYPE *B,
		    MARS_WIDE_BUS_TYPE *tmp,
		    MARS_WIDE_BUS_TYPE *x,
		    MARS_WIDE_BUS_TYPE *y)
{
#pragma HLS INTERFACE m_axi port=A offset=slave bundle=gmem0 max_read_burst_length=256 max_write_burst_length=256
#pragma HLS INTERFACE m_axi port=B offset=slave bundle=gmem1 max_read_burst_length=256 max_write_burst_length=256
#pragma HLS INTERFACE m_axi port=tmp offset=slave bundle=gmem2 max_read_burst_length=256 max_write_burst_length=256
#pragma HLS INTERFACE m_axi port=x offset=slave bundle=gmem3 max_read_burst_length=256 max_write_burst_length=256
#pragma HLS INTERFACE m_axi port=y offset=slave bundle=gmem4 max_read_burst_length=256 max_write_burst_length=256

    const int n = N;

    int i;

    // Stage the reused vector x[] into a local buffer to enable
    // partitioned parallel access in the unrolled inner loop.
    double x_local[N];
#pragma HLS ARRAY_PARTITION variable=x_local cyclic factor=20 dim=1

    // Local result buffers (coalesced store at the end).
    double tmp_local[N];
    double y_local[N];
#pragma HLS ARRAY_PARTITION variable=tmp_local cyclic factor=20 dim=1
#pragma HLS ARRAY_PARTITION variable=y_local cyclic factor=20 dim=1

    // --- LOAD x phase (coalesced) ---
    memcpy_wide_bus_read_double(x_local, x, 0, n);

    // Double-buffered tile buffers for one row of A and B.
    double A_local_1[N];
    double B_local_1[N];
    double A_local_2[N];
    double B_local_2[N];
#pragma HLS ARRAY_PARTITION variable=A_local_1 cyclic factor=20 dim=1
#pragma HLS ARRAY_PARTITION variable=B_local_1 cyclic factor=20 dim=1
#pragma HLS ARRAY_PARTITION variable=A_local_2 cyclic factor=20 dim=1
#pragma HLS ARRAY_PARTITION variable=B_local_2 cyclic factor=20 dim=1

    // --- Prologue: load row 0 into buffer set 0 ---
    load_row(A, B, A_local_1, B_local_1, A_local_2, B_local_2, 0, n, 0);

    // --- Main loop: overlap load of row i+1 with compute of row i ---
    for (i = 0; i < n; i++)
    {
#pragma HLS LOOP_TRIPCOUNT min=N max=N
        int flag = i % 2;

        // Load next row (i+1) into the opposite buffer set while
        // computing the current row from the current buffer set.
        if (i + 1 < n) {
            load_row(A, B, A_local_1, B_local_1, A_local_2, B_local_2,
                     i + 1, n, (i + 1) % 2);
        }

        compute_row(A_local_1, B_local_1, A_local_2, B_local_2,
                    x_local, alpha, beta, tmp_local, y_local, i, n, flag);
    }

    // --- STORE phase (coalesced) ---
    memcpy_wide_bus_write_double(tmp, tmp_local, 0, n);
    memcpy_wide_bus_write_double(y, y_local, 0, n);
}

// Top-level wrapper matching the header-declared signature. Reinterprets the
// original pointers as wide-bus words to drive the coalesced implementation.
void kernel_gesummv(
		    double alpha,
		    double beta,
		    double A[N + 0][N + 0],
		    double B[N + 0][N + 0],
		    double tmp[N + 0],
		    double x[N + 0],
		    double y[N + 0])
{
#pragma HLS INTERFACE m_axi port=A offset=slave bundle=gmem0 max_read_burst_length=256 max_write_burst_length=256
#pragma HLS INTERFACE m_axi port=B offset=slave bundle=gmem1 max_read_burst_length=256 max_write_burst_length=256
#pragma HLS INTERFACE m_axi port=tmp offset=slave bundle=gmem2 max_read_burst_length=256 max_write_burst_length=256
#pragma HLS INTERFACE m_axi port=x offset=slave bundle=gmem3 max_read_burst_length=256 max_write_burst_length=256
#pragma HLS INTERFACE m_axi port=y offset=slave bundle=gmem4 max_read_burst_length=256 max_write_burst_length=256

#pragma HLS INTERFACE s_axilite port=alpha bundle=control
#pragma HLS INTERFACE s_axilite port=beta bundle=control
#pragma HLS INTERFACE s_axilite port=A bundle=control
#pragma HLS INTERFACE s_axilite port=B bundle=control
#pragma HLS INTERFACE s_axilite port=tmp bundle=control
#pragma HLS INTERFACE s_axilite port=x bundle=control
#pragma HLS INTERFACE s_axilite port=y bundle=control
#pragma HLS INTERFACE s_axilite port=return bundle=control

    kernel_gesummv_wide(alpha, beta,
                        (MARS_WIDE_BUS_TYPE *)A,
                        (MARS_WIDE_BUS_TYPE *)B,
                        (MARS_WIDE_BUS_TYPE *)tmp,
                        (MARS_WIDE_BUS_TYPE *)x,
                        (MARS_WIDE_BUS_TYPE *)y);
}