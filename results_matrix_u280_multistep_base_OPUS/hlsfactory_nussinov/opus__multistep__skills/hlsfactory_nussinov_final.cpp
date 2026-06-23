#include "nussinov.h"
#include <cstring>
#include <cstdint>

#define TILE 256

// ---- Wide bus definitions (portable, no ap_int dependency) ----
#define LARGE_BUS 512
#define INTS_PER_BUS  (LARGE_BUS / 32)   // 16 ints per 512-bit word
#define BYTES_PER_BUS (LARGE_BUS / 8)    // 64 bytes per 512-bit word

// A 512-bit wide bus word represented as a struct of 16 32-bit ints.
typedef struct {
    int data[INTS_PER_BUS];
} MARS_WIDE_BUS_TYPE;

typedef unsigned long MARS_ADDR_TYPE;

// Read `byte_len` bytes of int data from wide bus `bus` starting at byte offset
// `byte_offset` into local int array `local`.
static void memcpy_wide_bus_read_int(int *local, MARS_WIDE_BUS_TYPE *bus,
                                     MARS_ADDR_TYPE byte_offset, MARS_ADDR_TYPE byte_len)
{
    int num_ints = byte_len / sizeof(int);
    MARS_ADDR_TYPE start_int = byte_offset / sizeof(int);

read_int_loop:
    for (int idx = 0; idx < num_ints; idx++) {
#pragma HLS LOOP_TRIPCOUNT min=1 max=180
#pragma HLS PIPELINE II=1
        MARS_ADDR_TYPE gi = start_int + idx;
        MARS_ADDR_TYPE word = gi / INTS_PER_BUS;
        int lane = gi % INTS_PER_BUS;
        local[idx] = bus[word].data[lane];
    }
}

// Write `byte_len` bytes of int data from local int array `local` to wide bus
// `bus` starting at byte offset `byte_offset`.
static void memcpy_wide_bus_write_int(MARS_WIDE_BUS_TYPE *bus, int *local,
                                      MARS_ADDR_TYPE byte_offset, MARS_ADDR_TYPE byte_len)
{
    int num_ints = byte_len / sizeof(int);
    MARS_ADDR_TYPE start_int = byte_offset / sizeof(int);

write_int_loop:
    for (int idx = 0; idx < num_ints; idx++) {
#pragma HLS LOOP_TRIPCOUNT min=1 max=180
#pragma HLS PIPELINE II=1
        MARS_ADDR_TYPE gi = start_int + idx;
        MARS_ADDR_TYPE word = gi / INTS_PER_BUS;
        int lane = gi % INTS_PER_BUS;
        bus[word].data[lane] = local[idx];
    }
}

// Read `byte_len` bytes of char data from wide bus `bus` into local char array.
static void memcpy_wide_bus_read_char(char *local, MARS_WIDE_BUS_TYPE *bus,
                                      MARS_ADDR_TYPE byte_offset, MARS_ADDR_TYPE byte_len)
{
    int num_bytes = byte_len;
    MARS_ADDR_TYPE start_byte = byte_offset;

read_char_loop:
    for (int idx = 0; idx < num_bytes; idx++) {
#pragma HLS LOOP_TRIPCOUNT min=1 max=180
#pragma HLS PIPELINE II=1
        MARS_ADDR_TYPE gb = start_byte + idx;
        MARS_ADDR_TYPE word = gb / BYTES_PER_BUS;
        int byte_in_word = gb % BYTES_PER_BUS;
        int lane = byte_in_word / sizeof(int);   // which int within the word
        int shift = (byte_in_word % sizeof(int)) * 8;
        unsigned int v = (unsigned int)bus[word].data[lane];
        local[idx] = (char)((v >> shift) & 0xFF);
    }
}

static void load_tile_db(int l_table[N][N], int tileA[TILE], int tileB[TILE],
                         int i, int j, int kt, int chunk)
{
load_tile:
    for (int p = 0; p < chunk; p++) {
#pragma HLS LOOP_TRIPCOUNT min=1 max=256
#pragma HLS PIPELINE II=1
#pragma HLS UNROLL factor=8
        int kk = kt + p;
        tileA[p] = l_table[i][kk];
        tileB[p] = l_table[kk + 1][j];
    }
}

static int compute_tile_db(int tileA[TILE], int tileB[TILE], int acc, int chunk)
{
compute_tile:
    for (int p = 0; p < chunk; p++) {
#pragma HLS LOOP_TRIPCOUNT min=1 max=256
#pragma HLS PIPELINE II=1
#pragma HLS UNROLL factor=8
        int cand = tileA[p] + tileB[p];
        acc = (acc >= cand) ? acc : cand;
    }
    return acc;
}

extern "C" {

// Top function: signature matches nussinov.h declaration.
// The wide-bus coalescing is achieved by reinterpreting the global memory
// pointers as 512-bit wide bus words for the load/store helpers.
void kernel_nussinov( char seq[ N + 0],
			   int table[ N + 0][N + 0])
{
#pragma HLS INTERFACE m_axi port=seq   offset=slave bundle=gmem0 max_read_burst_length=256 max_write_burst_length=256
#pragma HLS INTERFACE m_axi port=table offset=slave bundle=gmem1 max_read_burst_length=256 max_write_burst_length=256
#pragma HLS INTERFACE s_axilite port=seq    bundle=control
#pragma HLS INTERFACE s_axilite port=table  bundle=control
#pragma HLS INTERFACE s_axilite port=return bundle=control

    const int n = N;

    // Reinterpret global memory pointers as wide-bus words for coalesced access.
    MARS_WIDE_BUS_TYPE *seq_bus   = (MARS_WIDE_BUS_TYPE *)seq;
    MARS_WIDE_BUS_TYPE *table_bus = (MARS_WIDE_BUS_TYPE *)table;

    // Stage data into local buffers to enable reuse and partitioned parallel access.
    static char  l_seq[N];
    static int   l_table[N][N];
#pragma HLS ARRAY_PARTITION variable=l_table cyclic factor=8 dim=2
#pragma HLS ARRAY_PARTITION variable=l_table cyclic factor=8 dim=1

    // Load inputs into local memory via wide bus.
    memcpy_wide_bus_read_char(l_seq, seq_bus, 0, (MARS_ADDR_TYPE)n * sizeof(char));

    // Load table row by row using wide bus reads.
load_table:
    for (int r = 0; r < n; r++) {
        memcpy_wide_bus_read_int(&l_table[r][0], table_bus,
                                 (MARS_ADDR_TYPE)r * n * sizeof(int),
                                 (MARS_ADDR_TYPE)n * sizeof(int));
    }

    int i, j, k;

    for (i = n-1; i >= 0; i--) {
        for (j = i+1; j < n; j++) {
#pragma HLS LOOP_TRIPCOUNT min=1 max=180

            if (j-1 >= 0)
                l_table[i][j] = ((l_table[i][j] >= l_table[i][j-1]) ? l_table[i][j] : l_table[i][j-1]);
            if (i+1 < n)
                l_table[i][j] = ((l_table[i][j] >= l_table[i+1][j]) ? l_table[i][j] : l_table[i+1][j]);

            if (j-1 >= 0 && i+1 < n) {
                if (i < j-1)
                    l_table[i][j] = ((l_table[i][j] >= l_table[i+1][j-1]+(((l_seq[i])+(l_seq[j])) == 3 ? 1 : 0)) ? l_table[i][j] : l_table[i+1][j-1]+(((l_seq[i])+(l_seq[j])) == 3 ? 1 : 0));
                else
                    l_table[i][j] = ((l_table[i][j] >= l_table[i+1][j-1]) ? l_table[i][j] : l_table[i+1][j-1]);
            }

            int acc = l_table[i][j];

            // ---- Tiled reduction over k in [i+1, j) ----
            const int k_start = i + 1;
            const int k_end   = j;          // exclusive

            // Double-buffered local tile buffers (ping-pong).
            int tileA_1[TILE];  int tileB_1[TILE];
            int tileA_2[TILE];  int tileB_2[TILE];
#pragma HLS ARRAY_PARTITION variable=tileA_1 cyclic factor=8
#pragma HLS ARRAY_PARTITION variable=tileB_1 cyclic factor=8
#pragma HLS ARRAY_PARTITION variable=tileA_2 cyclic factor=8
#pragma HLS ARRAY_PARTITION variable=tileB_2 cyclic factor=8

            // Number of tiles for this reduction range.
            int total = k_end - k_start;
            int num_tiles = (total + TILE - 1) / TILE;
            if (num_tiles < 0) num_tiles = 0;

            // Software-pipelined ping-pong over tiles:
            // load tile (t) into one buffer set while computing tile (t-1)
            // from the other buffer set.
        tile_loop:
            for (int t = 0; t <= num_tiles; t++) {
#pragma HLS LOOP_TRIPCOUNT min=1 max=2

                int flag = t & 1;  // selects which buffer to LOAD into this iter

                // ---- LOAD phase for tile t (if it exists) ----
                if (t < num_tiles) {
                    int kt = k_start + t * TILE;
                    int chunk = k_end - kt;
                    if (chunk > TILE) chunk = TILE;
                    if (flag == 0)
                        load_tile_db(l_table, tileA_1, tileB_1, i, j, kt, chunk);
                    else
                        load_tile_db(l_table, tileA_2, tileB_2, i, j, kt, chunk);
                }

                // ---- COMPUTE phase for tile t-1 (already loaded last iter) ----
                if (t > 0) {
                    int pt = t - 1;
                    int kt_c = k_start + pt * TILE;
                    int chunk_c = k_end - kt_c;
                    if (chunk_c > TILE) chunk_c = TILE;
                    int pflag = pt & 1;  // buffer that tile t-1 was loaded into
                    if (pflag == 0)
                        acc = compute_tile_db(tileA_1, tileB_1, acc, chunk_c);
                    else
                        acc = compute_tile_db(tileA_2, tileB_2, acc, chunk_c);
                }
            }

            l_table[i][j] = acc;
        }
    }

    // Write results back to global memory via wide bus, row by row.
store_table:
    for (int r = 0; r < n; r++) {
        memcpy_wide_bus_write_int(table_bus, &l_table[r][0],
                                  (MARS_ADDR_TYPE)r * n * sizeof(int),
                                  (MARS_ADDR_TYPE)n * sizeof(int));
    }
}

} // extern "C"