// Proven 32x8 compact stream kernel (HLS-validated selected kernel).
// Live generator: compact_pe_instantiate.py (deterministic Python; not LLM).
// This file is the checked-in 32x8 reference, not token-replaced at search time.
// Placeholders the generator fills: PE_NUM, SIMD, ap_uint<pack_bits> / packN/unpackN,
// explicit mm_pe() count, and i-tile placement (around DATAFLOW vs inside tasks).
#include "kernel.h"
#include <hls_stream.h>
#include <ap_int.h>

#define PE_NUM 32
#define SIMD   8

typedef ap_uint<256> vec8_bits;

static vec8_bits pack8(data_t v0, data_t v1, data_t v2, data_t v3,
                       data_t v4, data_t v5, data_t v6, data_t v7) {
#pragma HLS INLINE
    union { unsigned u; float f; } c0, c1, c2, c3, c4, c5, c6, c7;
    c0.f = (float)v0;
    c1.f = (float)v1;
    c2.f = (float)v2;
    c3.f = (float)v3;
    c4.f = (float)v4;
    c5.f = (float)v5;
    c6.f = (float)v6;
    c7.f = (float)v7;

    vec8_bits w;
    w.range(31, 0)    = c0.u;
    w.range(63, 32)   = c1.u;
    w.range(95, 64)   = c2.u;
    w.range(127, 96)  = c3.u;
    w.range(159, 128) = c4.u;
    w.range(191, 160) = c5.u;
    w.range(223, 192) = c6.u;
    w.range(255, 224) = c7.u;
    return w;
}

static void unpack8(vec8_bits w, data_t &v0, data_t &v1, data_t &v2, data_t &v3,
                    data_t &v4, data_t &v5, data_t &v6, data_t &v7) {
#pragma HLS INLINE
    union { unsigned u; float f; } c0, c1, c2, c3, c4, c5, c6, c7;
    c0.u = (unsigned)w.range(31, 0);
    c1.u = (unsigned)w.range(63, 32);
    c2.u = (unsigned)w.range(95, 64);
    c3.u = (unsigned)w.range(127, 96);
    c4.u = (unsigned)w.range(159, 128);
    c5.u = (unsigned)w.range(191, 160);
    c6.u = (unsigned)w.range(223, 192);
    c7.u = (unsigned)w.range(255, 224);

    v0 = (data_t)c0.f;
    v1 = (data_t)c1.f;
    v2 = (data_t)c2.f;
    v3 = (data_t)c3.f;
    v4 = (data_t)c4.f;
    v5 = (data_t)c5.f;
    v6 = (data_t)c6.f;
    v7 = (data_t)c7.f;
}

static void load_A(data_t A[I][K], hls::stream<vec8_bits> fifo_A[PE_NUM], int i0) {
#pragma HLS INLINE off
    for (int k0 = 0; k0 < K; k0 += SIMD) {
        for (int p = 0; p < PE_NUM; ++p) {
#pragma HLS PIPELINE II=1
            fifo_A[p].write(pack8(
                A[i0 + p][k0 + 0], A[i0 + p][k0 + 1],
                A[i0 + p][k0 + 2], A[i0 + p][k0 + 3],
                A[i0 + p][k0 + 4], A[i0 + p][k0 + 5],
                A[i0 + p][k0 + 6], A[i0 + p][k0 + 7]));
        }
    }
}

static void load_B(data_t B[J][K], hls::stream<vec8_bits> &fifo_B) {
#pragma HLS INLINE off
    for (int k0 = 0; k0 < K; k0 += SIMD) {
        for (int j = 0; j < J; ++j) {
#pragma HLS PIPELINE II=1
            fifo_B.write(pack8(
                B[j][k0 + 0], B[j][k0 + 1],
                B[j][k0 + 2], B[j][k0 + 3],
                B[j][k0 + 4], B[j][k0 + 5],
                B[j][k0 + 6], B[j][k0 + 7]));
        }
    }
}

static void mm_pe(hls::stream<vec8_bits> &fifo_A,
                  hls::stream<vec8_bits> &fifo_B_in,
                  hls::stream<vec8_bits> &fifo_B_out,
                  hls::stream<data_t> &fifo_C) {
#pragma HLS INLINE off
    data_t Crow[J];
#pragma HLS BIND_STORAGE variable=Crow type=ram_2p impl=bram

    data_t a0, a1, a2, a3, a4, a5, a6, a7;

    for (int t = 0; t < (K / SIMD) * J; ++t) {
#pragma HLS PIPELINE II=1
        const int k_tile = t / J;
        const int j = t - k_tile * J;

        if (j == 0) {
            unpack8(fifo_A.read(), a0, a1, a2, a3, a4, a5, a6, a7);
        }

        vec8_bits bw = fifo_B_in.read();
        fifo_B_out.write(bw);

        data_t b0, b1, b2, b3, b4, b5, b6, b7;
        unpack8(bw, b0, b1, b2, b3, b4, b5, b6, b7);

        data_t partial =
            a0 * b0 + a1 * b1 + a2 * b2 + a3 * b3 +
            a4 * b4 + a5 * b5 + a6 * b6 + a7 * b7;

        data_t acc = (k_tile == 0) ? partial : (Crow[j] + partial);
        Crow[j] = acc;

        if (k_tile == (K / SIMD) - 1) {
            fifo_C.write(acc);
        }
    }
}

static void drain_B(hls::stream<vec8_bits> &fifo_B) {
#pragma HLS INLINE off
    for (int k0 = 0; k0 < K; k0 += SIMD) {
        for (int j = 0; j < J; ++j) {
#pragma HLS PIPELINE II=1
            (void)fifo_B.read();
        }
    }
}

static void store_C(data_t C[I][J], hls::stream<data_t> fifo_C[PE_NUM], int i0) {
#pragma HLS INLINE off
    for (int p = 0; p < PE_NUM; ++p) {
        for (int j = 0; j < J; ++j) {
#pragma HLS PIPELINE II=1
            C[i0 + p][j] = fifo_C[p].read();
        }
    }
}

extern "C" void autosa_mm(data_t A[I][K], data_t B[J][K], data_t C[I][J]) {
#pragma HLS INTERFACE m_axi port=A offset=slave bundle=gmem0 max_read_burst_length=64 num_read_outstanding=16
#pragma HLS INTERFACE m_axi port=B offset=slave bundle=gmem1 max_read_burst_length=64 num_read_outstanding=16
#pragma HLS INTERFACE m_axi port=C offset=slave bundle=gmem2 max_read_burst_length=64 max_write_burst_length=64 num_read_outstanding=16 num_write_outstanding=16
#pragma HLS INTERFACE s_axilite port=A bundle=control
#pragma HLS INTERFACE s_axilite port=B bundle=control
#pragma HLS INTERFACE s_axilite port=C bundle=control
#pragma HLS INTERFACE s_axilite port=return bundle=control

    hls::stream<vec8_bits> fifo_A[PE_NUM];
    hls::stream<vec8_bits> fifo_B[PE_NUM + 1];
    hls::stream<data_t> fifo_C[PE_NUM];

#pragma HLS STREAM variable=fifo_A depth=16
#pragma HLS STREAM variable=fifo_B depth=8
#pragma HLS STREAM variable=fifo_C depth=64
#pragma HLS ARRAY_PARTITION variable=fifo_A complete
#pragma HLS ARRAY_PARTITION variable=fifo_B complete
#pragma HLS ARRAY_PARTITION variable=fifo_C complete

    for (int i0 = 0; i0 < I; i0 += PE_NUM) {
#pragma HLS DATAFLOW
        load_A(A, fifo_A, i0);
        load_B(B, fifo_B[0]);

        mm_pe(fifo_A[0],  fifo_B[0],  fifo_B[1],  fifo_C[0]);
        mm_pe(fifo_A[1],  fifo_B[1],  fifo_B[2],  fifo_C[1]);
        mm_pe(fifo_A[2],  fifo_B[2],  fifo_B[3],  fifo_C[2]);
        mm_pe(fifo_A[3],  fifo_B[3],  fifo_B[4],  fifo_C[3]);
        mm_pe(fifo_A[4],  fifo_B[4],  fifo_B[5],  fifo_C[4]);
        mm_pe(fifo_A[5],  fifo_B[5],  fifo_B[6],  fifo_C[5]);
        mm_pe(fifo_A[6],  fifo_B[6],  fifo_B[7],  fifo_C[6]);
        mm_pe(fifo_A[7],  fifo_B[7],  fifo_B[8],  fifo_C[7]);
        mm_pe(fifo_A[8],  fifo_B[8],  fifo_B[9],  fifo_C[8]);
        mm_pe(fifo_A[9],  fifo_B[9],  fifo_B[10], fifo_C[9]);
        mm_pe(fifo_A[10], fifo_B[10], fifo_B[11], fifo_C[10]);
        mm_pe(fifo_A[11], fifo_B[11], fifo_B[12], fifo_C[11]);
        mm_pe(fifo_A[12], fifo_B[12], fifo_B[13], fifo_C[12]);
        mm_pe(fifo_A[13], fifo_B[13], fifo_B[14], fifo_C[13]);
        mm_pe(fifo_A[14], fifo_B[14], fifo_B[15], fifo_C[14]);
        mm_pe(fifo_A[15], fifo_B[15], fifo_B[16], fifo_C[15]);
        mm_pe(fifo_A[16], fifo_B[16], fifo_B[17], fifo_C[16]);
        mm_pe(fifo_A[17], fifo_B[17], fifo_B[18], fifo_C[17]);
        mm_pe(fifo_A[18], fifo_B[18], fifo_B[19], fifo_C[18]);
        mm_pe(fifo_A[19], fifo_B[19], fifo_B[20], fifo_C[19]);
        mm_pe(fifo_A[20], fifo_B[20], fifo_B[21], fifo_C[20]);
        mm_pe(fifo_A[21], fifo_B[21], fifo_B[22], fifo_C[21]);
        mm_pe(fifo_A[22], fifo_B[22], fifo_B[23], fifo_C[22]);
        mm_pe(fifo_A[23], fifo_B[23], fifo_B[24], fifo_C[23]);
        mm_pe(fifo_A[24], fifo_B[24], fifo_B[25], fifo_C[24]);
        mm_pe(fifo_A[25], fifo_B[25], fifo_B[26], fifo_C[25]);
        mm_pe(fifo_A[26], fifo_B[26], fifo_B[27], fifo_C[26]);
        mm_pe(fifo_A[27], fifo_B[27], fifo_B[28], fifo_C[27]);
        mm_pe(fifo_A[28], fifo_B[28], fifo_B[29], fifo_C[28]);
        mm_pe(fifo_A[29], fifo_B[29], fifo_B[30], fifo_C[29]);
        mm_pe(fifo_A[30], fifo_B[30], fifo_B[31], fifo_C[30]);
        mm_pe(fifo_A[31], fifo_B[31], fifo_B[32], fifo_C[31]);

        drain_B(fifo_B[32]);
        store_C(C, fifo_C, i0);
    }
}