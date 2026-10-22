#include "kernel.h"

// probe: cost of fp op implementations
extern "C" void autosa_mm(data_t A[I][K], data_t B[J][K], data_t C[I][J]) {
#pragma HLS INTERFACE m_axi port=A bundle=gmem0 offset=slave depth=1048576
#pragma HLS INTERFACE m_axi port=B bundle=gmem1 offset=slave depth=1048576
#pragma HLS INTERFACE m_axi port=C bundle=gmem2 offset=slave depth=1048576
  for (int n = 0; n < 1024; n++) {
#pragma HLS PIPELINE II=1
    data_t a0 = A[n][0], a1 = A[n][1], a2 = A[n][2], a3 = A[n][3];
    data_t b0 = B[n][0], b1 = B[n][1], b2 = B[n][2], b3 = B[n][3];
    data_t m_med = a0 * b0;
#pragma HLS BIND_OP variable=m_med op=fmul impl=meddsp
    data_t m_fab = a1 * b1;
#pragma HLS BIND_OP variable=m_fab op=fmul impl=fabric
    data_t m_full = a2 * b2;
#pragma HLS BIND_OP variable=m_full op=fmul impl=fulldsp
    data_t s_fab = m_med + m_fab;
#pragma HLS BIND_OP variable=s_fab op=fadd impl=fabric
    data_t s_full = m_full + a3;
#pragma HLS BIND_OP variable=s_full op=fadd impl=fulldsp
    data_t s3 = s_fab - s_full + b3;
    C[n][0] = s3;
  }
}
