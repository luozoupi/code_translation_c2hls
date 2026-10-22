#include "kernel.h"
#include <hls_stream.h>

struct vec16 { data_t v[16]; };

// probe: sequential read of B, chunked (non-sequential) read of A, chunked write of C
static void rdB(data_t B[J][K], hls::stream<vec16> &s) {
  for (int r = 0; r < J; r++)
    for (int c = 0; c < K / 16; c++) {
#pragma HLS PIPELINE II=1
      vec16 x;
      for (int u = 0; u < 16; u++) x.v[u] = B[r][c * 16 + u];
      s.write(x);
    }
}

static void rdA(data_t A[I][K], hls::stream<vec16> &s) {
  for (int g = 0; g < I / 16; g++)
    for (int c = 0; c < K / 16; c++)
      for (int r = 0; r < 16; r++) {
#pragma HLS PIPELINE II=1
        vec16 x;
        for (int u = 0; u < 16; u++) x.v[u] = A[g * 16 + r][c * 16 + u];
        s.write(x);
      }
}

static void mix(hls::stream<vec16> &a, hls::stream<vec16> &b, hls::stream<vec16> &c) {
  for (int n = 0; n < I * K / 16; n++) {
#pragma HLS PIPELINE II=1
    vec16 x = a.read(), y = b.read(), z;
    for (int u = 0; u < 16; u++) z.v[u] = x.v[u] + y.v[u];
    c.write(z);
  }
}

static void wrC(hls::stream<vec16> &s, data_t C[I][J]) {
  for (int g = 0; g < J / 16; g++)
    for (int r = 0; r < I; r++) {
#pragma HLS PIPELINE II=1
      vec16 x = s.read();
      for (int u = 0; u < 16; u++) C[r][g * 16 + u] = x.v[u];
    }
}

extern "C" void autosa_mm(data_t A[I][K], data_t B[J][K], data_t C[I][J]) {
#pragma HLS INTERFACE m_axi port=A bundle=gmem0 offset=slave depth=1048576
#pragma HLS INTERFACE m_axi port=B bundle=gmem1 offset=slave depth=1048576
#pragma HLS INTERFACE m_axi port=C bundle=gmem2 offset=slave depth=1048576
#pragma HLS DATAFLOW
  hls::stream<vec16> sa("sa"), sb("sb"), sc("sc");
  rdA(A, sa);
  rdB(B, sb);
  mix(sa, sb, sc);
  wrC(sc, C);
}
