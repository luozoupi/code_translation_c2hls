#include "lu.h"


void kernel_lu(
	       double A[ N + 0][N + 0])
{
#pragma HLS INLINE off

    const int n = N;

  int i, j, k;

  for (i = 0; i < n; i++) {
    for (j = 0; j <i; j++) {
       for (k = 0; k < j; k++) {
#pragma HLS PIPELINE II=1
          A[i][j] -= A[i][k] * A[k][j];
       }
        A[i][j] /= A[j][j];
    }
   for (j = i; j < n; j++) {
       for (k = 0; k < i; k++) {
#pragma HLS PIPELINE II=1
          A[i][j] -= A[i][k] * A[k][j];
       }
    }
  }
}

extern "C" {
void workload(
	       double A[ N + 0][N + 0])
{
#pragma HLS INTERFACE m_axi port=A offset=slave bundle=gmem
#pragma HLS INTERFACE s_axilite port=A bundle=control
#pragma HLS INTERFACE s_axilite port=return bundle=control

    kernel_lu(A);
}
}