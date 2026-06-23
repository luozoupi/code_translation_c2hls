#include "ludcmp.h"


void kernel_ludcmp(
		   double A[ N + 0][N + 0],
		   double b[ N + 0],
		   double x[ N + 0],
		   double y[ N + 0])
{
#pragma HLS INTERFACE m_axi port=A offset=slave bundle=gmem0
#pragma HLS INTERFACE m_axi port=b offset=slave bundle=gmem1
#pragma HLS INTERFACE m_axi port=x offset=slave bundle=gmem2
#pragma HLS INTERFACE m_axi port=y offset=slave bundle=gmem3
#pragma HLS INTERFACE s_axilite port=A bundle=control
#pragma HLS INTERFACE s_axilite port=b bundle=control
#pragma HLS INTERFACE s_axilite port=x bundle=control
#pragma HLS INTERFACE s_axilite port=y bundle=control
#pragma HLS INTERFACE s_axilite port=return bundle=control

    const int n = N;

  int i, j, k;

  double w;

  for (i = 0; i < n; i++) {
    for (j = 0; j <i; j++) {
       w = A[i][j];
       for (k = 0; k < j; k++) {
#pragma HLS PIPELINE II=1
          w -= A[i][k] * A[k][j];
       }
        A[i][j] = w / A[j][j];
    }
   for (j = i; j < n; j++) {
       w = A[i][j];
       for (k = 0; k < i; k++) {
#pragma HLS PIPELINE II=1
          w -= A[i][k] * A[k][j];
       }
       A[i][j] = w;
    }
  }

  for (i = 0; i < n; i++) {
     w = b[i];
     for (j = 0; j < i; j++) {
#pragma HLS PIPELINE II=1
        w -= A[i][j] * y[j];
     }
     y[i] = w;
  }

   for (i = n-1; i >=0; i--) {
     w = y[i];
     for (j = i+1; j < n; j++) {
#pragma HLS PIPELINE II=1
        w -= A[i][j] * x[j];
     }
     x[i] = w / A[i][i];
  }

}