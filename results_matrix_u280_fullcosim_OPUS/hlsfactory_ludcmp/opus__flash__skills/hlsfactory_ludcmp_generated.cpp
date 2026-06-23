#include "ludcmp.h"


extern "C" {
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

  // Local staging buffers to enable fast on-chip access and partitioning
  static double A_l[N][N];
#pragma HLS ARRAY_PARTITION variable=A_l cyclic factor=8 dim=2
  static double b_l[N];
  static double x_l[N];
  static double y_l[N];
#pragma HLS ARRAY_PARTITION variable=y_l cyclic factor=8 dim=1
#pragma HLS ARRAY_PARTITION variable=x_l cyclic factor=8 dim=1

  // Stage A from global memory
LOAD_A_I:
  for (i = 0; i < n; i++) {
  LOAD_A_J:
    for (j = 0; j < n; j++) {
#pragma HLS PIPELINE II=1
      A_l[i][j] = A[i][j];
    }
  }
LOAD_B:
  for (i = 0; i < n; i++) {
#pragma HLS PIPELINE II=1
    b_l[i] = b[i];
  }

  // LU decomposition
  for (i = 0; i < n; i++) {
    for (j = 0; j < i; j++) {
       w = A_l[i][j];
    LU_K_LOWER:
       for (k = 0; k < j; k++) {
#pragma HLS PIPELINE II=1
          w -= A_l[i][k] * A_l[k][j];
       }
        A_l[i][j] = w / A_l[j][j];
    }
   for (j = i; j < n; j++) {
       w = A_l[i][j];
    LU_K_UPPER:
       for (k = 0; k < i; k++) {
#pragma HLS PIPELINE II=1
          w -= A_l[i][k] * A_l[k][j];
       }
       A_l[i][j] = w;
    }
  }

  // Forward substitution
  for (i = 0; i < n; i++) {
     w = b_l[i];
  FWD_J:
     for (j = 0; j < i; j++) {
#pragma HLS PIPELINE II=1
        w -= A_l[i][j] * y_l[j];
     }
     y_l[i] = w;
  }

  // Backward substitution
   for (i = n-1; i >=0; i--) {
     w = y_l[i];
  BWD_J:
     for (j = i+1; j < n; j++) {
#pragma HLS PIPELINE II=1
        w -= A_l[i][j] * x_l[j];
     }
     x_l[i] = w / A_l[i][i];
  }

  // Write results back to global memory
STORE_A_I:
  for (i = 0; i < n; i++) {
  STORE_A_J:
    for (j = 0; j < n; j++) {
#pragma HLS PIPELINE II=1
      A[i][j] = A_l[i][j];
    }
  }
STORE_XY:
  for (i = 0; i < n; i++) {
#pragma HLS PIPELINE II=1
    x[i] = x_l[i];
    y[i] = y_l[i];
  }

}
}