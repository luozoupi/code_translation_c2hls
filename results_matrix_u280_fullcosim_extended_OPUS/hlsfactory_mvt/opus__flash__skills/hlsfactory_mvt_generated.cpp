#include "mvt.h"


void kernel_mvt(
		double x1[ N + 0],
		double x2[ N + 0],
		double y_1[ N + 0],
		double y_2[ N + 0],
		double A[ N + 0][N + 0])
{
#pragma HLS INLINE off

    const int n = N;

  int i, j;

  loop_i1: for (i = 0; i < n; i++) {
    double acc1 = x1[i];
    loop_j1: for (j = 0; j < n; j++) {
#pragma HLS PIPELINE II=1
      acc1 = acc1 + A[i][j] * y_1[j];
    }
    x1[i] = acc1;
  }

  loop_i2: for (i = 0; i < n; i++) {
    double acc2 = x2[i];
    loop_j2: for (j = 0; j < n; j++) {
#pragma HLS PIPELINE II=1
      acc2 = acc2 + A[j][i] * y_2[j];
    }
    x2[i] = acc2;
  }

}


extern "C" {
void workload(
		double x1[ N + 0],
		double x2[ N + 0],
		double y_1[ N + 0],
		double y_2[ N + 0],
		double A[ N + 0][N + 0])
{
#pragma HLS INTERFACE m_axi port=x1  offset=slave bundle=gmem0
#pragma HLS INTERFACE m_axi port=x2  offset=slave bundle=gmem1
#pragma HLS INTERFACE m_axi port=y_1 offset=slave bundle=gmem2
#pragma HLS INTERFACE m_axi port=y_2 offset=slave bundle=gmem3
#pragma HLS INTERFACE m_axi port=A   offset=slave bundle=gmem4

#pragma HLS INTERFACE s_axilite port=x1  bundle=control
#pragma HLS INTERFACE s_axilite port=x2  bundle=control
#pragma HLS INTERFACE s_axilite port=y_1 bundle=control
#pragma HLS INTERFACE s_axilite port=y_2 bundle=control
#pragma HLS INTERFACE s_axilite port=A   bundle=control
#pragma HLS INTERFACE s_axilite port=return bundle=control

    kernel_mvt(x1, x2, y_1, y_2, A);
}
}