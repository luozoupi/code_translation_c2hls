#include "durbin.h"


void kernel_durbin(
		   double r[ N + 0],
		   double y[ N + 0])
{
#pragma HLS INLINE off

    const int n = N;

 double z[N];
 double alpha;
 double beta;
 double sum;

 int i,k;

 y[0] = -r[0];
 beta = 1.0;
 alpha = -r[0];

 for (k = 1; k < n; k++) {
   beta = (1-alpha*alpha)*beta;
   sum = 0.0;
   // Serial FP reduction: keep order preserving for bit-exact result.
   for (i=0; i<k; i++) {
#pragma HLS PIPELINE II=1
#pragma HLS LOOP_TRIPCOUNT min=1 max=N
      sum += r[k-i-1]*y[i];
   }
   alpha = - (r[k] + sum)/beta;

   // Independent iterations: read from y, write to z.
   for (i=0; i<k; i++) {
#pragma HLS PIPELINE II=1
#pragma HLS LOOP_TRIPCOUNT min=1 max=N
      z[i] = y[i] + alpha*y[k-i-1];
   }
   for (i=0; i<k; i++) {
#pragma HLS PIPELINE II=1
#pragma HLS LOOP_TRIPCOUNT min=1 max=N
     y[i] = z[i];
   }
   y[k] = alpha;
 }

}


extern "C" {
void workload(
		   double r[ N + 0],
		   double y[ N + 0])
{
#pragma HLS INTERFACE m_axi port=r offset=slave bundle=gmem0
#pragma HLS INTERFACE m_axi port=y offset=slave bundle=gmem1
#pragma HLS INTERFACE s_axilite port=r bundle=control
#pragma HLS INTERFACE s_axilite port=y bundle=control
#pragma HLS INTERFACE s_axilite port=return bundle=control

    kernel_durbin(r, y);
}
}