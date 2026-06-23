#include "nussinov.h"


void kernel_nussinov( char seq[ N + 0],
			   int table[ N + 0][N + 0])
{
#pragma HLS INTERFACE m_axi port=seq offset=slave bundle=gmem0
#pragma HLS INTERFACE m_axi port=table offset=slave bundle=gmem1
#pragma HLS INTERFACE s_axilite port=seq bundle=control
#pragma HLS INTERFACE s_axilite port=table bundle=control
#pragma HLS INTERFACE s_axilite port=return bundle=control

    const int n = N;

  // Local buffers for parallel access
  static char l_seq[N];
#pragma HLS ARRAY_PARTITION variable=l_seq complete dim=1
  static int l_table[N][N];
#pragma HLS ARRAY_PARTITION variable=l_table cyclic factor=8 dim=2

  int i, j, k;

  // Load inputs into local buffers
  for (i = 0; i < n; i++) {
#pragma HLS PIPELINE II=1
    l_seq[i] = seq[i];
  }

  for (i = 0; i < n; i++) {
    for (j = 0; j < n; j++) {
#pragma HLS PIPELINE II=1
      l_table[i][j] = table[i][j];
    }
  }

 for (i = n-1; i >= 0; i--) {
  for (j=i+1; j<n; j++) {

   if (j-1>=0)
      l_table[i][j] = ((l_table[i][j] >= l_table[i][j-1]) ? l_table[i][j] : l_table[i][j-1]);
   if (i+1<n)
      l_table[i][j] = ((l_table[i][j] >= l_table[i+1][j]) ? l_table[i][j] : l_table[i+1][j]);

   if (j-1>=0 && i+1<n) {

     if (i<j-1)
        l_table[i][j] = ((l_table[i][j] >= l_table[i+1][j-1]+(((l_seq[i])+(l_seq[j])) == 3 ? 1 : 0)) ? l_table[i][j] : l_table[i+1][j-1]+(((l_seq[i])+(l_seq[j])) == 3 ? 1 : 0));
     else
        l_table[i][j] = ((l_table[i][j] >= l_table[i+1][j-1]) ? l_table[i][j] : l_table[i+1][j-1]);
   }

   for (k=i+1; k<j; k++) {
#pragma HLS PIPELINE II=1
      l_table[i][j] = ((l_table[i][j] >= l_table[i][k] + l_table[k+1][j]) ? l_table[i][j] : l_table[i][k] + l_table[k+1][j]);
   }
  }
 }

  // Write back results
  for (i = 0; i < n; i++) {
    for (j = 0; j < n; j++) {
#pragma HLS PIPELINE II=1
      table[i][j] = l_table[i][j];
    }
  }

}