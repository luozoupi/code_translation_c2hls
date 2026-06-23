#include "nussinov.h"


extern "C" {

void kernel_nussinov( char seq[ N + 0],
			   int table[ N + 0][N + 0])
{
#pragma HLS INTERFACE m_axi port=seq offset=slave bundle=gmem0
#pragma HLS INTERFACE m_axi port=table offset=slave bundle=gmem1
#pragma HLS INTERFACE s_axilite port=seq bundle=control
#pragma HLS INTERFACE s_axilite port=table bundle=control
#pragma HLS INTERFACE s_axilite port=return bundle=control

    const int n = N;

  int i, j, k;

  // Stage inputs into local buffers to enable fast on-chip access and reuse.
  static char seq_local[N];
  static int  table_local[N][N];
#pragma HLS ARRAY_PARTITION variable=table_local cyclic factor=2 dim=2

  load_seq:
  for (i = 0; i < n; i++) {
#pragma HLS PIPELINE II=1
    seq_local[i] = seq[i];
  }

  load_table:
  for (i = 0; i < n; i++) {
    for (j = 0; j < n; j++) {
#pragma HLS PIPELINE II=1
      table_local[i][j] = table[i][j];
    }
  }

 outer_i:
 for (i = n-1; i >= 0; i--) {
  inner_j:
  for (j=i+1; j<n; j++) {

   if (j-1>=0)
      table_local[i][j] = ((table_local[i][j] >= table_local[i][j-1]) ? table_local[i][j] : table_local[i][j-1]);
   if (i+1<n)
      table_local[i][j] = ((table_local[i][j] >= table_local[i+1][j]) ? table_local[i][j] : table_local[i+1][j]);

   if (j-1>=0 && i+1<n) {

     if (i<j-1)
        table_local[i][j] = ((table_local[i][j] >= table_local[i+1][j-1]+(((seq_local[i])+(seq_local[j])) == 3 ? 1 : 0)) ? table_local[i][j] : table_local[i+1][j-1]+(((seq_local[i])+(seq_local[j])) == 3 ? 1 : 0));
     else
        table_local[i][j] = ((table_local[i][j] >= table_local[i+1][j-1]) ? table_local[i][j] : table_local[i+1][j-1]);
   }

   inner_k:
   for (k=i+1; k<j; k++) {
#pragma HLS PIPELINE II=1
      table_local[i][j] = ((table_local[i][j] >= table_local[i][k] + table_local[k+1][j]) ? table_local[i][j] : table_local[i][k] + table_local[k+1][j]);
   }
  }
 }

  // Write results back to global memory.
  store_table:
  for (i = 0; i < n; i++) {
    for (j = 0; j < n; j++) {
#pragma HLS PIPELINE II=1
      table[i][j] = table_local[i][j];
    }
  }

}

}