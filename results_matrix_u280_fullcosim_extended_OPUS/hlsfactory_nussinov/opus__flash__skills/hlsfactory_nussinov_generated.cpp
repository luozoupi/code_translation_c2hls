#include "nussinov.h"


void kernel_nussinov( char seq[ N + 0],
			   int table[ N + 0][N + 0])
{
#pragma HLS INLINE off

    const int n = N;

  int i, j, k;

 for (i = n-1; i >= 0; i--) {
  for (j=i+1; j<n; j++) {

   if (j-1>=0)
      table[i][j] = ((table[i][j] >= table[i][j-1]) ? table[i][j] : table[i][j-1]);
   if (i+1<n)
      table[i][j] = ((table[i][j] >= table[i+1][j]) ? table[i][j] : table[i+1][j]);

   if (j-1>=0 && i+1<n) {

     if (i<j-1)
        table[i][j] = ((table[i][j] >= table[i+1][j-1]+(((seq[i])+(seq[j])) == 3 ? 1 : 0)) ? table[i][j] : table[i+1][j-1]+(((seq[i])+(seq[j])) == 3 ? 1 : 0));
     else
        table[i][j] = ((table[i][j] >= table[i+1][j-1]) ? table[i][j] : table[i+1][j-1]);
   }

   for (k=i+1; k<j; k++) {
#pragma HLS PIPELINE II=1
      table[i][j] = ((table[i][j] >= table[i][k] + table[k+1][j]) ? table[i][j] : table[i][k] + table[k+1][j]);
   }
  }
 }

}

extern "C" {
void workload( char seq[ N + 0],
			   int table[ N + 0][N + 0])
{
#pragma HLS INTERFACE m_axi port=seq offset=slave bundle=gmem0
#pragma HLS INTERFACE m_axi port=table offset=slave bundle=gmem1
#pragma HLS INTERFACE s_axilite port=seq bundle=control
#pragma HLS INTERFACE s_axilite port=table bundle=control
#pragma HLS INTERFACE s_axilite port=return bundle=control

    kernel_nussinov(seq, table);
}
}