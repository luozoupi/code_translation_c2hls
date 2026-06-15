#include "trmm.h"


void kernel_trmm( 
		 double alpha,
		 double A[ M + 0][M + 0],
		 double B[ M + 0][N + 0])
{


    const int m = M;
    const int n = N;

  int i, j, k;
  for (i = 0; i < m; i++)
     for (j = 0; j < n; j++) {
        for (k = i+1; k < m; k++)
           B[i][j] += A[k][i] * B[k][j];
        B[i][j] = alpha * B[i][j];
     }

}