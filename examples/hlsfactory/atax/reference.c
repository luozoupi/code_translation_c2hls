/* Reduced-shape C11/C++17 adapter of the preserved HLSFactory/PolyBench export.
 * Kernel body and double precision retained; see ../README.md and ../LICENSE.PolyBench.txt.
 * Portability baseline, not an LLM response or an optimized implementation.
 */
#include "kernel.h"


void kernel_atax( 
		 double A[ M + 0][N + 0],
		 double x[ N + 0],
		 double y[ N + 0],
		 double tmp[ M + 0])
{


    const int m = M;
    const int n = N;

  int i, j;

  for (i = 0; i < n; i++)
    y[i] = 0;
  for (i = 0; i < m; i++)
    {
      tmp[i] = 0.0;
      for (j = 0; j < n; j++)
	tmp[i] = tmp[i] + A[i][j] * x[j];
      for (j = 0; j < n; j++)
	y[j] = y[j] + A[i][j] * tmp[i];
    }

}