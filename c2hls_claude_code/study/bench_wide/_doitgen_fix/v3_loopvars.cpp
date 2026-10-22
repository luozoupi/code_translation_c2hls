#include "doitgen.h"


void kernel_doitgen(  
		    double A[ NR + 0][NQ + 0][NP + 0],
		    double C4[ NP + 0][NP + 0],
		    double sum[ NP + 0])
{
  #pragma HLS top name=kernel_doitgen

    const int nr = NR;
    const int nq = NQ;
    const int np = NP;


  for (int r = 0; r < nr; r++)
    for (int q = 0; q < nq; q++)  {
      for (int p = 0; p < np; p++)  {
	sum[p] = 0.0;
	for (int s = 0; s < np; s++)
	  sum[p] += A[r][q][s] * C4[s][p];
      }
      for (int p = 0; p < np; p++)
	A[r][q][p] = sum[p];
    }

}