#include "fdtd-2d.h"


void kernel_fdtd_2d(
		    
		    
		    double ex[ NX + 0][NY + 0],
		    double ey[ NX + 0][NY + 0],
		    double hz[ NX + 0][NY + 0],
		    double _fict_[ TMAX + 0])
{
#pragma HLS INTERFACE m_axi port=ex offset=slave bundle=gmem0
#pragma HLS INTERFACE m_axi port=ey offset=slave bundle=gmem1
#pragma HLS INTERFACE m_axi port=hz offset=slave bundle=gmem2
#pragma HLS INTERFACE m_axi port=_fict_ offset=slave bundle=gmem3
#pragma HLS INTERFACE s_axilite port=ex bundle=control
#pragma HLS INTERFACE s_axilite port=ey bundle=control
#pragma HLS INTERFACE s_axilite port=hz bundle=control
#pragma HLS INTERFACE s_axilite port=_fict_ bundle=control
#pragma HLS INTERFACE s_axilite port=return bundle=control


    const int tmax = TMAX;
    const int nx = NX;
    const int ny = NY;

  int t, i, j;


  for(t = 0; t < tmax; t++)
    {
      for (j = 0; j < ny; j++) {
#pragma HLS PIPELINE II=1
	ey[0][j] = _fict_[t];
      }
      for (i = 1; i < nx; i++)
	for (j = 0; j < ny; j++) {
#pragma HLS PIPELINE II=1
	  ey[i][j] = ey[i][j] - 0.5*(hz[i][j]-hz[i-1][j]);
	}
      for (i = 0; i < nx; i++)
	for (j = 1; j < ny; j++) {
#pragma HLS PIPELINE II=1
	  ex[i][j] = ex[i][j] - 0.5*(hz[i][j]-hz[i][j-1]);
	}
      for (i = 0; i < nx - 1; i++)
	for (j = 0; j < ny - 1; j++) {
#pragma HLS PIPELINE II=1
	  hz[i][j] = hz[i][j] - 0.7*  (ex[i][j+1] - ex[i][j] +
				       ey[i+1][j] - ey[i][j]);
	}
    }

}