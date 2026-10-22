#include "ludcmp.h"


void kernel_ludcmp(
		   double A[ N + 0][N + 0],
		   double b[ N + 0],
		   double x[ N + 0],
		   double y[ N + 0])
{


    const int n = N;

  int i, j, k;

  double w;

  for (i = 0; i < n; i++) {
    for (j = 0; j <i; j++) {
       w = A[i][j];
       for (k = 0; k < j; k++) {
          w -= A[i][k] * A[k][j];
       }
        A[i][j] = w / A[j][j];
    }
   for (j = i; j < n; j++) {
       w = A[i][j];
       for (k = 0; k < i; k++) {
          w -= A[i][k] * A[k][j];
       }
       A[i][j] = w;
    }
  }

  for (i = 0; i < n; i++) {
     w = b[i];
     for (j = 0; j < i; j++)
        w -= A[i][j] * y[j];
     y[i] = w;
  }

   for (i = n-1; i >=0; i--) {
     w = y[i];
     for (j = i+1; j < n; j++)
        w -= A[i][j] * x[j];
     x[i] = w / A[i][i];
  }

}