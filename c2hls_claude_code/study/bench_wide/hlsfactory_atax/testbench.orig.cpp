#include <stdio.h>
#include <string.h>
#include <math.h>
#include <stdlib.h>

#include "atax.h"


void init_array (int m, int n,
		 double A[ M + 0][N + 0],
		 double x[ N + 0])
{
  int i, j;
  double fn;
  fn = (double)n;

  for (i = 0; i < n; i++)
      x[i] = 1 + (i / fn);
  for (i = 0; i < m; i++)
    for (j = 0; j < n; j++)
      A[i][j] = (double) ((i+j) % n) / (5*m);
}


void print_array(int n,
		 double y[ N + 0])

{
  int i;

  fprintf(stderr, "==BEGIN DUMP_ARRAYS==\n");
  fprintf(stderr, "begin dump: %s", "y");
  for (i = 0; i < n; i++) {
    if (i % 20 == 0) fprintf (stderr, "\n");
    fprintf (stderr, "%0.6lf ", y[i]);
  }
  fprintf(stderr, "\nend   dump: %s\n", "y");
  fprintf(stderr, "==END   DUMP_ARRAYS==\n");
}


int main(int argc, char** argv)
{

  int m = M;
  int n = N;


   double A[ M + 0][N + 0];
   double x[ N + 0];
   double y[ N + 0];
   double tmp[ M + 0];


  init_array (m, n, A, x);


  kernel_atax (
	       A,
	       x,
	       y,
	       tmp);


  print_array(n, y);


  return 0;
}