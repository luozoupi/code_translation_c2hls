#include <stdio.h>
#include <string.h>
#include <math.h>
#include <stdlib.h>

#include "gesummv.h"


void init_array(int n,
		double *alpha,
		double *beta,
		double A[ N + 0][N + 0],
		double B[ N + 0][N + 0],
		double x[ N + 0])
{
  int i, j;

  *alpha = 1.5;
  *beta = 1.2;
  for (i = 0; i < n; i++)
    {
      x[i] = (double)( i % n) / n;
      for (j = 0; j < n; j++) {
	A[i][j] = (double) ((i*j+1) % n) / n;
	B[i][j] = (double) ((i*j+2) % n) / n;
      }
    }
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

/* ---- strict checker (bench_wide SPEC) ---- */
/* golden: static copy of plain.cpp top function, run on identical input copies */



static void kernel_gesummv_ref(
		    double alpha,
		    double beta,
		    double A[ N + 0][N + 0],
		    double B[ N + 0][N + 0],
		    double tmp[ N + 0],
		    double x[ N + 0],
		    double y[ N + 0])
{


    const int n = N;

  int i, j;

  for (i = 0; i < n; i++)
    {
      tmp[i] = 0.0;
      y[i] = 0.0;
      for (j = 0; j < n; j++)
	{
	  tmp[i] = A[i][j] * x[j] + tmp[i];
	  y[i] = B[i][j] * x[j] + y[i];
	}
      y[i] = alpha * tmp[i] + beta * y[i];
    }

}
static int g_errs = 0;
static void chk_d(const char *nm, const double *d, const double *r, size_t n)
{
  for (size_t q = 0; q < n; q++) {
    int bad = 0;
    if (isnan(d[q]) || isnan(r[q])) bad = 1; /* NaN in kernel (or golden) output is an error */
    else bad = !(fabs(d[q] - r[q]) <= 1e-6 * fmax(1.0, fabs(r[q])));
    if (bad) { if (g_errs < 10) printf("MISMATCH %s[%zu]: got %.9g expected %.9g\n", nm, q, d[q], r[q]); g_errs++; }
  }
}
static void chk_i(const char *nm, const int *d, const int *r, size_t n)
{
  for (size_t q = 0; q < n; q++)
    if (d[q] != r[q]) { if (g_errs < 10) printf("MISMATCH %s[%zu]: got %d expected %d\n", nm, q, d[q], r[q]); g_errs++; }
}
static void chk_c(const char *nm, const char *d, const char *r, size_t n)
{
  for (size_t q = 0; q < n; q++)
    if (d[q] != r[q]) { if (g_errs < 10) printf("MISMATCH %s[%zu]: got %d expected %d\n", nm, q, (int)d[q], (int)r[q]); g_errs++; }
}

int main(int argc, char** argv)
{

  int n = N;


  double alpha;
  double beta;
   double A[ N + 0][N + 0];
   double B[ N + 0][N + 0];
   double tmp[ N + 0];
   double x[ N + 0];
   double y[ N + 0];


  init_array (n, &alpha, &beta,
	      A,
	      B,
	      x);


    /* reference copies */
  static double A_ref[ N + 0][N + 0];
  static double B_ref[ N + 0][N + 0];
  static double tmp_ref[ N + 0];
  static double x_ref[ N + 0];
  static double y_ref[ N + 0];
  /* poison pure outputs */
  for (size_t q = 0; q < sizeof(tmp)/sizeof(double); q++) ((double*)&tmp)[q] = 12345.0;
  for (size_t q = 0; q < sizeof(y)/sizeof(double); q++) ((double*)&y)[q] = 12345.0;
  memcpy(&A_ref, &A, sizeof(A));
  memcpy(&B_ref, &B, sizeof(B));
  memcpy(&tmp_ref, &tmp, sizeof(tmp));
  memcpy(&x_ref, &x, sizeof(x));
  memcpy(&y_ref, &y, sizeof(y));
  /* golden */
  kernel_gesummv_ref (alpha, beta,
		  A_ref,
		  B_ref,
		  tmp_ref,
		  x_ref,
		  y_ref);
  /* kernel under test */
  kernel_gesummv (alpha, beta,
		  A,
		  B,
		  tmp,
		  x,
		  y);
  chk_d("y", (const double*)&y, (const double*)&y_ref, sizeof(y)/sizeof(double));
  if (g_errs) { printf("Failed with %d errors!\n", g_errs); return 1; }
  printf("Passed!\n");
  return 0;
}
