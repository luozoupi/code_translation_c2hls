#include <stdio.h>
#include <string.h>
#include <math.h>
#include <stdlib.h>

#include "gemver.h"


void init_array (int n,
		 double *alpha,
		 double *beta,
		 double A[ N + 0][N + 0],
		 double u1[ N + 0],
		 double v1[ N + 0],
		 double u2[ N + 0],
		 double v2[ N + 0],
		 double w[ N + 0],
		 double x[ N + 0],
		 double y[ N + 0],
		 double z[ N + 0])
{
  int i, j;

  *alpha = 1.5;
  *beta = 1.2;

  double fn = (double)n;

  for (i = 0; i < n; i++)
    {
      u1[i] = i;
      u2[i] = ((i+1)/fn)/2.0;
      v1[i] = ((i+1)/fn)/4.0;
      v2[i] = ((i+1)/fn)/6.0;
      y[i] = ((i+1)/fn)/8.0;
      z[i] = ((i+1)/fn)/9.0;
      x[i] = 0.0;
      w[i] = 0.0;
      for (j = 0; j < n; j++)
        A[i][j] = (double) (i*j % n) / n;
    }
}


void print_array(int n,
		 double w[ N + 0])
{
  int i;

  fprintf(stderr, "==BEGIN DUMP_ARRAYS==\n");
  fprintf(stderr, "begin dump: %s", "w");
  for (i = 0; i < n; i++) {
    if (i % 20 == 0) fprintf (stderr, "\n");
    fprintf (stderr, "%0.6lf ", w[i]);
  }
  fprintf(stderr, "\nend   dump: %s\n", "w");
  fprintf(stderr, "==END   DUMP_ARRAYS==\n");
}

/* ---- strict checker (bench_wide SPEC) ---- */
/* golden: static copy of plain.cpp top function, run on identical input copies */



static void kernel_gemver_ref(
		   double alpha,
		   double beta,
		   double A[ N + 0][N + 0],
		   double u1[ N + 0],
		   double v1[ N + 0],
		   double u2[ N + 0],
		   double v2[ N + 0],
		   double w[ N + 0],
		   double x[ N + 0],
		   double y[ N + 0],
		   double z[ N + 0])
{


    const int n = N;

  int i, j;


  for (i = 0; i < n; i++)
    for (j = 0; j < n; j++)
      A[i][j] = A[i][j] + u1[i] * v1[j] + u2[i] * v2[j];

  for (i = 0; i < n; i++)
    for (j = 0; j < n; j++)
      x[i] = x[i] + beta * A[j][i] * y[j];

  for (i = 0; i < n; i++)
    x[i] = x[i] + z[i];

  for (i = 0; i < n; i++)
    for (j = 0; j < n; j++)
      w[i] = w[i] +  alpha * A[i][j] * x[j];

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
   double u1[ N + 0];
   double v1[ N + 0];
   double u2[ N + 0];
   double v2[ N + 0];
   double w[ N + 0];
   double x[ N + 0];
   double y[ N + 0];
   double z[ N + 0];


  init_array (n, &alpha, &beta,
	      A,
	      u1,
	      v1,
	      u2,
	      v2,
	      w,
	      x,
	      y,
	      z);


    /* reference copies */
  static double A_ref[ N + 0][N + 0];
  static double u1_ref[ N + 0];
  static double v1_ref[ N + 0];
  static double u2_ref[ N + 0];
  static double v2_ref[ N + 0];
  static double w_ref[ N + 0];
  static double x_ref[ N + 0];
  static double y_ref[ N + 0];
  static double z_ref[ N + 0];
  /* poison pure outputs */
  memcpy(&A_ref, &A, sizeof(A));
  memcpy(&u1_ref, &u1, sizeof(u1));
  memcpy(&v1_ref, &v1, sizeof(v1));
  memcpy(&u2_ref, &u2, sizeof(u2));
  memcpy(&v2_ref, &v2, sizeof(v2));
  memcpy(&w_ref, &w, sizeof(w));
  memcpy(&x_ref, &x, sizeof(x));
  memcpy(&y_ref, &y, sizeof(y));
  memcpy(&z_ref, &z, sizeof(z));
  /* golden */
  kernel_gemver_ref (alpha, beta,
		 A_ref,
		 u1_ref,
		 v1_ref,
		 u2_ref,
		 v2_ref,
		 w_ref,
		 x_ref,
		 y_ref,
		 z_ref);
  /* kernel under test */
  kernel_gemver (alpha, beta,
		 A,
		 u1,
		 v1,
		 u2,
		 v2,
		 w,
		 x,
		 y,
		 z);
  chk_d("w", (const double*)&w, (const double*)&w_ref, sizeof(w)/sizeof(double));
  if (g_errs) { printf("Failed with %d errors!\n", g_errs); return 1; }
  printf("Passed!\n");
  return 0;
}
