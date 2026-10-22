#include <stdio.h>
#include <string.h>
#include <math.h>
#include <stdlib.h>

#include "trisolv.h"


void init_array(int n,
		double L[ N + 0][N + 0],
		double x[ N + 0],
		double b[ N + 0])
{
  int i, j;

  for (i = 0; i < n; i++)
    {
      x[i] = - 999;
      b[i] =  i ;
      for (j = 0; j <= i; j++)
	L[i][j] = (double) (i+n-j+1)*2/n;
    }
}


void print_array(int n,
		 double x[ N + 0])

{
  int i;

  fprintf(stderr, "==BEGIN DUMP_ARRAYS==\n");
  fprintf(stderr, "begin dump: %s", "x");
  for (i = 0; i < n; i++) {
    fprintf (stderr, "%0.6lf ", x[i]);
    if (i % 20 == 0) fprintf (stderr, "\n");
  }
  fprintf(stderr, "\nend   dump: %s\n", "x");
  fprintf(stderr, "==END   DUMP_ARRAYS==\n");
}

/* ---- strict checker (bench_wide SPEC) ---- */
/* golden: static copy of plain.cpp top function, run on identical input copies */



static void kernel_trisolv_ref(
		    double L[ N + 0][N + 0],
		    double x[ N + 0],
		    double b[ N + 0])
{


    const int n = N;

  int i, j;

  for (i = 0; i < n; i++)
    {
      x[i] = b[i];
      for (j = 0; j <i; j++)
        x[i] -= L[i][j] * x[j];
      x[i] = x[i] / L[i][i];
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


   double L[ N + 0][N + 0];
   double x[ N + 0];
   double b[ N + 0];


  init_array (n, L, x, b);


    /* reference copies */
  static double L_ref[ N + 0][N + 0];
  static double x_ref[ N + 0];
  static double b_ref[ N + 0];
  /* poison pure outputs */
  for (size_t q = 0; q < sizeof(x)/sizeof(double); q++) ((double*)&x)[q] = 12345.0;
  memcpy(&L_ref, &L, sizeof(L));
  memcpy(&x_ref, &x, sizeof(x));
  memcpy(&b_ref, &b, sizeof(b));
  /* golden */
  kernel_trisolv_ref (L_ref, x_ref, b_ref);
  /* kernel under test */
  kernel_trisolv (L, x, b);
  chk_d("x", (const double*)&x, (const double*)&x_ref, sizeof(x)/sizeof(double));
  if (g_errs) { printf("Failed with %d errors!\n", g_errs); return 1; }
  printf("Passed!\n");
  return 0;
}
