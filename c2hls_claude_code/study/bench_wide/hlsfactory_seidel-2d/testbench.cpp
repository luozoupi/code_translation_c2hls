#include <stdio.h>
#include <string.h>
#include <math.h>
#include <stdlib.h>

#include "seidel-2d.h"


void init_array (int n,
		 double A[ N + 0][N + 0])
{
  int i, j;

  for (i = 0; i < n; i++)
    for (j = 0; j < n; j++)
      A[i][j] = ((double) i*(j+2) + 2) / n;
}


void print_array(int n,
		 double A[ N + 0][N + 0])

{
  int i, j;

  fprintf(stderr, "==BEGIN DUMP_ARRAYS==\n");
  fprintf(stderr, "begin dump: %s", "A");
  for (i = 0; i < n; i++)
    for (j = 0; j < n; j++) {
      if ((i * n + j) % 20 == 0) fprintf(stderr, "\n");
      fprintf(stderr, "%0.6lf ", A[i][j]);
    }
  fprintf(stderr, "\nend   dump: %s\n", "A");
  fprintf(stderr, "==END   DUMP_ARRAYS==\n");
}

/* ---- strict checker (bench_wide SPEC) ---- */
/* golden: static copy of plain.cpp top function, run on identical input copies */



static void kernel_seidel_2d_ref(
		      
		      double A[ N + 0][N + 0])
{


    const int n = N;
    const int tsteps = TSTEPS;

  int t, i, j;

  for (t = 0; t <= tsteps - 1; t++)
    for (i = 1; i<= n - 2; i++)
      for (j = 1; j <= n - 2; j++)
	A[i][j] = (A[i-1][j-1] + A[i-1][j] + A[i-1][j+1]
		   + A[i][j-1] + A[i][j] + A[i][j+1]
		   + A[i+1][j-1] + A[i+1][j] + A[i+1][j+1])/9.0;

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
  int tsteps = TSTEPS;


   double A[ N + 0][N + 0];


  init_array (n, A);


    /* reference copies */
  static double A_ref[ N + 0][N + 0];
  /* init_array data is bilinear: the 9-point average leaves it (almost) unchanged, so a no-op kernel would pass. Perturb deterministically. */
  for (int i = 0; i < N; i++) for (int j = 0; j < N; j++) A[i][j] += (double)((i*7 + j*13) % 17) / 17.0;
  /* poison pure outputs */
  memcpy(&A_ref, &A, sizeof(A));
  /* golden */
  kernel_seidel_2d_ref ( A_ref);
  /* kernel under test */
  kernel_seidel_2d ( A);
  chk_d("A", (const double*)&A, (const double*)&A_ref, sizeof(A)/sizeof(double));
  if (g_errs) { printf("Failed with %d errors!\n", g_errs); return 1; }
  printf("Passed!\n");
  return 0;
}
