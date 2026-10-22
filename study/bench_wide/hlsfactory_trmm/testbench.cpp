#include <stdio.h>
#include <string.h>
#include <math.h>
#include <stdlib.h>

#include "trmm.h"


void init_array(int m, int n,
		double *alpha,
		double A[ M + 0][M + 0],
		double B[ M + 0][N + 0])
{
  int i, j;

  *alpha = 1.5;
  for (i = 0; i < m; i++) {
    for (j = 0; j < i; j++) {
      A[i][j] = (double)((i+j) % m)/m;
    }
    A[i][i] = 1.0;
    for (j = 0; j < n; j++) {
      B[i][j] = (double)((n+(i-j)) % n)/n;
    }
 }

}


void print_array(int m, int n,
		 double B[ M + 0][N + 0])
{
  int i, j;

  fprintf(stderr, "==BEGIN DUMP_ARRAYS==\n");
  fprintf(stderr, "begin dump: %s", "B");
  for (i = 0; i < m; i++)
    for (j = 0; j < n; j++) {
	if ((i * m + j) % 20 == 0) fprintf (stderr, "\n");
	fprintf (stderr, "%0.6lf ", B[i][j]);
    }
  fprintf(stderr, "\nend   dump: %s\n", "B");
  fprintf(stderr, "==END   DUMP_ARRAYS==\n");
}

/* ---- strict checker (bench_wide SPEC) ---- */
/* golden: static copy of plain.cpp top function, run on identical input copies */



static void kernel_trmm_ref( 
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

  int m = M;
  int n = N;


  double alpha;
  double A[ M + 0][M + 0];
  double B[ M + 0][N + 0];


  init_array (m, n, &alpha, A, B);


    /* reference copies */
  static double A_ref[ M + 0][M + 0];
  static double B_ref[ M + 0][N + 0];
  /* poison pure outputs */
  memcpy(&A_ref, &A, sizeof(A));
  memcpy(&B_ref, &B, sizeof(B));
  /* golden */
  kernel_trmm_ref (alpha, A_ref, B_ref);
  /* kernel under test */
  kernel_trmm (alpha, A, B);
  chk_d("B", (const double*)&B, (const double*)&B_ref, sizeof(B)/sizeof(double));
  if (g_errs) { printf("Failed with %d errors!\n", g_errs); return 1; }
  printf("Passed!\n");
  return 0;
}
