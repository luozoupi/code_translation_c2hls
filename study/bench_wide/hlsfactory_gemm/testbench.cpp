#include <stdio.h>
#include <string.h>
#include <math.h>
#include <stdlib.h>

#include "gemm.h"


void init_array(int ni, int nj, int nk,
		double *alpha,
		double *beta,
		double C[ NI + 0][NJ + 0],
		double A[ NI + 0][NK + 0],
		double B[ NK + 0][NJ + 0])
{
  int i, j;

  *alpha = 1.5;
  *beta = 1.2;
  for (i = 0; i < ni; i++)
    for (j = 0; j < nj; j++)
      C[i][j] = (double) ((i*j+1) % ni) / ni;
  for (i = 0; i < ni; i++)
    for (j = 0; j < nk; j++)
      A[i][j] = (double) (i*(j+1) % nk) / nk;
  for (i = 0; i < nk; i++)
    for (j = 0; j < nj; j++)
      B[i][j] = (double) (i*(j+2) % nj) / nj;
}


void print_array(int ni, int nj,
		 double C[ NI + 0][NJ + 0])
{
  int i, j;

  fprintf(stderr, "==BEGIN DUMP_ARRAYS==\n");
  fprintf(stderr, "begin dump: %s", "C");
  for (i = 0; i < ni; i++)
    for (j = 0; j < nj; j++) {
	if ((i * ni + j) % 20 == 0) fprintf (stderr, "\n");
	fprintf (stderr, "%0.6lf ", C[i][j]);
    }
  fprintf(stderr, "\nend   dump: %s\n", "C");
  fprintf(stderr, "==END   DUMP_ARRAYS==\n");
}

/* ---- strict checker (bench_wide SPEC) ---- */
/* golden: static copy of plain.cpp top function, run on identical input copies */



static void kernel_gemm_ref(  
		 double alpha,
		 double beta,
		 double C[ NI + 0][NJ + 0],
		 double A[ NI + 0][NK + 0],
		 double B[ NK + 0][NJ + 0])
{


    const int ni = NI;
    const int nj = NJ;
    const int nk = NK;

  int i, j, k;
  for (i = 0; i < ni; i++) {
    for (j = 0; j < nj; j++)
	C[i][j] *= beta;
    for (k = 0; k < nk; k++) {
       for (j = 0; j < nj; j++)
	  C[i][j] += alpha * A[i][k] * B[k][j];
    }
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

  int ni = NI;
  int nj = NJ;
  int nk = NK;


  double alpha;
  double beta;
  double C[ NI + 0][NJ + 0];
  double A[ NI + 0][NK + 0];
  double B[ NK + 0][NJ + 0];


  init_array (ni, nj, nk, &alpha, &beta,
	      C,
	      A,
	      B);


    /* reference copies */
  static double C_ref[ NI + 0][NJ + 0];
  static double A_ref[ NI + 0][NK + 0];
  static double B_ref[ NK + 0][NJ + 0];
  /* poison pure outputs */
  memcpy(&C_ref, &C, sizeof(C));
  memcpy(&A_ref, &A, sizeof(A));
  memcpy(&B_ref, &B, sizeof(B));
  /* golden */
  kernel_gemm_ref (
	       alpha, beta,
	       C_ref,
	       A_ref,
	       B_ref);
  /* kernel under test */
  kernel_gemm (
	       alpha, beta,
	       C,
	       A,
	       B);
  chk_d("C", (const double*)&C, (const double*)&C_ref, sizeof(C)/sizeof(double));
  if (g_errs) { printf("Failed with %d errors!\n", g_errs); return 1; }
  printf("Passed!\n");
  return 0;
}
