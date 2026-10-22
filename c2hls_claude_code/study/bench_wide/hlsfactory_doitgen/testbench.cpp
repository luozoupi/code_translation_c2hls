#include <stdio.h>
#include <string.h>
#include <math.h>
#include <stdlib.h>

#include "doitgen.h"


void init_array(int nr, int nq, int np,
		double A[ NR + 0][NQ + 0][NP + 0],
		double C4[ NP + 0][NP + 0])
{
  int i, j, k;

  for (i = 0; i < nr; i++)
    for (j = 0; j < nq; j++)
      for (k = 0; k < np; k++)
	A[i][j][k] = (double) ((i*j + k)%np) / np;
  for (i = 0; i < np; i++)
    for (j = 0; j < np; j++)
      C4[i][j] = (double) (i*j % np) / np;
}


void print_array(int nr, int nq, int np,
		 double A[ NR + 0][NQ + 0][NP + 0])
{
  int i, j, k;

  fprintf(stderr, "==BEGIN DUMP_ARRAYS==\n");
  fprintf(stderr, "begin dump: %s", "A");
  for (i = 0; i < nr; i++)
    for (j = 0; j < nq; j++)
      for (k = 0; k < np; k++) {
	if ((i*nq*np+j*np+k) % 20 == 0) fprintf (stderr, "\n");
	fprintf (stderr, "%0.6lf ", A[i][j][k]);
      }
  fprintf(stderr, "\nend   dump: %s\n", "A");
  fprintf(stderr, "==END   DUMP_ARRAYS==\n");
}

/* ---- strict checker (bench_wide SPEC) ---- */
/* golden: static copy of plain.cpp top function, run on identical input copies */



static void kernel_doitgen_ref(  
		    double A[ NR + 0][NQ + 0][NP + 0],
		    double C4[ NP + 0][NP + 0],
		    double sum[ NP + 0])
{


    const int nr = NR;
    const int nq = NQ;
    const int np = NP;

  int r, q, p, s;

  for (r = 0; r < nr; r++)
    for (q = 0; q < nq; q++)  {
      for (p = 0; p < np; p++)  {
	sum[p] = 0.0;
	for (s = 0; s < np; s++)
	  sum[p] += A[r][q][s] * C4[s][p];
      }
      for (p = 0; p < np; p++)
	A[r][q][p] = sum[p];
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

  int nr = NR;
  int nq = NQ;
  int np = NP;


  double A[ NR + 0][NQ + 0][NP + 0];
  double sum[ NP + 0];
  double C4[ NP + 0][NP + 0];


  init_array (nr, nq, np,
	      A,
	      C4);


    /* reference copies */
  static double A_ref[ NR + 0][NQ + 0][NP + 0];
  static double sum_ref[ NP + 0];
  static double C4_ref[ NP + 0][NP + 0];
  /* poison pure outputs */
  for (size_t q = 0; q < sizeof(sum)/sizeof(double); q++) ((double*)&sum)[q] = 12345.0;
  memcpy(&A_ref, &A, sizeof(A));
  memcpy(&sum_ref, &sum, sizeof(sum));
  memcpy(&C4_ref, &C4, sizeof(C4));
  /* golden */
  kernel_doitgen_ref (
		  A_ref,
		  C4_ref,
		  sum_ref);
  /* kernel under test */
  kernel_doitgen (
		  A,
		  C4,
		  sum);
  chk_d("A", (const double*)&A, (const double*)&A_ref, sizeof(A)/sizeof(double));
  if (g_errs) { printf("Failed with %d errors!\n", g_errs); return 1; }
  printf("Passed!\n");
  return 0;
}
