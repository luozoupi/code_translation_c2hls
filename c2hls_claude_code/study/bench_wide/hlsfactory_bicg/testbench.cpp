#include <stdio.h>
#include <string.h>
#include <math.h>
#include <stdlib.h>

#include "bicg.h"


void init_array (int m, int n,
		 double A[ N + 0][M + 0],
		 double r[ N + 0],
		 double p[ M + 0])
{
  int i, j;

  for (i = 0; i < m; i++)
    p[i] = (double)(i % m) / m;
  for (i = 0; i < n; i++) {
    r[i] = (double)(i % n) / n;
    for (j = 0; j < m; j++)
      A[i][j] = (double) (i*(j+1) % n)/n;
  }
}


void print_array(int m, int n,
		 double s[ M + 0],
		 double q[ N + 0])

{
  int i;

  fprintf(stderr, "==BEGIN DUMP_ARRAYS==\n");
  fprintf(stderr, "begin dump: %s", "s");
  for (i = 0; i < m; i++) {
    if (i % 20 == 0) fprintf (stderr, "\n");
    fprintf (stderr, "%0.6lf ", s[i]);
  }
  fprintf(stderr, "\nend   dump: %s\n", "s");
  fprintf(stderr, "begin dump: %s", "q");
  for (i = 0; i < n; i++) {
    if (i % 20 == 0) fprintf (stderr, "\n");
    fprintf (stderr, "%0.6lf ", q[i]);
  }
  fprintf(stderr, "\nend   dump: %s\n", "q");
  fprintf(stderr, "==END   DUMP_ARRAYS==\n");
}

/* ---- strict checker (bench_wide SPEC) ---- */
/* golden: static copy of plain.cpp top function, run on identical input copies */



static void kernel_bicg_ref( 
		 double A[ N + 0][M + 0],
		 double s[ M + 0],
		 double q[ N + 0],
		 double p[ M + 0],
		 double r[ N + 0])
{


    const int n = N;
    const int m = M;

  int i, j;

  for (i = 0; i < m; i++)
    s[i] = 0;
  for (i = 0; i < n; i++)
    {
      q[i] = 0.0;
      for (j = 0; j < m; j++)
	{
	  s[j] = s[j] + r[i] * A[i][j];
	  q[i] = q[i] + A[i][j] * p[j];
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

  int n = N;
  int m = M;


   double A[ N + 0][M + 0];
   double s[ M + 0];
   double q[ N + 0];
   double p[ M + 0];
   double r[ N + 0];


  init_array (m, n,
	      A,
	      r,
	      p);


    /* reference copies */
  static double A_ref[ N + 0][M + 0];
  static double s_ref[ M + 0];
  static double q_ref[ N + 0];
  static double p_ref[ M + 0];
  static double r_ref[ N + 0];
  /* poison pure outputs */
  for (size_t q = 0; q < sizeof(s)/sizeof(double); q++) ((double*)&s)[q] = 12345.0;
  for (size_t q = 0; q < sizeof(q)/sizeof(double); q++) ((double*)&q)[q] = 12345.0;
  memcpy(&A_ref, &A, sizeof(A));
  memcpy(&s_ref, &s, sizeof(s));
  memcpy(&q_ref, &q, sizeof(q));
  memcpy(&p_ref, &p, sizeof(p));
  memcpy(&r_ref, &r, sizeof(r));
  /* golden */
  kernel_bicg_ref (
	       A_ref,
	       s_ref,
	       q_ref,
	       p_ref,
	       r_ref);
  /* kernel under test */
  kernel_bicg (
	       A,
	       s,
	       q,
	       p,
	       r);
  chk_d("s", (const double*)&s, (const double*)&s_ref, sizeof(s)/sizeof(double));
  chk_d("q", (const double*)&q, (const double*)&q_ref, sizeof(q)/sizeof(double));
  if (g_errs) { printf("Failed with %d errors!\n", g_errs); return 1; }
  printf("Passed!\n");
  return 0;
}
