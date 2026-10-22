#include <stdio.h>
#include <string.h>
#include <math.h>
#include <stdlib.h>

#include "durbin.h"


void init_array (int n,
		 double r[ N + 0])
{
  int i, j;

  for (i = 0; i < n; i++)
    {
      r[i] = (n+1-i);
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



static void kernel_durbin_ref(
		   double r[ N + 0],
		   double y[ N + 0])
{


    const int n = N;

 double z[N];
 double alpha;
 double beta;
 double sum;

 int i,k;

 y[0] = -r[0];
 beta = 1.0;
 alpha = -r[0];

 for (k = 1; k < n; k++) {
   beta = (1-alpha*alpha)*beta;
   sum = 0.0;
   for (i=0; i<k; i++) {
      sum += r[k-i-1]*y[i];
   }
   alpha = - (r[k] + sum)/beta;

   for (i=0; i<k; i++) {
      z[i] = y[i] + alpha*y[k-i-1];
   }
   for (i=0; i<k; i++) {
     y[i] = z[i];
   }
   y[k] = alpha;
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


   double r[ N + 0];
   double y[ N + 0];


  init_array (n, r);


    /* reference copies */
  static double r_ref[ N + 0];
  static double y_ref[ N + 0];
  /* poison pure outputs */
  for (size_t q = 0; q < sizeof(y)/sizeof(double); q++) ((double*)&y)[q] = 12345.0;
  memcpy(&r_ref, &r, sizeof(r));
  memcpy(&y_ref, &y, sizeof(y));
  /* golden */
  kernel_durbin_ref (
		 r_ref,
		 y_ref);
  /* kernel under test */
  kernel_durbin (
		 r,
		 y);
  chk_d("y", (const double*)&y, (const double*)&y_ref, sizeof(y)/sizeof(double));
  if (g_errs) { printf("Failed with %d errors!\n", g_errs); return 1; }
  printf("Passed!\n");
  return 0;
}
