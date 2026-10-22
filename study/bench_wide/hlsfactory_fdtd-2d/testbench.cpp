#include <stdio.h>
#include <string.h>
#include <math.h>
#include <stdlib.h>

#include "fdtd-2d.h"


void init_array (int tmax,
		 int nx,
		 int ny,
		 double ex[ NX + 0][NY + 0],
		 double ey[ NX + 0][NY + 0],
		 double hz[ NX + 0][NY + 0],
		 double _fict_[ TMAX + 0])
{
  int i, j;

  for (i = 0; i < tmax; i++)
    _fict_[i] = (double) i;
  for (i = 0; i < nx; i++)
    for (j = 0; j < ny; j++)
      {
	ex[i][j] = ((double) i*(j+1)) / nx;
	ey[i][j] = ((double) i*(j+2)) / ny;
	hz[i][j] = ((double) i*(j+3)) / nx;
      }
}


void print_array(int nx,
		 int ny,
		 double ex[ NX + 0][NY + 0],
		 double ey[ NX + 0][NY + 0],
		 double hz[ NX + 0][NY + 0])
{
  int i, j;

  fprintf(stderr, "==BEGIN DUMP_ARRAYS==\n");
  fprintf(stderr, "begin dump: %s", "ex");
  for (i = 0; i < nx; i++)
    for (j = 0; j < ny; j++) {
      if ((i * nx + j) % 20 == 0) fprintf(stderr, "\n");
      fprintf(stderr, "%0.6lf ", ex[i][j]);
    }
  fprintf(stderr, "\nend   dump: %s\n", "ex");
  fprintf(stderr, "==END   DUMP_ARRAYS==\n");

  fprintf(stderr, "begin dump: %s", "ey");
  for (i = 0; i < nx; i++)
    for (j = 0; j < ny; j++) {
      if ((i * nx + j) % 20 == 0) fprintf(stderr, "\n");
      fprintf(stderr, "%0.6lf ", ey[i][j]);
    }
  fprintf(stderr, "\nend   dump: %s\n", "ey");

  fprintf(stderr, "begin dump: %s", "hz");
  for (i = 0; i < nx; i++)
    for (j = 0; j < ny; j++) {
      if ((i * nx + j) % 20 == 0) fprintf(stderr, "\n");
      fprintf(stderr, "%0.6lf ", hz[i][j]);
    }
  fprintf(stderr, "\nend   dump: %s\n", "hz");
}

/* ---- strict checker (bench_wide SPEC) ---- */
/* golden: static copy of plain.cpp top function, run on identical input copies */



static void kernel_fdtd_2d_ref(
		    
		    
		    double ex[ NX + 0][NY + 0],
		    double ey[ NX + 0][NY + 0],
		    double hz[ NX + 0][NY + 0],
		    double _fict_[ TMAX + 0])
{


    const int tmax = TMAX;
    const int nx = NX;
    const int ny = NY;

  int t, i, j;


  for(t = 0; t < tmax; t++)
    {
      for (j = 0; j < ny; j++)
	ey[0][j] = _fict_[t];
      for (i = 1; i < nx; i++)
	for (j = 0; j < ny; j++)
	  ey[i][j] = ey[i][j] - 0.5*(hz[i][j]-hz[i-1][j]);
      for (i = 0; i < nx; i++)
	for (j = 1; j < ny; j++)
	  ex[i][j] = ex[i][j] - 0.5*(hz[i][j]-hz[i][j-1]);
      for (i = 0; i < nx - 1; i++)
	for (j = 0; j < ny - 1; j++)
	  hz[i][j] = hz[i][j] - 0.7*  (ex[i][j+1] - ex[i][j] +
				       ey[i+1][j] - ey[i][j]);
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

  int tmax = TMAX;
  int nx = NX;
  int ny = NY;


  double ex[ NX + 0][NY + 0];
  double ey[ NX + 0][NY + 0];
  double hz[ NX + 0][NY + 0];
  double _fict_[ TMAX + 0];


  init_array (tmax, nx, ny,
	      ex,
	      ey,
	      hz,
	      _fict_);


    /* reference copies */
  static double ex_ref[ NX + 0][NY + 0];
  static double ey_ref[ NX + 0][NY + 0];
  static double hz_ref[ NX + 0][NY + 0];
  static double _fict__ref[ TMAX + 0];
  /* poison pure outputs */
  memcpy(&ex_ref, &ex, sizeof(ex));
  memcpy(&ey_ref, &ey, sizeof(ey));
  memcpy(&hz_ref, &hz, sizeof(hz));
  memcpy(&_fict__ref, &_fict_, sizeof(_fict_));
  /* golden */
  kernel_fdtd_2d_ref (
		  ex_ref,
		  ey_ref,
		  hz_ref,
		  _fict__ref);
  /* kernel under test */
  kernel_fdtd_2d (
		  ex,
		  ey,
		  hz,
		  _fict_);
  chk_d("ex", (const double*)&ex, (const double*)&ex_ref, sizeof(ex)/sizeof(double));
  chk_d("ey", (const double*)&ey, (const double*)&ey_ref, sizeof(ey)/sizeof(double));
  chk_d("hz", (const double*)&hz, (const double*)&hz_ref, sizeof(hz)/sizeof(double));
  if (g_errs) { printf("Failed with %d errors!\n", g_errs); return 1; }
  printf("Passed!\n");
  return 0;
}
