/*
 * Strict Vitis HLS C-simulation testbench for machsuite_backprop.
 * MachSuite harness (input.data -> run_benchmark -> check), with:
 *  - golden = check.data (parsed with the harness's output_to_data; it covers every output / in-out array);
 *    every element is compared (no early exit), prints at most 10 mismatches, then exactly
 *    "Passed!" (return 0) or "Failed with N errors!" (return 1) instead of the harness's check_data()/Success. text.
 *  - all six weight/bias arrays are IN-OUT (updated in place with -= and renormalised, real data from input.data), so
 *    no pure output exists and nothing is poisoned; training_data/training_targets are inputs.
 *  - double tolerance 1e-6*fmax(1,|ref|) (SPEC), NaN = error.
 */
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <unistd.h>
#include <fcntl.h>
#include <sys/stat.h>
#include <assert.h>
#include "backprop.h"

/* MachSuite support is C; force C linkage from this C++ TB. */
extern "C" {
#include "support.h"
}

/* ---- checking helpers (golden = check.data) ---- */
static int errors = 0;
#define REPORT(...) do { if (errors < 10) printf(__VA_ARGS__); errors++; } while (0)
static int chk_d(const char *name, long i, double d, double r) {
    if (!isfinite(r) || !(fabs(d - r) <= 1e-6 * fmax(1.0, fabs(r)))) {   /* NaN/inf in DUT output is an error */
        REPORT("%s[%ld]: got %.9g expected %.9g\n", name, i, d, r);
        return 1;
    }
    return 0;
}

/* ---- end ---- */


int main(int argc, char **argv)
{
  const char *in_file = "input.data";
  const char *check_file = "check.data";
  assert(argc < 4 && "Usage: ./benchmark <input_file> <check_file>");
  if (argc > 1) in_file = argv[1];
  if (argc > 2) check_file = argv[2];

  char *data = (char *)malloc(INPUT_SIZE);
  char *gold = (char *)malloc(INPUT_SIZE);
  char *chk = (char *)malloc(INPUT_SIZE);
  assert(data && gold && chk && "Out of memory");
  int in_fd = open(in_file, O_RDONLY);
  assert(in_fd > 0 && "Couldn't open input data file");
  input_to_data(in_fd, data);
  close(in_fd);

  struct bench_args_t *d = (struct bench_args_t *)data;
  struct bench_args_t *g = (struct bench_args_t *)gold;
  struct bench_args_t *c = (struct bench_args_t *)chk;

  /* identical inputs / in-out arrays for reference and kernel */
  memcpy(gold, data, INPUT_SIZE);
  /* golden is check.data (covers every output / in-out array) */

  /* poison pure outputs of the kernel under test (after inputs are loaded) */
/* no pure outputs: weights/biases are in-out */

  run_benchmark(data);

  int check_fd = open(check_file, O_RDONLY);
  assert(check_fd > 0 && "Couldn't open check data file");
  output_to_data(check_fd, chk);
  close(check_fd);

  for (int i = 0; i < input_dimension*nodes_per_layer; i++) chk_d("weights1", i, d->weights1[i], c->weights1[i]);
  for (int i = 0; i < nodes_per_layer*nodes_per_layer; i++) chk_d("weights2", i, d->weights2[i], c->weights2[i]);
  for (int i = 0; i < nodes_per_layer*possible_outputs; i++) chk_d("weights3", i, d->weights3[i], c->weights3[i]);
  for (int i = 0; i < nodes_per_layer; i++) chk_d("biases1", i, d->biases1[i], c->biases1[i]);
  for (int i = 0; i < nodes_per_layer; i++) chk_d("biases2", i, d->biases2[i], c->biases2[i]);
  for (int i = 0; i < possible_outputs; i++) chk_d("biases3", i, d->biases3[i], c->biases3[i]);
  free(data); free(gold); free(chk);
  if (errors) {
    printf("Failed with %d errors!\n", errors);
    return 1;
  }
  printf("Passed!\n");
  return 0;
}
