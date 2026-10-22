/*
 * Strict Vitis HLS C-simulation testbench for machsuite_sort_radix.
 * MachSuite harness (input.data -> run_benchmark -> check), with:
 *  - independent golden: plain.cpp's ss_sort() (and helpers, renamed ref_*) run on a copy of the same input,
 *    element-by-element comparison of a[] (the sorted result), also compared with check.data (sorted array);
 *  - a[] is in-out (input data, sorted in place); b[], bucket[], sum[] are pure scratch in plain.cpp (written before
 *    read) and are poisoned with 0x5A5A5A5A; only a[] is the functional output, so scratch is not compared.
 */
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <unistd.h>
#include <fcntl.h>
#include <sys/stat.h>
#include <assert.h>
#include "sort.h"

/* MachSuite support is C; force C linkage from this C++ TB. */
extern "C" {
#include "support.h"
}

/* ---- reference: copy of plain.cpp ---- */
/*
Implementation based on algorithm described in:
A. Danalis, G. Marin, C. McCurdy, J. S. Meredith, P. C. Roth, K. Spafford, V. Tipparaju, and J. S. Vetter.
The scalable heterogeneous computing (shoc) benchmark suite.
In Proceedings of the 3rd Workshop on General-Purpose Computation on Graphics Processing Units, 2010
*/


static void ref_local_scan(int bucket[BUCKETSIZE])
{
    int radixID, i, bucket_indx;
    local_1 : for (radixID=0; radixID<SCAN_RADIX; radixID++) {
        local_2 : for (i=1; i<SCAN_BLOCK; i++){
            bucket_indx = radixID*SCAN_BLOCK + i;
            bucket[bucket_indx] += bucket[bucket_indx-1];
        }
    }
}

static void ref_sum_scan(int sum[SCAN_RADIX], int bucket[BUCKETSIZE])
{
    int radixID, bucket_indx;
    sum[0] = 0;
    sum_1 : for (radixID=1; radixID<SCAN_RADIX; radixID++) {
        bucket_indx = radixID*SCAN_BLOCK - 1;
        sum[radixID] = sum[radixID-1] + bucket[bucket_indx];
    }
}

static void ref_last_step_scan(int bucket[BUCKETSIZE], int sum[SCAN_RADIX])
{
    int radixID, i, bucket_indx;
    last_1:for (radixID=0; radixID<SCAN_RADIX; radixID++) {
        last_2:for (i=0; i<SCAN_BLOCK; i++) {
            bucket_indx = radixID * SCAN_BLOCK + i;
            bucket[bucket_indx] = bucket[bucket_indx] + sum[radixID];
         }
    }
}

static void ref_init(int bucket[BUCKETSIZE])
{
    int i;
    init_1 : for (i=0; i<BUCKETSIZE; i++) {
        bucket[i] = 0;
    }
}

static void ref_hist(int bucket[BUCKETSIZE], int a[SIZE], int exp)
{
    int blockID, i, bucket_indx, a_indx;
    blockID = 0;
    hist_1 : for (blockID=0; blockID<NUMOFBLOCKS; blockID++) {
        hist_2 : for(i=0; i<4; i++) {
            a_indx = blockID * ELEMENTSPERBLOCK + i;
            bucket_indx = ((a[a_indx] >> exp) & 0x3)*NUMOFBLOCKS + blockID + 1;
            bucket[bucket_indx]++;
        }
    }
}

static void ref_update(int b[SIZE], int bucket[BUCKETSIZE], int a[SIZE], int exp)
{
    int i, blockID, bucket_indx, a_indx;
    blockID = 0;

    update_1 : for (blockID = 0; blockID < NUMOFBLOCKS; blockID++) {
        update_2 : for(i=0; i<4; i++) {
            bucket_indx = ((a[blockID * ELEMENTSPERBLOCK + i] >> exp) & 0x3)*NUMOFBLOCKS + blockID;
            a_indx = blockID * ELEMENTSPERBLOCK + i;
            b[bucket[bucket_indx]] = a[a_indx];
            bucket[bucket_indx]++;
        }
    }
}

static void ref_ss_sort(int a[SIZE], int b[SIZE], int bucket[BUCKETSIZE], int sum[SCAN_RADIX]){
    int exp=0;
    int valid_buffer=0;
    #define BUFFER_A 0
    #define BUFFER_B 1

    sort_1 : for (exp=0; exp<32; exp+=2) {
        ref_init(bucket);
        if (valid_buffer == BUFFER_A) {
            ref_hist(bucket, a, exp);
        } else {
            ref_hist(bucket, b, exp);
        }

        ref_local_scan(bucket);
        ref_sum_scan(sum, bucket);
        ref_last_step_scan(bucket, sum);

        if (valid_buffer==BUFFER_A) {
            ref_update(b, bucket, a, exp);
            valid_buffer = BUFFER_B;
        } else {
            ref_update(a, bucket, b, exp);
            valid_buffer = BUFFER_A;
        }
    }
    // If trip count is even, buffer A will be valid at the end.
}
/* ---- end reference ---- */

static int errors = 0;
#define REPORT(...) do { if (errors < 10) printf(__VA_ARGS__); errors++; } while (0)

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
  ref_ss_sort(g->a, g->b, g->bucket, g->sum);

  /* poison pure outputs of the kernel under test (after inputs are loaded) */
  for (int i = 0; i < SIZE; i++) d->b[i] = 0x5A5A5A5A;
  for (int i = 0; i < BUCKETSIZE; i++) d->bucket[i] = 0x5A5A5A5A;
  for (int i = 0; i < SCAN_RADIX; i++) d->sum[i] = 0x5A5A5A5A;

  run_benchmark(data);

  int check_fd = open(check_file, O_RDONLY);
  assert(check_fd > 0 && "Couldn't open check data file");
  output_to_data(check_fd, chk);
  close(check_fd);

  for (int i = 0; i < SIZE; i++) {
    if (d->a[i] != g->a[i] || d->a[i] != c->a[i])
      REPORT("a[%d]: got %d expected %d (check.data %d)\n", i, d->a[i], g->a[i], c->a[i]);
  }
  free(data); free(gold); free(chk);
  if (errors) {
    printf("Failed with %d errors!\n", errors);
    return 1;
  }
  printf("Passed!\n");
  return 0;
}
