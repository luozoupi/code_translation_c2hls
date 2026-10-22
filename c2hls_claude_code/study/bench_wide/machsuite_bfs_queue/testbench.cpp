/*
 * Strict Vitis HLS C-simulation testbench for machsuite_bfs_queue.
 * MachSuite harness (input.data -> run_benchmark -> check), with:
 *  - independent golden: plain.cpp's bfs() copied below as bfs_ref(), run on a copy of the same input;
 *  - every level[] entry and every level_counts[] entry is compared against the reference
 *    (BFS distances are unique, so this pins the result completely), plus check.data level_counts.
 *  level[] is in-out (initialised to MAX_LEVEL by input_to_data, read before written) and level_counts[]
 *  is in-out as well (zero-initialised by input_to_data; unwritten tail entries must stay 0), so neither
 *  is poisoned; there is no pure-output array in this kernel.
 */
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <unistd.h>
#include <fcntl.h>
#include <sys/stat.h>
#include <assert.h>
#include "bfs.h"

/* MachSuite support is C; force C linkage from this C++ TB. */
extern "C" {
#include "support.h"
}

/* ---- reference: body of plain.cpp bfs() ---- */
/*
Implementation based on:
Hong, Oguntebi, Olukotun. "Efficient Parallel Graph Exploration on Multi-Core CPU and GPU." PACT, 2011.
*/


#define Q_PUSH(node) { queue[q_in==0?N_NODES-1:q_in-1]=node; q_in=(q_in+1)%N_NODES; }
#define Q_PEEK() (queue[q_out])
#define Q_POP() { q_out = (q_out+1)%N_NODES; }
#define Q_EMPTY() (q_in>q_out ? q_in==q_out+1 : (q_in==0)&&(q_out==N_NODES-1))
#define MAX_N_EDGES 158

static void bfs_ref(node_t nodes[N_NODES], edge_t edges[N_EDGES],  node_index_t starting_node, level_t level[N_NODES],  edge_index_t level_counts[N_LEVELS]){
  node_index_t queue[N_NODES];
  node_index_t q_in, q_out;
  node_index_t dummy;
  node_index_t n;
  edge_index_t e;
  int i = 0;
  /*init_levels: for( n=0; n<N_NODES; n++ )*/
  /*level[n] = MAX_LEVEL;*/
  /*init_horizons: for( i=0; i<N_LEVELS; i++ )*/
  /*level_counts[i] = 0;*/

  q_in = 1;
  q_out = 0;
  level[starting_node] = 0;
  level_counts[0] = 1;
  Q_PUSH(starting_node);

  loop_queue: for( dummy=0; dummy<N_NODES; dummy++ ) { // Typically while(not_empty(queue)){
    if( Q_EMPTY() )
      break;
    n = Q_PEEK();
    Q_POP();
    e = nodes[n].edge_begin;
    edge_index_t tmp_end = nodes[n].edge_end;
    loop_neighbors: for( i=0; i<MAX_N_EDGES; i++ ) {
      if (e < tmp_end) {
        node_index_t tmp_dst = edges[e].dst;
        level_t tmp_level = level[tmp_dst];

        if( tmp_level ==MAX_LEVEL ) { // Unmarked
          level_t tmp_level = level[n]+1;
          level[tmp_dst] = tmp_level;
          ++level_counts[tmp_level];
          Q_PUSH(tmp_dst);
        }
      }
      e++;
    }
  }

  /*
  printf("Horizons:");
  for( i=0; i<N_LEVELS; i++ )
    printf(" %d", level_counts[i]);
  printf("\n");
  */
}
/* ---- end reference ---- */

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

  /* identical inputs / in-out arrays for reference and kernel */
  memcpy(gold, data, INPUT_SIZE);
  struct bench_args_t *g = (struct bench_args_t *)gold;
  bfs_ref(g->nodes, g->edges, g->starting_node, g->level, g->level_counts);

  run_benchmark(data);
  struct bench_args_t *d = (struct bench_args_t *)data;

  int check_fd = open(check_file, O_RDONLY);
  assert(check_fd > 0 && "Couldn't open check data file");
  output_to_data(check_fd, chk);
  close(check_fd);
  struct bench_args_t *c = (struct bench_args_t *)chk;

  int errors = 0;
  for (int i = 0; i < N_NODES; i++) {
    if (d->level[i] != g->level[i]) {
      if (errors < 10) printf("level[%d]: got %d expected %d\n", i, (int)d->level[i], (int)g->level[i]);
      errors++;
    }
  }
  for (int i = 0; i < N_LEVELS; i++) {
    if (d->level_counts[i] != g->level_counts[i] || d->level_counts[i] != c->level_counts[i]) {
      if (errors < 10) printf("level_counts[%d]: got %llu expected %llu (check.data %llu)\n", i,
                              (unsigned long long)d->level_counts[i], (unsigned long long)g->level_counts[i],
                              (unsigned long long)c->level_counts[i]);
      errors++;
    }
  }
  free(data); free(gold); free(chk);
  if (errors) {
    printf("Failed with %d errors!\n", errors);
    return 1;
  }
  printf("Passed!\n");
  return 0;
}
