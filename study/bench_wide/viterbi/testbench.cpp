/*
 * Strict Vitis HLS C-simulation testbench for viterbi.
 * Golden: viterbi_ref() (same algorithm and arithmetic order as plain.cpp's viterbi, so exact ties resolve identically)
 * on an identical deterministic HMM (srand(42)). workload(obs, init, transition, emission, path): obs, init,
 * transition, emission are inputs; path[N_OBS] is a pure output, poisoned with 0x5A (90, not a state id < 64) for ref
 * and DUT. Every path entry is compared exactly. Prints "Passed!" or "Failed with N errors!".
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "viterbi.h"

/* Golden reference: plain-C Viterbi (identical algorithm) */
static void viterbi_ref(tok_t obs[N_OBS], prob_t init[N_STATES],
                        prob_t transition[N_STATES * N_STATES],
                        prob_t emission[N_STATES * N_TOKENS],
                        state_t path[N_OBS]) {
    prob_t llike[N_OBS][N_STATES];
    int t;
    uint8_t prev, curr, s;
    prob_t min_p, p;
    uint8_t min_s;

    for (s = 0; s < N_STATES; s++)
        llike[0][s] = init[s] + emission[s * N_TOKENS + obs[0]];

    for (t = 1; t < N_OBS; t++) {
        for (curr = 0; curr < N_STATES; curr++) {
            min_p = llike[t-1][0] + transition[0 * N_STATES + curr]
                    + emission[curr * N_TOKENS + obs[t]];
            for (prev = 1; prev < N_STATES; prev++) {
                p = llike[t-1][prev] + transition[prev * N_STATES + curr]
                    + emission[curr * N_TOKENS + obs[t]];
                if (p < min_p) min_p = p;
            }
            llike[t][curr] = min_p;
        }
    }

    min_s = 0;
    min_p = llike[N_OBS - 1][0];
    for (s = 1; s < N_STATES; s++) {
        p = llike[N_OBS - 1][s];
        if (p < min_p) { min_p = p; min_s = s; }
    }
    path[N_OBS - 1] = min_s;

    for (t = N_OBS - 2; t >= 0; t--) {
        min_s = 0;
        min_p = llike[t][0] + transition[0 * N_STATES + path[t + 1]];
        for (s = 1; s < N_STATES; s++) {
            p = llike[t][s] + transition[s * N_STATES + path[t + 1]];
            if (p < min_p) { min_p = p; min_s = s; }
        }
        path[t] = min_s;
    }
}

extern "C" void workload(tok_t* obs, prob_t* init, prob_t* transition,
              prob_t* emission, state_t* path);

static int errors = 0;
static void chk_i(const char *name, long i, long long d, long long r) {
    if (d != r) {
        if (errors < 10) printf("%s[%ld]: got %lld expected %lld\n", name, i, d, r);
        errors++;
    }
}

int main() {

    tok_t obs[N_OBS];
    prob_t init[N_STATES];
    prob_t transition[N_STATES * N_STATES];
    prob_t emission[N_STATES * N_TOKENS];
    state_t ref_path[N_OBS], dut_path[N_OBS];

    /* Generate deterministic random HMM parameters (in -log space) */
    srand(42);
    for (int i = 0; i < N_OBS; i++)
        obs[i] = (tok_t)(rand() % N_TOKENS);
    for (int i = 0; i < N_STATES; i++)
        init[i] = (prob_t)(rand() % 100) / 10.0;
    for (int i = 0; i < N_STATES * N_STATES; i++)
        transition[i] = (prob_t)(rand() % 100) / 10.0;
    for (int i = 0; i < N_STATES * N_TOKENS; i++)
        emission[i] = (prob_t)(rand() % 100) / 10.0;

    /* Compute golden reference */
    memset(ref_path, 0x5A, sizeof(ref_path));   /* poison */
    viterbi_ref(obs, init, transition, emission, ref_path);

    /* Call HLS design under test */
    memset(dut_path, 0x5A, sizeof(dut_path));   /* poison */
    workload(obs, init, transition, emission, dut_path);

    /* Compare paths */
    for (int i = 0; i < N_OBS; i++) chk_i("path", i, dut_path[i], ref_path[i]);
    if (errors) { printf("Failed with %d errors!\n", errors); return 1; }
    printf("Passed!\n");
    return 0;
}
