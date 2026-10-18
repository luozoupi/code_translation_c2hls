#include "viterbi.h"

int viterbi(tok_t obs[N_OBS], prob_t init[N_STATES],
            prob_t transition[N_STATES * N_STATES],
            prob_t emission[N_STATES * N_TOKENS],
            state_t path[N_OBS]) {
    prob_t llike[N_OBS][N_STATES];
    step_t t;
    state_t prev, curr;
    prob_t min_p, p;
    state_t min_s, s;

    L_init: for (s = 0; s < N_STATES; s++) {
        llike[0][s] = init[s] + emission[s * N_TOKENS + obs[0]];
    }

    L_timestep: for (t = 1; t < N_OBS; t++) {
        L_curr_state: for (curr = 0; curr < N_STATES; curr++) {
            prev = 0;
            min_p = llike[t - 1][prev] +
                    transition[prev * N_STATES + curr] +
                    emission[curr * N_TOKENS + obs[t]];
            L_prev_state: for (prev = 1; prev < N_STATES; prev++) {
                p = llike[t - 1][prev] +
                    transition[prev * N_STATES + curr] +
                    emission[curr * N_TOKENS + obs[t]];
                if (p < min_p) {
                    min_p = p;
                }
            }
            llike[t][curr] = min_p;
        }
    }

    min_s = 0;
    min_p = llike[N_OBS - 1][min_s];
    L_end: for (s = 1; s < N_STATES; s++) {
        p = llike[N_OBS - 1][s];
        if (p < min_p) {
            min_p = p;
            min_s = s;
        }
    }
    path[N_OBS - 1] = min_s;

    L_backtrack: for (t = N_OBS - 2; t >= 0; t--) {
        min_s = 0;
        min_p = llike[t][min_s] + transition[min_s * N_STATES + path[t + 1]];
        L_state: for (s = 1; s < N_STATES; s++) {
            p = llike[t][s] + transition[s * N_STATES + path[t + 1]];
            if (p < min_p) {
                min_p = p;
                min_s = s;
            }
        }
        path[t] = min_s;
    }

    return 0;
}

extern "C" {
void workload(tok_t* obs, prob_t* init, prob_t* transition,
              prob_t* emission, state_t* path) {
#pragma HLS INTERFACE m_axi port=obs offset=slave bundle=gmem0
#pragma HLS INTERFACE m_axi port=init offset=slave bundle=gmem1
#pragma HLS INTERFACE m_axi port=transition offset=slave bundle=gmem1
#pragma HLS INTERFACE m_axi port=emission offset=slave bundle=gmem2
#pragma HLS INTERFACE m_axi port=path offset=slave bundle=gmem3
#pragma HLS INTERFACE s_axilite port=obs bundle=control
#pragma HLS INTERFACE s_axilite port=init bundle=control
#pragma HLS INTERFACE s_axilite port=transition bundle=control
#pragma HLS INTERFACE s_axilite port=emission bundle=control
#pragma HLS INTERFACE s_axilite port=path bundle=control
#pragma HLS INTERFACE s_axilite port=return bundle=control

    tok_t l_obs[N_OBS];
    prob_t l_init[N_STATES];
    prob_t l_transition[N_STATES * N_STATES];
    prob_t l_emission[N_STATES * N_TOKENS];
    state_t l_path[N_OBS];
    int i;

    for (i = 0; i < N_OBS; i++) l_obs[i] = obs[i];
    for (i = 0; i < N_STATES; i++) l_init[i] = init[i];
    for (i = 0; i < N_STATES * N_STATES; i++) l_transition[i] = transition[i];
    for (i = 0; i < N_STATES * N_TOKENS; i++) l_emission[i] = emission[i];

    viterbi(l_obs, l_init, l_transition, l_emission, l_path);

    for (i = 0; i < N_OBS; i++) path[i] = l_path[i];
}
}
