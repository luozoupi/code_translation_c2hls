#include "3mm.h"

// Tile sizes
#define TILE_I 8
#define TILE_J 10
#define TILE_K 10
#define TILE_L 10
#define TILE_M 10

// Unroll factors
#define UNROLL_J 2
#define UNROLL_L 2
#define UNROLL_K 10
#define UNROLL_M 10

extern "C" {

void kernel_3mm(    
		double E[ NI + 0][NJ + 0],
		double A[ NI + 0][NK + 0],
		double B[ NK + 0][NJ + 0],
		double F[ NJ + 0][NL + 0],
		double C[ NJ + 0][NM + 0],
		double D[ NM + 0][NL + 0],
		double G[ NI + 0][NL + 0])
{
#pragma HLS INTERFACE m_axi port=E offset=slave bundle=gmem0
#pragma HLS INTERFACE m_axi port=A offset=slave bundle=gmem1
#pragma HLS INTERFACE m_axi port=B offset=slave bundle=gmem2
#pragma HLS INTERFACE m_axi port=F offset=slave bundle=gmem3
#pragma HLS INTERFACE m_axi port=C offset=slave bundle=gmem4
#pragma HLS INTERFACE m_axi port=D offset=slave bundle=gmem5
#pragma HLS INTERFACE m_axi port=G offset=slave bundle=gmem6
#pragma HLS INTERFACE s_axilite port=E bundle=control
#pragma HLS INTERFACE s_axilite port=A bundle=control
#pragma HLS INTERFACE s_axilite port=B bundle=control
#pragma HLS INTERFACE s_axilite port=F bundle=control
#pragma HLS INTERFACE s_axilite port=C bundle=control
#pragma HLS INTERFACE s_axilite port=D bundle=control
#pragma HLS INTERFACE s_axilite port=G bundle=control
#pragma HLS INTERFACE s_axilite port=return bundle=control

    // Intermediate full arrays stored locally
    double local_E[NI][NJ];
    double local_F[NJ][NL];
    double local_G[NI][NL];

    // Partition dim=2 by factor matching unroll on j/l dimensions
#pragma HLS ARRAY_PARTITION variable=local_E cyclic factor=8 dim=2
#pragma HLS ARRAY_PARTITION variable=local_F cyclic factor=8 dim=2
#pragma HLS ARRAY_PARTITION variable=local_G cyclic factor=8 dim=2

    // Tile buffers for A and B (used in E = A*B computation)
    double tile_A[TILE_I][TILE_K];
    double tile_B[TILE_K][TILE_J];
    double tile_E[TILE_I][TILE_J];

#pragma HLS ARRAY_PARTITION variable=tile_A complete dim=2
#pragma HLS ARRAY_PARTITION variable=tile_B complete dim=1
#pragma HLS ARRAY_PARTITION variable=tile_B complete dim=2
#pragma HLS ARRAY_PARTITION variable=tile_E complete dim=2

    // Tile buffers for C and D (used in F = C*D computation)
    double tile_C[TILE_J][TILE_M];
    double tile_D[TILE_M][TILE_L];
    double tile_F[TILE_J][TILE_L];

#pragma HLS ARRAY_PARTITION variable=tile_C complete dim=2
#pragma HLS ARRAY_PARTITION variable=tile_D complete dim=1
#pragma HLS ARRAY_PARTITION variable=tile_D complete dim=2
#pragma HLS ARRAY_PARTITION variable=tile_F complete dim=2

    // Tile buffers for E and F (used in G = E*F computation)
    double tile_EG[TILE_I][TILE_J];
    double tile_FG[TILE_J][TILE_L];
    double tile_G[TILE_I][TILE_L];

#pragma HLS ARRAY_PARTITION variable=tile_EG complete dim=2
#pragma HLS ARRAY_PARTITION variable=tile_FG complete dim=1
#pragma HLS ARRAY_PARTITION variable=tile_FG complete dim=2
#pragma HLS ARRAY_PARTITION variable=tile_G complete dim=2

    // =========================================================
    // Phase 1: Compute E = A * B using tiled matrix multiply
    // E[i][j] = sum_k A[i][k] * B[k][j]
    // =========================================================
    init_E: for (int i = 0; i < NI; i++) {
        init_E_j: for (int j = 0; j < NJ; j++) {
#pragma HLS PIPELINE II=1
#pragma HLS LOOP_TRIPCOUNT min=NJ max=NJ
#pragma HLS DEPENDENCE variable=local_E inter false
            local_E[i][j] = 0.0;
        }
    }

    tile_E_i: for (int ii = 0; ii < NI; ii += TILE_I) {
#pragma HLS LOOP_TRIPCOUNT min=NI/TILE_I max=NI/TILE_I
        tile_E_j: for (int jj = 0; jj < NJ; jj += TILE_J) {
#pragma HLS LOOP_TRIPCOUNT min=NJ/TILE_J max=NJ/TILE_J
            tile_E_k: for (int kk = 0; kk < NK; kk += TILE_K) {
#pragma HLS LOOP_TRIPCOUNT min=NK/TILE_K max=NK/TILE_K

                // Load tile of A: A[ii..ii+TILE_I][kk..kk+TILE_K]
                load_tA_i: for (int i = 0; i < TILE_I; i++) {
                    load_tA_k: for (int k = 0; k < TILE_K; k++) {
#pragma HLS PIPELINE II=1
#pragma HLS LOOP_TRIPCOUNT min=TILE_K max=TILE_K
#pragma HLS DEPENDENCE variable=tile_A inter false
                        int gi = ii + i;
                        int gk = kk + k;
                        tile_A[i][k] = (gi < NI && gk < NK) ? A[gi][gk] : 0.0;
                    }
                }

                // Load tile of B: B[kk..kk+TILE_K][jj..jj+TILE_J]
                load_tB_k: for (int k = 0; k < TILE_K; k++) {
                    load_tB_j: for (int j = 0; j < TILE_J; j++) {
#pragma HLS PIPELINE II=1
#pragma HLS UNROLL factor=2
#pragma HLS LOOP_TRIPCOUNT min=TILE_J max=TILE_J
#pragma HLS DEPENDENCE variable=tile_B inter false
                        int gk = kk + k;
                        int gj = jj + j;
                        tile_B[k][j] = (gk < NK && gj < NJ) ? B[gk][gj] : 0.0;
                    }
                }

                // Compute partial tile_E += tile_A * tile_B
                compute_tE_i: for (int i = 0; i < TILE_I; i++) {
                    compute_tE_j: for (int j = 0; j < TILE_J; j++) {
#pragma HLS PIPELINE II=1
#pragma HLS UNROLL factor=2
#pragma HLS LOOP_TRIPCOUNT min=TILE_J max=TILE_J
#pragma HLS DEPENDENCE variable=local_E inter false
                        double sum = 0.0;
                        compute_tE_k: for (int k = 0; k < TILE_K; k++) {
#pragma HLS UNROLL factor=10
                            sum += tile_A[i][k] * tile_B[k][j];
                        }
                        int gi = ii + i;
                        int gj = jj + j;
                        if (gi < NI && gj < NJ) {
                            local_E[gi][gj] += sum;
                        }
                    }
                }
            }
        }
    }

    // =========================================================
    // Phase 2: Compute F = C * D using tiled matrix multiply
    // F[j][l] = sum_m C[j][m] * D[m][l]
    // =========================================================
    init_F: for (int j = 0; j < NJ; j++) {
        init_F_l: for (int l = 0; l < NL; l++) {
#pragma HLS PIPELINE II=1
#pragma HLS LOOP_TRIPCOUNT min=NL max=NL
#pragma HLS DEPENDENCE variable=local_F inter false
            local_F[j][l] = 0.0;
        }
    }

    tile_F_j: for (int jj = 0; jj < NJ; jj += TILE_J) {
#pragma HLS LOOP_TRIPCOUNT min=NJ/TILE_J max=NJ/TILE_J
        tile_F_l: for (int ll = 0; ll < NL; ll += TILE_L) {
#pragma HLS LOOP_TRIPCOUNT min=NL/TILE_L max=NL/TILE_L
            tile_F_m: for (int mm = 0; mm < NM; mm += TILE_M) {
#pragma HLS LOOP_TRIPCOUNT min=NM/TILE_M max=NM/TILE_M

                // Load tile of C: C[jj..jj+TILE_J][mm..mm+TILE_M]
                load_tC_j: for (int j = 0; j < TILE_J; j++) {
                    load_tC_m: for (int m = 0; m < TILE_M; m++) {
#pragma HLS PIPELINE II=1
#pragma HLS LOOP_TRIPCOUNT min=TILE_M max=TILE_M
#pragma HLS DEPENDENCE variable=tile_C inter false
                        int gj = jj + j;
                        int gm = mm + m;
                        tile_C[j][m] = (gj < NJ && gm < NM) ? C[gj][gm] : 0.0;
                    }
                }

                // Load tile of D: D[mm..mm+TILE_M][ll..ll+TILE_L]
                load_tD_m: for (int m = 0; m < TILE_M; m++) {
                    load_tD_l: for (int l = 0; l < TILE_L; l++) {
#pragma HLS PIPELINE II=1
#pragma HLS UNROLL factor=2
#pragma HLS LOOP_TRIPCOUNT min=TILE_L max=TILE_L
#pragma HLS DEPENDENCE variable=tile_D inter false
                        int gm = mm + m;
                        int gl = ll + l;
                        tile_D[m][l] = (gm < NM && gl < NL) ? D[gm][gl] : 0.0;
                    }
                }

                // Compute partial tile_F += tile_C * tile_D
                compute_tF_j: for (int j = 0; j < TILE_J; j++) {
                    compute_tF_l: for (int l = 0; l < TILE_L; l++) {
#pragma HLS PIPELINE II=1
#pragma HLS UNROLL factor=2
#pragma HLS LOOP_TRIPCOUNT min=TILE_L max=TILE_L
#pragma HLS DEPENDENCE variable=local_F inter false
                        double sum = 0.0;
                        compute_tF_m: for (int m = 0; m < TILE_M; m++) {
#pragma HLS UNROLL factor=10
                            sum += tile_C[j][m] * tile_D[m][l];
                        }
                        int gj = jj + j;
                        int gl = ll + l;
                        if (gj < NJ && gl < NL) {
                            local_F[gj][gl] += sum;
                        }
                    }
                }
            }
        }
    }

    // =========================================================
    // Phase 3: Compute G = E * F using tiled matrix multiply
    // G[i][l] = sum_j E[i][j] * F[j][l]
    // =========================================================
    init_G: for (int i = 0; i < NI; i++) {
        init_G_l: for (int l = 0; l < NL; l++) {
#pragma HLS PIPELINE II=1
#pragma HLS LOOP_TRIPCOUNT min=NL max=NL
#pragma HLS DEPENDENCE variable=local_G inter false
            local_G[i][l] = 0.0;
        }
    }

    tile_G_i: for (int ii = 0; ii < NI; ii += TILE_I) {
#pragma HLS LOOP_TRIPCOUNT min=NI/TILE_I max=NI/TILE_I
        tile_G_l: for (int ll = 0; ll < NL; ll += TILE_L) {
#pragma HLS LOOP_TRIPCOUNT min=NL/TILE_L max=NL/TILE_L
            tile_G_j: for (int jj = 0; jj < NJ; jj += TILE_J) {
#pragma HLS LOOP_TRIPCOUNT min=NJ/TILE_J max=NJ/TILE_J

                // Load tile of E: local_E[ii..ii+TILE_I][jj..jj+TILE_J]
                load_tEG_i: for (int i = 0; i < TILE_I; i++) {
                    load_tEG_j: for (int j = 0; j < TILE_J; j++) {
#pragma HLS PIPELINE II=1
#pragma HLS LOOP_TRIPCOUNT min=TILE_J max=TILE_J
#pragma HLS DEPENDENCE variable=tile_EG inter false
#pragma HLS DEPENDENCE variable=local_E inter false
                        int gi = ii + i;
                        int gj = jj + j;
                        tile_EG[i][j] = (gi < NI && gj < NJ) ? local_E[gi][gj] : 0.0;
                    }
                }

                // Load tile of F: local_F[jj..jj+TILE_J][ll..ll+TILE_L]
                load_tFG_j: for (int j = 0; j < TILE_J; j++) {
                    load_tFG_l: for (int l = 0; l < TILE_L; l++) {
#pragma HLS PIPELINE II=1
#pragma HLS UNROLL factor=2
#pragma HLS LOOP_TRIPCOUNT min=TILE_L max=TILE_L
#pragma HLS DEPENDENCE variable=tile_FG inter false
#pragma HLS DEPENDENCE variable=local_F inter false
                        int gj = jj + j;
                        int gl = ll + l;
                        tile_FG[j][l] = (gj < NJ && gl < NL) ? local_F[gj][gl] : 0.0;
                    }
                }

                // Compute partial tile_G += tile_EG * tile_FG
                compute_tG_i: for (int i = 0; i < TILE_I; i++) {
                    compute_tG_l: for (int l = 0; l < TILE_L; l++) {
#pragma HLS PIPELINE II=1
#pragma HLS UNROLL factor=2
#pragma HLS LOOP_TRIPCOUNT min=TILE_L max=TILE_L
#pragma HLS DEPENDENCE variable=local_G inter false
                        double sum = 0.0;
                        compute_tG_j: for (int j = 0; j < TILE_J; j++) {
#pragma HLS UNROLL factor=10
                            sum += tile_EG[i][j] * tile_FG[j][l];
                        }
                        int gi = ii + i;
                        int gl = ll + l;
                        if (gi < NI && gl < NL) {
                            local_G[gi][gl] += sum;
                        }
                    }
                }
            }
        }
    }

    // =========================================================
    // Store phase: write E, F, G back to global memory
    // =========================================================
    write_E_i: for (int i = 0; i < NI; i++) {
        write_E_j: for (int j = 0; j < NJ; j++) {
#pragma HLS PIPELINE II=1
#pragma HLS LOOP_TRIPCOUNT min=NJ max=NJ
#pragma HLS DEPENDENCE variable=local_E inter false
            E[i][j] = local_E[i][j];
        }
    }

    write_F_i: for (int i = 0; i < NJ; i++) {
        write_F_j: for (int j = 0; j < NL; j++) {
#pragma HLS PIPELINE II=1
#pragma HLS LOOP_TRIPCOUNT min=NL max=NL
#pragma HLS DEPENDENCE variable=local_F inter false
            F[i][j] = local_F[i][j];
        }
    }

    write_G_i: for (int i = 0; i < NI; i++) {
        write_G_j: for (int j = 0; j < NL; j++) {
#pragma HLS PIPELINE II=1
#pragma HLS LOOP_TRIPCOUNT min=NL max=NL
#pragma HLS DEPENDENCE variable=local_G inter false
            G[i][j] = local_G[i][j];
        }
    }
}

} // extern "C"