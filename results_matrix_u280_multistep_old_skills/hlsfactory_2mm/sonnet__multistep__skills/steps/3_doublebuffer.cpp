#include "2mm.h"

// Tile sizes increased to improve parallelism with unrolling
#define TI 8
#define TJ 8
#define TK 8
#define TL 8

// Unroll factors for the pipelined i-loop
#define UI 2

extern "C" {

void kernel_2mm(   
		double alpha,
		double beta,
		double tmp[ NI + 0][NJ + 0],
		double A[ NI + 0][NK + 0],
		double B[ NK + 0][NJ + 0],
		double C[ NJ + 0][NL + 0],
		double D[ NI + 0][NL + 0])
{
#pragma HLS INTERFACE s_axilite port=alpha    bundle=control
#pragma HLS INTERFACE s_axilite port=beta     bundle=control
#pragma HLS INTERFACE m_axi     port=tmp      offset=slave bundle=gmem
#pragma HLS INTERFACE s_axilite port=tmp      bundle=control
#pragma HLS INTERFACE m_axi     port=A        offset=slave bundle=gmem
#pragma HLS INTERFACE s_axilite port=A        bundle=control
#pragma HLS INTERFACE m_axi     port=B        offset=slave bundle=gmem
#pragma HLS INTERFACE s_axilite port=B        bundle=control
#pragma HLS INTERFACE m_axi     port=C        offset=slave bundle=gmem
#pragma HLS INTERFACE s_axilite port=C        bundle=control
#pragma HLS INTERFACE m_axi     port=D        offset=slave bundle=gmem
#pragma HLS INTERFACE s_axilite port=D        bundle=control
#pragma HLS INTERFACE s_axilite port=return   bundle=control

    // -------------------------------------------------------
    // Phase 1: tmp[i][j] = alpha * sum_k( A[i][k] * B[k][j] )
    // Tile over i, j, k dimensions with double buffering on k tiles
    // -------------------------------------------------------

    double acc_tmp[TI][TJ];

    // Double-buffered tile arrays for A and B
    double tile_A_0[TI][TK];
    double tile_A_1[TI][TK];
    double tile_B_0[TK][TJ];
    double tile_B_1[TK][TJ];

#pragma HLS ARRAY_PARTITION variable=acc_tmp  cyclic factor=UI dim=1
#pragma HLS ARRAY_PARTITION variable=acc_tmp  complete dim=2
#pragma HLS ARRAY_PARTITION variable=tile_A_0 cyclic factor=UI dim=1
#pragma HLS ARRAY_PARTITION variable=tile_A_0 complete dim=2
#pragma HLS ARRAY_PARTITION variable=tile_A_1 cyclic factor=UI dim=1
#pragma HLS ARRAY_PARTITION variable=tile_A_1 complete dim=2
#pragma HLS ARRAY_PARTITION variable=tile_B_0 complete dim=1
#pragma HLS ARRAY_PARTITION variable=tile_B_0 complete dim=2
#pragma HLS ARRAY_PARTITION variable=tile_B_1 complete dim=1
#pragma HLS ARRAY_PARTITION variable=tile_B_1 complete dim=2

    phase1_i:
    for (int ii = 0; ii < NI; ii += TI) {
#pragma HLS LOOP_TRIPCOUNT min=5 max=5
        phase1_j:
        for (int jj = 0; jj < NJ; jj += TJ) {
#pragma HLS LOOP_TRIPCOUNT min=7 max=7

            // Initialize accumulator tile
            init_acc:
            for (int i = 0; i < TI; i++) {
#pragma HLS PIPELINE II=1
#pragma HLS UNROLL factor=UI
#pragma HLS LOOP_TRIPCOUNT min=TI max=TI
                for (int j = 0; j < TJ; j++) {
#pragma HLS UNROLL
                    acc_tmp[i][j] = 0.0;
                }
            }

            // Preload first k tile into buffer 0
            {
                int kk = 0;
                load_A_pre:
                for (int i = 0; i < TI; i++) {
#pragma HLS PIPELINE II=1
#pragma HLS UNROLL factor=UI
#pragma HLS LOOP_TRIPCOUNT min=TI max=TI
                    for (int k = 0; k < TK; k++) {
#pragma HLS UNROLL
                        int gi = ii + i;
                        int gk = kk + k;
                        tile_A_0[i][k] = (gi < NI && gk < NK) ? A[gi][gk] : 0.0;
                    }
                }
                load_B_pre:
                for (int k = 0; k < TK; k++) {
#pragma HLS PIPELINE II=1
#pragma HLS LOOP_TRIPCOUNT min=TK max=TK
                    for (int j = 0; j < TJ; j++) {
#pragma HLS UNROLL
                        int gk = kk + k;
                        int gj = jj + j;
                        tile_B_0[k][j] = (gk < NK && gj < NJ) ? B[gk][gj] : 0.0;
                    }
                }
            }

            // Accumulate over k tiles with double buffering
            phase1_k:
            for (int kk = 0; kk < NK; kk += TK) {
#pragma HLS LOOP_TRIPCOUNT min=9 max=9

                int buf_sel = (kk / TK) % 2; // which buffer holds current tile
                int next_kk = kk + TK;

                // Load next tile into the other buffer (if not last)
                if (next_kk < NK) {
                    if (buf_sel == 0) {
                        // Load next into buffer 1
                        load_A_next_0:
                        for (int i = 0; i < TI; i++) {
#pragma HLS PIPELINE II=1
#pragma HLS UNROLL factor=UI
#pragma HLS LOOP_TRIPCOUNT min=TI max=TI
                            for (int k = 0; k < TK; k++) {
#pragma HLS UNROLL
                                int gi = ii + i;
                                int gk = next_kk + k;
                                tile_A_1[i][k] = (gi < NI && gk < NK) ? A[gi][gk] : 0.0;
                            }
                        }
                        load_B_next_0:
                        for (int k = 0; k < TK; k++) {
#pragma HLS PIPELINE II=1
#pragma HLS LOOP_TRIPCOUNT min=TK max=TK
                            for (int j = 0; j < TJ; j++) {
#pragma HLS UNROLL
                                int gk = next_kk + k;
                                int gj = jj + j;
                                tile_B_1[k][j] = (gk < NK && gj < NJ) ? B[gk][gj] : 0.0;
                            }
                        }
                    } else {
                        // Load next into buffer 0
                        load_A_next_1:
                        for (int i = 0; i < TI; i++) {
#pragma HLS PIPELINE II=1
#pragma HLS UNROLL factor=UI
#pragma HLS LOOP_TRIPCOUNT min=TI max=TI
                            for (int k = 0; k < TK; k++) {
#pragma HLS UNROLL
                                int gi = ii + i;
                                int gk = next_kk + k;
                                tile_A_0[i][k] = (gi < NI && gk < NK) ? A[gi][gk] : 0.0;
                            }
                        }
                        load_B_next_1:
                        for (int k = 0; k < TK; k++) {
#pragma HLS PIPELINE II=1
#pragma HLS LOOP_TRIPCOUNT min=TK max=TK
                            for (int j = 0; j < TJ; j++) {
#pragma HLS UNROLL
                                int gk = next_kk + k;
                                int gj = jj + j;
                                tile_B_0[k][j] = (gk < NK && gj < NJ) ? B[gk][gj] : 0.0;
                            }
                        }
                    }
                }

                // Compute from current buffer
                if (buf_sel == 0) {
                    compute1_0:
                    for (int i = 0; i < TI; i++) {
#pragma HLS PIPELINE II=1
#pragma HLS UNROLL factor=UI
#pragma HLS LOOP_TRIPCOUNT min=TI max=TI
#pragma HLS DEPENDENCE variable=acc_tmp inter false
                        for (int j = 0; j < TJ; j++) {
#pragma HLS UNROLL
                            double sum = 0.0;
                            for (int k = 0; k < TK; k++) {
#pragma HLS UNROLL
                                sum += alpha * tile_A_0[i][k] * tile_B_0[k][j];
                            }
                            acc_tmp[i][j] += sum;
                        }
                    }
                } else {
                    compute1_1:
                    for (int i = 0; i < TI; i++) {
#pragma HLS PIPELINE II=1
#pragma HLS UNROLL factor=UI
#pragma HLS LOOP_TRIPCOUNT min=TI max=TI
#pragma HLS DEPENDENCE variable=acc_tmp inter false
                        for (int j = 0; j < TJ; j++) {
#pragma HLS UNROLL
                            double sum = 0.0;
                            for (int k = 0; k < TK; k++) {
#pragma HLS UNROLL
                                sum += alpha * tile_A_1[i][k] * tile_B_1[k][j];
                            }
                            acc_tmp[i][j] += sum;
                        }
                    }
                }

            } // kk

            // Store accumulated tmp tile to global memory
            store_tmp:
            for (int i = 0; i < TI; i++) {
#pragma HLS PIPELINE II=1
#pragma HLS UNROLL factor=UI
#pragma HLS LOOP_TRIPCOUNT min=TI max=TI
                for (int j = 0; j < TJ; j++) {
#pragma HLS UNROLL
                    int gi = ii + i;
                    int gj = jj + j;
                    if (gi < NI && gj < NJ)
                        tmp[gi][gj] = acc_tmp[i][j];
                }
            }

        } // jj
    } // ii

    // -------------------------------------------------------
    // Phase 2: D[i][l] = beta * D[i][l] + sum_k( tmp[i][k] * C[k][l] )
    // Tile over i, l, j dimensions with double buffering on j tiles
    // -------------------------------------------------------

    double acc_D[TI][TL];

    // Double-buffered tile arrays for tmp and C
    double tile_tmp_0[TI][TJ];
    double tile_tmp_1[TI][TJ];
    double tile_C_0[TJ][TL];
    double tile_C_1[TJ][TL];

#pragma HLS ARRAY_PARTITION variable=acc_D     cyclic factor=UI dim=1
#pragma HLS ARRAY_PARTITION variable=acc_D     complete dim=2
#pragma HLS ARRAY_PARTITION variable=tile_tmp_0 cyclic factor=UI dim=1
#pragma HLS ARRAY_PARTITION variable=tile_tmp_0 complete dim=2
#pragma HLS ARRAY_PARTITION variable=tile_tmp_1 cyclic factor=UI dim=1
#pragma HLS ARRAY_PARTITION variable=tile_tmp_1 complete dim=2
#pragma HLS ARRAY_PARTITION variable=tile_C_0  complete dim=1
#pragma HLS ARRAY_PARTITION variable=tile_C_0  complete dim=2
#pragma HLS ARRAY_PARTITION variable=tile_C_1  complete dim=1
#pragma HLS ARRAY_PARTITION variable=tile_C_1  complete dim=2

    phase2_i:
    for (int ii = 0; ii < NI; ii += TI) {
#pragma HLS LOOP_TRIPCOUNT min=5 max=5
        phase2_l:
        for (int ll = 0; ll < NL; ll += TL) {
#pragma HLS LOOP_TRIPCOUNT min=10 max=10

            // Load and initialize accumulator: acc_D = beta * D[ii:ii+TI][ll:ll+TL]
            init_D:
            for (int i = 0; i < TI; i++) {
#pragma HLS PIPELINE II=1
#pragma HLS UNROLL factor=UI
#pragma HLS LOOP_TRIPCOUNT min=TI max=TI
                for (int l = 0; l < TL; l++) {
#pragma HLS UNROLL
                    int gi = ii + i;
                    int gl = ll + l;
                    acc_D[i][l] = (gi < NI && gl < NL) ? beta * D[gi][gl] : 0.0;
                }
            }

            // Preload first j tile into buffer 0
            {
                int jj = 0;
                load_tmp_pre:
                for (int i = 0; i < TI; i++) {
#pragma HLS PIPELINE II=1
#pragma HLS UNROLL factor=UI
#pragma HLS LOOP_TRIPCOUNT min=TI max=TI
                    for (int j = 0; j < TJ; j++) {
#pragma HLS UNROLL
                        int gi = ii + i;
                        int gj = jj + j;
                        tile_tmp_0[i][j] = (gi < NI && gj < NJ) ? tmp[gi][gj] : 0.0;
                    }
                }
                load_C_pre:
                for (int j = 0; j < TJ; j++) {
#pragma HLS PIPELINE II=1
#pragma HLS LOOP_TRIPCOUNT min=TJ max=TJ
                    for (int l = 0; l < TL; l++) {
#pragma HLS UNROLL
                        int gj = jj + j;
                        int gl = ll + l;
                        tile_C_0[j][l] = (gj < NJ && gl < NL) ? C[gj][gl] : 0.0;
                    }
                }
            }

            // Accumulate over j tiles with double buffering
            phase2_k:
            for (int jj = 0; jj < NJ; jj += TJ) {
#pragma HLS LOOP_TRIPCOUNT min=7 max=7

                int buf_sel = (jj / TJ) % 2;
                int next_jj = jj + TJ;

                // Load next tile into the other buffer (if not last)
                if (next_jj < NJ) {
                    if (buf_sel == 0) {
                        // Load next into buffer 1
                        load_tmp_next_0:
                        for (int i = 0; i < TI; i++) {
#pragma HLS PIPELINE II=1
#pragma HLS UNROLL factor=UI
#pragma HLS LOOP_TRIPCOUNT min=TI max=TI
                            for (int j = 0; j < TJ; j++) {
#pragma HLS UNROLL
                                int gi = ii + i;
                                int gj = next_jj + j;
                                tile_tmp_1[i][j] = (gi < NI && gj < NJ) ? tmp[gi][gj] : 0.0;
                            }
                        }
                        load_C_next_0:
                        for (int j = 0; j < TJ; j++) {
#pragma HLS PIPELINE II=1
#pragma HLS LOOP_TRIPCOUNT min=TJ max=TJ
                            for (int l = 0; l < TL; l++) {
#pragma HLS UNROLL
                                int gj = next_jj + j;
                                int gl = ll + l;
                                tile_C_1[j][l] = (gj < NJ && gl < NL) ? C[gj][gl] : 0.0;
                            }
                        }
                    } else {
                        // Load next into buffer 0
                        load_tmp_next_1:
                        for (int i = 0; i < TI; i++) {
#pragma HLS PIPELINE II=1
#pragma HLS UNROLL factor=UI
#pragma HLS LOOP_TRIPCOUNT min=TI max=TI
                            for (int j = 0; j < TJ; j++) {
#pragma HLS UNROLL
                                int gi = ii + i;
                                int gj = next_jj + j;
                                tile_tmp_0[i][j] = (gi < NI && gj < NJ) ? tmp[gi][gj] : 0.0;
                            }
                        }
                        load_C_next_1:
                        for (int j = 0; j < TJ; j++) {
#pragma HLS PIPELINE II=1
#pragma HLS LOOP_TRIPCOUNT min=TJ max=TJ
                            for (int l = 0; l < TL; l++) {
#pragma HLS UNROLL
                                int gj = next_jj + j;
                                int gl = ll + l;
                                tile_C_0[j][l] = (gj < NJ && gl < NL) ? C[gj][gl] : 0.0;
                            }
                        }
                    }
                }

                // Compute from current buffer
                if (buf_sel == 0) {
                    compute2_0:
                    for (int i = 0; i < TI; i++) {
#pragma HLS PIPELINE II=1
#pragma HLS UNROLL factor=UI
#pragma HLS LOOP_TRIPCOUNT min=TI max=TI
#pragma HLS DEPENDENCE variable=acc_D inter false
                        for (int l = 0; l < TL; l++) {
#pragma HLS UNROLL
                            double sum = 0.0;
                            for (int j = 0; j < TJ; j++) {
#pragma HLS UNROLL
                                sum += tile_tmp_0[i][j] * tile_C_0[j][l];
                            }
                            acc_D[i][l] += sum;
                        }
                    }
                } else {
                    compute2_1:
                    for (int i = 0; i < TI; i++) {
#pragma HLS PIPELINE II=1
#pragma HLS UNROLL factor=UI
#pragma HLS LOOP_TRIPCOUNT min=TI max=TI
#pragma HLS DEPENDENCE variable=acc_D inter false
                        for (int l = 0; l < TL; l++) {
#pragma HLS UNROLL
                            double sum = 0.0;
                            for (int j = 0; j < TJ; j++) {
#pragma HLS UNROLL
                                sum += tile_tmp_1[i][j] * tile_C_1[j][l];
                            }
                            acc_D[i][l] += sum;
                        }
                    }
                }

            } // jj

            // Store D tile back to global memory
            store_D:
            for (int i = 0; i < TI; i++) {
#pragma HLS PIPELINE II=1
#pragma HLS UNROLL factor=UI
#pragma HLS LOOP_TRIPCOUNT min=TI max=TI
                for (int l = 0; l < TL; l++) {
#pragma HLS UNROLL
                    int gi = ii + i;
                    int gl = ll + l;
                    if (gi < NI && gl < NL)
                        D[gi][gl] = acc_D[i][l];
                }
            }

        } // ll
    } // ii
}

} // extern "C"