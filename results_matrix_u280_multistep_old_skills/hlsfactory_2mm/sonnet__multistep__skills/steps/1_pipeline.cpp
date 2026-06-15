#include "2mm.h"

// Tile sizes chosen to fit in local BRAM
#define TI 8
#define TJ 8
#define TK 8
#define TL 8

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
    // Tile over i, j, k dimensions
    // -------------------------------------------------------

    // Accumulator tile for tmp (one (TI x TJ) output tile at a time)
    double acc_tmp[TI][TJ];
    double tile_A[TI][TK];
    double tile_B[TK][TJ];
#pragma HLS ARRAY_PARTITION variable=acc_tmp complete dim=1
#pragma HLS ARRAY_PARTITION variable=acc_tmp complete dim=2
#pragma HLS ARRAY_PARTITION variable=tile_A  complete dim=1
#pragma HLS ARRAY_PARTITION variable=tile_A  complete dim=2
#pragma HLS ARRAY_PARTITION variable=tile_B  complete dim=1
#pragma HLS ARRAY_PARTITION variable=tile_B  complete dim=2

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
#pragma HLS LOOP_TRIPCOUNT min=TI max=TI
                for (int j = 0; j < TJ; j++) {
#pragma HLS UNROLL
                    acc_tmp[i][j] = 0.0;
                }
            }

            // Accumulate over k tiles
            phase1_k:
            for (int kk = 0; kk < NK; kk += TK) {
#pragma HLS LOOP_TRIPCOUNT min=9 max=9

                // Load tile of A[ii:ii+TI][kk:kk+TK]
                load_A:
                for (int i = 0; i < TI; i++) {
#pragma HLS PIPELINE II=1
#pragma HLS LOOP_TRIPCOUNT min=TI max=TI
                    for (int k = 0; k < TK; k++) {
#pragma HLS UNROLL
                        int gi = ii + i;
                        int gk = kk + k;
                        tile_A[i][k] = (gi < NI && gk < NK) ? A[gi][gk] : 0.0;
                    }
                }

                // Load tile of B[kk:kk+TK][jj:jj+TJ]
                load_B:
                for (int k = 0; k < TK; k++) {
#pragma HLS PIPELINE II=1
#pragma HLS LOOP_TRIPCOUNT min=TK max=TK
                    for (int j = 0; j < TJ; j++) {
#pragma HLS UNROLL
                        int gk = kk + k;
                        int gj = jj + j;
                        tile_B[k][j] = (gk < NK && gj < NJ) ? B[gk][gj] : 0.0;
                    }
                }

                // Compute tile contribution
                // Pipeline the i-loop; unroll j and k to enable full parallelism
                compute1:
                for (int i = 0; i < TI; i++) {
#pragma HLS PIPELINE II=1
#pragma HLS LOOP_TRIPCOUNT min=TI max=TI
#pragma HLS DEPENDENCE variable=acc_tmp inter false
                    for (int j = 0; j < TJ; j++) {
#pragma HLS UNROLL
                        double sum = 0.0;
                        for (int k = 0; k < TK; k++) {
#pragma HLS UNROLL
                            sum += alpha * tile_A[i][k] * tile_B[k][j];
                        }
                        acc_tmp[i][j] += sum;
                    }
                }
            } // kk

            // Store accumulated tmp tile to global memory
            store_tmp:
            for (int i = 0; i < TI; i++) {
#pragma HLS PIPELINE II=1
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
    // Tile over i, l, k(=j) dimensions
    // -------------------------------------------------------

    double acc_D[TI][TL];
    double tile_tmp[TI][TJ];
    double tile_C[TJ][TL];
#pragma HLS ARRAY_PARTITION variable=acc_D    complete dim=1
#pragma HLS ARRAY_PARTITION variable=acc_D    complete dim=2
#pragma HLS ARRAY_PARTITION variable=tile_tmp complete dim=1
#pragma HLS ARRAY_PARTITION variable=tile_tmp complete dim=2
#pragma HLS ARRAY_PARTITION variable=tile_C   complete dim=1
#pragma HLS ARRAY_PARTITION variable=tile_C   complete dim=2

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
#pragma HLS LOOP_TRIPCOUNT min=TI max=TI
                for (int l = 0; l < TL; l++) {
#pragma HLS UNROLL
                    int gi = ii + i;
                    int gl = ll + l;
                    acc_D[i][l] = (gi < NI && gl < NL) ? beta * D[gi][gl] : 0.0;
                }
            }

            // Accumulate over j tiles (k in the second loop nest uses j as reduction var)
            phase2_k:
            for (int jj = 0; jj < NJ; jj += TJ) {
#pragma HLS LOOP_TRIPCOUNT min=7 max=7

                // Load tile of tmp[ii:ii+TI][jj:jj+TJ]
                load_tmp:
                for (int i = 0; i < TI; i++) {
#pragma HLS PIPELINE II=1
#pragma HLS LOOP_TRIPCOUNT min=TI max=TI
                    for (int j = 0; j < TJ; j++) {
#pragma HLS UNROLL
                        int gi = ii + i;
                        int gj = jj + j;
                        tile_tmp[i][j] = (gi < NI && gj < NJ) ? tmp[gi][gj] : 0.0;
                    }
                }

                // Load tile of C[jj:jj+TJ][ll:ll+TL]
                load_C:
                for (int j = 0; j < TJ; j++) {
#pragma HLS PIPELINE II=1
#pragma HLS LOOP_TRIPCOUNT min=TJ max=TJ
                    for (int l = 0; l < TL; l++) {
#pragma HLS UNROLL
                        int gj = jj + j;
                        int gl = ll + l;
                        tile_C[j][l] = (gj < NJ && gl < NL) ? C[gj][gl] : 0.0;
                    }
                }

                // Compute tile contribution
                // Pipeline the i-loop; unroll l and j to enable full parallelism
                compute2:
                for (int i = 0; i < TI; i++) {
#pragma HLS PIPELINE II=1
#pragma HLS LOOP_TRIPCOUNT min=TI max=TI
#pragma HLS DEPENDENCE variable=acc_D inter false
                    for (int l = 0; l < TL; l++) {
#pragma HLS UNROLL
                        double sum = 0.0;
                        for (int j = 0; j < TJ; j++) {
#pragma HLS UNROLL
                            sum += tile_tmp[i][j] * tile_C[j][l];
                        }
                        acc_D[i][l] += sum;
                    }
                }
            } // jj

            // Store D tile back to global memory
            store_D:
            for (int i = 0; i < TI; i++) {
#pragma HLS PIPELINE II=1
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