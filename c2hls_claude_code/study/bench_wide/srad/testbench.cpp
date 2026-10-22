/*
 * Strict Vitis HLS C-simulation testbench for srad.
 * Independent golden: plain.cpp (srad_core1/2, srad_kernel2, workload) copied below with _ref suffix and run on
 * identical copies of the deterministic inputs (srand(42)). workload(J, Jout), both (ROWS+3)*COLS:
 *   - J    : in-out (real data in, rows 1..ROWS overwritten by the second pass) -> same data for ref and DUT
 *   - Jout : rows 1..ROWS are pure output (written before read) -> poisoned with 12345.0f; rows 0, ROWS+1, ROWS+2
 *            are read by the second pass but never written, so they are in-out border data (a copy of J).
 * ALL (ROWS+3)*COLS elements of J and of Jout are checked (rows plain.cpp does not write must keep their initial
 * values). Float tolerance 1e-3*fmax(1,|ref|) (SPEC); NaN is an error.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "srad.h"

extern "C" void workload(float J[(ROWS+3)*COLS], float Jout[(ROWS+3)*COLS]);

/* ---- reference: copy of plain.cpp ---- */

static float srad_core1_ref (float dN, float dS, float dW, float dE,
		  float Jc, float q0sqr) {
  float G2, L, num, den, qsqr, c;
  
  G2 = (dN*dN + dS*dS + dW*dW + dE*dE) / (Jc*Jc);

  L = (dN + dS + dW + dE) / Jc;

  num  = (0.5*G2) - ((1.0/16.0)*(L*L)) ;
  den  = 1 + (.25*L);
  qsqr = num/(den*den);
 
  // diffusion coefficent (equ 33)
  den = (qsqr-q0sqr) / (q0sqr * (1+q0sqr)) ;
  c = 1.0 / (1.0+den) ;
  //printf("core1: d = %.16f, %.16f, %.16f, %.16f; Jc = %.16f, q0sqr = %.16f, den = %.16f, c = %.16f\n", dN, dS, dW, dE, Jc, q0sqr, den, c);
  return c;
}

static float srad_core2_ref (float dN, float dS, float dW, float dE,
		  float cN, float cS, float cW, float cE,
		  float J) {
  float D, Jout;
  // divergence (equ 58)
  D = cN * dN + cS * dS + cW * dW + cE * dE;
  //printf("core2: c = %.16f, %.16f, %.16f, %.16f; d = %.16f, %.16f, %.16f, %.16f\n", cN, cS, cW, cE, dN, dS, dW, dE);
  //printf("core2: D = %.16f\n", D);              
  // image update (equ 61)
  Jout = J + 0.25*LAMBDA*D;
  return Jout;
}



static void srad_kernel2_ref(float J[(TILE_ROWS+3)*COLS], float Jout[TILE_ROWS*COLS], float q0sqr, int tile){
  int i, ii, j, k, iN, iS, jW, jE;

  float cN, cS, cW, cE, D;

  float J_top[PARA_FACTOR], J_left[PARA_FACTOR], J_right[PARA_FACTOR], J_bottom[PARA_FACTOR], J_center[PARA_FACTOR], c_tmp[PARA_FACTOR];

  float J_rf[PARA_FACTOR][COLS * 2 / PARA_FACTOR + 1];

  float dN[(TILE_ROWS+1)*COLS];
  
  float dS[(TILE_ROWS+1)*COLS];
  
  float dW[(TILE_ROWS+1)*COLS];
  
  float dE[(TILE_ROWS+1)*COLS];
  
  float c[(TILE_ROWS+1)*COLS];
  
  //initialize the line buffer
  /*KERNEL1: for (i = 0; i < COLS * 2 / PARA_FACTOR + 1; i++) {
    for (ii = 0; ii < PARA_FACTOR; ii++) {
      J_rf[ii][i] = J[i*PARA_FACTOR + ii];
    }
  }*/

  /*printf ("========q0sqr = %.16f\n", q0sqr);
  for (i = 0; i < (TILE_ROWS+3)*COLS; i++)
  printf("J[%d] = %.16f\n", i, J[i]);*/
  
  MAIN_KERNEL1: for (i = -2*COLS/PARA_FACTOR-1; i < COLS / PARA_FACTOR * (TILE_ROWS+1); i++) {
    for (k = 0; k < PARA_FACTOR; k++) {
      //read from line buffer, handle borders as well
      J_center[k]  = J_rf[k][COLS / PARA_FACTOR];     
      J_top[k]     = (tile == TOP_TILE && i < COLS / PARA_FACTOR) ? J_center[k] : J_rf[k][0];
      J_left[k]    = ((i % (COLS / PARA_FACTOR)) == 0 && k == 0) ? J_center[k] : J_rf[(k - 1 + PARA_FACTOR) % PARA_FACTOR][COLS / PARA_FACTOR - (k == 0) ];
      J_right[k]   = ((i % (COLS / PARA_FACTOR)) == (COLS / PARA_FACTOR - 1) && k == PARA_FACTOR - 1) ? J_center[k] : J_rf[(k + 1 + PARA_FACTOR) % PARA_FACTOR][COLS / PARA_FACTOR + (k == (PARA_FACTOR - 1)) ];
      J_bottom[k]  = (tile == BOTTOM_TILE && i >= COLS / PARA_FACTOR * (TILE_ROWS - 1)) ? J_center[k] : J_rf[k][COLS / PARA_FACTOR * 2];

      if (i >= 0) {
	// directional derivates
	// note that in srad, we have two stencil cores
	// and we have to store the intermediate data
      	dN[i*PARA_FACTOR+k] = J_top[k] - J_center[k];
      	dS[i*PARA_FACTOR+k] = J_bottom[k] - J_center[k];
      	dW[i*PARA_FACTOR+k] = J_left[k] - J_center[k];
      	dE[i*PARA_FACTOR+k] = J_right[k] - J_center[k];

	// call the stencil core
      	c_tmp[k] = srad_core1_ref(dN[i*PARA_FACTOR+k],
      			      dS[i*PARA_FACTOR+k],
      			      dW[i*PARA_FACTOR+k],
      			      dE[i*PARA_FACTOR+k],
      			      J_center[k], q0sqr);
                
	// saturate diffusion coefficent
      	if (c_tmp[k] < 0) {
          c[i*PARA_FACTOR+k] = 0;
        }
      	else if (c_tmp[k] > 1) {
          c[i*PARA_FACTOR+k] = 1;
        }
      	else {
          c[i*PARA_FACTOR+k] = c_tmp[k];
        }
	//printf("index = %d, c_tmp = %.16f, c = %.16f\n", i*PARA_FACTOR+k, c_tmp[k], c[i*PARA_FACTOR+k]);
      }
    }

    //shift the line buffer one by one
    for (k = 0; k < PARA_FACTOR; k++) {
      for (j = 0; j < COLS * 2 / PARA_FACTOR; j++) {
        J_rf[k][j] = J_rf[k][j + 1];
      }

      J_rf[k][COLS * 2 / PARA_FACTOR] = J[2*COLS + (i + 1) * PARA_FACTOR + k];
    }
  }//*/
  float c_right[PARA_FACTOR], c_bottom[PARA_FACTOR], c_center[PARA_FACTOR];

  float c_rf[PARA_FACTOR][COLS / PARA_FACTOR + 1];
  
  //initialize the line buffer
  /*KERNEL2: for (i = 0; i < COLS / PARA_FACTOR + 1; i++) {
    for (ii = 0; ii < PARA_FACTOR; ii++) {
      c_rf[ii][i] = c[i*PARA_FACTOR + ii];
    }
  }*/
  
  MAIN_KERNEL2: for (i = -COLS/PARA_FACTOR-1; i < COLS / PARA_FACTOR * TILE_ROWS; i++) {
    for (k = 0; k < PARA_FACTOR; k++) {
      //read from line buffer, handle borders as well
      c_center[k]  = c_rf[k][0];
      c_right[k]   = ((i % (COLS / PARA_FACTOR)) == (COLS / PARA_FACTOR - 1) && k == PARA_FACTOR - 1) ? c_center[k] : c_rf[(k + 1 + PARA_FACTOR) % PARA_FACTOR][ (k == (PARA_FACTOR - 1)) ];
      c_bottom[k]  = (tile == BOTTOM_TILE && i >= COLS / PARA_FACTOR * (TILE_ROWS - 1)) ? c_center[k] : c_rf[k][COLS / PARA_FACTOR];

      if (i >= 0) {
        Jout[i*PARA_FACTOR+k] = srad_core2_ref(dN[i*PARA_FACTOR+k], dS[i*PARA_FACTOR+k],
					   dW[i*PARA_FACTOR+k], dE[i*PARA_FACTOR+k],
					   c_center[k], c_bottom[k], c_center[k], c_right[k],
					   J[COLS+i*PARA_FACTOR+k]);
        //printf("========inside Jout[%d]=%.16f\n", i*PARA_FACTOR+k, Jout[i*PARA_FACTOR+k]);
      }
    }

    //shift the line buffer one by one
    for (k = 0; k < PARA_FACTOR; k++) {
      for (j = 0; j < COLS / PARA_FACTOR; j++) {
        c_rf[k][j] = c_rf[k][j + 1];
      }

      c_rf[k][COLS / PARA_FACTOR] = c[COLS + (i + 1) * PARA_FACTOR + k];
    }
  }
}


  
static void workload_ref(float J[(ROWS+3)*COLS], float Jout[(ROWS+3)*COLS]) {
  
  
  float J_buf[(TILE_ROWS+3)*COLS];
  //assume C2-C1 > PARA_FACTOR and (C2-C1)%PARA_FACTOR == 0
  float Jout_buf[TILE_ROWS*COLS];

  int iter, t=0;
  float v0sqr = 0.0870038941502571;
  //assume NITER%2 == 0
  ITER_LOOP: for (iter=0; iter<NITER/2; iter++){
    //srad_kernel1(J, &v0sqr);
    //assume ROWS%TILE_ROWS == 0
    for (t = 0; t < ROWS/TILE_ROWS; t++) {
      memcpy(J_buf, J+t*TILE_ROWS*COLS, (TILE_ROWS+3)*COLS*sizeof(float));
      srad_kernel2_ref(J_buf, Jout_buf, v0sqr, t);
      memcpy(Jout+(t*TILE_ROWS+1)*COLS, Jout_buf, TILE_ROWS*COLS*sizeof(float));
    }
    //srad_kernel1(Jout, &v0sqr);
    for (t = 0; t < ROWS/TILE_ROWS; t++) {
      memcpy(J_buf, Jout+t*TILE_ROWS*COLS, (TILE_ROWS+3)*COLS*sizeof(float));
      srad_kernel2_ref(J_buf, Jout_buf, v0sqr, t);
      memcpy(J+(t*TILE_ROWS+1)*COLS, Jout_buf, TILE_ROWS*COLS*sizeof(float));
    }
  }

  return;
}
  

/* ---- end reference ---- */

static int errors = 0;
static void chk_f(const char *name, int i, float d, float r) {
    if (!(fabsf(d - r) <= 1e-3f * fmaxf(1.0f, fabsf(r)))) {   /* also catches NaN */
        if (errors < 10) printf("%s[%d]: got %g expected %g\n", name, i, (double)d, (double)r);
        errors++;
    }
}

int main() {
    int total = (ROWS+3) * COLS;
    srand(42);

    float* J_ref   = new float[total];
    float* J_dut   = new float[total];
    float* Out_ref = new float[total];
    float* Out_dut = new float[total];

    /* positive image data for valid diffusion */
    for (int i = 0; i < total; i++)
        J_ref[i] = 50.0f + (float)(rand() % 200) / 10.0f;

    memcpy(J_dut, J_ref, total * sizeof(float));
    /* border rows of Jout (0, ROWS+1, ROWS+2) are in-out data; rows 1..ROWS are poisoned */
    memcpy(Out_ref, J_ref, total * sizeof(float));
    memcpy(Out_dut, J_ref, total * sizeof(float));
    for (int r = 1; r <= ROWS; r++)
        for (int c = 0; c < COLS; c++) { Out_ref[r*COLS + c] = 12345.0f; Out_dut[r*COLS + c] = 12345.0f; }

    workload_ref(J_ref, Out_ref);
    workload(J_dut, Out_dut);

    for (int i = 0; i < total; i++) chk_f("Jout", i, Out_dut[i], Out_ref[i]);
    for (int i = 0; i < total; i++) chk_f("J", i, J_dut[i], J_ref[i]);

    delete[] J_ref; delete[] J_dut;
    delete[] Out_ref; delete[] Out_dut;
    if (errors) { printf("Failed with %d errors!\n", errors); return 1; }
    printf("Passed!\n");
    return 0;
}
