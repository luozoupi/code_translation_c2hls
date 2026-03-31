#include <stdio.h>
#include <stdlib.h>
#include <string.h>

//-----------------------------------------------
#define TILE_ROWS 256
#define ROWS 256
#define COLS 256
//-----------------------------------------------
#define R1 0
#define R2 127
#define C1 0
#define C2 127
#define LAMBDA 0.5 
#define PARA_FACTOR 16
#define NITER 2
#define TYPE float
#define TOP_TILE 0
#define BOTTOM_TILE (ROWS/TILE_ROWS - 1)

float srad_core1(float dN, float dS, float dW, float dE,
		  float Jc, float q0sqr);
float srad_core2 (float dN, float dS, float dW, float dE,
		  float cN, float cS, float cW, float cE,
		  float J);
void srad_kernel2(float J[(TILE_ROWS+3)*COLS], float Jout[TILE_ROWS*COLS], float q0sqr, int tile);

////////////////////////////////////////////////////////////////////////////////
// Test harness interface code.

struct bench_args_t {
  float J[(ROWS+3)*COLS];
  float Jout[(ROWS+3)*COLS];
};

// ============================================================================
// HLS-Optimized Kernel Implementation
// ============================================================================

float srad_core1 (float dN, float dS, float dW, float dE,
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
  return c;
}

float srad_core2 (float dN, float dS, float dW, float dE,
		  float cN, float cS, float cW, float cE,
		  float J) {
  float D, Jout;
  // divergence (equ 58)
  D = cN * dN + cS * dS + cW * dW + cE * dE;
  // image update (equ 61)
  Jout = J + 0.25*LAMBDA*D;
  return Jout;
}

void srad_kernel2(float J[(TILE_ROWS+3)*COLS], float Jout[TILE_ROWS*COLS], float q0sqr, int tile){
  int i, ii, j, k, iN, iS, jW, jE;

  float cN, cS, cW, cE, D;

  float J_top[PARA_FACTOR], J_left[PARA_FACTOR], J_right[PARA_FACTOR], J_bottom[PARA_FACTOR], J_center[PARA_FACTOR], c_tmp[PARA_FACTOR];
  #pragma HLS ARRAY_PARTITION variable=J_top complete dim=1
  #pragma HLS ARRAY_PARTITION variable=J_left complete dim=1
  #pragma HLS ARRAY_PARTITION variable=J_right complete dim=1
  #pragma HLS ARRAY_PARTITION variable=J_bottom complete dim=1
  #pragma HLS ARRAY_PARTITION variable=J_center complete dim=1
  #pragma HLS ARRAY_PARTITION variable=c_tmp complete dim=1

  float J_rf[PARA_FACTOR][COLS * 2 / PARA_FACTOR + 1];
  #pragma HLS ARRAY_PARTITION variable=J_rf complete dim=1
  
  float dN[(TILE_ROWS+1)*COLS];
  float dS[(TILE_ROWS+1)*COLS];
  float dW[(TILE_ROWS+1)*COLS];
  float dE[(TILE_ROWS+1)*COLS];
  float c[(TILE_ROWS+1)*COLS];
  
  // Initialize line buffer for J
  for (i = 0; i < COLS * 2 / PARA_FACTOR + 1; i++) {
    #pragma HLS PIPELINE II=1
    for (ii = 0; ii < PARA_FACTOR; ii++) {
      #pragma HLS UNROLL
      if (i * PARA_FACTOR + ii < (TILE_ROWS + 3) * COLS) {
        J_rf[ii][i] = J[i*PARA_FACTOR + ii];
      }
    }
  }

  // First pass: compute directional derivatives and diffusion coefficients
  for (i = -2*COLS/PARA_FACTOR-1; i < COLS / PARA_FACTOR * (TILE_ROWS+1); i++) {
    #pragma HLS PIPELINE II=1
    for (k = 0; k < PARA_FACTOR; k++) {
      #pragma HLS UNROLL
      //read from line buffer, handle borders as well
      J_center[k]  = J_rf[k][COLS / PARA_FACTOR];     
      J_top[k]     = (tile == TOP_TILE && i < COLS / PARA_FACTOR) ? J_center[k] : J_rf[k][0];
      J_left[k]    = ((i % (COLS / PARA_FACTOR)) == 0 && k == 0) ? J_center[k] : J_rf[(k - 1 + PARA_FACTOR) % PARA_FACTOR][COLS / PARA_FACTOR - (k == 0) ];
      J_right[k]   = ((i % (COLS / PARA_FACTOR)) == (COLS / PARA_FACTOR - 1) && k == PARA_FACTOR - 1) ? J_center[k] : J_rf[(k + 1 + PARA_FACTOR) % PARA_FACTOR][COLS / PARA_FACTOR + (k == (PARA_FACTOR - 1)) ];
      J_bottom[k]  = (tile == BOTTOM_TILE && i >= COLS / PARA_FACTOR * (TILE_ROWS - 1)) ? J_center[k] : J_rf[k][COLS / PARA_FACTOR * 2];

      if (i >= 0 && i < COLS / PARA_FACTOR * (TILE_ROWS+1)) {
	// directional derivates
      	dN[i*PARA_FACTOR+k] = J_top[k] - J_center[k];
      	dS[i*PARA_FACTOR+k] = J_bottom[k] - J_center[k];
      	dW[i*PARA_FACTOR+k] = J_left[k] - J_center[k];
      	dE[i*PARA_FACTOR+k] = J_right[k] - J_center[k];

	// call the stencil core
      	c_tmp[k] = srad_core1(dN[i*PARA_FACTOR+k],
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
      }
    }

    //shift the line buffer one by one
    for (k = 0; k < PARA_FACTOR; k++) {
      #pragma HLS UNROLL
      for (j = 0; j < COLS * 2 / PARA_FACTOR; j++) {
        #pragma HLS PIPELINE II=1
        J_rf[k][j] = J_rf[k][j + 1];
      }
      if ((i + 1) * PARA_FACTOR + k < (TILE_ROWS + 3) * COLS) {
        J_rf[k][COLS * 2 / PARA_FACTOR] = J[2*COLS + (i + 1) * PARA_FACTOR + k];
      }
    }
  }

  float c_right[PARA_FACTOR], c_bottom[PARA_FACTOR], c_center[PARA_FACTOR];
  #pragma HLS ARRAY_PARTITION variable=c_right complete dim=1
  #pragma HLS ARRAY_PARTITION variable=c_bottom complete dim=1
  #pragma HLS ARRAY_PARTITION variable=c_center complete dim=1
  
  float c_rf[PARA_FACTOR][COLS / PARA_FACTOR + 1];
  #pragma HLS ARRAY_PARTITION variable=c_rf complete dim=1
  
  // Initialize line buffer for c
  for (i = 0; i < COLS / PARA_FACTOR + 1; i++) {
    #pragma HLS PIPELINE II=1
    for (ii = 0; ii < PARA_FACTOR; ii++) {
      #pragma HLS UNROLL
      if (i * PARA_FACTOR + ii < (TILE_ROWS + 1) * COLS) {
        c_rf[ii][i] = c[i*PARA_FACTOR + ii];
      }
    }
  }
  
  // Second pass: compute output using divergence
  for (i = -COLS/PARA_FACTOR-1; i < COLS / PARA_FACTOR * TILE_ROWS; i++) {
    #pragma HLS PIPELINE II=1
    for (k = 0; k < PARA_FACTOR; k++) {
      #pragma HLS UNROLL
      //read from line buffer, handle borders as well
      c_center[k]  = c_rf[k][0];
      c_right[k]   = ((i % (COLS / PARA_FACTOR)) == (COLS / PARA_FACTOR - 1) && k == PARA_FACTOR - 1) ? c_center[k] : c_rf[(k + 1 + PARA_FACTOR) % PARA_FACTOR][ (k == (PARA_FACTOR - 1)) ];
      c_bottom[k]  = (tile == BOTTOM_TILE && i >= COLS / PARA_FACTOR * (TILE_ROWS - 1)) ? c_center[k] : c_rf[k][COLS / PARA_FACTOR];

      if (i >= 0 && i < COLS / PARA_FACTOR * TILE_ROWS) {
        Jout[i*PARA_FACTOR+k] = srad_core2(dN[i*PARA_FACTOR+k], dS[i*PARA_FACTOR+k],
					   dW[i*PARA_FACTOR+k], dE[i*PARA_FACTOR+k],
					   c_center[k], c_bottom[k], c_center[k], c_right[k],
					   J[COLS+i*PARA_FACTOR+k]);
      }
    }

    //shift the line buffer one by one
    for (k = 0; k < PARA_FACTOR; k++) {
      #pragma HLS UNROLL
      for (j = 0; j < COLS / PARA_FACTOR; j++) {
        #pragma HLS PIPELINE II=1
        c_rf[k][j] = c_rf[k][j + 1];
      }
      if (COLS + (i + 1) * PARA_FACTOR + k < (TILE_ROWS + 1) * COLS) {
        c_rf[k][COLS / PARA_FACTOR] = c[COLS + (i + 1) * PARA_FACTOR + k];
      }
    }
  }
}

extern "C" {

void workload(float *J, float *Jout) {
  #pragma HLS INTERFACE m_axi port=J offset=slave bundle=gmem max_write_burst_length=256 max_read_burst_length=256
  #pragma HLS INTERFACE m_axi port=Jout offset=slave bundle=gmem max_write_burst_length=256 max_read_burst_length=256
  #pragma HLS INTERFACE s_axilite port=J bundle=control
  #pragma HLS INTERFACE s_axilite port=Jout bundle=control
  #pragma HLS INTERFACE s_axilite port=return bundle=control
  
  float J_buf[(TILE_ROWS+3)*COLS];
  #pragma HLS ARRAY_PARTITION variable=J_buf cyclic factor=16 dim=1
  
  float Jout_buf[TILE_ROWS*COLS];
  #pragma HLS ARRAY_PARTITION variable=Jout_buf cyclic factor=16 dim=1

  int iter, t=0;
  float v0sqr = 0.0870038941502571;
  
  for (iter=0; iter<NITER/2; iter++){
    #pragma HLS LOOP_TRIPCOUNT min=1 max=1
    for (t = 0; t < ROWS/TILE_ROWS; t++) {
      #pragma HLS LOOP_TRIPCOUNT min=1 max=1
      memcpy(J_buf, J+t*TILE_ROWS*COLS, (TILE_ROWS+3)*COLS*sizeof(float));
      srad_kernel2(J_buf, Jout_buf, v0sqr, t);
      memcpy(Jout+(t*TILE_ROWS+1)*COLS, Jout_buf, TILE_ROWS*COLS*sizeof(float));
    }
    for (t = 0; t < ROWS/TILE_ROWS; t++) {
      #pragma HLS LOOP_TRIPCOUNT min=1 max=1
      memcpy(J_buf, Jout+t*TILE_ROWS*COLS, (TILE_ROWS+3)*COLS*sizeof(float));
      srad_kernel2(J_buf, Jout_buf, v0sqr, t);
      memcpy(J+(t*TILE_ROWS+1)*COLS, Jout_buf, TILE_ROWS*COLS*sizeof(float));
    }
  }

  return;
}

} // extern "C"