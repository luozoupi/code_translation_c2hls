#include "fft.h"

#define THREADS 64
#define cmplx_M_x(a_x, a_y, b_x, b_y) (a_x*b_x - a_y*b_y)
#define cmplx_M_y(a_x, a_y, b_x, b_y) (a_x*b_y + a_y*b_x)
#define cmplx_mul_x(a_x, a_y, b_x, b_y) (a_x*b_x - a_y*b_y)
#define cmplx_mul_y(a_x, a_y, b_x, b_y) (a_x*b_y + a_y*b_x)
#define cmplx_add_x(a_x, b_x) (a_x + b_x)
#define cmplx_add_y(a_y, b_y) (a_y + b_y)
#define cmplx_sub_x(a_x, b_x) (a_x - b_x)
#define cmplx_sub_y(a_y, b_y) (a_y - b_y)
#define cm_fl_mul_x(a_x, b) (b*a_x)
#define cm_fl_mul_y(a_y, b) (b*a_y)

void twiddles8(TYPE a_x[8], TYPE a_y[8], int i, int n) {
    int reversed8[8] = {0,4,2,6,1,5,3,7};
    int j;
    TYPE phi, tmp, phi_x, phi_y;
    twiddles: for (j = 1; j < 8; j++) {
        phi = ((-2*PI*reversed8[j]/n)*i);
        phi_x = cos(phi);
        phi_y = sin(phi);
        tmp = a_x[j];
        a_x[j] = cmplx_M_x(a_x[j], a_y[j], phi_x, phi_y);
        a_y[j] = cmplx_M_y(tmp, a_y[j], phi_x, phi_y);
    }
}

#define FF2(a0_x, a0_y, a1_x, a1_y){ \
    TYPE c0_x = *a0_x; TYPE c0_y = *a0_y; \
    *a0_x = cmplx_add_x(c0_x, *a1_x); *a0_y = cmplx_add_y(c0_y, *a1_y); \
    *a1_x = cmplx_sub_x(c0_x, *a1_x); *a1_y = cmplx_sub_y(c0_y, *a1_y); \
}

#define FFT4(a0_x, a0_y, a1_x, a1_y, a2_x, a2_y, a3_x, a3_y){ \
    TYPE exp_1_44_x, exp_1_44_y, tmp; \
    exp_1_44_x = 0.0; exp_1_44_y = -1.0; \
    FF2(a0_x, a0_y, a2_x, a2_y); FF2(a1_x, a1_y, a3_x, a3_y); \
    tmp = *a3_x; *a3_x = *a3_x*exp_1_44_x-*a3_y*exp_1_44_y; \
    *a3_y = tmp*exp_1_44_y - *a3_y*exp_1_44_x; \
    FF2(a0_x, a0_y, a1_x, a1_y); FF2(a2_x, a2_y, a3_x, a3_y); \
}

#define FFT8(a_x, a_y) { \
    TYPE exp_1_8_x=1, exp_1_4_x=0, exp_3_8_x=-1; \
    TYPE exp_1_8_y=-1, exp_1_4_y=-1, exp_3_8_y=-1; \
    TYPE tmp_1; \
    FF2(&a_x[0],&a_y[0],&a_x[4],&a_y[4]); FF2(&a_x[1],&a_y[1],&a_x[5],&a_y[5]); \
    FF2(&a_x[2],&a_y[2],&a_x[6],&a_y[6]); FF2(&a_x[3],&a_y[3],&a_x[7],&a_y[7]); \
    tmp_1=a_x[5]; a_x[5]=cm_fl_mul_x(cmplx_mul_x(a_x[5],a_y[5],exp_1_8_x,exp_1_8_y),M_SQRT1_2); \
    a_y[5]=cm_fl_mul_y(cmplx_mul_y(tmp_1,a_y[5],exp_1_8_x,exp_1_8_y),M_SQRT1_2); \
    tmp_1=a_x[6]; a_x[6]=cmplx_mul_x(a_x[6],a_y[6],exp_1_4_x,exp_1_4_y); \
    a_y[6]=cmplx_mul_y(tmp_1,a_y[6],exp_1_4_x,exp_1_4_y); \
    tmp_1=a_x[7]; a_x[7]=cm_fl_mul_x(cmplx_mul_x(a_x[7],a_y[7],exp_3_8_x,exp_3_8_y),M_SQRT1_2); \
    a_y[7]=cm_fl_mul_y(cmplx_mul_y(tmp_1,a_y[7],exp_3_8_x,exp_3_8_y),M_SQRT1_2); \
    FFT4(&a_x[0],&a_y[0],&a_x[1],&a_y[1],&a_x[2],&a_y[2],&a_x[3],&a_y[3]); \
    FFT4(&a_x[4],&a_y[4],&a_x[5],&a_y[5],&a_x[6],&a_y[6],&a_x[7],&a_y[7]); \
}

void loady8(TYPE a_y[], TYPE x[], int offset, int sx) {
    for (int k = 0; k < 8; k++) {
        a_y[k] = x[k*sx+offset];
    }
}

void fft1D_512(TYPE work_x[512], TYPE work_y[512]) {
    int tid, hi, lo, stride;
    int reversed[] = {0,4,2,6,1,5,3,7};
    TYPE DATA_x[THREADS*8], DATA_y[THREADS*8];
    TYPE data_x[8], data_y[8];
    #pragma HLS ARRAY_PARTITION variable=data_x complete dim=1
    #pragma HLS ARRAY_PARTITION variable=data_y complete dim=1
    
    TYPE smem[8*8*9];
    stride = THREADS;

    loop1: for (tid = 0; tid < THREADS; tid++) {
        for (int k = 0; k < 8; k++) {
            data_x[k] = work_x[k*stride+tid];
            data_y[k] = work_y[k*stride+tid];
        }
        FFT8(data_x, data_y);
        twiddles8(data_x, data_y, tid, 512);
        for (int k = 0; k < 8; k++) {
            DATA_x[tid*8+k] = data_x[k];
            DATA_y[tid*8+k] = data_y[k];
        }
    }
    
    int sx, offset;
    sx = 66;
    loop2: for (tid = 0; tid < 64; tid++) {
        hi=tid>>3; lo=tid&7; offset=hi*8+lo;
        smem[0*sx+offset]=DATA_x[tid*8+0]; smem[4*sx+offset]=DATA_x[tid*8+1];
        smem[1*sx+offset]=DATA_x[tid*8+4]; smem[5*sx+offset]=DATA_x[tid*8+5];
        smem[2*sx+offset]=DATA_x[tid*8+2]; smem[6*sx+offset]=DATA_x[tid*8+3];
        smem[3*sx+offset]=DATA_x[tid*8+6]; smem[7*sx+offset]=DATA_x[tid*8+7];
    }
    
    sx = 8;
    loop3: for (tid = 0; tid < 64; tid++) {
        hi=tid>>3; lo=tid&7; offset=lo*66+hi;
        DATA_x[tid*8+0]=smem[0*sx+offset]; DATA_x[tid*8+4]=smem[4*sx+offset];
        DATA_x[tid*8+1]=smem[1*sx+offset]; DATA_x[tid*8+5]=smem[5*sx+offset];
        DATA_x[tid*8+2]=smem[2*sx+offset]; DATA_x[tid*8+6]=smem[6*sx+offset];
        DATA_x[tid*8+3]=smem[3*sx+offset]; DATA_x[tid*8+7]=smem[7*sx+offset];
    }
    
    sx = 66;
    loop4: for (tid = 0; tid < 64; tid++) {
        hi=tid>>3; lo=tid&7; offset=hi*8+lo;
        smem[0*sx+offset]=DATA_y[tid*8+0]; smem[4*sx+offset]=DATA_y[tid*8+1];
        smem[1*sx+offset]=DATA_y[tid*8+4]; smem[5*sx+offset]=DATA_y[tid*8+5];
        smem[2*sx+offset]=DATA_y[tid*8+2]; smem[6*sx+offset]=DATA_y[tid*8+3];
        smem[3*sx+offset]=DATA_y[tid*8+6]; smem[7*sx+offset]=DATA_y[tid*8+7];
    }
    
    loop5: for (tid = 0; tid < 64; tid++) {
        hi=tid>>3; lo=tid&7;
        loady8(data_y, smem, lo*66+hi, 8);
        for (int k = 0; k < 8; k++) {
            DATA_y[tid*8+k] = data_y[k];
        }
    }
    
    loop6: for (tid = 0; tid < 64; tid++) {
        for (int k = 0; k < 8; k++) {
            data_x[k]=DATA_x[tid*8+k];
            data_y[k]=DATA_y[tid*8+k];
        }
        FFT8(data_x, data_y);
        hi=tid>>3;
        twiddles8(data_x, data_y, hi, 64);
        for (int k = 0; k < 8; k++) {
            DATA_x[tid*8+k]=data_x[k];
            DATA_y[tid*8+k]=data_y[k];
        }
    }
    
    sx = 72;
    loop7: for (tid = 0; tid < 64; tid++) {
        hi=tid>>3; lo=tid&7; offset=hi*8+lo;
        smem[0*sx+offset]=DATA_x[tid*8+0]; smem[4*sx+offset]=DATA_x[tid*8+1];
        smem[1*sx+offset]=DATA_x[tid*8+4]; smem[5*sx+offset]=DATA_x[tid*8+5];
        smem[2*sx+offset]=DATA_x[tid*8+2]; smem[6*sx+offset]=DATA_x[tid*8+3];
        smem[3*sx+offset]=DATA_x[tid*8+6]; smem[7*sx+offset]=DATA_x[tid*8+7];
    }
    
    sx = 8;
    loop8: for (tid = 0; tid < 64; tid++) {
        hi=tid>>3; lo=tid&7; offset=hi*72+lo;
        DATA_x[tid*8+0]=smem[0*sx+offset]; DATA_x[tid*8+4]=smem[4*sx+offset];
        DATA_x[tid*8+1]=smem[1*sx+offset]; DATA_x[tid*8+5]=smem[5*sx+offset];
        DATA_x[tid*8+2]=smem[2*sx+offset]; DATA_x[tid*8+6]=smem[6*sx+offset];
        DATA_x[tid*8+3]=smem[3*sx+offset]; DATA_x[tid*8+7]=smem[7*sx+offset];
    }
    
    sx = 72;
    loop9: for (tid = 0; tid < 64; tid++) {
        hi=tid>>3; lo=tid&7; offset=hi*8+lo;
        smem[0*sx+offset]=DATA_y[tid*8+0]; smem[4*sx+offset]=DATA_y[tid*8+1];
        smem[1*sx+offset]=DATA_y[tid*8+4]; smem[5*sx+offset]=DATA_y[tid*8+5];
        smem[2*sx+offset]=DATA_y[tid*8+2]; smem[6*sx+offset]=DATA_y[tid*8+3];
        smem[3*sx+offset]=DATA_y[tid*8+6]; smem[7*sx+offset]=DATA_y[tid*8+7];
    }
    
    loop10: for (tid = 0; tid < 64; tid++) {
        hi=tid>>3; lo=tid&7;
        loady8(data_y, smem, hi*72+lo, 8);
        for (int k = 0; k < 8; k++) {
            DATA_y[tid*8+k] = data_y[k];
        }
    }
    
    loop11: for (tid = 0; tid < 64; tid++) {
        for (int k = 0; k < 8; k++) {
            data_x[k]=DATA_x[tid*8+k];
            data_y[k]=DATA_y[tid*8+k];
        }
        FFT8(data_x, data_y);
        for (int k = 0; k < 8; k++) {
            work_x[k*stride+tid]=data_x[reversed[k]];
            work_y[k*stride+tid]=data_y[reversed[k]];
        }
    }
}

extern "C" {
void workload(TYPE* work_x, TYPE* work_y) {
    #pragma HLS INTERFACE m_axi port=work_x offset=slave bundle=gmem
    #pragma HLS INTERFACE m_axi port=work_y offset=slave bundle=gmem
    #pragma HLS INTERFACE s_axilite port=work_x bundle=control
    #pragma HLS INTERFACE s_axilite port=work_y bundle=control
    #pragma HLS INTERFACE s_axilite port=return bundle=control

    TYPE l_x[512], l_y[512];
    
    int i;
    copy_in: for (i = 0; i < 512; i++) {
        l_x[i] = work_x[i];
        l_y[i] = work_y[i];
    }
    
    fft1D_512(l_x, l_y);
    
    copy_out: for (i = 0; i < 512; i++) {
        work_x[i] = l_x[i];
        work_y[i] = l_y[i];
    }
}
}