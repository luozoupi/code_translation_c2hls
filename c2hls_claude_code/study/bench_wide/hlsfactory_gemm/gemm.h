#pragma once
// >>> c2hls auto-macro guards (do not edit between markers)
#ifndef NI
#define NI 60
#endif
#ifndef NJ
#define NJ 70
#endif
#ifndef NK
#define NK 80
#endif
// <<< c2hls auto-macro guards
#include <cmath>


extern "C" {
void kernel_gemm(  
		 double alpha,
		 double beta,
		 double C[ NI + 0][NJ + 0],
		 double A[ NI + 0][NK + 0],
		 double B[ NK + 0][NJ + 0]);
}