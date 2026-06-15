#pragma once
// >>> c2hls auto-macro guards (do not edit between markers)
#ifndef NI
#define NI 40
#endif
#ifndef NJ
#define NJ 50
#endif
#ifndef NK
#define NK 70
#endif
#ifndef NL
#define NL 80
#endif
// <<< c2hls auto-macro guards
#include <cmath>


extern "C" {
void kernel_2mm(   
		double alpha,
		double beta,
		double tmp[ NI + 0][NJ + 0],
		double A[ NI + 0][NK + 0],
		double B[ NK + 0][NJ + 0],
		double C[ NJ + 0][NL + 0],
		double D[ NI + 0][NL + 0]);
}