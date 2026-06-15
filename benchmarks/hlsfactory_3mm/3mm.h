#pragma once
// >>> c2hls auto-macro guards (do not edit between markers)
#ifndef NI
#define NI 40
#endif
#ifndef NJ
#define NJ 50
#endif
#ifndef NK
#define NK 60
#endif
#ifndef NL
#define NL 70
#endif
#ifndef NM
#define NM 80
#endif
// <<< c2hls auto-macro guards
#include <cmath>


extern "C" {
void kernel_3mm(    
		double E[ NI + 0][NJ + 0],
		double A[ NI + 0][NK + 0],
		double B[ NK + 0][NJ + 0],
		double F[ NJ + 0][NL + 0],
		double C[ NJ + 0][NM + 0],
		double D[ NM + 0][NL + 0],
		double G[ NI + 0][NL + 0]);
}