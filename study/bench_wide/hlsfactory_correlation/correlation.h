#pragma once
// >>> c2hls auto-macro guards (do not edit between markers)
#ifndef N
#define N 100
#endif
#ifndef M
#define M 80
#endif
// <<< c2hls auto-macro guards
#include <cmath>


extern "C" {
void kernel_correlation( 
			double float_n,
			double data[ N + 0][M + 0],
			double corr[ M + 0][M + 0],
			double mean[ M + 0],
			double stddev[ M + 0]);
}