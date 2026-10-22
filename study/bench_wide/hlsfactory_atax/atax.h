#pragma once
// >>> c2hls auto-macro guards (do not edit between markers)
#ifndef M
#define M 116
#endif
#ifndef N
#define N 124
#endif
// <<< c2hls auto-macro guards
#include <cmath>


extern "C" {
void kernel_atax( 
		 double A[ M + 0][N + 0],
		 double x[ N + 0],
		 double y[ N + 0],
		 double tmp[ M + 0]);
}