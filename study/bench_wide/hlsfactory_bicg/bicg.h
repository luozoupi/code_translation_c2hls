#pragma once
// >>> c2hls auto-macro guards (do not edit between markers)
#ifndef N
#define N 124
#endif
#ifndef M
#define M 116
#endif
// <<< c2hls auto-macro guards
#include <cmath>


extern "C" {
void kernel_bicg( 
		 double A[ N + 0][M + 0],
		 double s[ M + 0],
		 double q[ N + 0],
		 double p[ M + 0],
		 double r[ N + 0]);
}