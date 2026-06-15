#pragma once
// >>> c2hls auto-macro guards (do not edit between markers)
#ifndef N
#define N 180
#endif
// <<< c2hls auto-macro guards
#include <cmath>


extern "C" {
void kernel_nussinov( char seq[ N + 0],
			   int table[ N + 0][N + 0]);
}