#pragma once
// >>> c2hls auto-macro guards (do not edit between markers)
#ifndef NR
#define NR 25
#endif
#ifndef NQ
#define NQ 20
#endif
#ifndef NP
#define NP 30
#endif
// <<< c2hls auto-macro guards
#include <cmath>


extern "C" {
void kernel_doitgen(  
		    double A[ NR + 0][NQ + 0][NP + 0],
		    double C4[ NP + 0][NP + 0],
		    double sum[ NP + 0]);
}