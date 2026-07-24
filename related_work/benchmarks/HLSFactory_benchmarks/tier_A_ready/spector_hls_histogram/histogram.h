#ifndef HISTOGRAM_H_
#define HISTOGRAM_H_
#include "params.h"

void histogram_hls(
    unsigned char data[DATA_SIZE / KNOB_NUM_WORK_GROUPS],
    unsigned int histogram[KNOB_HIST_SIZE],
    unsigned long offset);

#endif
