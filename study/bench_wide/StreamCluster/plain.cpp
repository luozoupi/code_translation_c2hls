#include "streamcluster.h"


void workload(    
    float* coord,                      
    float* weight,                      
    float* cost, 
    float* target,
    int* assign,
    int* center_table,
    char* switch_membership,
    float* work_mem,
    int num,
    float* cost_of_opening_x,
    int numcenter            
)
{


    int i, j;
    for( i = 0; i < BATCH_SIZE; i++ ) {
        float sum = 0;
        for( j = 0; j < DIM; j++ ) {
            float a = coord[i * DIM + j] - target[j];
            sum += a * a;
        }
        float current_cost = sum * weight[i] - cost[i];
        int local_center_index = center_table[assign[i]];
        if (current_cost < 0) {

            // point i would save cost just by switching to x
            // (note that i cannot be a median, 
            // or else dist(p[i], p[x]) would be 0)
      
            switch_membership[i] = 1;
            cost_of_opening_x[0] += current_cost;

        } 
        else {

            // cost of assigning i to x is at least current assignment cost of i

            // consider the savings that i's **current** median would realize
            // if we reassigned that median and all its members to x;
            // note we've already accounted for the fact that the median
            // would save z by closing; now we have to subtract from the savings
            // the extra cost of reassigning that median and its members 
            work_mem[local_center_index] -= current_cost;
        }
    }
    
    return;
}