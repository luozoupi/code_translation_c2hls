#include "kmeans.h"
#include <string.h>

// Load a tile of features from global memory into local buffer
static void load_features(float *feature, float local_features[TILE_SIZE][NFEATURES], int tile_idx)
{
    int base = tile_idx * TILE_SIZE * NFEATURES;
    for (int i = 0; i < TILE_SIZE; i++) {
        for (int k = 0; k < NFEATURES; k++) {
#pragma HLS PIPELINE II=1
            local_features[i][k] = feature[base + i * NFEATURES + k];
        }
    }
}

// Load clusters from global memory into local buffer
static void load_clusters(float *clusters, float local_clusters[NCLUSTERS][NFEATURES])
{
    for (int j = 0; j < NCLUSTERS; j++) {
        for (int k = 0; k < NFEATURES; k++) {
#pragma HLS PIPELINE II=1
            local_clusters[j][k] = clusters[j * NFEATURES + k];
        }
    }
}

// Compute membership for all points in the current tile
static void compute_membership(float local_features[TILE_SIZE][NFEATURES],
                                float local_clusters[NCLUSTERS][NFEATURES],
                                int local_membership[TILE_SIZE])
{
#pragma HLS ARRAY_PARTITION variable=local_clusters cyclic factor=4 dim=2
#pragma HLS ARRAY_PARTITION variable=local_features cyclic factor=4 dim=2

    for (int i = 0; i < TILE_SIZE; i++) {
#pragma HLS LOOP_TRIPCOUNT min=4096 max=4096
        float min_dist = FLT_MAX;
        int index = 0;

        for (int j = 0; j < NCLUSTERS; j++) {
#pragma HLS LOOP_TRIPCOUNT min=5 max=5
            float dist = 0.0f;

            for (int k = 0; k < NFEATURES; k++) {
#pragma HLS LOOP_TRIPCOUNT min=34 max=34
#pragma HLS PIPELINE II=1
#pragma HLS DEPENDENCE variable=dist inter false
                float diff = local_features[i][k] - local_clusters[j][k];
                dist += diff * diff;
            }

            if (dist < min_dist) {
                min_dist = dist;
                index = j;
            }
        }

        local_membership[i] = index;
    }
}

// Store membership results for the current tile back to global memory
static void store_membership(int *membership, int local_membership[TILE_SIZE], int tile_idx)
{
    int base = tile_idx * TILE_SIZE;
    for (int i = 0; i < TILE_SIZE; i++) {
#pragma HLS PIPELINE II=1
        membership[base + i] = local_membership[i];
    }
}

extern "C" void workload(float *feature, float *clusters, int *membership)
{
#pragma HLS INTERFACE m_axi port=feature    offset=slave bundle=gmem
#pragma HLS INTERFACE m_axi port=clusters   offset=slave bundle=gmem
#pragma HLS INTERFACE m_axi port=membership offset=slave bundle=gmem
#pragma HLS INTERFACE s_axilite port=feature    bundle=control
#pragma HLS INTERFACE s_axilite port=clusters   bundle=control
#pragma HLS INTERFACE s_axilite port=membership bundle=control
#pragma HLS INTERFACE s_axilite port=return     bundle=control

    // Local buffers
    float local_features[TILE_SIZE][NFEATURES];
    float local_clusters[NCLUSTERS][NFEATURES];
    int   local_membership[TILE_SIZE];

    // Load clusters once (small buffer, reused for all tiles)
    load_clusters(clusters, local_clusters);

    // Process data tile by tile
    TILE_LOOP: for (int t = 0; t < NUM_TILES; t++) {
#pragma HLS LOOP_TRIPCOUNT min=100 max=100
        // Phase 1: Load tile of features from global memory
        load_features(feature, local_features, t);

        // Phase 2: Compute membership for all points in this tile
        compute_membership(local_features, local_clusters, local_membership);

        // Phase 3: Store membership results back to global memory
        store_membership(membership, local_membership, t);
    }
}