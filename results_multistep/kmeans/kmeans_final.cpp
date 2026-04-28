#include "kmeans.h"

extern "C" {

/* Load function: reads feature data into the specified buffer set */
void load_features(float *feature, float *local_features_1, float *local_features_2, 
                   int tile_id, int buffer_flag) {
    int start_idx = tile_id * TILE_SIZE;
    int end_idx = start_idx + TILE_SIZE;
    
    if (buffer_flag == 0) {
        LOAD_FEATURES_1: for (int i = start_idx, local_i = 0; i < end_idx; i++, local_i++) {
            #pragma HLS PIPELINE II=1
            for (int k = 0; k < NFEATURES; k++) {
                #pragma HLS UNROLL factor=2
                local_features_1[local_i * NFEATURES + k] = feature[i * NFEATURES + k];
            }
        }
    } else {
        LOAD_FEATURES_2: for (int i = start_idx, local_i = 0; i < end_idx; i++, local_i++) {
            #pragma HLS PIPELINE II=1
            for (int k = 0; k < NFEATURES; k++) {
                #pragma HLS UNROLL factor=2
                local_features_2[local_i * NFEATURES + k] = feature[i * NFEATURES + k];
            }
        }
    }
}

/* Compute function: processes feature data and writes membership */
void compute_membership(float *local_clusters, float *local_features_1, float *local_features_2,
                       int *membership, int tile_id, int buffer_flag) {
    int start_idx = tile_id * TILE_SIZE;
    int end_idx = start_idx + TILE_SIZE;
    
    if (buffer_flag == 0) {
        COMPUTE_TILE_1: for (int i = start_idx, local_i = 0; i < end_idx; i++, local_i++) {
            #pragma HLS LOOP_TRIPCOUNT min=4096 max=4096
            float min_dist = FLT_MAX;
            int index = 0;

            /* find the cluster center id with min distance to pt */
            MIN_1: for (int j = 0; j < NCLUSTERS; j++) {
                #pragma HLS LOOP_TRIPCOUNT min=5 max=5
                #pragma HLS DEPENDENCE variable=min_dist inter false
                #pragma HLS DEPENDENCE variable=index inter false
                float dist = 0.0;

                /* Unroll feature distance computation by factor of 2 */
                DIST_1: for (int k = 0; k < NFEATURES; k += 2) {
                    #pragma HLS LOOP_TRIPCOUNT min=17 max=17
                    #pragma HLS UNROLL factor=2
                    #pragma HLS DEPENDENCE variable=dist inter false
                    
                    float diff0 = local_features_1[local_i * NFEATURES + k] - 
                                  local_clusters[NFEATURES * j + k];
                    float diff1 = local_features_1[local_i * NFEATURES + k + 1] - 
                                  local_clusters[NFEATURES * j + k + 1];
                    dist += diff0 * diff0 + diff1 * diff1;
                }
                if (dist < min_dist) {
                    min_dist = dist;
                    index = j;
                }
            }
            /* assign the membership to object i */
            membership[i] = index;
        }
    } else {
        COMPUTE_TILE_2: for (int i = start_idx, local_i = 0; i < end_idx; i++, local_i++) {
            #pragma HLS LOOP_TRIPCOUNT min=4096 max=4096
            float min_dist = FLT_MAX;
            int index = 0;

            /* find the cluster center id with min distance to pt */
            MIN_2: for (int j = 0; j < NCLUSTERS; j++) {
                #pragma HLS LOOP_TRIPCOUNT min=5 max=5
                #pragma HLS DEPENDENCE variable=min_dist inter false
                #pragma HLS DEPENDENCE variable=index inter false
                float dist = 0.0;

                /* Unroll feature distance computation by factor of 2 */
                DIST_2: for (int k = 0; k < NFEATURES; k += 2) {
                    #pragma HLS LOOP_TRIPCOUNT min=17 max=17
                    #pragma HLS UNROLL factor=2
                    #pragma HLS DEPENDENCE variable=dist inter false
                    
                    float diff0 = local_features_2[local_i * NFEATURES + k] - 
                                  local_clusters[NFEATURES * j + k];
                    float diff1 = local_features_2[local_i * NFEATURES + k + 1] - 
                                  local_clusters[NFEATURES * j + k + 1];
                    dist += diff0 * diff0 + diff1 * diff1;
                }
                if (dist < min_dist) {
                    min_dist = dist;
                    index = j;
                }
            }
            /* assign the membership to object i */
            membership[i] = index;
        }
    }
}

void workload(float  *feature, /* [npoints][nfeatures] */
			  float  *clusters, /* [n_clusters][n_features] */
			  int *membership)
{
	#pragma HLS INTERFACE m_axi port=feature offset=slave bundle=gmem max_read_burst_length=256
	#pragma HLS INTERFACE m_axi port=clusters offset=slave bundle=gmem max_read_burst_length=256
	#pragma HLS INTERFACE m_axi port=membership offset=slave bundle=gmem max_write_burst_length=256
	#pragma HLS INTERFACE s_axilite port=feature bundle=control
	#pragma HLS INTERFACE s_axilite port=clusters bundle=control
	#pragma HLS INTERFACE s_axilite port=membership bundle=control
	#pragma HLS INTERFACE s_axilite port=return bundle=control

	/* Local buffer for cluster centers - buffer into on-chip memory */
	float local_clusters[NCLUSTERS * NFEATURES];
	#pragma HLS ARRAY_PARTITION variable=local_clusters cyclic factor=17

	/* Double buffering: two sets of feature buffers */
	float local_features_1[TILE_SIZE * NFEATURES];
	#pragma HLS ARRAY_PARTITION variable=local_features_1 cyclic factor=17
	
	float local_features_2[TILE_SIZE * NFEATURES];
	#pragma HLS ARRAY_PARTITION variable=local_features_2 cyclic factor=17

	/* Load clusters into local buffer once */
	LOAD_CLUSTERS: for (int i = 0; i < NCLUSTERS * NFEATURES; i++) {
		#pragma HLS PIPELINE II=1
		local_clusters[i] = clusters[i];
	}

	/* Main tiled loop with double buffering */
	TILE_LOOP: for (int tile = 0; tile < NUM_TILES; tile++) {
		#pragma HLS LOOP_TRIPCOUNT min=100 max=100
		int buffer_flag = tile % 2;

		/* Load next tile while computing current tile */
		if (tile < NUM_TILES - 1) {
			/* Load tile+1 into alternate buffer */
			load_features(feature, local_features_1, local_features_2, tile + 1, (buffer_flag + 1) % 2);
		}

		/* Compute current tile from the buffer that was loaded previously */
		if (tile == 0) {
			/* First iteration: load tile 0 first, then compute */
			load_features(feature, local_features_1, local_features_2, 0, 0);
			compute_membership(local_clusters, local_features_1, local_features_2, membership, 0, 0);
		} else {
			/* Subsequent iterations: compute from previously loaded tile */
			compute_membership(local_clusters, local_features_1, local_features_2, membership, tile, (buffer_flag + 1) % 2);
		}
	}
}

}