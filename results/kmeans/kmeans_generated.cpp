#include "kmeans.h"

extern "C" {

void workload(float  *clusters, /* [n_clusters][n_features] */
			float  *feature, /* [npoints][nfeatures] */
			int *membership)
{
#pragma HLS INTERFACE m_axi port=clusters offset=slave bundle=gmem max_read_burst_length=256 max_write_burst_length=256
#pragma HLS INTERFACE m_axi port=feature offset=slave bundle=gmem max_read_burst_length=256 max_write_burst_length=256
#pragma HLS INTERFACE m_axi port=membership offset=slave bundle=gmem max_read_burst_length=256 max_write_burst_length=256
#pragma HLS INTERFACE s_axilite port=clusters bundle=control
#pragma HLS INTERFACE s_axilite port=feature bundle=control
#pragma HLS INTERFACE s_axilite port=membership bundle=control
#pragma HLS INTERFACE s_axilite port=return bundle=control

	UPDATE_MEMBER: for (int i = 0; i < NPOINTS; i++) {
#pragma HLS LOOP_TRIPCOUNT min=409600 max=409600 avg=409600
#pragma HLS PIPELINE II=35
		float min_dist = FLT_MAX;
		int index = 0;

		/* find the cluster center id with min distance to pt */
		MIN: for (int j = 0; j < NCLUSTERS; j++) {
#pragma HLS LOOP_TRIPCOUNT min=5 max=5 avg=5
			float dist = 0.0;

			DIST: for (int k = 0; k < NFEATURES; k++) {
#pragma HLS LOOP_TRIPCOUNT min=34 max=34 avg=34
				float diff = feature[NFEATURES * i + k] - clusters[NFEATURES * j + k];
				dist += diff * diff;
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