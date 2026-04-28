#include "kmeans.h"

extern "C" {

void workload(float  *feature, /* [npoints][nfeatures] */
			  float  *clusters, /* [n_clusters][n_features] */
			  int *membership)
{
	#pragma HLS INTERFACE m_axi port=feature offset=slave bundle=gmem
	#pragma HLS INTERFACE m_axi port=clusters offset=slave bundle=gmem
	#pragma HLS INTERFACE m_axi port=membership offset=slave bundle=gmem
	#pragma HLS INTERFACE s_axilite port=feature bundle=control
	#pragma HLS INTERFACE s_axilite port=clusters bundle=control
	#pragma HLS INTERFACE s_axilite port=membership bundle=control
	#pragma HLS INTERFACE s_axilite port=return bundle=control

	/* Local buffer for cluster centers - buffer into on-chip memory */
	float local_clusters[NCLUSTERS * NFEATURES];
	#pragma HLS ARRAY_PARTITION variable=local_clusters cyclic factor=34 dim=1

	/* Load clusters into local buffer once */
	LOAD_CLUSTERS: for (int i = 0; i < NCLUSTERS * NFEATURES; i++) {
		#pragma HLS PIPELINE II=1
		local_clusters[i] = clusters[i];
	}

	/* Process points with pipelined distance computation */
	UPDATE_MEMBER: for (int i = 0; i < NPOINTS; i++) {
		#pragma HLS LOOP_TRIPCOUNT min=409600 max=409600
		#pragma HLS PIPELINE II=1
		float min_dist = FLT_MAX;
		int index = 0;

		/* find the cluster center id with min distance to pt */
		MIN: for (int j = 0; j < NCLUSTERS; j++) {
			#pragma HLS LOOP_TRIPCOUNT min=5 max=5
			#pragma HLS DEPENDENCE variable=min_dist inter false
			#pragma HLS DEPENDENCE variable=index inter false
			float dist = 0.0;

			DIST: for (int k = 0; k < NFEATURES; k++) {
				#pragma HLS LOOP_TRIPCOUNT min=34 max=34
				#pragma HLS UNROLL factor=8
				#pragma HLS DEPENDENCE variable=dist inter false
				float diff = feature[NFEATURES * i + k] - local_clusters[NFEATURES * j + k];
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