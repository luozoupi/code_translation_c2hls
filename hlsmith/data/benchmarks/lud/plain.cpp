#include"lud.h"

void lud(float result[GRID_ROWS * GRID_COLS])
	{
		int i, j, k; 
		float sum;
	 
		for (i=0; i<SIZE; i++){
		     for (j=i; j<SIZE; j++){
		         sum=result[i*SIZE+j];
		         for (k=0; k<i; k++) sum -= result[i*SIZE+k]*result[k*SIZE+j];
		         result[i*SIZE+j]=sum;
		     }

		     for (j=i+1;j<SIZE; j++){
		         sum=result[j*SIZE+i];
		         for (k=0; k<i; k++) sum -= result[j*SIZE+k]*result[k*SIZE+i];
		         result[j*SIZE+i]=sum/result[i*SIZE+i];
		     }
		 }
		
		 return;
	}

	void workload(float result[GRID_ROWS * GRID_COLS])
	{


		lud(result);

		return;

	}