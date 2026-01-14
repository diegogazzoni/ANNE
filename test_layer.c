#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include "anne.h"
#include "train.h"

#define N_UNITS 2
#define N_INPUT 5

int main() {
	srand(time(NULL));
	
	// Layer with 3 neurons, connected to a 5-unit input. Here we will have 15 weights.
    ANN_layer* layer = (ANN_layer*) malloc(sizeof(ANN_layer));
    init_layer(layer, N_UNITS, N_INPUT);

	double input[N_INPUT] = {1, 1, 1, 1, 1}; 
	eval_layer(layer, input, N_INPUT);
	
	fprintf(stdout, "S = [");
	for (int i=0; i<N_UNITS; i++)	
		printf("%f ", layer->s[i]);
	printf("]\n"); // MUST BE 5 if W=1

	fprintf(stdout, "A = [");
	for (int i=0; i<N_UNITS; i++)
    	printf("%f ", layer->a[i]);
	printf("]\n");
	
	// Printing weight matrix
	printf("Weights: \n");
	for (int i=0; i<N_INPUT; i++) {
		for (int j=0; j<N_UNITS; j++) {
			printf("%.3f ", layer->w[j + i*N_UNITS]);
		}
		printf("\n");
	}

    return 0;
}
