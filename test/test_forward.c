#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include "anne.h"
#include "train.h"

int main() {
    srand(time(NULL));

    int inputs = 5;
    int outputs = 3;
    int n_layers = 3;
	int spec[n_layers]; 
   
	spec[0] = 5;
	spec[1] = 1;
	spec[2] = outputs;

    ANN* net = (ANN*) malloc(sizeof(ANN));
    net->n_inputs = inputs;
    net->n_layers = n_layers;
    net->n_outputs = outputs;
    net->layer = (ANN_layer**) malloc(sizeof(ANN_layer*) * n_layers);

    int prev_units = inputs;
    for (int i = 0; i < n_layers; i++) {
        net->layer[i] = (ANN_layer*) malloc(sizeof(ANN_layer));
        init_layer(net->layer[i], spec[i], prev_units);
        prev_units = spec[i];
    }

	double test_in[5] = {0.091, 0.21, 0.45, 0.75, 2.5}; // Dovrebbe essere vicino a 0
    forward(net, test_in);
	
	fprintf(stdout, "OUT = [");
	for (int i=0; i<net->n_outputs; i++)
    	printf("%f ", net->layer[n_layers-1]->a[i]);
	printf("]\n");

    return 0;
}
