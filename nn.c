#include <stdio.h>
#include <stdlib.h>

#include "nnm.h"
#include "nn.h"

/* 
	Initializes a single layer. 
*/
void init_layer(ANN_layer* layer, int n_units, int n_units_prev) {
	int n_weights = n_units * n_units_prev; // we have n_units * n_units_prev weights
	if (n_weights == 0) {
		printf("ERROR: IMPOSSIBLE TO INSTANCE LAYER!\n");
		return;
	}

	layer->n_units = n_units;
	layer->w = (double*) malloc(sizeof(double)*n_weights);
	layer->b = (double*) malloc(sizeof(double)*n_units);
	layer->a = (double*) malloc(sizeof(double)*n_units);
	layer->s = (double*) malloc(sizeof(double)*n_units);
	layer->delta = (double*) malloc(sizeof(double)*n_units);

	for (int i=0;i<n_units*n_units_prev;i++) {
		// Weight initialization must be changed to improve training...
		layer->w[i]	= rand_double(); // -1, +1
		if (i < n_units)
			layer->b[i] = rand_double();	
	}
}

/* 
	Computes the sum inside neurons. NB: weight matrix has n_prev * n_units dims. 
*/
void eval_layer(ANN_layer* layer, double* input, int n_inputs) {
	// Cycling on the neurons.
	for (int j=0; j<layer->n_units; j++) {
		// Cycling on all the other neurons connected to it, and computing the sum. It is initialized with the bias.
		double tmp_sum = layer->b[j];
		for (int i=0;i<n_inputs; i++) {
			tmp_sum += layer->w[j+i*layer->n_units]*input[i];
		}
		layer->s[j] = tmp_sum; // linear combination
		layer->a[j] = sigmoid(tmp_sum); // activation
	}	
}

/* 
	Deallocates a single layer. 
*/
void destroy_layer(ANN_layer* layer) {
	free(layer->w);
	free(layer->b);
	free(layer->a);
	free(layer->s);
	free(layer->delta);
	free(layer);
}

/* 
	Computes the activations and so the network output.
*/
void forward(ANN* ann, double* input) {
	// Activations on the first layer
	eval_layer(ann->layer[0], input, ann->n_inputs);	
	// Activations on deep and output layers
	for (int l=1; l<ann->n_layers; l++) {
		eval_layer(ann->layer[l], ann->layer[l-1]->a, ann->layer[l-1]->n_units);	
	}
}

/* 
	Computes the errors and backpropagates it into the network. Actually it uses the MSE function. 
	NOTE: consider to pass 'loss_fn' and use it to compute the error.
*/
void backward(ANN* ann) {
	//... 
}

void destroy_ANN(ANN* ann) {
	for (int l=0; l<ann->n_layers; l++)
		destroy_layer(ann->layer[l]);
	free(ann->layer);
	free(ann);
}
