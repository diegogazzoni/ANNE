#include <stdio.h>
#include <stdlib.h>

#include "utils.h"
#include "anne.h"

/* Initializes a single layer. */
void init_layer(ANN_layer* layer, int n_units, int n_units_prev) {
	int n_weights = n_units * n_units_prev; // we have n_units * n_units_prev weights
	if (n_weights == 0) {
		fprintf(stderr, "FATAL: Impossible to instance layer\n");
		return;
	}
	layer->n_units = n_units;
	layer->n_weights = n_weights;
	layer->ba = (double*) calloc(n_units, sizeof(double));
	layer->ga = (double*) calloc(n_weights, sizeof(double));
    layer->w  = (double*) calloc(n_weights, sizeof(double));
	layer->b  = (double*) calloc(n_units, sizeof(double));
	layer->a  = (double*) calloc(n_units, sizeof(double));
	layer->s  = (double*) calloc(n_units, sizeof(double));
	layer->delta = (double*) calloc(n_units, sizeof(double));
	for (int i=0;i<n_units*n_units_prev;i++) {
		// Weight initialization must be changed to improve training...
		layer->w[i]	= xavier_glorot(n_units_prev, n_units); //rand_double(); // -1, +1
		if (i < n_units)
			layer->b[i] = drand();	
	}
}

/* Computes the sum inside neurons. NB: weight matrix has n_prev * n_units dims. */
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

/* Computes the activations and so the network output.*/
void forward(ANN* ann, double* input) {
	// Activations on the first layer
	eval_layer(ann->layer[0], input, ann->n_inputs);	
	// Activations on deep and output layers
	for (int l=1; l<ann->n_layers; l++) {
		eval_layer(ann->layer[l], ann->layer[l-1]->a, ann->layer[l-1]->n_units);	
	}
}

/* Computes the errors and backpropagates it into the network. Actually it uses the SQUARE SUM function. 
	NOTE: consider to pass 'loss_fn' and use it to compute the error.*/
void backward(ANN* ann, double* true_vals) {
	int n_layers = ann->n_layers;
	for (int l=n_layers-1; l>=0; l--) {
		int n_units = ann->layer[l]->n_units;
		// The first layer must have a proper computation which depends on the loss function. Others are recursive.		
		for (int j=0;j<n_units; j++) {
			double unit_a = ann->layer[l]->a[j];
			double unit_s = ann->layer[l]->s[j];
			if (l == n_layers-1) {
				ann->layer[l]->delta[j] = 2.0 * (unit_a - true_vals[j]) * d_sigmoid(unit_s); // dL/dI * da/ds
			} else {	
				int n_units_next = ann->layer[l+1]->n_units;
				double e = 0.0;
				// Using next layer
				for (int k=0;k<n_units_next; k++) {
					e += ann->layer[l+1]->delta[k] * ann->layer[l+1]->w[k+j*n_units_next]; // !!!!  CRITICAL  !!!!
				}
				e *= d_sigmoid(ann->layer[l]->s[j]);
				ann->layer[l]->delta[j] = e;
			}	
		} 	
	}
}

/* Deallocates a single layer. */
void destroy_layer(ANN_layer* layer) {
	free(layer->w);
	free(layer->b);
	free(layer->a);
	free(layer->s);
	free(layer->delta);
	free(layer->ba);
	free(layer->ga);
	free(layer);
}

/* Deallocates the entire network. */
void destroy_ANN(ANN* ann) {
	for (int l=0; l<ann->n_layers; l++)
		destroy_layer(ann->layer[l]);
	free(ann->layer);
	free(ann);
}
