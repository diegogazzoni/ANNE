#include <stdio.h>
#include <stdlib.h>

#include "aux.h"
#include "anne.h"
#include "dataset.h"

/*
	Initializes a single layer, allocating arrays for weights, outputs and setting the activation function.
	Weight initialization is implemented using the standard xavier-glorot algorithm
*/
void init_layer(ANN_layer* layer, double (*act_fn)(double), double (*act_d_fn)(double), int n_units, int n_units_prev) {
	int n_weights = n_units * n_units_prev; // we have n_units * n_units_prev weights
	if (n_weights == 0) {
		fprintf(stderr, "n_weights must be grater than zero.\n");
		exit(-1);
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
	layer->fn = act_fn;
	layer->d_fn = act_d_fn;

	for (int i=0;i<n_units*n_units_prev;i++) {
		layer->w[i]	= xavier_glorot(n_units_prev, n_units);
		if (i < n_units)
			layer->b[i] = 0.0;
	}
}

/*
 * Initializes the ANN structure
 */
void init_ANN(ANN* ann, int n_inputs, int n_outputs, int n_layers, double (*loss_fn) (double*, double*, int), double (*loss_fn_grad) (double, double, int)) {
	ann->n_inputs = n_inputs;
	ann->n_outputs = n_outputs;
	ann->n_layers = n_layers;
	ann->loss_fn = loss_fn;
	ann->loss_fn_grad = loss_fn_grad;
}

/*
 * Computes sums and activations inside neurons for a given input.
 * NOTE: weight matrix dims are (n_prev, n_units).
 */
void eval_layer(ANN_layer* layer, double* input, int input_dim) {
	// Cycling on the neurons.
	// NOTE: input_dim is the number of previous neurons connected to this layer. This is coherent even with inputs, if they are considered as coming from other "unknown" neurons.
	if (input_dim == 0) {
		fprintf(stderr, "input_dim=%d must be instanced or greater than zero.\n", input_dim);
		exit(-1);
	}

	for (int j=0; j<layer->n_units; j++) {
		// Cycling on all the other neurons connected to it, and computing the sum. It is initialized with the bias.
		double tmp_sum = layer->b[j];
		for (int i=0;i<input_dim; i++) {
			tmp_sum += layer->w[j+i*layer->n_units]*input[i];
		}
		layer->s[j] = tmp_sum; // linear combination
		layer->a[j] = layer->fn(tmp_sum); // activation
	}
}

/*
 * Computes the activations inside all the network, so its output.
 */
void forward(ANN* ann, double* input) {
	if (input == NULL) {
		fprintf(stderr, "Input must be instanced\n");
		exit(-1);
	}
	// Activations on the first layer
	eval_layer(ann->layer[0], input, ann->n_inputs);
	// Activations on deep and output layers
	for (int l=1; l<ann->n_layers; l++) {
		eval_layer(ann->layer[l], ann->layer[l-1]->a, ann->layer[l-1]->n_units);
	}
}

/*
 * Computes the errors and backpropagates it into the network.
 */
void backward(ANN* ann, double* true_vals) {
	int n_layers = ann->n_layers;

	if (true_vals == NULL) {
		fprintf(stderr, "Reference values must be not NULL\n");
	}

	for (int l=n_layers-1; l>=0; l--) {
		int n_units = ann->layer[l]->n_units;
		// The first layer must have a proper computation which depends on the loss function. Others are recursive.
		// NOTE: test here the case when the reference (true_vals) have a different dim than inputs.
    for (int j=0;j<n_units; j++) {
			double unit_a = ann->layer[l]->a[j];
			double unit_s = ann->layer[l]->s[j];
			if (l == n_layers-1) {
				double d_loss = ann->loss_fn_grad(unit_a, true_vals[j], n_units);
                double d_act = ann->layer[l]->d_fn(unit_s);
				ann->layer[l]->delta[j] =  d_loss * d_act;
			} else {
				int n_units_next = ann->layer[l+1]->n_units;
				double e = 0.0;
				// Using next layer (so here we read 'backwards' with respect to the cycle direction)
				for (int k=0;k<n_units_next; k++) {
					e += ann->layer[l+1]->delta[k] * ann->layer[l+1]->w[k+j*n_units_next]; // !!!!  CRITICAL  !!!! k+j*n_units_next
				}
				e *= ann->layer[l]->d_fn(unit_s);
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
	for (int l=0; l<ann->n_layers; l++) {
		destroy_layer(ann->layer[l]);
	}
	free(ann->layer);
	free(ann);
}
