#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "train.h"
#include "anne.h"

/* As the function name says, it resets gradient accumulators to zero. */
void zero_grad_ANN(ANN* ann) {
	// Resets gradients for each layer
	for (int l=0; l<ann->n_layers; l++) {
		memset(ann->layer[l]->ga, 0, sizeof(double) * ann->layer[l]->n_weights);
		memset(ann->layer[l]->ba, 0, sizeof(double) * ann->layer[l]->n_units);	
	}		
}

/* Trains a neural network using the stochastic gradient descent method with data batches. */
void train_minibatch_sgd(ANN* ann, ANN_dataset* dataset, int n_epochs, double lr) {
	int epoch = 0;
	do {
		fprintf(stdout, "Epoch n. %d\n", epoch);
		// Firstly, get current batch index.
		for (int b=0; b<dataset->n_samples; b += dataset->batch_size) {
            // Before the network eats a batch, its gradient accumulators need to be zeroed.
			zero_grad_ANN(ann);
			// Secondly, get the s-th sample inside the batch
			for (int s=0; s<dataset->batch_size; s++) {
				int s_idx = b+s;
				double* sample = dataset->data[s_idx]; // data with sample_dim elements
				double* sample_ref = dataset->ref[s_idx];
				
				// Computing activations inside the network
				forward(ann, sample); // ANN saves n_inputs --> it MUST match with sample_dim!
				
				// Computing backward pass to get errors
				backward(ann, sample_ref);
				
				// Updating gradients in each layer.
				for (int l=0;l<ann->n_layers; l++) {
					int n_units = ann->layer[l]->n_units;
					int n_units_prev = ann->n_inputs; 
					double* input = sample;

					// If deep layers, use previous act as input
					if (l > 0) {
						n_units_prev = ann->layer[l-1]->n_units;
						input = ann->layer[l-1]->a;
					}

					// Accumulating weights gradients
					for (int i=0;i<n_units_prev; i++) {
						for (int j=0;j<n_units;j++) {
							ann->layer[l]->ga[j+i*n_units] += (input[i] * ann->layer[l]->delta[j]);
						}	
					}

					// Accumulating bias gradients
                    for (int j=0;j<n_units; j++) {
                        ann->layer[l]->ba[j] += ann->layer[l]->delta[j];
                    }
                }
			}
			// Update weights and biases
			for (int l=0; l<ann->n_layers; l++) {
                for (int n=0; n<ann->layer[l]->n_weights; n++) 
                    ann->layer[l]->w[n] -= lr * ann->layer[l]->ga[n] / dataset->batch_size;
                for (int n=0; n<ann->layer[l]->n_units; n++) 
                    ann->layer[l]->b[n] -= lr * ann->layer[l]->ba[n] / dataset->batch_size;
            }
		}
		epoch++;
	} while(epoch < n_epochs);
}
