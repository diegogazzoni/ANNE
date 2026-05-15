#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "train.h"
#include "anne.h"

/*
 * This is a util function with the purpose of zero-ing all the stored gradients.
 * It is a necessary operation to have a successful training, since gradients are accumulated over samples inside a batch.
 */
void zero_grad_ANN(ANN* ann) {
	for (int l=0; l<ann->n_layers; l++) {
		memset(ann->layer[l]->ga, 0, sizeof(double) * ann->layer[l]->n_weights);
		memset(ann->layer[l]->ba, 0, sizeof(double) * ann->layer[l]->n_units);	
	}		
}

/* 
 * Core routine to implement the (Batch) Stochastic Gradient Descent algorithm. 
 * 
 * Data is treated in batches here. A batch is a collection of training data used to gather information for the final weights update.
 * The weight update is performed with the following mathematical rule:
 *                        w_next = w_current - lr * sum_m=1^batch_size (grad(L(w_current))_m)/batch_size
 * Where: lr is the learning ratio (1e-3, typical value), batch_size the number of batches and w_* are weights vectors. L is the loss function.
 */
double train_sgd_epoch(ANN* ann, ANN_dataset* dataset, double lr) {
	double loss = 0.0;
	int n_samples = dataset->n_samples;
	int batch_size = dataset->batch_size;
	if (n_samples % batch_size != 0) {
		fprintf(stderr, "Batch size must divide n_samples equally.");
		return -1;
	}
  
  // The computation needs three "spatial" cycles
  // - One cycle for the batches
  // - One for the data inside the selected batch
  // - One to traverse the network

	int n_batches = 0;

	// Cycle on batches
	for (int b=0; b<n_samples; b += batch_size) {
    double loss_batch = 0.0;
    n_batches++;
    // In first instance, before the network phagocytises a batch, its gradient accumulators need to be zero.
		zero_grad_ANN(ann);
		// So, we can gee the s-th sample inside the batch.
		for (int s=0; s<batch_size; s++) {
			int s_idx = b+s;
			double* sample = dataset->data[s_idx]; // data with sample_dim elements
			double* sample_ref = dataset->ref[s_idx]; // this COULD have the same dimensions of sample (e.g, in autoencoders), but it's not general (e.g, classifiers).
			
			// Computing activations inside the network and the loss score (note: this last is not necessary).
			forward(ann, sample); // ANN saves n_inputs --> it MUST match with sample_dim!
			double* pred_sample = ann->layer[ann->n_layers-1]->a;		
			loss_batch += ann->loss_fn(pred_sample, sample, ann->layer[ann->n_layers-1]->n_units); // to compute the average loss on a batch

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
		loss_batch /= batch_size;
		loss += loss_batch;

		// Update weights and biases
		for (int l=0; l<ann->n_layers; l++) {
        for (int n=0; n<ann->layer[l]->n_weights; n++) 
            ann->layer[l]->w[n] -= lr * ann->layer[l]->ga[n] / dataset->batch_size;
        for (int n=0; n<ann->layer[l]->n_units; n++) 
            ann->layer[l]->b[n] -= lr * ann->layer[l]->ba[n] / dataset->batch_size;
    }
	}
	return loss / (double) n_batches;
}

/* Trains a neural network using the stochastic gradient descent method with data batches. */
void train(ANN* ann, ANN_dataset* dataset, int n_epochs, double lr) {
	for (int e = 1; e < n_epochs+1; e++) {
		fprintf(stdout, "======= epoch n. %d =======\n", e);
		double loss =  train_sgd_epoch(ann, dataset, lr); 
		fprintf(stdout, "loss = %lf\n", loss);
	}
}
