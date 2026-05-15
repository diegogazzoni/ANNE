#ifndef TRAIN_H
#define TRAIN_H

#include "anne.h"

/* 
 * This structure represents a base collection of data that can traverse the network. 
 * It does not contain information about splits, so to create a "higher level dataset" we'll need another structure (however, it's not necessary).
 */
typedef struct ANN_dataset {
	int n_batches;
	int n_samples;
	int batch_size;
	int sample_dim;
  int ref_dim;

	double** data; // eg: {{data1}, {data2}, ...}
	double** ref;
} ANN_dataset;

/* 
 * Defining functions for implementing Batch Stochastic Gradient Descent.
 */
void zero_grad_ANN(ANN* ann);
double train_sgd_epoch(ANN* ann, ANN_dataset* dataset, double lr);
void train(ANN* ann, ANN_dataset* dataset, int n_epochs, double lr);

#endif
