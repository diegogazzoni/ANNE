#ifndef TRAIN_H
#define TRAIN_H

#include "anne.h"

/* Dataset abstractions */
typedef struct ANN_dataset {
	int n_batches;
	int n_samples;
	int batch_size;
	int sample_dim;

	double** data; // eg: {{data1}, {data2}, ...}
	double** ref;
} ANN_dataset;

/* Minibatch SGD functions */
void zero_grad_ANN(ANN* ann);
double train_sgd_epoch(ANN* ann, ANN_dataset* dataset, double lr);
void train(ANN* ann, ANN_dataset* dataset, int n_epochs, double lr);

#endif
