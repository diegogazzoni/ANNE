#include "nn.h"

/* Dataset abstractions */
typedef struct ANN_dataset {
	int n_batches;
	int n_samples;
	int batch_size;
	int sample_dim;

	double** data; // eg: {{data1}, {data2}, ...}
	double** ref;
} ANN_dataset;

double* get_data_sample(ANN_dataset* dataset, int i_batch, i_sample);

/* Minibatch SGD functions */
void zero_grad_ANN(ANN* ann);
void train_minibatch_sgd(ANN* ann, ANN_dataset* dataset, int n_epochs, double lr);
