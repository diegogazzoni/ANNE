#include "nn.h"

/* 
	A simple structure to manager datasets
*/
typedef struct ANN_dataset {
	int n_batches;
	int n_samples;
	int batch_size;
	
	double** data; // eg: {{data1}, {data2}, ...}
}

/* Minibatch SGD functions */
void update_weights(ANN* ann);
void train_minibatch_sgd(ANN* ann, ANN_dataset* dataset);
