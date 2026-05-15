#ifndef DATASET_H
#define DATASET_H

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

#endif
