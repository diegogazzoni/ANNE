typedef struct ANN_layer {
	int n_units; // number of neurons
	
	double* w; // weights connecting the i-1 layer with the current one i-th. Dimensions: n_prev x n_units
	double* b; // biases 
	double* a; // activation fun(sum). Sigmoid actually
	double* s; // weighted sum
	double* delta; // error vector for backprop
} ANN_layer;

typedef struct ANN {
	ANN_layer** layer;
	int n_layers;
	int n_inputs;
	int n_outputs;
	
	double loss_score;
} ANN;

typedef struct ANN_dataset {
	int n_batch;
	int n_samples;
	int batch_size;
	
	double** data;
}

void init_layer(ANN_layer* layer, int n_units, int n_units_prev);
void eval_layer(ANN_layer* layer, double* input, int n_inputs);
void destroy_layer(ANN_layer* layer);

void forward(ANN* ann, double* input);
void backward(ANN* ann, double* true_vals);
void destroy_ANN(ANN* ann);

void train_ANN(ANN* ann, ANN_dataset* dataset);
