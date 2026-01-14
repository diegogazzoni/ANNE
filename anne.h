typedef struct ANN_layer {
	int n_units; // number of neurons
	int n_weights; // number of weights

	double* w; // weights connecting the i-1 layer with the current one i-th. Dimensions: n_prev x n_units
	double* b; // biases 
	double* a; // activation fun(sum). Sigmoid actually
	double* s; // weighted sum
	double* delta; // error vector for backprop
    
    double* ba; // accumulator for bias
	double* ga; // gradient accumulator for SGD
} ANN_layer;

typedef struct ANN {
	ANN_layer** layer;
	int n_layers;
	int n_inputs;
	int n_outputs;
	
	double loss_score;
} ANN;

void init_layer(ANN_layer* layer, int n_units, int n_units_prev);
void eval_layer(ANN_layer* layer, double* input, int n_inputs);
void destroy_layer(ANN_layer* layer);

void forward(ANN* ann, double* input);
void backward(ANN* ann, double* true_vals);
void destroy_ANN(ANN* ann);
