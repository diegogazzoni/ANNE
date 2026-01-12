// Project: ANNE Artificial Neural Network Exercise

// =================================== Layers ======================================
typedef struct ANN_layer {
	int n_units; // number of neurons
	
	double* w; // weights connecting the i-1 layer with the current one i-th. Dimensions: n_prev x n_units
	double* b; // biases 
	double* a; // activation fun(sum). Sigmoid actually
	double* s; // weighted sum
	double* delta; // error vector for backprop
	
} ANN_layer;

/* Creates a neural layer with n_units neurons. */
void init_layer(ANN_layer* layer, int n_units, int n_units_prev);

/* Evaluates the activations inside the referenced layer */
void eval_layer(ANN_layer* layer, double* input, int n_inputs);

/* Deallocate the layer. Pay attention if it is used inside a network. */
void destroy_layer(ANN_layer* layer);

// =================================== Network =====================================
typedef struct ANN {
	int n_layers;
	int n_inputs;
	int n_outputs;
	ANN_layer** layer;

	double loss_score;
} ANN;

void forward(ANN* ann, double* input);
void backward(ANN* ann, double* true_vals);
void destroy_ANN(ANN* ann);

void train_ANN(ANN* ann, void* data); // TODO: think how to pass the dataset. Create a struct? Is it necessary?
