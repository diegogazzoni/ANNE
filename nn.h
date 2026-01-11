// Project: ANNE Artificial Neural Network Exercise

// =================================== Layers ======================================
typedef struct ANN_layer {
	int n_units;

	double* w; // weights connecting the i-1 layer with the current one i-th. Dimensions: n_prev x n_units
	double* b; // biases on the current layer
	double* act; // activation fun(sum). Sigmoid actually
	double* sum; // weighted sum
	double* delta; // error vector

} ANN_layer;

/* Creates a neural layer. If has_prev, the layer is internal and weights are assigned. If not, weights are not assigned since it is the first layer. */
void init_ANN_layer(ANN_layer* layer, int n_units, uint8_t has_prev);
/* Deallocate the layer. Pay attention if it is used inside a network. */
void destroy_ANN_layer(ANN_layer* layer);

// =================================== Network =====================================
typedef struct ANN {
	int n_layers;	
	ANN_layer* layer;

	double loss_score;
} ANN;

void destroy_ANN(ANN* ann);
void forward_ANN(ANN* ann);
void train_ANN(ANN* ann, void* data); // TODO: think how to pass the dataset. Create a struct? Is it necessary?
