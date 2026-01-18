#ifndef ANNE_H
#define ANNE_H

typedef struct ANN_layer {
	char *descr;

    int n_units; // number of neurons
	int n_weights; // number of weights
	
	double (*fn)(double); // function pointer to the selected activation
	double (*d_fn)(double); // derivative for bp

	double* w; // weights connecting the i-1 layer with the current one i-th. Dimensions: n_prev x n_units
	double* b; // biases 
	double* a; // activation fun(sum). Sigmoid actually
	double* s; // weighted sum
	double* delta; // error vector for backprop
    
    double* ba; // accumulator for bias
	double* ga; // gradient accumulator for SGD
} ANN_layer;

typedef struct ANN {
	ANN_layer** layer; // Network layers
	int n_layers;
	int n_inputs;
	int n_outputs;
	
	double (*loss_fn) (double*, double*, int); // loss function
	double (*loss_fn_grad) (double, double, int); // derivative of the loss function
} ANN;

void init_layer(ANN_layer* layer, double (*act_fn)(double), double (*act_d_fn)(double), int n_units, int n_units_prev);
void eval_layer(ANN_layer* layer, double* input, int n_inputs);
void destroy_layer(ANN_layer* layer);

void init_ANN(ANN* ann, int n_inputs, int n_outputs, int n_layers, double (*loss_fn) (double*, double*, int), double (*loss_fn_grad) (double, double, int));
void forward(ANN* ann, double* input);
void backward(ANN* ann, double* true_vals);
void destroy_ANN(ANN* ann);

#endif
