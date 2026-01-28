#include <math.h>

/*  Activation Functions */
static double zero(double x) {
    return 0;
}

// NB: tanh is already present in math.h
static double d_tanh(double x) {
    return 1.0 - tanh(x)*tanh(x);
}

static double sigmoid(double x) {
	return 1.0 / (1.0 + exp(-x));
}

static double d_sigmoid(double x) {
	double s = sigmoid(x);
	return s * (1.0 - s);
}

static double relu(double x) {
    return x > 0 ? x : 0;
}

static double d_relu(double x) {
    return x > 0 ? 1 : 0; 
}

/* Loss functions */
static double fn_MSE(double *y_pred, double *y_true, int n_samples) {
	double mse = 0.0;

	for (int n=0;n<n_samples;n++) {
		mse += (y_pred[n]-y_true[n])*(y_pred[n]-y_true[n]);
	}
	return mse / (double) n_samples;
}

static double grad_MSE(double y_pred, double y_true, int n_samples) {
	return 2.0 * (y_pred - y_true) / n_samples;
}
