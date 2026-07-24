#include <math.h>

/*******************************************/
/* Consider to implement proper structures */
/* for functions                           */
/*******************************************/

/* Activation Functions */
static inline double zero(double x) {
    return 0;
}

// NB: tanh is already present in math.h
static inline double d_tanh(double x) {
    return 1.0 - tanh(x)*tanh(x);
}

static inline double sigmoid(double x) {
	return 1.0 / (1.0 + exp(-x));
}

static inline double d_sigmoid(double x) {
	double s = sigmoid(x);
	return s * (1.0 - s);
}

static inline double relu(double x) {
    return x > 0 ? x : 0;
}

static inline double d_relu(double x) {
    return x > 0 ? 1 : 0; 
}

/* Loss functions */

// Mean Square Error
static double fn_MSE(double *y_pred, double *y_true, int n_samples) {
	double mse = 0.0;

	for (int n=0;n<n_samples;n++) {
		mse += (y_pred[n]-y_true[n])*(y_pred[n]-y_true[n]);
	}
	return mse / (double) n_samples;
}

static inline double grad_MSE(double y_pred, double y_true, int n_samples) {
	return 2.0 * (y_pred - y_true);
}

// Binary Cross Entropy
static double fn_BCE(double *y_pred, double *y_true, int n_samples) {
  double bce = 0.0;

  for (int n=0;n<n_samples;n++) {
    bce += y_true[n]*log(y_pred[n]) + (1.0-y_true[n])*log(1.0-y_pred[n]);
  }

  return -bce/(double)n_samples; 
}

static inline double grad_BCE(double y_pred, double y_true, int n_samples) {
  return -(y_true/y_pred - (1.0-y_true)/(1-y_pred));
} 
