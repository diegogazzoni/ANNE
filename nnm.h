#include <math.h>
#include <stdlib.h>

// ================== Utils ==================
double rand_double() {
	double d = (double) RAND_MAX / 2.0;
	double r = (double) rand();
	return (r - d) / d;
}

// =============== Activations ===============
double sigmoid(double x) {
	return 1.0/(1.0 + exp(-x));
}

double d_sigmoid(double x) {
	return sigmoid(x) * (1.0-sigmoid(x));
}
