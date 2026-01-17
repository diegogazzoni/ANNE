#include <math.h>
#include <stdlib.h>

#define EPS 1e-3

/* Weight initialization */
static double drand() {
	double d = (double) RAND_MAX / 2.0;
	double r = (double) rand();
	return (r - d) / d;
}

static double xavier_glorot(int n_in, int n_out) {
	double u1 = ((double)rand() / (double)RAND_MAX);
	double u2 = ((double)rand() / (double)RAND_MAX);
	
	if (u1 < EPS) u1 = EPS;
	if (u1 > 1.0 - EPS) u1 = 1.0 - EPS;
	if (u2 < EPS) u2 = EPS;
	if (u2 > 1.0 - EPS) u2 = 1.0 - EPS;

	double n = sqrt(-2.0 * log(u1)) * cos(2.0 * M_PI * u2);
	double sigma = sqrt(2.0 / (double)(n_in + n_out));
	return n * sigma;
}

/*  Activation Functions */
static double sigmoid(double x) {
	return 1.0 / (1.0 + exp(-x));
}

static double d_sigmoid(double x) {
	double s = sigmoid(x);
	return s * (1.0 - s);
}
