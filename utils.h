#ifndef UTILS_H
#define UTILS_H

#include <math.h>
#include <stdlib.h>

#define EPS 1e-12

/* Weight initialization */
double drand() {
	double d = (double) RAND_MAX / 2.0;
	double r = (double) rand();
	return (r - d) / d;
}

double xavier_glorot(int n_in, int n_out) {
	double d1 = (drand() + 1.0) / 2.0 + EPS;
	double d2 = (drand() + 1.0) / 2.0 + EPS;
	double n = sqrt(-2.0*log(d1)*cos(2.0*M_PI*d2)); // Box-Muller transform
	double sigma = sqrt(2.0/(n_in+n_out));
	return n * sigma;
}

/*  Activation Functions */
double sigmoid(double x) {
	return 1.0/(1.0 + exp(-x));
}

double d_sigmoid(double x) {
	double s = sigmoid(x);
	return s * (1.0-s);
}

#endif
