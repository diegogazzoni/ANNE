#ifndef TRAIN_H
#define TRAIN_H

#include "anne.h"
#include "dataset.h"

/* 
 * Defining functions for implementing Batch Stochastic Gradient Descent.
 */
void zero_grad_ANN(ANN* ann);
double train_sgd_epoch(ANN* ann,  int n_samples, int batch_size, double** input, double** ref, double lr);
void train(ANN* ann,  int n_samples, int batch_size, double** input, double** ref, int n_epochs, double lr);

#endif
