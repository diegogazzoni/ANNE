#ifndef TRAIN_H
#define TRAIN_H

#include "anne.h"

/* 
 * Defining functions for implementing Batch Stochastic Gradient Descent.
 */
void zero_grad_ANN(ANN* ann);
double train_sgd_epoch(ANN* ann, ANN_dataset* dataset, double lr);
void train(ANN* ann, ANN_dataset* dataset, int n_epochs, double lr);

#endif
