#ifndef TENSOR_H
#define TENSOR_H

#define N_MAX_OP 2

typedef struct TensorComponent {
	double* data;
	int len;
} TensorComponent;

typedef struct Tensor {
	int* sizes;
	int* shape;
	TensorComponent* components;
} Tensor;

void tensor_alloc(Tensor* dst, int* sizes, int* lens);
void tensor_del(Tensor* tensor);

#endif
