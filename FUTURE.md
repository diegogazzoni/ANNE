# Tensors
It is time to introduce a proper representation of **tensors** to generalise all the operations inside the network. Tensors must be ideally implemented with a top layer and a backend targeting a particular architecture (CPU, GPU).

This might be a template for beginning the implementation of tensors (`utensor.c`):
```C
typedef struct Tensor {
    double* data; // data store, row-major access? These data will be used by a particular backend to compute operations
    int ndim; // dimension of the tensor
    int *strides; // number of items to jump to access each dimension
    int *shapes; // number of elements for each dimension (cardinality)
}
```
