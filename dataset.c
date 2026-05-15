#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <dirent.h>

#include "train.h"

#define MAX_FILES 1e5

static ANN_dataset* load_dataset(const char* data_dir, const char* ref_dir, int batch_size, int data_dim, int ref_dim) {
    // Try to open the directory.
    DIR* d_data = opendir(data_dir);
    DIR* d_ref = NULL;
    if (!d_data) {
      printf("Failed to open dataset directory");
      return NULL;
    }

    if (ref_dir != NULL) {
      d_ref = opendir(ref_dir)
      if (!d_ref) {
        printf("Failed to open dataset reference directory");
        return NULL;
      }
    } else {
      printf("Warning: no reference indicated. Loading dataset without it.");
    }
    
    // Create an array containing all the paths of the files, it will be used to read them.
    // NOTE: each filepath is not allocated here, pay attention for bugs.
    char** paths = (char**) malloc(sizeof(char*) * MAX_FILES);
    int n = 0;
    
    struct dirent* ent;
    while ((ent = readdir(d)) && n < MAX_FILES) {
        if (is_pgm(ent->d_name)) {
            paths[n] = (char*) malloc(256); // each file name must have max. 256 characters.
            snprintf(paths[n], 256, "%s/%s", dir, ent->d_name);
            n++;
        }
    }
    closedir(d);
    
    if (n == 0) {
        free(paths);
        return NULL;
    }
    
    // Create data and reference data. Reference data can be labels or data. 
    double** data = (double**) malloc(sizeof(double*) * n);
    double** ref = (double**) malloc(sizeof(double*) * n);
    
    for (int i=0; i<n; i++) {
        data[i] = load_pgm(paths[i]);
        free(paths[i]);
    }
    free(paths);
    
    // Adjust batch size to not exceed dataset size
    if (batch_size > n) batch_size = n;
    
    ANN_dataset* ds = (ANN_dataset*) malloc(sizeof(ANN_dataset));
    ds->n_samples = n;
    ds->batch_size = batch_size;
    ds->n_batches = (n + batch_size - 1) / batch_size;
    ds->sample_dim = data_dim;
    ds->ref_dim = ref_dim;
    ds->data = data;
    ds->ref = ref;
    
    return ds;
}

static void free_dataset(ANN_dataset* ds) {
    for (int i=0; i<ds->n_samples; i++)
        free(ds->data[i]);
    free(ds->data);
    free(ds->ref);
    free(ds);
}
