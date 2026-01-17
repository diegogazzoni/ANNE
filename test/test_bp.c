#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>

#include "anne.h"
#include "train.h"

#define N_INPUTS 100
#define N_SAMPLES 50000
#define BATCH_SIZE 100
#define N_EPOCHS 20
#define LEARNING_RATE 0.01

/* Computes Mean Squared Error loss */
double compute_mse(ANN* ann, ANN_dataset* dataset) {
    double total_loss = 0.0;
    for (int i = 0; i < dataset->n_samples; i++) {
        forward(ann, dataset->data[i]);
        double sample_loss = 0.0;
        for (int j = 0; j < ann->n_outputs; j++) {
            double diff = ann->layer[ann->n_layers-1]->a[j] - dataset->ref[i][j];
            sample_loss += diff * diff;
        }
        total_loss += sample_loss;
    }
    return total_loss / dataset->n_samples;
}

void generate_dataset(ANN_dataset* dataset) {
    srand(42); // Fixed seed for reproducibility
    
    // Create 5 basic pattern templates (each repeated 10 times)
    double templates[5][N_INPUTS];
    
    for (int t = 0; t < 5; t++) {
        for (int j = 0; j < N_INPUTS; j++) {
            templates[t][j] = 0.0;
        }
        
        // Template 0: First 20 neurons active
        if (t == 0) {
            for (int j = 0; j < 20; j++) templates[t][j] = 0.8;
        }
        // Template 1: Middle 20 neurons active
        else if (t == 1) {
            for (int j = 40; j < 60; j++) templates[t][j] = 0.8;
        }
        // Template 2: Last 20 neurons active
        else if (t == 2) {
            for (int j = 80; j < 100; j++) templates[t][j] = 0.8;
        }
        // Template 3: Even indices active
        else if (t == 3) {
            for (int j = 0; j < N_INPUTS; j += 2) templates[t][j] = 0.7;
        }
        // Template 4: Odd indices active
        else if (t == 4) {
            for (int j = 1; j < N_INPUTS; j += 2) templates[t][j] = 0.7;
        }
    }
    
    // Generate samples by adding noise to templates
    for (int i = 0; i < dataset->n_samples; i++) {
        dataset->data[i] = (double*) calloc(dataset->sample_dim, sizeof(double));
        dataset->ref[i] = (double*) calloc(dataset->sample_dim, sizeof(double));
        
        int template_idx = i % 5; // Cycle through templates
        
        for (int j = 0; j < dataset->sample_dim; j++) {
            // Add small random noise to template
            double noise = ((double)rand() / RAND_MAX - 0.5) * 0.2;
            double val = templates[template_idx][j] + noise;
            
            // Clamp to [0, 1]
            if (val < 0.0) val = 0.0;
            if (val > 1.0) val = 1.0;
            
            dataset->data[i][j] = val;
            dataset->ref[i][j] = val; // Autoencoder: output = input
        }
    }
}

/* Frees dataset memory */
void destroy_dataset(ANN_dataset* dataset) {
    for (int i = 0; i < dataset->n_samples; i++) {
        free(dataset->data[i]);
        free(dataset->ref[i]);
    }
    free(dataset->data);
    free(dataset->ref);
}

/* Debug function to check for NaN/Inf in layer */
void check_layer_health(ANN_layer* layer, int layer_idx) {
    for (int i = 0; i < layer->n_weights; i++) {
        if (isnan(layer->w[i]) || isinf(layer->w[i])) {
            fprintf(stderr, "ERROR: Layer %d weight[%d] = %f\n", layer_idx, i, layer->w[i]);
        }
    }
    for (int i = 0; i < layer->n_units; i++) {
        if (isnan(layer->a[i]) || isinf(layer->a[i])) {
            fprintf(stderr, "ERROR: Layer %d activation[%d] = %f\n", layer_idx, i, layer->a[i]);
        }
        if (isnan(layer->delta[i]) || isinf(layer->delta[i])) {
            fprintf(stderr, "ERROR: Layer %d delta[%d] = %f\n", layer_idx, i, layer->delta[i]);
        }
    }
}

int main() {
    srand(time(NULL));
    
    fprintf(stdout, "=== Autoencoder Training Test ===\n");
    fprintf(stdout, "Architecture: %d -> 50 -> 20 -> 50 -> %d\n", N_INPUTS, N_INPUTS);
    fprintf(stdout, "Samples: %d, Batch size: %d, Epochs: %d, LR: %.3f\n\n", 
            N_SAMPLES, BATCH_SIZE, N_EPOCHS, LEARNING_RATE);
    
    fprintf(stdout, "DEBUG: Checking for potential issues...\n");
    
    // Define network architecture (encoder-decoder structure)
    int n_layers = 4; // FIXED: era 3, ma servono 4 layer per 100->50->20->50->100
    int spec[4] = {50, 20, 50, N_INPUTS}; // 100->50->20->50->100
    
    // Create network
    ANN* net = (ANN*) malloc(sizeof(ANN));
    net->n_inputs = N_INPUTS;
    net->n_layers = n_layers;
    net->n_outputs = N_INPUTS;
    net->layer = (ANN_layer**) malloc(sizeof(ANN_layer*) * n_layers);
    
    int prev_units = N_INPUTS;
    for (int i = 0; i < n_layers; i++) {
        net->layer[i] = (ANN_layer*) malloc(sizeof(ANN_layer));
        init_layer(net->layer[i], spec[i], prev_units);
        prev_units = spec[i];
    }
    
    fprintf(stdout, "Network initialized successfully.\n");
    
    // Debug: Check initial weights
    fprintf(stdout, "Checking initial weights for NaN/Inf...\n");
    for (int i = 0; i < n_layers; i++) {
        check_layer_health(net->layer[i], i);
    }
    
    // Create dataset
    ANN_dataset* dataset = (ANN_dataset*) malloc(sizeof(ANN_dataset));
    dataset->n_samples = N_SAMPLES;
    dataset->batch_size = BATCH_SIZE;
    dataset->n_batches = N_SAMPLES / BATCH_SIZE;
    dataset->sample_dim = N_INPUTS;
    dataset->data = (double**) malloc(sizeof(double*) * N_SAMPLES);
    dataset->ref = (double**) malloc(sizeof(double*) * N_SAMPLES);
    
    generate_dataset(dataset);
    fprintf(stdout, "Dataset generated (%d samples).\n", N_SAMPLES);
    
    // Debug: Check dataset values
    fprintf(stdout, "Sample data range check:\n");
    double min_val = dataset->data[0][0], max_val = dataset->data[0][0];
    for (int i = 0; i < dataset->n_samples; i++) {
        for (int j = 0; j < dataset->sample_dim; j++) {
            if (dataset->data[i][j] < min_val) min_val = dataset->data[i][j];
            if (dataset->data[i][j] > max_val) max_val = dataset->data[i][j];
        }
    }
    fprintf(stdout, "Data range: [%.6f, %.6f]\n\n", min_val, max_val);
    
    // Compute initial loss
    double initial_loss = compute_mse(net, dataset);
    fprintf(stdout, "Initial MSE Loss: %.6f\n", initial_loss);
    
    // Debug: Check activations after first forward pass
    fprintf(stdout, "Checking network health after initial forward pass...\n");
    for (int i = 0; i < n_layers; i++) {
        check_layer_health(net->layer[i], i);
    }
    fprintf(stdout, "\n");
    
    // Train the network
    fprintf(stdout, "Starting training...\n");
    train_minibatch_sgd(net, dataset, N_EPOCHS, LEARNING_RATE);
    
    // Compute final loss
    double final_loss = compute_mse(net, dataset);
    fprintf(stdout, "\nFinal MSE Loss: %.6f\n", final_loss);
    fprintf(stdout, "Loss reduction: %.2f%%\n", 
            100.0 * (initial_loss - final_loss) / initial_loss);
    
    // Test reconstruction on first sample
    fprintf(stdout, "\n=== Sample Reconstruction Test ===\n");
    forward(net, dataset->data[0]);
    
    fprintf(stdout, "First 10 values:\n");
    fprintf(stdout, "Input:  ");
    for (int i = 0; i < 10; i++)
        fprintf(stdout, "%.3f ", dataset->data[0][i]);
    fprintf(stdout, "\nOutput: ");
    for (int i = 0; i < 10; i++)
        fprintf(stdout, "%.3f ", net->layer[n_layers-1]->a[i]);
    fprintf(stdout, "\n");
    
    // Cleanup
    destroy_dataset(dataset);
    free(dataset);
    destroy_ANN(net);
    
    fprintf(stdout, "\nTest completed successfully.\n");
    
    return 0;
}
