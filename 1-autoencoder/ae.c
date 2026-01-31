#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <dirent.h>
#include <sys/stat.h>

#include "anne.h"
#include "train.h"
#include "fn.h"

#define IMG_DIM 28
#define N_PIXELS (IMG_DIM*IMG_DIM)
#define MAX_FILES 1024*8

#define N_EPOCHS 300
#define LEARNING_RATE 0.025
#define BATCH_SIZE 32

#define LATENT_DIM 16
#define LAST_LAYER 5
#define DATASET "./mnist"

ANN* create_AE(int n_inputs) {
    ANN* ae = (ANN*) malloc(sizeof(ANN));
    init_ANN(ae, n_inputs, n_inputs, 6, fn_MSE, grad_MSE);
    
    ae->layer = (ANN_layer**) malloc(sizeof(ANN_layer*) * 6);
    ae->layer[0] = (ANN_layer*) malloc(sizeof(ANN_layer));
    ae->layer[1] = (ANN_layer*) malloc(sizeof(ANN_layer));
    ae->layer[2] = (ANN_layer*) malloc(sizeof(ANN_layer));
    ae->layer[3] = (ANN_layer*) malloc(sizeof(ANN_layer));
		ae->layer[4] = (ANN_layer*) malloc(sizeof(ANN_layer));
		ae->layer[5] = (ANN_layer*) malloc(sizeof(ANN_layer));

		// Encoder	
    init_layer(ae->layer[0], relu, d_relu, 128, n_inputs);
    init_layer(ae->layer[1], relu, d_relu, 64, 128);
    init_layer(ae->layer[2], relu, d_relu, LATENT_DIM, 64);
		// Decoder
    init_layer(ae->layer[3], relu, d_relu, 64, LATENT_DIM);
		init_layer(ae->layer[4], relu, d_relu, 128, 64);
    init_layer(ae->layer[5], relu, d_relu, n_inputs, 128);
    
    return ae;
}

static double* load_pgm(const char* path) {
    FILE* f = fopen(path, "rb");
    if (!f) {
        fprintf(stderr, "Cannot open: %s\n", path);
        return NULL;
    }
    
    char magic[3], line[256];
    int w, h, maxval;
    
    // Magic number
    if (!fgets(line, sizeof(line), f)) {
        fprintf(stderr, "Cannot read magic: %s\n", path);
        fclose(f);
        return NULL;
    }
    sscanf(line, "%2s", magic);
    
    // Skip comments and read dimensions
    while (fgets(line, sizeof(line), f)) {
        if (line[0] != '#') {
            sscanf(line, "%d %d", &w, &h);
            break;
        }
    }
    
    // Maxval (skip comments)
    while (fgets(line, sizeof(line), f)) {
        if (line[0] != '#') {
            sscanf(line, "%d", &maxval);
            break;
        }
    }
    
    if ((strcmp(magic, "P5") && strcmp(magic, "P2")) || w <= 0 || h <= 0) {
        fprintf(stderr, "Invalid PGM: %s (magic=%s, w=%d, h=%d)\n", path, magic, w, h);
        fclose(f);
        return NULL;
    }
    
    printf("  Loading %s: %s %dx%d\n", path, magic, w, h);
    
    // Load original image
    unsigned char* buf = (unsigned char*) malloc(w * h);
    if (!buf) {
        fprintf(stderr, "Cannot allocate buffer for: %s\n", path);
        fclose(f);
        return NULL;
    }
    
    if (strcmp(magic, "P5") == 0) {
        // Binary
        fread(buf, 1, w * h, f);
    } else {
        // ASCII
        for (int i=0; i<w*h; i++) {
            int val;
            if (fscanf(f, "%d", &val) != 1) {
                fprintf(stderr, "Read error at pixel %d in %s\n", i, path);
                free(buf);
                fclose(f);
                return NULL;
            }
            buf[i] = (unsigned char)val;
        }
    }
    fclose(f);
    
    // Resize to IMG_DIM x IMG_DIM using nearest neighbor
    double* img = (double*) malloc(N_PIXELS * sizeof(double));
    if (!img) {
        fprintf(stderr, "Cannot allocate output image\n");
        free(buf);
        return NULL;
    }
    
    float scale_x = (float)w / IMG_DIM;
    float scale_y = (float)h / IMG_DIM;
    
    for (int y=0; y<IMG_DIM; y++) {
        for (int x=0; x<IMG_DIM; x++) {
            int src_x = (int)(x * scale_x);
            int src_y = (int)(y * scale_y);
            img[y * IMG_DIM + x] = buf[src_y * w + src_x] / 255.0;
        }
    }
    
    free(buf);
    return img;
}

static void save_pgm(const char* path, double* img) {
    FILE* f = fopen(path, "wb");
    fprintf(f, "P5\n%d %d\n255\n", IMG_DIM, IMG_DIM);
    unsigned char buf[N_PIXELS];
    for (int i=0; i<N_PIXELS; i++) {
        double v = img[i] * 255.0;
        buf[i] = v < 0 ? 0 : (v > 255 ? 255 : (unsigned char)v);
    }
    fwrite(buf, 1, N_PIXELS, f);
    fclose(f);
}

static int is_pgm(const char* name) {
    int len = strlen(name);
    return len > 4 && strcmp(name + len - 4, ".pgm") == 0;
}

static ANN_dataset* load_dataset(const char* dir, int batch_size) {
    DIR* d = opendir(dir);
    if (!d) return NULL;
    
    char** paths = (char**) malloc(sizeof(char*) * MAX_FILES);
    int n = 0;
    
    struct dirent* ent;
    while ((ent = readdir(d)) && n < MAX_FILES) {
        if (is_pgm(ent->d_name)) {
            paths[n] = (char*) malloc(256);
            snprintf(paths[n], 256, "%s/%s", dir, ent->d_name);
            n++;
        }
    }
    closedir(d);
    
    if (n == 0) {
        free(paths);
        return NULL;
    }
    
    double** data = (double**) malloc(sizeof(double*) * n);
    double** ref = (double**) malloc(sizeof(double*) * n);
    
    for (int i=0; i<n; i++) {
        data[i] = load_pgm(paths[i]);
        ref[i] = data[i];
        free(paths[i]);
    }
    free(paths);
    
    // Adjust batch size to not exceed dataset size
    if (batch_size > n) batch_size = n;
    
    ANN_dataset* ds = (ANN_dataset*) malloc(sizeof(ANN_dataset));
    ds->n_samples = n;
    ds->batch_size = batch_size;
    ds->n_batches = (n + batch_size - 1) / batch_size;
    ds->sample_dim = N_PIXELS;
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

static double eval_mse(ANN* ae, ANN_dataset* ds) {
    double total = 0.0;
    for (int i=0; i<ds->n_samples; i++) {
        forward(ae, ds->data[i]);
        total += fn_MSE(ae->layer[LAST_LAYER]->a, ds->ref[i], N_PIXELS);
    }
    return total / ds->n_samples;
}

static void test_and_save(ANN* ae, const char* test_dir, const char* out_dir) {
    DIR* d = opendir(test_dir);
    if (!d) return;
    
    mkdir(out_dir, 0755);
    
    struct dirent* ent;
    double total_mse = 0.0;
    int n = 0;
    
    while ((ent = readdir(d))) {
        if (!is_pgm(ent->d_name)) continue;
        if (n > MAX_FILES) break;
        
        char in_path[256], out_path[256];
        snprintf(in_path, 256, "%s/%s", test_dir, ent->d_name);
        snprintf(out_path, 256, "%s/recon_%s", out_dir, ent->d_name);
        
        double* img = load_pgm(in_path);
        if (!img) continue;
        
        forward(ae, img);
        double mse = fn_MSE(ae->layer[LAST_LAYER]->a, img, N_PIXELS);
        total_mse += mse;
        n++;
        
        save_pgm(out_path, ae->layer[LAST_LAYER]->a);
        printf("  %s -> MSE: %.6f\n", ent->d_name, mse);
        
        free(img);
    }
    closedir(d);
    
    if (n > 0)
        printf("\nTest MSE: %.6f\n", total_mse / n);
}

int main() {
    printf("=== Autoencoder Training ===\n");
    printf("Loading dataset from train/...\n");
    
    ANN_dataset* train_ds = load_dataset(DATASET, BATCH_SIZE);
    if (!train_ds) {
        fprintf(stderr, "Failed to load train dataset\n");
        return 1;
    }
    
    printf("Loaded %d training images\n", train_ds->n_samples);
    printf("Sample dim: %d, Batch size: %d\n", train_ds->sample_dim, train_ds->batch_size);
    
    printf("\nCreating autoencoder network...\n");
    ANN* ae = create_AE(N_PIXELS);
    printf("Network created successfully\n");
    
    printf("\nStarting training (%d epochs, lr=%.4f)...\n", N_EPOCHS, LEARNING_RATE);
    train_minibatch_sgd(ae, train_ds, N_EPOCHS, LEARNING_RATE);
    
    double final_mse = eval_mse(ae, train_ds);
    printf("\nFinal Train MSE: %.6f\n", final_mse);
		
    printf("\nTesting...\n");
    test_and_save(ae, DATASET, "output");
    
    printf("\nCleaning up...\n");
    free_dataset(train_ds);
    destroy_ANN(ae);
    printf("Done!\n");
    return 0;
}
