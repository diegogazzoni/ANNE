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
#define MAX_FILES 1024*16

#define N_EPOCHS 50
#define LEARNING_RATE 0.01
#define BATCH_SIZE 32

#define LATENT_DIM 16
#define LAST_LAYER 5
#define DATASET "./mnist"
