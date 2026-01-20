#include <stdio.h>
#include <stdlib.h>
#include <ctype.h>

#define MAX_LEN 70

typedef struct {
	char *magic;
	int w, h;
	int max;
	double *data;
} PGM;

int read(PGM* dest, char* filename) {

}

int main() {
	PGM pgm;
	//read(&pgm, "./data/baboon.ascii.pgm");

	return 0;
}
