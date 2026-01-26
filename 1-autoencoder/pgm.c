#include <stdio.h>
#include <stdlib.h>
#include <ctype.h>
#include <stdint.h>

#define MAX_CHARS_LINE 70

typedef struct {
	char *magic;
	unsigned int w, h;
	unsigned int max;
	unsigned int *data;
	unsigned int n_lines;
} PGM;

int read(PGM* pgm, char* filename) {
	FILE* fp = fopen(filename, "r"); // Open the file and create a buffer
	
	uint8_t magic_found, dim_found, max_found; 
	char magic[3]; // 3 because I need to store \0
	unsigned int w, h;
	unsigned int max;
	unsigned int n_lines = 0;
	unsigned int* data;
	char line[MAX_CHARS_LINE+1];
	
	while(fgets(line, MAX_CHARS_LINE+1, fp)) {
		char c;
		sscanf(line, "%1c", &c);
		if (c != '#') {  // if line is not a comment
			n_lines++;
			if (!magic_found) {
				sscanf(line, "%2c", magic);	// magic is the first, if not found due to comments is the first to check
				magic_found = 1;
			} else if (!dim_found) { // find dimensions
				sscanf(line, "%d", &w);
				sscanf(line, "%d", &h);
				dim_found = 1;
				// Now we can alloc data structure for pixels
				data = malloc(sizeof(unsigned int) * w * h);
			} else if (!max_found) { // find max
				sscanf(line, "%d", &max);
				max_found = 1;
			} else { // read bytes
				// Now I need to read each integer in this line.
				unsigned int pix;
				unsigned int n = 0;
				while(sscanf(line, "%d", &pix) != 1) {	
					data[n++] = pix;
				}
			}
		} 		
	}
	
	if ((magic_found+dim_found+max_found) != 3) {
		fprintf(stdout, "Wrong header.");
		return -1;
	}

	pgm->magic = magic;
	pgm->w = w;
	pgm->h = h;
	pgm->max = max;
	pgm->n_lines = n_lines;
	pgm->data = data;
	return 1;
}

void dump(PGM* pgm) {
	if (pgm == NULL) {
		fprintf(stdout, "Impossible to dump PGM, it must be initialized.");
		return;
	}
	fprintf(stdout, "MAGIC=%s\nW=%d\nH=%d\nMAX=%d\nN.LINES=%d\n", pgm->magic, pgm->w, pgm->h, pgm->max, pgm->n_lines);
}

int main() {
	PGM pgm;
	read(&pgm, "./data/baboon.ascii.pgm");
	dump(&pgm);

	return 0;
}
