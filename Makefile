# Produces libanne.a (static)
CC = gcc
CFLAGS = -Wall -Wextra -O3
LIBS = -lm -I.
COMPILE = $(CC) $(CFLAGS) $(LIBS)   

lib: anne.c train.c aux.h fn.h
	$(CC) -c anne.c train.c fn.h $(CFLAGS) $(LIBS)
	ar rs libanne.a anne.o train.o

TEST_FOLDER = ./1-autoencoder
TEST_LIBS = -L. -lanne
test: lib $(TEST_FOLDER)/ae.c
	$(CC) -o ${TEST_FOLDER}/ae $(TEST_FOLDER)/ae.c $(TEST_LIBS) $(LIBS) $(CFLAGS)

clean: 
	rm -f *.o
	rm -f *.gch
