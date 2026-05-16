# Source folders
SRC = src
INCLUDE = include
BUILD = build

# Library target (produces libanne.a) 
CC = gcc
CFLAGS = -Wall -Wextra -O3 -I. -Iinclude/
LIBS = -lm
COMPILE = $(CC) $(CFLAGS) $(LIBS)   

lib: $(SRC)/anne.c $(SRC)/train.c $(INCLUDE)/aux.h $(INCLUDE)/fn.h
	$(CC) -c $(SRC)/anne.c $(SRC)/train.c $(INCLUDE)/fn.h $(CFLAGS) $(LIBS)
	ar rs $(BUILD)/libanne.a anne.o train.o

# Test target
TEST_FOLDER = ./1-autoencoder
TEST_LIBS = -L. -l:./build/libanne.a
test: lib $(TEST_FOLDER)/ae.c
	$(CC) -o ${TEST_FOLDER}/ae $(TEST_FOLDER)/ae.c $(TEST_LIBS) $(LIBS) $(CFLAGS)

clean: 
	rm -f *.o
	rm -f *.gch