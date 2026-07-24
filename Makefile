# Source folders
SRC = src
BUILD = build

# Library target (produces libanne.a) 
CC = gcc
CFLAGS = -Wall -Wextra -O3 -I. -Isrc/support/ -Isrc/network -Isrc/datastruct
LIBS = -lm
COMPILE = $(CC) $(CFLAGS) $(LIBS)   

lib: $(SRC)/network/anne.c $(SRC)/network/train.c $(SRC)/support/aux.h $(SRC)/support/fn.h
	$(CC) -c $(SRC)/network/anne.c $(SRC)/network/train.c $(SRC)/support/fn.h $(CFLAGS) $(LIBS)
	ar rs $(BUILD)/libanne.a anne.o train.o

# Test target
TEST_FOLDER = ./1-autoencoder
TEST_LIBS = -L. -l:./build/libanne.a
test: lib $(TEST_FOLDER)/ae.c
	$(CC) -o ${TEST_FOLDER}/ae $(TEST_FOLDER)/ae.c $(TEST_LIBS) $(LIBS) $(CFLAGS)

clean: 
	rm -f *.o
	rm -f *.gch
