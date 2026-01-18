# Produces libanne.a (static)
CC = gcc
CFLAGS = -Wall -Wextra -O3
LIBS = -lm -I.
COMPILE = $(CC) $(CFLAGS) $(LIBS)   

lib: anne.c train.c aux.h fn.h
	$(CC) -c anne.c train.c fn.h $(CFLAGS) $(LIBS)
	ar rs libanne.a anne.o train.o

TEST_FOLDER = ./test
TEST_LIBS = -L. -lanne
test: lib $(TEST_FOLDER)/test_layer.c $(TEST_FOLDER)/test_bp.c
	$(CC) -o test/exec-test_layer $(TEST_FOLDER)/test_layer.c $(TEST_LIBS) $(LIBS) $(CFLAGS)
	$(CC) -o test/exec-test_bp $(TEST_FOLDER)/test_bp.c $(TEST_LIBS) $(LIBS) $(CFLAGS)

clean: 
	rm -f *.o
