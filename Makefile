CC = gcc
CFLAGS = -Wall -Wextra -O3
LIBS = -lm

libanne.a: anne.o train.o
	$(CC) anne.o train.o $(CFLAGS) $(LIBS)

clean:
	rm -f $(OBJS) $(LIB_NAME)

