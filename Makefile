CC = gcc
CFLAGS = -Wall -Wextra -O2
LDFLAGS = -lm

OBJ = anne.o train.o cnet.o

TARGET = anne

all: $(TARGET)

$(TARGET): $(OBJ)
	$(CC) $(OBJ) -o $(TARGET) $(LDFLAGS)

%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

anne.o: anne.c anne.h utils.h
train.o: train.c train.h anne.h
cnet.o: cnet.c anne.h train.h utils.h

# Pulizia dei file temporanei
clean:
	rm -f *.o $(TARGET)
