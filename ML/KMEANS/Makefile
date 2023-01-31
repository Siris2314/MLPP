CC=g++
SRC := $(ML_ROOT)/KMEANS/src
CFLAGS := -std=c++11
INCLUDE_DIR := $(PWD)/include/*

all: main

main: $(SRC)/kmeans.cc
	$(CC) $(CFLAGS) $(SRC)/kmeans.cc -o main -L${ML_ROOT}/lib/ -I${INCLUDE_DIR} \
	-I${ML_ROOT}/include -ldata

clean:
	rm main