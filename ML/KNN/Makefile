CC=g++
SRC := $(ML_ROOT)/KNN/src
CFLAGS := -std=c++11 -DEUCLID -g
INCLUDE_DIR := $(PWD)/KNN/include/*

all: main

main: $(SRC)/knn.cc
	$(CC) $(CFLAGS) $(SRC)/knn.cc -o main -L${ML_ROOT}/lib/ -I${INCLUDE_DIR} \
	-I${ML_ROOT}/include -ldata

clean:
	rm main