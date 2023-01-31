CC=g++
SRC := $(PWD)/src
CFLAGS := -std=c++11 -g -DMNIST
INCLUDE_DIR := $(PWD)/include/

all: main

main : obj $(SRC)/nn.cc obj/layer.o obj/neuron.o
	$(CC) $(CFLAGS) $(SRC)/nn.cc -o main -L$(ML_ROOT)/lib/ -I$(INCLUDE_DIR) \
	-I$(ML_ROOT)/include -ldata ./obj/*.o

obj:
	if [ ! -d "./obj" ]; then mkdir obj; fi

obj/layer.o : $(SRC)/layer.cc
	$(CC) -c $(CFLAGS) $(SRC)/layer.cc -o obj/layer.o -I$(INCLUDE_DIR) \
	-I$(ML_ROOT)/include

obj/neuron.o : $(SRC)/neuron.cc
	$(CC) -c $(CFLAGS) $(SRC)/neuron.cc -o obj/neuron.o -I$(INCLUDE_DIR) \
	-I$(ML_ROOT)/include

clean :
	rm -r main obj