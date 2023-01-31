CC=g++
INCLUDE_DIR := $(ML_ROOT)/include
src := $(ML_ROOT)/src
CFLAGS := -shared -std=c++11 -g
LIB_DATA := libdata.so

all : $(LIB_DATA)

$(LIB_DATA): libdir objdir obj/handle_data.o obj/data.o obj/common.o
	$(CC) $(CFLAGS) -o $(ML_ROOT)/lib/$(LIB_DATA) obj/*.o
	rm -r ${ML_ROOT}/obj

libdir:
	mkdir -p $(ML_ROOT)/lib

objdir:
	mkdir -p $(ML_ROOT)/obj

obj/handle_data.o: $(src)/handle_data.cc
	$(CC) -fPIC ${CFLAGS} -o obj/handle_data.o -I$(INCLUDE_DIR) -c $(src)/handle_data.cc

obj/data.o: $(src)/data.cc
	$(CC) -fPIC ${CFLAGS} -o obj/data.o -I$(INCLUDE_DIR) -c $(src)/data.cc

obj/common.o: $(src)/common.cc
	$(CC) -fPIC ${CFLAGS} -o obj/common.o -I$(INCLUDE_DIR) -c $(src)/common.cc

clean:
	rm -r $(ML_ROOT)/lib
	rm -r $(ML_ROOT)/obj