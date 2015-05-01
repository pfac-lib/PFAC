# 
#  Top-level Makefile
#
#  recursive Make process
#
#

include ../common.mk

OBJ_DIR = ../obj
EXE_DIR = ../bin
LIB_DIR = ../lib

TARGET = simple_example simple_example_reduce  profiling  omp_PFAC  SimpleMultiGPU_pthread  

.PHONY: all $(TARGET)

exe_files = $(patsubst %,%.exe,$(TARGET))

all: $(exe_files)
 
%.exe: $(OBJ_DIR)/%.o
	$(LINK) -o $(EXE_DIR)/$@ $< -L$(LIB_DIR) -lpfac $(LIBS)

####### Build rules

$(OBJ_DIR)/%.o: %.cpp
	$(CXX) -c $(CXXFLAGS) $(INCPATH) -o $@ $<

####### Implicit rules

.SUFFIXES: .o .c .cpp .cc .cxx .C

.cpp.o:
	$(CXX) -c $(CXXFLAGS) $(INCPATH) -o "$@" "$<"

.cc.o:
	$(CXX) -c $(CXXFLAGS) $(INCPATH) -o "$@" "$<"

.cxx.o:
	$(CXX) -c $(CXXFLAGS) $(INCPATH) -o "$@" "$<"

.C.o:
	$(CXX) -c $(CXXFLAGS) $(INCPATH) -o "$@" "$<"

.c.o:
	$(CC) -c $(CFLAGS) $(INCPATH) -o "$@" "$<"
 
