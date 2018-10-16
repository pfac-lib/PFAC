#
# Makefile in directory src
#
# resource usage:
#
# To compile a dynamic module
# (1) nvcc cannot accept -fPIC, so compile .cu to .cu.cpp first
#     nvcc -arch=sm_20 -cuda ../src/PFAC_kernel.cu
#
# (2) then use g++ to comple PFAC_notex_shared_reorder.cu.cpp
#     g++ -fPIC -c PFAC_kernel.cu.cpp
#
# (3) finally combine two object files to a .so library
#     g++ -shared -o libpfac.so  $(LIBS) PFAC_kernel.cu.o ...
#
#     $(LIBS) is necessary when compiling PFAC library on 32-bit machine
#

include ../common.mk

INC_DIR = ../include
LIB_DIR = ../lib
OBJ_DIR = ../obj

INCPATH += -I../include/

CU_SRC = PFAC_kernel.cu
CU_SRC += PFAC_reduce_kernel.cu
CU_SRC += PFAC_reduce_inplace_kernel.cu
CU_SRC += PFAC_kernel_spaceDriven.cu

CPP_SRC  = PFAC_reorder_Table.cpp
CPP_SRC += PFAC_CPU.cpp
CPP_SRC += PFAC_CPU_OMP.cpp
CPP_SRC += PFAC.cpp

inc_files = $(INC_DIR)/PFAC_P.h  $(INC_DIR)/PFAC.h

CU_CPP = $(patsubst %.cu,%.cu.cpp,$(CU_SRC)) 

CPP_OBJ = $(patsubst %.cpp,%.o,$(CPP_SRC)) 

cu_cpp_obj_loc = $(patsubst %.cpp,$(OBJ_DIR)/$(sm)_%.cpp.o,$(CU_CPP))

cppobj_loc = $(patsubst %.o,$(OBJ_DIR)/%.o,$(CPP_OBJ))

cppobj_fpic_loc = $(patsubst %.o,$(OBJ_DIR)/%_fpic.o,$(CPP_OBJ))

all:  mk_libso_sm mk_lib_fpic

mk_libso_sm:
	$(foreach sm, $(sm_support), $(info Compiling to $(sm)) $(cu_cpp) $(cu_cpp_link))

cu_cpp = $(foreach file, $(CU_SRC), $(cu_cpp_file))

cu_cpp_file = $(shell $(NVCC) -arch=$(sm) -cuda $(INCPATH) -o $(OBJ_DIR)/$(sm)_$(file).cpp $(file) && \
		$(CXX) -fPIC -O2 -c -o $(OBJ_DIR)/$(sm)_$(file).cpp.o $(OBJ_DIR)/$(sm)_$(file).cpp \
)

cu_cpp_link = $(shell $(CXX) -shared -o $(LIB_DIR)/libpfac_$(sm).so $(LIBS) $(cu_cpp_obj_loc))

mk_liba: $(cppobj_loc)
	ar cru $(LIB_DIR)/libpfac.a  $(cppobj_loc)
	ranlib $(LIB_DIR)/libpfac.a

mk_lib_fpic: $(cppobj_fpic_loc)
	$(CXX) -shared -o $(LIB_DIR)/libpfac.so  $(cppobj_fpic_loc) $(LIBS)


$(OBJ_DIR)/%_fpic.o: %.cpp  $(inc_files)
	$(CXX) -fPIC -c $(CXXFLAGS) $(INCPATH) -o $@ $<

$(OBJ_DIR)/PFAC_CPU_OMP_reorder_fpic.o: PFAC_CPU_OMP_reorder.cpp  $(inc_files)
	$(CXX) -fPIC -c $(CXXFLAGS) $(INCPATH) -o $@ $<

$(OBJ_DIR)/PFAC_CPU_OMP_reorder.o: PFAC_CPU_OMP_reorder.cpp  $(inc_files)
	$(CXX) -c $(CXXFLAGS) $(INCPATH) -o $@ $<

$(OBJ_DIR)/%.o: %.cpp  $(inc_files)
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

####### Build rules
