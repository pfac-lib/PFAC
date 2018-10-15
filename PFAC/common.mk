#
#   Common part of Makefile
#
#   default compiler is gcc/g++ targeting on 32-bit platform (-m32) or 64-bit platform (-m64)
#   default host compiler of nvcc is also g++
# 
#   Remark: users must set path of CUDA library such that nvcc can be found by
#      >which nvcc
#
#      or making process cannot continue
#
#   if users want to compile host code by 3rd party compiler, then CC, CXX and LINK must be modified
#
#   For example: if users want to use Intel C++ compiler, then 
#
#       CC   = icc -fopenmp
#       CXX  = icpc -openmp
#       LINK = icpc -openmp -Wall
#
#   note that we use machine type {x86_64 or i386} to choose either -m64 or -m32.
#

PFAC_LIB_ROOT := $(PWD)

CFLAGS        = -O2 -D_REENTRANT -Wall
CXXFLAGS      = -O2 -D_REENTRANT -Wall

#
# CUDA toolkit may not be installed in default directory /usr/local/cuda 
# variable 'cudalib_path' finds correct path of CUDA library 
#
# machine = {x86_64, i386 or i686}
#
nvcc_loc := $(shell which nvcc)
machine  := $(shell uname -m) 
cudalib_path := $(if $(filter $(machine),x86_64),$(patsubst %/bin/nvcc,%/lib64,$(nvcc_loc)),$(patsubst %/bin/nvcc,%/lib,$(nvcc_loc)))
cudainc_path := $(patsubst %/bin/nvcc,%/include,$(nvcc_loc))

CC   := $(if $(filter $(machine),x86_64),gcc -m64 -fopenmp,gcc -m32 -fopenmp)
CXX  := $(if $(filter $(machine),x86_64),g++ -m64 -fopenmp,g++ -m32 -fopenmp)
LINK := $(if $(filter $(machine),x86_64),g++ -m64 -fopenmp -Wall,g++ -m32 -fopenmp -Wall)

NVCC := $(if $(filter $(machine),x86_64),nvcc -m64,nvcc -m32)

sm_support = $(shell nvcc -h | grep --only-matching "sm_[0-9][0-9]" | sort --unique)

#
# default CUDA library path is /usr/local/cuda
#
#LIBS          = -L/usr/local/cuda/lib64 -lcudart -lcublas -ldl -lpthread
#INCPATH       = -I/usr/local/cuda/include

LIBS          = -L$(cudalib_path) -lcudart -ldl -lpthread
INCPATH       = -I$(cudainc_path)

#
# add PFAC include directory
#
INCPATH += -I$(PFAC_LIB_ROOT)/include
