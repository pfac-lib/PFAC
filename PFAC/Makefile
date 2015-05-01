# 
#  Top-level Makefile
#
#  recursive Make process
#
#

include common.mk

src_dir = src

test_dir = test 


.PHONY: all $(src_dir) $(test_dir) 

all: mkdir_obj_bin $(src_dir) $(test_dir) 


OBJ_DIR = obj
EXE_DIR = bin
LIB_DIR = lib

#
# Generate sub-directories bin, lib and obj
#
all_files = $(shell ls) 
binDir_matched = $(filter bin, $(all_files))
objDir_matched = $(filter obj, $(all_files))
libDir_matched = $(filter lib, $(all_files))

mkdir_obj_bin:
	$(if $(binDir_matched), , $(shell mkdir bin))
	$(if $(objDir_matched), , $(shell mkdir obj))
	$(if $(libDir_matched), , $(shell mkdir lib))

$(src_dir) $(test_dir):
	$(MAKE) --directory=$@
	
$(test_dir): $(src_dir)
       
####### Build rules

clean :
	rm -f *.linkinfo
	rm -f $(OBJ_DIR)/*
	rm -f $(EXE_DIR)/*
	rm -f $(LIB_DIR)/*
