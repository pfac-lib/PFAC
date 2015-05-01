/*
 *  Copyright 2011 Chen-Hsiung Liu, Lung-Sheng Chien, Cheng-Hung Lin,and Shih-Chieh Chang
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 */

/*
 * The example shows Unified Virtual Address Space
 *   
 * [quote of section 3.2.7 of CUDA programming guide 4.0]
 *     For devices of compute capability 2.0 and above and when the application is run
 * as a 64-bit peocess on Windows Vista/7 in TCC mode (only supported for devices from Tesla
 * series), on Windows XP, or on Linux, a single virtual address space is used for all allocations
 * made in host memory via cudaHostAlloc() and in any of the device memories via cudaMalloc*().
 * [/quote]
 *
 * Objective: PFAC context binds to GPU 0 and input/output data, d_input_string and d_matched_result
 * are allocated from GPU 1. This does not work on any CUDA version prior to CUDA 4.0.
 * In this example, we show how to do peer-to-peer memory access
 * (please refer to section 3.2.6.4 of CUDA programming guide 4.0)
 *
 * Requirement:  
 * 1) two GPUs are devices of compute capability 2.0, in our testing machine, 
 *    GPU 0 is GTX480 and GPU 1 is TeslaC2070
 * 2) 64-bit OS, which is listed in section 3.2.7 of CUDA programming guide 4.0
 * 3) CUDA 4.0 
 *    CUDA 4.0 RC2 is available on http://developer.nvidia.com/cuda-downloads
 *
 * How to compile
 *     g++ -m64 -fopenmp -I$(PFAC_LIB_ROOT)/include -I$(CUDA_ROOT)/include -o bin/UVA.exe test/UVA.cpp -L$(PFAC_LIB_ROOT)/lib -lpfac -L$(CUDA_ROOT)/lib64 -lcudart  
 *  
 *
 * This example shows following steps:
 * 1. set device to GPU 0, then
 *    bind PFAC context to GPU 0 and read pattern file
 * 2. set device to GPU 1, then
 *    allocate d_input_string and d_matched_result on GPU 1
 * 3. set device to GPU 0 again, then
 *    enable peer-to-peer access with device 1
 * 4. call PFAC_matchFromDevice()
 *
 */
 
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <cuda_runtime.h>
 
#include "../include/PFAC.h"

void show_memoryType( void *ptr );

int main(int argc, char **argv)
{
    char inputFile[] = "../test/data/example_input";
    char patternFile[] = "../test/pattern/example_pattern";
    PFAC_handle_t handle;
    PFAC_status_t PFAC_status;
    int input_size;
    char *h_input_string = NULL;
    int  *h_matched_result = NULL;

    char *d_input_string;
    int *d_matched_result;
    cudaError_t cuda_status; 

    // step 1: create PFAC handle and bind PFAC context to GPU 0 
    cuda_status = cudaSetDevice(0);
    assert( cudaSuccess == cuda_status );

    PFAC_status = PFAC_create( &handle ); 
    assert( PFAC_STATUS_SUCCESS == PFAC_status );

    PFAC_status = PFAC_readPatternFromFile(handle, patternFile);
    assert ( PFAC_STATUS_SUCCESS == PFAC_status );
  
    // prepare input stream h_input_string in host memory and
    // allocate h_matched_result to contain the matched results
    FILE* fpin = fopen( inputFile, "rb");
    assert ( NULL != fpin ) ;
    fseek (fpin , 0 , SEEK_END); 
    input_size = ftell (fpin); // obtain file size
    rewind (fpin);
    h_input_string = (char *) malloc (sizeof(char)*input_size);
    assert( NULL != h_input_string );

    h_matched_result = (int *) malloc (sizeof(int)*input_size);
    assert( NULL != h_matched_result );
    memset( h_matched_result, 0, sizeof(int)*input_size ) ;
     
    // copy the file into the buffer
    input_size = fread(h_input_string, 1, input_size, fpin);
    fclose(fpin); 

    // step 2: set device to GPU 1, allocate d_input_string and d_matched_result on GPU 1
    cuda_status = cudaSetDevice(1) ;
    assert( cudaSuccess == cuda_status ) ; 

    cuda_status = cudaMalloc((void **) &d_input_string, input_size);
    if ( cudaSuccess != cuda_status ){
        printf("Error: %s\n", cudaGetErrorString(cuda_status));
        exit(1) ;
    }
    printf("d_input_string = %p, its UVA info:\n", d_input_string );
    show_memoryType( (void *)d_input_string );
    
    // allocate GPU memory for matched result
    cuda_status = cudaMalloc((void **) &d_matched_result, sizeof(int)*input_size);
    if ( cudaSuccess != cuda_status ){
        printf("Error: %s\n", cudaGetErrorString(cuda_status));
        exit(1) ;
    }
    printf("d_matched_result = %p, its UVA info:\n", d_matched_result );
    show_memoryType( (void *)d_matched_result );
    
    // copy input string from host to device
    cudaMemcpy(d_input_string, h_input_string, input_size, cudaMemcpyHostToDevice);

   /*
    * step 3: set device to GPU 0 again, enable peer-to-peer access with device 1
    *    PFAC binds to GPU 0 and wants to access d_input_string and d_matched_result in GPU 1
    * WARNING: if cudaDeviceEnablePeerAccess() is disable, then error "unspecified launch failure" occurs
    */
    cuda_status = cudaSetDevice(0) ;
    assert( cudaSuccess == cuda_status ); 
    cuda_status = cudaDeviceEnablePeerAccess(1, 0); // enable peer to peer access 
    assert( cudaSuccess == cuda_status ); 
          
    // step 4: run PFAC on GPU by calling PFAC_matchFromDevice          
    PFAC_status = PFAC_matchFromDevice( handle, d_input_string, input_size, d_matched_result ) ;
    if ( PFAC_STATUS_SUCCESS != PFAC_status ){
        printf("Error: PFAC_matchFromDevice failed, %s\n", PFAC_getErrorString(PFAC_status) );
        exit(1) ;	
    }     
    
    cudaThreadSynchronize();
    cuda_status = cudaGetLastError();
    if ( cudaSuccess != cuda_status ){
        printf("Error: PFAC_matchFromDevice failed, %s\n", cudaGetErrorString(cuda_status));
        exit(1) ;        	
    }
    
    // copy the result data from device to host
    cudaMemcpy(h_matched_result, d_matched_result, sizeof(int)*input_size, cudaMemcpyDeviceToHost);

    // step 5: output matched result
    for (int i = 0; i < input_size; i++) {
        if (h_matched_result[i] != 0) {
            printf("At position %4d, match pattern %d\n", i, h_matched_result[i]);
        }
    }

    PFAC_destroy( handle ) ;

    free(h_input_string);
    free(h_matched_result); 
    cudaFree(d_input_string);
    cudaFree(d_matched_result);
    
    cudaThreadExit();
    
    return 0;
}


void show_memoryType( void *ptr )
{
    cudaError_t cuda_status ;
    struct cudaPointerAttributes  attributes;
    
    cuda_status = cudaPointerGetAttributes( &attributes, ptr);
    assert( cudaSuccess == cuda_status ) ;
    
    if ( cudaMemoryTypeHost == attributes.memoryType ){
        printf("\tptr belongs to host memory, device = %d, hostPointer = %p\n", 
            attributes.device, attributes.hostPointer );
    }else if ( cudaMemoryTypeDevice == attributes.memoryType ){
        printf("\tptr belongs to device memory, device = %d, devicePointer = %p\n", 
            attributes.device, attributes.devicePointer );
    }else{
        printf("Error: unknown .memoryType %d\n", attributes.memoryType);	
        exit(1);
    }
    
}

