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

#include "../include/PFAC_P.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <dlfcn.h>

#include <vector>

using namespace std ;

/* This is missing from very old Linux libc. */
#ifndef RTLD_NOW
#define RTLD_NOW 2
#endif

#include <limits.h>
#ifndef PATH_MAX
#define PATH_MAX 255
#endif

//#define DEBUG_MSG

void  PFAC_freeResource( PFAC_handle_t handle );

PFAC_status_t  PFAC_bindCudaArray( PFAC_handle_t handle );
PFAC_status_t  PFAC_bindLinearMem( PFAC_handle_t handle );
void  PFAC_freeCUDAarray( PFAC_handle_t handle );
void  PFAC_freeCUDALinearMem( PFAC_handle_t handle );
PFAC_status_t  PFAC_bindTable( PFAC_handle_t handle );


PFAC_status_t  PFAC_create( PFAC_handle_t *handle )
{
    *handle = (PFAC_handle_t) malloc( sizeof(PFAC_context) ) ;

    if ( NULL == *handle ){
        return PFAC_STATUS_ALLOC_FAILED ;
    }

    memset( *handle, 0, sizeof(PFAC_context) ) ;

    // bind proper library sm_20, sm_13, sm_11 ...
    char modulepath[1+ PATH_MAX];
    void *module = NULL;

    int device ;
    cudaError_t cuda_status = cudaGetDevice( &device ) ;
    if ( cudaSuccess != cuda_status ){
        return (PFAC_status_t)cuda_status ;
    }

    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, device);

#ifdef DEBUG_MSG
    printf("major = %d, minor = %d\n", deviceProp.major, deviceProp.minor );
#endif

    int device_no = 10*deviceProp.major + deviceProp.minor ;
    if ( 21 == device_no ){
        strcpy (modulepath, "libpfac_sm21.so");    
    }else if ( 20 == device_no ){
        strcpy (modulepath, "libpfac_sm20.so");
    }else if ( 13 == device_no ){
        strcpy (modulepath, "libpfac_sm13.so");
    }else if ( 12 == device_no ){
        strcpy (modulepath, "libpfac_sm12.so");
    }else if ( 11 == device_no ){
        strcpy (modulepath, "libpfac_sm11.so");
    }else{
        return PFAC_STATUS_ARCH_MISMATCH ;
    }
    
    (*handle)->device_no = device_no ;
    
#ifdef DEBUG_MSG
    printf("load module %s \n", modulepath );
#endif

    // Load the module.
    module = dlopen (modulepath, RTLD_NOW);
    if (!module){
#ifdef DEBUG_MSG
        printf("Error: modulepath(%s) cannot load module\n", modulepath );
#endif
        return PFAC_STATUS_LIB_NOT_EXIST ;
    }

    // Find entry point of PFAC_kernel
    (*handle)->kernel_ptr = (PFAC_kernel_protoType) dlsym (module, "PFAC_kernel");
    if ( NULL == (*handle)->kernel_ptr ){
#ifdef DEBUG_MSG
        printf("Error: cannot load PFAC_kernel, error = %s\n", dlerror() );
#endif
        return PFAC_STATUS_INTERNAL_ERROR ;
    }

    // Find entry point of PFAC_reduce_kernel
    (*handle)->reduce_kernel_ptr = (PFAC_reduce_kernel_protoType) dlsym (module, "PFAC_reduce_kernel");
    if ( NULL == (*handle)->reduce_kernel_ptr ){
#ifdef DEBUG_MSG
        printf("Error: cannot load PFAC_reduce_kernel, error = %s\n", dlerror() );
#endif
        return PFAC_STATUS_INTERNAL_ERROR ;
    }

  
    (*handle)->reduce_inplace_kernel_ptr = (PFAC_reduce_kernel_protoType) dlsym (module, "PFAC_reduce_inplace_kernel");
    if ( NULL == (*handle)->reduce_inplace_kernel_ptr ){
#ifdef DEBUG_MSG
        printf("Error: cannot load PFAC_reduce_inplace_kernel, error = %s\n", dlerror() );
#endif
        return PFAC_STATUS_INTERNAL_ERROR ;
    }

    return PFAC_STATUS_SUCCESS ;
}


PFAC_status_t  PFAC_destroy( PFAC_handle_t handle )
{
    if ( NULL == handle ){
        return PFAC_STATUS_INVALID_HANDLE ;
    }

    PFAC_freeResource( handle ) ;

    free( handle ) ;

    return PFAC_STATUS_SUCCESS ;
}


void  PFAC_freeCUDAarray( PFAC_handle_t handle )
{
    if ( NULL != handle->d_PFAC_table_array ){
        cudaFreeArray(handle->d_PFAC_table_array);
        handle->d_PFAC_table_array = NULL ;
    }
}


void  PFAC_freeCUDALinearMem( PFAC_handle_t handle )
{
    if ( NULL != handle->d_PFAC_table ){
        cudaFree(handle->d_PFAC_table);
        handle->d_PFAC_table= NULL ;
    }
}


void  PFAC_freeResource( PFAC_handle_t handle )
{
    // free host resource
    if ( NULL != handle->valPtr ){
        free( handle->valPtr );
        handle->valPtr = NULL ;
    }
    if ( NULL != handle->rowPtr ){
        free( handle->rowPtr );
        handle->rowPtr = NULL ;
    }
    if ( NULL != handle->PFAC_table ){
        free( handle->PFAC_table ) ;
        handle->PFAC_table = NULL ;
    }
    if ( NULL != handle->patternLen_table ){
        free( handle->patternLen_table ) ;
        handle->patternLen_table = NULL ;
    }
    if ( NULL != handle->patternID_table ){
        free( handle->patternID_table );
        handle->patternID_table = NULL ;
    }

    // free device resource
    PFAC_freeCUDALinearMem( handle ) ;
    PFAC_freeCUDAarray( handle ) ;

    handle->isPatternsReady = false ;
}


PFAC_status_t  PFAC_bindCudaArray( PFAC_handle_t handle )
{
    if ( NULL == handle ){
        return PFAC_STATUS_INVALID_HANDLE ;
    }

    if ( !(handle->isPatternsReady) ){
        return PFAC_STATUS_PATTERNS_NOT_READY ;
    }

    if ( NULL != handle->d_PFAC_table_array ){
        return     PFAC_STATUS_SUCCESS ;
    }

    int state_num = handle->state_num ;

    // set texture memory for PFAC table on device
    handle->channelDesc = cudaCreateChannelDesc (sizeof(int)*8, 0, 0, 0, cudaChannelFormatKindSigned);
    cudaError_t cuda_status = cudaMallocArray(&handle->d_PFAC_table_array, &handle->channelDesc, CHAR_SET, state_num); // d_PFAC_table[state_num][CHAR_SET]
    if ( cudaSuccess != cuda_status ){
#ifdef DEBUG_MSG
        printf("Error: cudaMallocArray(PFAC_table(%d,%d)): %s\n", state_num, CHAR_SET, cudaGetErrorString(cuda_status));
        size_t  free_size, total_size ;
        cuda_status = cudaMemGetInfo( &free_size, &total_size );
        double  free_db = (double)free_size / 1024. /1024. ;
        double  total_db = (double)total_size / 1024. /1024. ;
        printf("free:%5.2f MB, total:%5.2f MB\n", free_db, total_db ) ;
#endif
        return PFAC_STATUS_CUDA_ALLOC_FAILED ;
    }
    cuda_status = cudaMemcpyToArray(handle->d_PFAC_table_array, 0, 0, handle->PFAC_table, sizeof(int)*CHAR_SET*state_num, cudaMemcpyHostToDevice);
    if ( cudaSuccess != cuda_status ){
        return PFAC_STATUS_INTERNAL_ERROR ;
    }

    return PFAC_STATUS_SUCCESS ;
}


PFAC_status_t  PFAC_bindLinearMem( PFAC_handle_t handle )
{
    if ( NULL == handle ){
        return PFAC_STATUS_INVALID_HANDLE ;
    }

    if ( !(handle->isPatternsReady) ){
        return PFAC_STATUS_PATTERNS_NOT_READY ;
    }

    if ( NULL != handle->d_PFAC_table ){
        return PFAC_STATUS_SUCCESS ;
    }

    int state_num = handle->state_num ;

    cudaError_t cuda_status = cudaMalloc((void **) &handle->d_PFAC_table, sizeof(int)*CHAR_SET*state_num);
    if ( cudaSuccess != cuda_status ){
        return PFAC_STATUS_CUDA_ALLOC_FAILED ;
    }

    cuda_status = cudaMemcpy(handle->d_PFAC_table, handle->PFAC_table,
        sizeof(int)*CHAR_SET*state_num, cudaMemcpyHostToDevice);
    if ( cudaSuccess != cuda_status ){
        return PFAC_STATUS_INTERNAL_ERROR ;
    }

    return PFAC_STATUS_SUCCESS ;
}


PFAC_status_t  PFAC_bindTable( PFAC_handle_t handle )
{
    PFAC_status_t  PFAC_status ;

    if ( NULL == handle ){
        return PFAC_STATUS_INVALID_HANDLE ;
    }
    if ( PFAC_AUTOMATIC == handle->textureMode ){
        // bind texture first
        PFAC_status = PFAC_bindCudaArray( handle ) ;
        if ( PFAC_STATUS_SUCCESS != PFAC_status ){
            cudaGetLastError(); // clear last error
            PFAC_freeCUDAarray( handle ) ;
            // bind linear memory
#ifdef DEBUG_MSG
            printf("WARNING:PFAC_bindCudaArray fails, bind PFAC_bindLinearMem\n");
#endif
            PFAC_status = PFAC_bindLinearMem( handle ) ;
            if ( PFAC_STATUS_SUCCESS != PFAC_status){
                PFAC_freeCUDALinearMem( handle );
                handle->isPatternsReady = false ;
                return PFAC_status ;
            }else{
                handle->textureMode = PFAC_TEXTURE_OFF ;
            }
        }else{
            handle->textureMode = PFAC_TEXTURE_ON ;
        }
    }else if ( PFAC_TEXTURE_ON == handle->textureMode ){
        PFAC_status = PFAC_bindCudaArray( handle ) ;
        if ( PFAC_STATUS_SUCCESS != PFAC_status ){
            PFAC_freeCUDAarray( handle ) ;
            handle->isPatternsReady = false ;
            return PFAC_status ;
        }
    }else{
        // PFAC_TEXTURE_OFF
        PFAC_status = PFAC_bindLinearMem( handle ) ;
        if ( PFAC_STATUS_SUCCESS != PFAC_status){
            PFAC_freeCUDALinearMem( handle );
            handle->isPatternsReady = false ;
            return PFAC_status ;
        }
    }
    return PFAC_STATUS_SUCCESS ;
}


PFAC_status_t  PFAC_readPatternFromFile( PFAC_handle_t handle, char *filename )
{
    if ( NULL == handle ){
        return PFAC_STATUS_INVALID_HANDLE ;
    }

    if ( NULL == filename ){
        return PFAC_STATUS_INVALID_PARAMETER ;
    }

    if ( handle->isPatternsReady ){
        // free previous pattern
        PFAC_freeResource( handle );
    }

    if ( FILENAME_LEN > strlen(filename) ){
        strcpy( handle->patternFile, filename ) ;
    }else{
        return PFAC_STATUS_INTERNAL_ERROR ;
    }

    PFAC_status_t PFAC_status = parsePatternFile( filename,
        &handle->rowPtr, &handle->valPtr, &handle->patternID_table, &handle->patternLen_table,
        &handle->max_state_num, &handle->pattern_num ) ;

    if ( PFAC_STATUS_SUCCESS != PFAC_status ){
        PFAC_freeResource( handle );
        return PFAC_status ;
    }

    int max_state_num = handle->max_state_num ;
    int pattern_num   = handle->pattern_num ;

    // compute maximum pattern length
    handle->maxPatternLen = 0 ;
    for(int i = 1 ; i <= pattern_num ; i++ ){
        if ( handle->maxPatternLen < (handle->patternLen_table)[i] ){
            handle->maxPatternLen = (handle->patternLen_table)[i];
        }
    }

    handle->initial_state  = handle->pattern_num + 1 ;
    handle->num_finalState = handle->pattern_num ;

    handle->PFAC_table = (int*) malloc( sizeof(int)*max_state_num*CHAR_SET) ;
    if ( NULL == handle->PFAC_table ){
        PFAC_freeResource( handle );
        return PFAC_STATUS_ALLOC_FAILED ;
    }

    // step 2: create PFAC table
    PFAC_status = create_PFACTable_reorder( (const char**)handle->rowPtr,
        (const int*)handle->patternLen_table, (const int*)handle->patternID_table,
        handle->max_state_num,
        handle->pattern_num, handle->initial_state,
        &handle->state_num,
        handle->PFAC_table ) ;

    if ( PFAC_STATUS_SUCCESS != PFAC_status ){
        PFAC_freeResource( handle );
        return PFAC_status ;
    }

    // step 3: copy data to device memory
    handle->isPatternsReady = true ;

    PFAC_status = PFAC_bindTable( handle ) ;
    if ( PFAC_STATUS_SUCCESS != PFAC_status ){
        PFAC_freeResource( handle );
        handle->isPatternsReady = false ;
        return PFAC_status ;
    }

    return PFAC_STATUS_SUCCESS ;
}


PFAC_status_t  PFAC_setPlatform( PFAC_handle_t handle, PFAC_platform_t platform)
{
    if ( NULL == handle ){
        return PFAC_STATUS_INVALID_HANDLE ;
    }

    if ( (PFAC_PLATFORM_CPU != platform) &&
         (PFAC_PLATFORM_CPU_OMP != platform) &&
         (PFAC_PLATFORM_GPU != platform)
       ){
        return PFAC_STATUS_INVALID_PARAMETER ;
    }

    handle->platform = (int) platform ;

    return PFAC_STATUS_SUCCESS ;
}


PFAC_status_t  PFAC_setTextureMode( PFAC_handle_t handle, PFAC_textureMode_t textureModeSel )
{
    if ( NULL == handle ){
        return PFAC_STATUS_INVALID_HANDLE ;
    }
    if ( (textureModeSel != PFAC_AUTOMATIC  ) &&
         (textureModeSel != PFAC_TEXTURE_ON ) &&
         (textureModeSel != PFAC_TEXTURE_OFF)
       ){
        return PFAC_STATUS_INVALID_PARAMETER ;
    }

    handle->textureMode = textureModeSel ;

    // allocate device memory if patterns is ready
    if ( handle->isPatternsReady ){
        PFAC_status_t PFAC_status = PFAC_bindTable( handle ) ;
        if ( PFAC_STATUS_SUCCESS != PFAC_status ){
            return PFAC_status ;
        }
    }
    return PFAC_STATUS_SUCCESS ;
}


PFAC_status_t  PFAC_setPerfMode( PFAC_handle_t handle, PFAC_perfMode_t perfModeSel ) 
{
    if ( NULL == handle ){
        return PFAC_STATUS_INVALID_HANDLE ;
    }	
    if ( (PFAC_TIME_DRIVEN  != perfModeSel) &&
    	   (PFAC_SPACE_DRIVEN != perfModeSel)
    	 ){
        return PFAC_STATUS_INVALID_PARAMETER ;
    }

    handle->perfMode = perfModeSel ; 
    
    return PFAC_STATUS_SUCCESS ;
}

/*
 *  platform is immaterial, do matching on GPU
 */
PFAC_status_t  PFAC_matchFromDevice( PFAC_handle_t handle, char *d_input_string, size_t input_size,
    int *d_matched_result )
{
    if ( NULL == handle ){
        return PFAC_STATUS_INVALID_HANDLE ;
    }
    if ( !(handle->isPatternsReady) ){
        return PFAC_STATUS_PATTERNS_NOT_READY ;
    }
    if ( NULL == d_input_string ){
        return PFAC_STATUS_INVALID_PARAMETER ;
    }
    if ( NULL == d_matched_result ){
        return PFAC_STATUS_INVALID_PARAMETER ;
    }

    if ( 0 == input_size ){ 
        return PFAC_STATUS_SUCCESS ;	
    }

    PFAC_status_t PFAC_status = (*(handle->kernel_ptr))( handle, d_input_string, input_size, d_matched_result );
    
    return PFAC_status ;
}


PFAC_status_t  PFAC_matchFromHost( PFAC_handle_t handle, char *h_input_string, size_t input_size,
    int *h_matched_result )
{
    if ( NULL == handle ){
        return PFAC_STATUS_INVALID_HANDLE ;
    }
    if ( !(handle->isPatternsReady) ){
        return PFAC_STATUS_PATTERNS_NOT_READY ;
    }
    if ( NULL == h_input_string ){
        return PFAC_STATUS_INVALID_PARAMETER ;
    }
    if ( NULL == h_matched_result ){
        return PFAC_STATUS_INVALID_PARAMETER ;
    }

    if ( 0 == input_size ){ 
        return PFAC_STATUS_SUCCESS ;	
    }

    if ( PFAC_PLATFORM_CPU == handle->platform ){
        
        return PFAC_CPU(h_input_string, input_size, handle->PFAC_table,
            handle->num_finalState, handle->initial_state, h_matched_result) ;
    }else if ( PFAC_PLATFORM_CPU_OMP == handle->platform){
        char *omp_var_str = getenv( "OMP_NUM_THREADS" ) ;
        if ( NULL == omp_var_str ){
#ifdef DEBUG_MSG
            printf("environment variable OMP_NUM_THREADS is missing, call non-openmp version \n");
#endif
            return PFAC_CPU(h_input_string, input_size, handle->PFAC_table,
                handle->num_finalState, handle->initial_state, h_matched_result) ;
        }else {
#ifdef DEBUG_MSG
            printf("environment variable OMP_NUM_THREADS = %s, call openmp version \n", omp_var_str );
#endif
            return PFAC_CPU_OMP(h_input_string, input_size, handle->PFAC_table,
                handle->num_finalState, handle->initial_state, h_matched_result) ;
        }
    }

    char *d_input_string  = NULL;
    int *d_matched_result = NULL;

    // n_hat = number of integers of input string
    int n_hat = (input_size + sizeof(int)-1)/sizeof(int) ;

    // allocate memory for input string and result
    // basic unit of d_input_string is integer
    cudaError_t cuda_status1 = cudaMalloc((void **) &d_input_string,        n_hat*sizeof(int) );
    cudaError_t cuda_status2 = cudaMalloc((void **) &d_matched_result, input_size*sizeof(int) );
    if ( (cudaSuccess != cuda_status1) || (cudaSuccess != cuda_status2) ){
    	  if ( NULL != d_input_string   ) { cudaFree(d_input_string); }
    	  if ( NULL != d_matched_result ) { cudaFree(d_matched_result); }
        return PFAC_STATUS_CUDA_ALLOC_FAILED;
    }

    // copy input string from host to device
    cuda_status1 = cudaMemcpy(d_input_string, h_input_string, input_size, cudaMemcpyHostToDevice);
    if ( cudaSuccess != cuda_status1 ){
    	  cudaFree(d_input_string); 
    	  cudaFree(d_matched_result);
        return PFAC_STATUS_INTERNAL_ERROR ;
    }

    PFAC_status_t PFAC_status = PFAC_matchFromDevice( handle, d_input_string, input_size,
        d_matched_result ) ;

    if ( PFAC_STATUS_SUCCESS != PFAC_status ){
        cudaFree(d_input_string);
        cudaFree(d_matched_result);
        return PFAC_status ;
    }

    // copy the result data from device to host
    cuda_status1 = cudaMemcpy(h_matched_result, d_matched_result, input_size*sizeof(int), cudaMemcpyDeviceToHost);
    if ( cudaSuccess != cuda_status1 ){
    	  cudaFree(d_input_string);
    	  cudaFree(d_matched_result);
        return PFAC_STATUS_INTERNAL_ERROR;
    }

    cudaFree(d_input_string);
    cudaFree(d_matched_result);

    return PFAC_STATUS_SUCCESS ;
}

const char* PFAC_getErrorString( PFAC_status_t status )
{
    static char PFAC_success_str[] = "PFAC_STATUS_SUCCESS: operation is successful" ;
    static char PFAC_alloc_failed_str[] = "PFAC_STATUS_ALLOC_FAILED: allocation fails on host memory" ;
    static char PFAC_cuda_alloc_failed_str[] = "PFAC_STATUS_CUDA_ALLOC_FAILED: allocation fails on device memory" ;
    static char PFAC_invalid_handle_str[] = "PFAC_STATUS_INVALID_HANDLE: handle is invalid (NULL)" ;
    static char PFAC_invalid_parameter_str[] = "PFAC_STATUS_INVALID_PARAMETER: parameter is invalid" ;
    static char PFAC_patterns_not_ready_str[] = "PFAC_STATUS_PATTERNS_NOT_READY: please call PFAC_readPatternFromFile() first" ;
    static char PFAC_file_open_error_str[] = "PFAC_STATUS_FILE_OPEN_ERROR: pattern file does not exist" ;
    static char PFAC_lib_not_exist_str[] = "PFAC_STATUS_LIB_NOT_EXIST: cannot find PFAC library, please check LD_LIBRARY_PATH" ;
    static char PFAC_arch_mismatch_str[] = "PFAC_STATUS_ARCH_MISMATCH: sm1.0 is not supported" ;
    static char PFAC_internal_error_str[] = "PFAC_STATUS_INTERNAL_ERROR: please report bugs" ;

    if ( PFAC_STATUS_SUCCESS == status ){
        return PFAC_success_str ;
    }
    if ( PFAC_STATUS_BASE > status ){
        return cudaGetErrorString( (cudaError_t) status ) ;
    }

    switch(status){
    case PFAC_STATUS_ALLOC_FAILED:
        return PFAC_alloc_failed_str ;
        break ;
    case PFAC_STATUS_CUDA_ALLOC_FAILED:
        return PFAC_cuda_alloc_failed_str;
        break ;
    case PFAC_STATUS_INVALID_HANDLE:
        return PFAC_invalid_handle_str ;
        break ;
    case PFAC_STATUS_INVALID_PARAMETER:
        return PFAC_invalid_parameter_str ;
        break ;
    case PFAC_STATUS_PATTERNS_NOT_READY:
        return PFAC_patterns_not_ready_str ;
        break ;
    case PFAC_STATUS_FILE_OPEN_ERROR:
        return PFAC_file_open_error_str ;
        break ;
    case PFAC_STATUS_LIB_NOT_EXIST:
        return PFAC_lib_not_exist_str ;
        break ;
    case PFAC_STATUS_ARCH_MISMATCH:
        return PFAC_arch_mismatch_str ;
        break ;
    default : // PFAC_STATUS_INTERNAL_ERROR:
        return PFAC_internal_error_str ;
    }
}


#define  PFAC_TABLE_MAP( i , j )   (i)*CHAR_SET + (j)

PFAC_status_t  PFAC_dumpTransitionTable( PFAC_handle_t handle, FILE *fp )
{
    if ( NULL == handle ){
        return PFAC_STATUS_INVALID_HANDLE ;
    }

    if ( NULL == fp ){
       fp = stdout ;
    }
    int state_num = handle->state_num ;
    int num_finalState = handle->num_finalState ;
    int initial_state = handle->initial_state ;
    int *patternLen_table = handle->patternLen_table ;
    int *PFAC_table = handle->PFAC_table ;
    int *patternID_table = handle->patternID_table ;

    fprintf(fp,"# Transition table: number of states = %d, initial state = %d\n", state_num, initial_state );
    fprintf(fp,"# (current state, input character) -> next state \n");
    for(int state = 0 ; state < state_num ; state++ ){
        for( int ch = 0 ; ch < CHAR_SET ; ch++ ){
            unsigned int nextState = PFAC_table[ PFAC_TABLE_MAP( state, ch ) ] ;
            if ( TRAP_STATE != nextState ){
                if ( (32 <= ch) && (126 >= ch) ){
                    fprintf(fp,"(%4d,%4c) -> %d \n", state, ch, nextState );
                }else{
                    fprintf(fp,"(%4d,%4.2x) -> %d \n", state, ch, nextState );
                }
            }
        }  // for each input character
    }  // for each state

    vector< char* > origin_patterns(num_finalState) ;
    for( int i = 0 ; i < num_finalState ; i++){
        char *pos = (handle->rowPtr)[i] ;
        int patternID = patternID_table[i] ;
        origin_patterns[patternID-1] = pos ;
    }

    fprintf(fp,"# Output table: number of final states = %d\n", num_finalState );
    fprintf(fp,"# [final state] [matched pattern ID] [pattern length] [pattern(string literal)] \n");

    for( int state = 1 ; state <= num_finalState ; state++){
        int patternID = state;
        int len = patternLen_table[patternID];
        if ( 0 != patternID ){
            fprintf(fp, "%5d %5d %5d    ", state, patternID, len );
            char *pos = origin_patterns[patternID-1] ;
            printStringEndNewLine( pos, fp );
            fprintf(fp, "\n" );
        }else{
            return PFAC_STATUS_INTERNAL_ERROR ;
        }
    }

    return PFAC_STATUS_SUCCESS ;
}



PFAC_status_t  PFAC_matchFromDeviceReduce( PFAC_handle_t handle, char *d_input_string, size_t input_size,
    int *d_matched_result, int *d_pos, int *h_num_matched )
{
    if ( NULL == handle ){
        return PFAC_STATUS_INVALID_HANDLE ;
    }
    if ( NULL == d_input_string ){
        return PFAC_STATUS_INVALID_PARAMETER ;
    }
    if ( NULL == h_num_matched ){
        return PFAC_STATUS_INVALID_PARAMETER ;
    }            
    if ( NULL == d_pos ){
        return PFAC_STATUS_INVALID_PARAMETER ;
    }
    if ( NULL == h_num_matched ){
    	  return PFAC_STATUS_INVALID_PARAMETER ;
    }
    
    if ( 0 == input_size ){ 
        return PFAC_STATUS_SUCCESS ;	
    }
    
    PFAC_status_t PFAC_status ;
    
    if ( PFAC_TIME_DRIVEN == handle->perfMode ){
      
        PFAC_status = (*(handle->reduce_kernel_ptr))( 
            handle, (int*)d_input_string, input_size,
            d_matched_result,  d_pos,  h_num_matched, NULL, NULL );
            
    }else if ( PFAC_SPACE_DRIVEN == handle->perfMode ){
    	
        PFAC_status = (*(handle->reduce_inplace_kernel_ptr))( 
            handle, (int*)d_input_string, input_size,
            d_matched_result,  d_pos,  h_num_matched, NULL, NULL );    	
    }else{
        return PFAC_STATUS_INTERNAL_ERROR ;	
    }    
  
    return PFAC_status;
}

PFAC_status_t  PFAC_matchFromHostReduce( PFAC_handle_t handle, char *h_input_string, size_t input_size,
    int *h_matched_result, int *h_pos, int *h_num_matched )
{
    PFAC_status_t PFAC_status ;
	  
    if ( NULL == handle ){
        return PFAC_STATUS_INVALID_HANDLE ;
    }
    if ( !handle->isPatternsReady ){
        return PFAC_STATUS_PATTERNS_NOT_READY ;
    }
    if ( NULL == h_input_string ){
        return PFAC_STATUS_INVALID_PARAMETER ;
    }
    if ( NULL == h_matched_result ){
        return PFAC_STATUS_INVALID_PARAMETER ;
    }
    if ( NULL == h_pos ){
        return PFAC_STATUS_INVALID_PARAMETER ;
    }
    if ( NULL == h_num_matched ){
        return PFAC_STATUS_INVALID_PARAMETER ;
    }

    if ( 0 == input_size ){ 
        return PFAC_STATUS_SUCCESS ;	
    }
    
    if ( PFAC_PLATFORM_GPU != handle->platform ){
        if ( PFAC_PLATFORM_CPU == handle->platform ){
        	  
            PFAC_status = PFAC_CPU(h_input_string, input_size, handle->PFAC_table,
                handle->num_finalState, handle->initial_state, h_matched_result) ;
        }else if ( PFAC_PLATFORM_CPU_OMP == handle->platform){
            char *omp_var_str = getenv( "OMP_NUM_THREADS" ) ;
            if ( NULL == omp_var_str ){
#ifdef DEBUG_MSG
                printf("environment variable OMP_NUM_THREADS is missing, call non-openmp version \n");
#endif
                PFAC_status = PFAC_CPU(h_input_string, input_size, handle->PFAC_table,
                handle->num_finalState, handle->initial_state, h_matched_result) ;
            }else {
#ifdef DEBUG_MSG
                printf("environment variable OMP_NUM_THREADS = %s, call openmp version \n", omp_var_str );
#endif
                PFAC_status = PFAC_CPU_OMP(h_input_string, input_size, handle->PFAC_table,
                handle->num_finalState, handle->initial_state, h_matched_result) ;
            } 
        }
        if ( PFAC_STATUS_SUCCESS != PFAC_status ) { return PFAC_status ; }
        // compresss h_matched_result and construct h_pos
        int zip_idx = 0 ;
        for (int i = 0 ; i < input_size ; i++){
            int matched = h_matched_result[i];
            if ( 0 < matched ){
                h_matched_result[zip_idx] = matched ;
                h_pos[zip_idx] = i ;
                zip_idx++ ;
            }	
        } 	 
        *h_num_matched = zip_idx ;
        return PFAC_STATUS_SUCCESS ;
    } // CPU version
   
    char *d_input_string = NULL;    // copy of h_input_string
    int  *d_matched_result = NULL;  // working space
    int  *d_pos = NULL;             // working space 

    // n_hat = number of integers of input string
    int n_hat = (input_size + sizeof(int)-1)/sizeof(int) ;

    // allocate memory for input string and result
    // basic unit of d_input_string is integer
    cudaError_t cuda_status1 = cudaMalloc((void **) &d_input_string, n_hat*sizeof(int) );
    cudaError_t cuda_status2 = cudaMalloc((void **) &d_matched_result, input_size*sizeof(int) );
    cudaError_t cuda_status3 = cudaMalloc((void **) &d_pos, input_size*sizeof(int) );
    if ( (cudaSuccess != cuda_status1) || (cudaSuccess != cuda_status2) || (cudaSuccess != cuda_status3) ){
    	  if ( NULL != d_input_string   ) { cudaFree(d_input_string); }
    	  if ( NULL != d_matched_result ) { cudaFree(d_matched_result); }
    	  if ( NULL != d_pos ) { cudaFree(d_pos); }
        return PFAC_STATUS_CUDA_ALLOC_FAILED;
    }

    // copy input string from host to device
    cuda_status1 = cudaMemcpy(d_input_string, h_input_string, input_size, cudaMemcpyHostToDevice);
    if ( cudaSuccess != cuda_status1 ){
        cudaFree(d_input_string);
        cudaFree(d_matched_result);
        cudaFree(d_pos); 
        return PFAC_STATUS_INTERNAL_ERROR ;
    }

    if ( PFAC_TIME_DRIVEN == handle->perfMode ){
        
        PFAC_status = (*(handle->reduce_kernel_ptr))( 
            handle, (int*)d_input_string, input_size,
            d_matched_result,  d_pos,  h_num_matched, h_matched_result, h_pos );
            
    }else if ( PFAC_SPACE_DRIVEN == handle->perfMode ){
    	    
        PFAC_status = (*(handle->reduce_inplace_kernel_ptr))( 
            handle, (int*)d_input_string, input_size,
            d_matched_result,  d_pos,  h_num_matched, h_matched_result, h_pos );    	
    }else{
        return PFAC_STATUS_INTERNAL_ERROR ;	
    }    
       
    if ( PFAC_STATUS_SUCCESS != PFAC_status ){
        cudaFree(d_input_string);
        cudaFree(d_matched_result);
        cudaFree(d_pos);
        return PFAC_status ;
    }

    cudaFree(d_input_string);
    cudaFree(d_matched_result);
    cudaFree(d_pos);

    return PFAC_STATUS_SUCCESS ;
}


