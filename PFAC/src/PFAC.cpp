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


#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <dlfcn.h>

#include <vector>

using namespace std ;

#include "../include/PFAC_P.h"

/* This is missing from very old Linux libc. */
#ifndef RTLD_NOW
#define RTLD_NOW 2
#endif

#include <limits.h>
#ifndef PATH_MAX
#define PATH_MAX 255
#endif

/* maximum width for a 1D texture reference bound to linear memory, independent of size of element*/
#define  MAXIMUM_WIDTH_1DTEX    (1 << 27)


//#define DEBUG_MSG


PFAC_status_t  PFAC_CPU_OMP(PFAC_handle_t handle, char *input_string, const int input_size, int *h_matched_result );
PFAC_status_t  PFAC_CPU( PFAC_handle_t handle, char *h_input_string, const int input_size, int *h_matched_result ) ;

void  PFAC_freeTable( PFAC_handle_t handle );
void  PFAC_freeResource( PFAC_handle_t handle );
PFAC_status_t  PFAC_bindTable( PFAC_handle_t handle );
PFAC_status_t  PFAC_create2DTable( PFAC_handle_t handle );
PFAC_status_t  PFAC_createHashTable( PFAC_handle_t handle );

/*
 *  Given k = pattern_number patterns in rowPtr[0:k-1] with lexicographic order and
 *  patternLen_table[1:k+1], patternID_table[0:k-1]
 *
 *  user specified a initial state "initial_state",
 *  construct
 *  (1) PFAC_table: DFA of PFAC with k final states labeled from 0:k-1
 *  (2) output_table[0:k-1]:  output_table[j] contains pattern number corresponding to
 *      final state j
 *
 *  WARNING: initial_state >= k, and size(output_table) >= k
 */
PFAC_status_t create_PFACTable_spaceDriven(const char** rowPtr, const int *patternLen_table, const int *patternID_table,
    const int max_state_num,
    const int pattern_num, const int initial_state, const int baseOfUsableStateID, 
    int *state_num_ptr,
    vector< vector<TableEle> > &PFAC_table );

/*
 *  CUDA 4.0 can supports one host thread to multiple GPU contexts.
 *  PFAC library still binds one PFAC handle to one GPU context.
 *
 *  consider followin example
 *  ----------------------------------------------------------------------
 *  cudaSetDevice(0);
 *  PFAC_create( PFAC_handle0 );
 *  PFAC_readPatternFromFile( PFAC_handle0, pattern_file )
 *  cudaSetDevice(1);
 *  PFAC_matchFromHost( PFAC_handle0, h_input_string, input_size, h_matched_result )
 *  ----------------------------------------------------------------------
 *
 *  Then PFAC library does not work because transition table of DFA is in GPU0 
 *  but d_input_string and d_matched_result are in GPU1.
 *  You can create two PFAC handles corresponding to different GPUs.
 *  ----------------------------------------------------------------------
 *  cudaSetDevice(0);
 *  PFAC_create( PFAC_handle0 );
 *  PFAC_readPatternFromFile( PFAC_handle0, pattern_file )
 *  cudaSetDevice(1);
 *  PFAC_create( PFAC_handle1 );
 *  PFAC_readPatternFromFile( PFAC_handle1, pattern_file ) 
 *  cudaSetDevice(0);
 *  PFAC_matchFromHost( PFAC_handle0, h_input_string, input_size, h_matched_result )
 *  cudaSetDevice(1);
 *  PFAC_matchFromHost( PFAC_handle1, h_input_string, input_size, h_matched_result ) 
 *  ---------------------------------------------------------------------- 
 *    
 */
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
    (*handle)->kernel_time_driven_ptr = (PFAC_kernel_protoType) dlsym (module, "PFAC_kernel_timeDriven_warpper");
    if ( NULL == (*handle)->kernel_time_driven_ptr ){
#ifdef DEBUG_MSG
        printf("Error: cannot load PFAC_kernel_timeDriven_warpper, error = %s\n", dlerror() );
#endif
        return PFAC_STATUS_INTERNAL_ERROR ;
    }
    
    (*handle)->kernel_space_driven_ptr = (PFAC_kernel_protoType) dlsym (module, "PFAC_kernel_spaceDriven_warpper");
    if ( NULL == (*handle)->kernel_space_driven_ptr ){
#ifdef DEBUG_MSG
        printf("Error: cannot load PFAC_kernel_spaceDriven_warpper, error = %s\n", dlerror() );
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

    // Find entry point of PFAC_reduce_inplace_kernel
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


void  PFAC_freeResource( PFAC_handle_t handle )
{
    // resource of patterns
    if ( NULL != handle->rowPtr ){
        free( handle->rowPtr );
        handle->rowPtr = NULL ;
    }
    
    if ( NULL != handle->valPtr ){
        free( handle->valPtr );
        handle->valPtr = NULL ;
    }

    if ( NULL != handle->patternLen_table ){
        free( handle->patternLen_table ) ;
        handle->patternLen_table = NULL ;
    }
    
    if ( NULL != handle->patternID_table ){
        free( handle->patternID_table );
        handle->patternID_table = NULL ;
    }
    
    if ( NULL != handle->table_compact ){
        delete 	handle->table_compact ;
        handle->table_compact = NULL ;
    }

    PFAC_freeTable( handle );
 
    handle->isPatternsReady = false ;
}

void  PFAC_freeTable( PFAC_handle_t handle )
{
    if ( NULL != handle->h_PFAC_table ){
        free( handle->h_PFAC_table ) ;
        handle->h_PFAC_table = NULL ;
    }

    if ( NULL != handle->h_hashRowPtr ){
        free( handle->h_hashRowPtr );
        handle->h_hashRowPtr = NULL ;	
    }
    
    if ( NULL != handle->h_hashValPtr ){
        free( handle->h_hashValPtr );
        handle->h_hashValPtr = NULL ;	
    }
    
    if ( NULL != handle->h_tableOfInitialState){
        free(handle->h_tableOfInitialState);
        handle->h_tableOfInitialState = NULL ; 
    }
    
    // free device resource
    if ( NULL != handle->d_PFAC_table ){
        cudaFree(handle->d_PFAC_table);
        handle->d_PFAC_table= NULL ;
    }
    
    if ( NULL != handle->d_hashRowPtr ){
        cudaFree( handle->d_hashRowPtr );
        handle->d_hashRowPtr = NULL ;
    }

    if ( NULL != handle->d_hashValPtr ){
        cudaFree( handle->d_hashValPtr );
        handle->d_hashValPtr = NULL ;	
    }
    
    if ( NULL != handle->d_tableOfInitialState ){
        cudaFree(handle->d_tableOfInitialState);
        handle->d_tableOfInitialState = NULL ;
    }	
}


/*
 *  suppose N = number of states
 *          C = number of character set = 256
 *
 *  TIME-DRIVEN:
 *     allocate a explicit 2-D table with N*C integers.
 *     host: 
 *          h_PFAC_table
 *     device: 
 *          d_PFAC_table
 *
 *  SPACE-DRIVEN:
 *     allocate a hash table (hashRowPtr, hashValPtr)
 *     host:
 *          h_hashRowPtr
 *          h_hashValPtr
 *          h_tableOfInitialState
 *     device:
 *          d_hashRowPtr
 *          d_hashValPtr
 *          d_tableOfInitialState         
 */
PFAC_status_t  PFAC_bindTable( PFAC_handle_t handle )
{
    if ( NULL == handle ){
        return PFAC_STATUS_INVALID_HANDLE ;
    }
    
    PFAC_status_t PFAC_status ;
    
    if (PFAC_TIME_DRIVEN == handle->perfMode){
        PFAC_status = PFAC_create2DTable(handle);
    }else{
        // PFAC_SPACE_DRIVEN
        PFAC_status = PFAC_createHashTable(handle);
    }

    if ( PFAC_STATUS_SUCCESS != PFAC_status ){
#ifdef DEBUG_MSG    	
        printf("Error: cannot create transistion table \n");	
#endif 
        return PFAC_status ;
    }

    return PFAC_STATUS_SUCCESS ;
}

 
PFAC_status_t  PFAC_create2DTable( PFAC_handle_t handle )
{
    if ( !(handle->isPatternsReady) ){
        return PFAC_STATUS_PATTERNS_NOT_READY ;
    }	
    
    /* perfMode is PFAC_TIME_DRIVEN, we don't need to allocate 2-D table again */
    if ( NULL != handle->d_PFAC_table ){
        return PFAC_STATUS_SUCCESS ;
    }	

    const int numOfStates = handle->numOfStates ;

    handle->numOfTableEntry = CHAR_SET*numOfStates ; 
    handle->sizeOfTableEntry = sizeof(int) ; 
    handle->sizeOfTableInBytes = (handle->numOfTableEntry) * (handle->sizeOfTableEntry) ; 

#define  PFAC_TABLE_MAP( i , j )   (i)*CHAR_SET + (j)    

    if ( NULL == handle->h_PFAC_table){    
        handle->h_PFAC_table = (int*) malloc( handle->sizeOfTableInBytes ) ;
        if ( NULL == handle->h_PFAC_table ){
            return PFAC_STATUS_ALLOC_FAILED ;
        }
    
        // initialize PFAC table to TRAP_STATE
        for (int i = 0; i < numOfStates ; i++) {
            for (int j = 0; j < CHAR_SET; j++) {
                (handle->h_PFAC_table)[ PFAC_TABLE_MAP( i , j ) ] = TRAP_STATE ;
            }
        }
        for(int i = 0 ; i < numOfStates ; i++ ){
            for(int j = 0 ; j < (int)(*(handle->table_compact))[i].size(); j++){
                TableEle ele = (*(handle->table_compact))[i][j];
                (handle->h_PFAC_table)[ PFAC_TABLE_MAP( i , ele.ch ) ] = ele.nextState;  	
            }
        }
    }

    cudaError_t cuda_status = cudaMalloc((void **) &handle->d_PFAC_table, handle->sizeOfTableInBytes );
    if ( cudaSuccess != cuda_status ){
        free(handle->h_PFAC_table);
        handle->h_PFAC_table = NULL ;
        return PFAC_STATUS_CUDA_ALLOC_FAILED ;
    }

    cuda_status = cudaMemcpy(handle->d_PFAC_table, handle->h_PFAC_table,
        handle->sizeOfTableInBytes, cudaMemcpyHostToDevice);
    if ( cudaSuccess != cuda_status ){
        free(handle->h_PFAC_table);
        handle->h_PFAC_table = NULL ;
        cudaFree(handle->d_PFAC_table);
        handle->d_PFAC_table = NULL;    	
        return PFAC_STATUS_INTERNAL_ERROR ;
    }
    
    return PFAC_STATUS_SUCCESS ;
}


/*
 *
 *  Element of h_hashRowPtr is int2, which is equivalent to
 *  typedef struct{
 *     int offset ;
 *     int k_sminus1 ;
 *  } 
 *
 *  we encode (k,s-1) by a 32-bit integer, k occupies most significant 16 bits and 
 *  (s-1) occupies Least significant 16 bits.
 * 
 *  Element of h_hashValPtr is int2, equivalent to
 *  tyepdef struct{
 *     int nextState ;
 *     int ch ;
 *  }  
 */
PFAC_status_t  PFAC_createHashTable( PFAC_handle_t handle )
{
    if ( !(handle->isPatternsReady) ){
        return PFAC_STATUS_PATTERNS_NOT_READY ;
    }
    
    /* perfMode is PFAC_SPACE_DRIVEN, we don't need to allocate hash table again */
    if ( NULL != handle->h_hashRowPtr ){
        if ( NULL == handle->h_hashValPtr ){ return PFAC_STATUS_INTERNAL_ERROR ; }
        if ( NULL == handle->h_tableOfInitialState ){ return PFAC_STATUS_INTERNAL_ERROR ; }
        return PFAC_STATUS_SUCCESS ;	
    }
    	
    const int numOfStates = handle->numOfStates ;
    
    handle->hash_m = 8 ;
    handle->hash_p = 257 ; // p = 2^m + 1
    
    handle->h_hashRowPtr = (int2*)malloc(sizeof(int2)*numOfStates);
    if ( NULL == handle->h_hashRowPtr ){
        return PFAC_STATUS_ALLOC_FAILED ; 	
    }
    //memset(handle->h_hashRowPtr, 0xFF, sizeof(int2)*numOfStates);

    vector< vector<TableEle> > *table_compact = handle->table_compact ;
    int totalEles = 0 ;
    for(int i = 0 ; i < numOfStates ; i++ ){
        const int Bi = (*table_compact)[i].size();
        int Si ;
        if ( 0 == Bi ){
            Si = 0 ;
        }else if ( 1 == Bi){
            Si = 1 ;	
        }else if ( 2 >= Bi ){
            Si = 4 ;	
        }else if ( 4 >= Bi ){
            Si = 16 ;	
        }else if ( 5 == Bi ){
            Si = 32 ;	
        }else if ( 8 >= Bi ){ // Si = {6, 7, 8}
            Si = 64 ;	
        }else if ( 11 >= Bi ){ // Si = {9, 10, 11}
            Si = 128 ;	
        }else if ( 255 >= Bi ){ // Si = {12, 13, ..., 255}
            Si = 256 ;	
        }else {
#ifdef DEBUG_MSG
            printf("Error: Bi (%d) is out-of-array bound\n", Bi);
#endif            
            free(handle->h_hashRowPtr);
            handle->h_hashRowPtr = NULL ;
            return PFAC_STATUS_INTERNAL_ERROR ;
        }
        int2 rowEle ;
        if ( 0 == Bi ){ // no valid transitions in state s{i}
            rowEle.x = -1 ;
            rowEle.y = -1 ;
        }else{
            rowEle.x = totalEles ; // offset of state s{i}
            rowEle.y = Si - 1; // information of k is filled out later
            totalEles += Si ;
        }
        (handle->h_hashRowPtr)[i] = rowEle ;
    }

    handle->numOfTableEntry = totalEles ;
    handle->sizeOfTableEntry = sizeof(int2) ; 
    handle->sizeOfTableInBytes = (handle->numOfTableEntry) * (handle->sizeOfTableEntry) ; 

    handle->h_hashValPtr = (int2*)malloc(handle->sizeOfTableInBytes) ;
    if ( NULL == handle->h_hashValPtr ){
        free(handle->h_hashRowPtr);
        handle->h_hashRowPtr = NULL ;
        return PFAC_STATUS_ALLOC_FAILED ;
    }
    memset( handle->h_hashValPtr, 0xFF, handle->sizeOfTableInBytes );
    
    for(int i = 0 ; i < numOfStates ; i++ ){
        const int Bi = (*table_compact)[i].size();
        const int Si = (handle->h_hashRowPtr)[i].y + 1 ; 
        const int offset = (handle->h_hashRowPtr)[i].x ; 
        if ( 0 == Bi ){ continue ;}

        vector<TableEle> elesOfRow = (*table_compact)[i];
        // Si = 1 or 256, then ki = 1 
        if ( (1 == Si) || (256 == Si) ){
            const int ki = 1 ;
            for(int j = 0 ; j < (int)elesOfRow.size(); j++){
                TableEle ele = elesOfRow[j];	
                int2 valEle ;
                valEle.x = ele.nextState ;
                valEle.y = ele.ch ;
                int pos = ( (ki * ele.ch) % handle->hash_p ) % Si ;
                (handle->h_hashValPtr)[offset + pos] = valEle ;
            }
            (handle->h_hashRowPtr)[i].y |= (ki << HASH_KEY_K_MASKBITS);	
            continue ;
        }
        // find a 0 < ki <= 256 such that (ki*c % p) % si are non-repeated.
        int ki = -1 ;
        vector<int> bin_flag ;
        for( int k = 1 ; k <= 256 ; k++){
            bool found = true; ;
            bin_flag.clear();
            for(int j = 0 ; j < Si; j++){
                bin_flag.push_back(-1);
            }	
            for(int j = 0 ; j < (int)elesOfRow.size(); j++){
                TableEle ele = elesOfRow[j];
                int pos = ((k*ele.ch) % handle->hash_p) % Si ;
                if ( 0 > bin_flag[pos] ){
                    bin_flag[pos] = 1 ;
                }else{
                    found = false ; // collision happens, try next k
                    break ;
                }
            }
            if (found){
                ki = k ;
                break ;	
            }
        }
        if ( 0 > ki ){
#ifdef DEBUG_MSG
            printf("Error: cannot find a k <= 256 to seperate elements \n");
#endif  
            free(handle->h_hashRowPtr);
            handle->h_hashRowPtr = NULL ;
            free(handle->h_hashValPtr);
            handle->h_hashValPtr = NULL ;
            return PFAC_STATUS_INTERNAL_ERROR ;
        }
        
        for(int j = 0 ; j < (int)elesOfRow.size(); j++){
            TableEle ele = elesOfRow[j];
            int2 valEle ;
            valEle.x = ele.nextState ;
            valEle.y = ele.ch ;
            int pos = ( (ki * ele.ch) % handle->hash_p ) % Si ;
            (handle->h_hashValPtr)[offset + pos] = valEle ;
        }
        (handle->h_hashRowPtr)[i].y |= (ki << HASH_KEY_K_MASKBITS);	
    }
    
    handle->h_tableOfInitialState = (int*)malloc(sizeof(int)*CHAR_SET) ;
    if ( NULL == handle->h_tableOfInitialState ){
        free(handle->h_hashRowPtr);
        handle->h_hashRowPtr = NULL ;
        free(handle->h_hashValPtr);
        handle->h_hashValPtr = NULL ;
        return PFAC_STATUS_ALLOC_FAILED ;    	
    }

    int2 rowEle = (handle->h_hashRowPtr)[ handle->initial_state ];
    int offset = rowEle.x ;
    if ( -1 == offset ){
        for (int j = 0; j < CHAR_SET; j++) {
           (handle->h_tableOfInitialState)[j] = TRAP_STATE ;
        }
    }else{
        int k_sminus1 = rowEle.y ;
    	  int sminus1 = k_sminus1 & HASH_KEY_S_MASK ; 
        int k = k_sminus1 >> HASH_KEY_K_MASKBITS ;
        for (int j = 0; j < CHAR_SET; j++) {
            int pos = ( ( k * j ) % handle->hash_p ) &  sminus1 ;
            int2 valEle = (handle->h_hashValPtr)[offset + pos];
            int nextState = valEle.x ;
            int ch = valEle.y ;
            if ( ch == j ){
                (handle->h_tableOfInitialState)[j] = nextState ;	
            }else{
                (handle->h_tableOfInitialState)[j] = TRAP_STATE ;	
            }
        }
    }

    // allocate device memory
    cudaError_t cuda_status1 = cudaMalloc((void **) &handle->d_hashRowPtr, sizeof(int2)*numOfStates );
    cudaError_t cuda_status2 = cudaMalloc((void **) &handle->d_hashValPtr, handle->sizeOfTableInBytes );
    cudaError_t cuda_status3 = cudaMalloc((void **) &handle->d_tableOfInitialState, sizeof(int)*CHAR_SET ); 
    if ( (cudaSuccess != cuda_status1) || (cudaSuccess != cuda_status2) || (cudaSuccess != cuda_status3) ){
        if (NULL != handle->d_hashRowPtr){
            cudaFree(handle->d_hashRowPtr) ; 
            handle->d_hashRowPtr = NULL ;	
        }
        if (NULL != handle->d_hashValPtr){
            cudaFree(handle->d_hashValPtr) ;
            handle->d_hashValPtr = NULL ;	
        }
        if (NULL != handle->d_tableOfInitialState){
            cudaFree(handle->d_tableOfInitialState);
            handle->d_tableOfInitialState = NULL ;	
        }
        free(handle->h_hashRowPtr);
        handle->h_hashRowPtr = NULL ;
        free(handle->h_hashValPtr);
        handle->h_hashValPtr = NULL ;
        free(handle->h_tableOfInitialState);
        handle->h_tableOfInitialState = NULL ;
        return PFAC_STATUS_CUDA_ALLOC_FAILED ;
    }

    cuda_status1 = cudaMemcpy(handle->d_hashRowPtr, handle->h_hashRowPtr,
        sizeof(int2)*numOfStates, cudaMemcpyHostToDevice);
    cuda_status2 = cudaMemcpy(handle->d_hashValPtr, handle->h_hashValPtr,
        handle->sizeOfTableInBytes , cudaMemcpyHostToDevice);
    cuda_status3 = cudaMemcpy(handle->d_tableOfInitialState, handle->h_tableOfInitialState,
        sizeof(int)*CHAR_SET , cudaMemcpyHostToDevice);    
    if ( (cudaSuccess != cuda_status1) || (cudaSuccess != cuda_status2) || (cudaSuccess != cuda_status3) ){
        cudaFree(handle->d_hashRowPtr) ; 
        handle->d_hashRowPtr = NULL ;	
        cudaFree(handle->d_hashValPtr) ;
        handle->d_hashValPtr = NULL ;	
        cudaFree(handle->d_tableOfInitialState);
        handle->d_tableOfInitialState = NULL ;	
        free(handle->h_hashRowPtr);
        handle->h_hashRowPtr = NULL ;
        free(handle->h_hashValPtr);
        handle->h_hashValPtr = NULL ;
        free(handle->h_tableOfInitialState);
        handle->h_tableOfInitialState = NULL ; 
        return PFAC_STATUS_INTERNAL_ERROR ;
    }

    return PFAC_STATUS_SUCCESS ;   	
}

/*
 *  if return status is not PFAC_STATUS_SUCCESS, then all reousrces are free.
 */
PFAC_status_t  PFAC_readPatternFromFile( PFAC_handle_t handle, char *filename )
{
    if ( NULL == handle ){
        return PFAC_STATUS_INVALID_HANDLE ;
    }

    if ( NULL == filename ){
        return PFAC_STATUS_INVALID_PARAMETER ;
    }

    if ( handle->isPatternsReady ){
        // free previous patterns, including transition tables in host and device memory
        PFAC_freeResource( handle );
    }

    if ( FILENAME_LEN > strlen(filename) ){
        strcpy( handle->patternFile, filename ) ;
    }else{
        return PFAC_STATUS_INTERNAL_ERROR ;
    }

    PFAC_status_t PFAC_status = parsePatternFile( filename,
        &handle->rowPtr, &handle->valPtr, &handle->patternID_table, &handle->patternLen_table,
        &handle->max_numOfStates, &handle->numOfPatterns ) ;

    if ( PFAC_STATUS_SUCCESS != PFAC_status ){
        PFAC_freeResource( handle );
        return PFAC_status ;
    }

    int pattern_num = handle->numOfPatterns ;
    
    // compute maximum pattern length
    handle->maxPatternLen = 0 ;
    for(int i = 1 ; i <= pattern_num ; i++ ){
        if ( handle->maxPatternLen < (handle->patternLen_table)[i] ){
            handle->maxPatternLen = (handle->patternLen_table)[i];
        }
    }

    handle->initial_state  = handle->numOfPatterns + 1 ;
    handle->numOfFinalStates = handle->numOfPatterns ;

    // step 2: create PFAC table
    handle->table_compact = new vector< vector<TableEle> > ;
    if ( NULL == handle->table_compact ){
        PFAC_freeResource( handle );
        return PFAC_STATUS_ALLOC_FAILED ;
    }
    
    int baseOfUsableStateID = handle->initial_state + 1 ; // assume initial_state = handle->numOfFinalStates + 1
    PFAC_status = create_PFACTable_spaceDriven((const char**)handle->rowPtr,
        (const int*)handle->patternLen_table, (const int*)handle->patternID_table,
        handle->max_numOfStates, handle->numOfPatterns, handle->initial_state, baseOfUsableStateID, 
        &handle->numOfStates, *(handle->table_compact) );

    if ( PFAC_STATUS_SUCCESS != PFAC_status ){
        PFAC_freeResource( handle );
        return PFAC_status ;
    }
    
    // compute numOfLeaves = number of leaf nodes
    // leaf node only appears in the final states
    handle->numOfLeaves = 0 ;
    for(int i = 1 ; i <= handle->numOfPatterns ; i++ ){
        // s0 is useless, so ignore s0
        if ( 0 == (*handle->table_compact)[i].size() ){
            handle->numOfLeaves ++ ;	
        }
    }
    
    // step 3: copy data to device memory
    handle->isPatternsReady = true ;

    PFAC_status = PFAC_bindTable( handle ) ;
    if ( PFAC_STATUS_SUCCESS != PFAC_status){
         PFAC_freeResource( handle );
         handle->isPatternsReady = false ;
         return PFAC_status ;
    }
        
    return PFAC_STATUS_SUCCESS ;
}

/*
 *  no need to change memory layout when platform is changed because
 *  we keep the same memory layout in host and device memory
 */
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

/*
 *  no need to change memory layout when platform is changed because
 *  we replace 2-D texture by 1-D texture, and bind to texture when 
 *  PFAC_matchFromHost/device[Reduce] is called. 
 */
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

    // reset transition table if patterns are ready and user change perfMode
    bool resetTable = false ;
    if ( handle->isPatternsReady ){
        if ( perfModeSel != handle->perfMode ){
#ifdef DEBUG_MSG 	
            printf("reset transition table \n");
#endif            
            resetTable = true ;
        }
    }
    
    handle->perfMode = perfModeSel ; 
    
    // we only free resources on transition table
    if ( resetTable ){
        PFAC_freeTable( handle );	
        PFAC_status_t PFAC_status = PFAC_bindTable( handle ) ;
        if ( PFAC_STATUS_SUCCESS != PFAC_status){
            PFAC_freeTable( handle );	
            return PFAC_status ;
        }
    }

    return PFAC_STATUS_SUCCESS ;
}

inline void correctTextureMode(PFAC_handle_t handle)
{		
    /* maximum width for a 1D texture reference is independent of type */
    if ( PFAC_AUTOMATIC == handle->textureMode ){
        if ( handle->numOfTableEntry < MAXIMUM_WIDTH_1DTEX ){ 
            handle->textureMode = PFAC_TEXTURE_ON ;
        }else{
            handle->textureMode = PFAC_TEXTURE_OFF ;
        }
    }
}

/*
 *  platform is immaterial, do matching on GPU
 *
 *  WARNING: d_input_string is allocated by caller, the size may not be multiple of 4.
 *  if shared mmeory version is chosen (for example, maximum pattern length is less than 512), then
 *  it is out-of-array bound logically, but it may not happen physically because basic unit of cudaMalloc() 
 *  is 256 bytes.  
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

    correctTextureMode(handle);
    
    PFAC_status_t PFAC_status ;
    
    if ( PFAC_TIME_DRIVEN == handle->perfMode ){
        
        PFAC_status = (*(handle->kernel_time_driven_ptr))( handle, d_input_string, input_size, d_matched_result );
    
    }else if ( PFAC_SPACE_DRIVEN == handle->perfMode ){
    	
        PFAC_status = (*(handle->kernel_space_driven_ptr))( handle, d_input_string, input_size, d_matched_result );
    
    }else{
        return PFAC_STATUS_INTERNAL_ERROR ;	
    }
  
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
        
        return PFAC_CPU(handle, h_input_string, input_size, h_matched_result);

    }else if ( PFAC_PLATFORM_CPU_OMP == handle->platform){
        char *omp_var_str = getenv( "OMP_NUM_THREADS" ) ;
        if ( NULL == omp_var_str ){
#ifdef DEBUG_MSG
            printf("environment variable OMP_NUM_THREADS is missing, call non-openmp version \n");
#endif
            return PFAC_CPU(handle, h_input_string, input_size, h_matched_result);

        }else {
#ifdef DEBUG_MSG
            printf("environment variable OMP_NUM_THREADS = %s, call openmp version \n", omp_var_str );
#endif
            return PFAC_CPU_OMP(handle, h_input_string, input_size, h_matched_result );
       
        }
    }

    // platform is GPU
    
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

    correctTextureMode(handle) ;
    
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
    PFAC_status_t PFAC_status = PFAC_STATUS_SUCCESS ;
	  
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
        	  
            PFAC_status = PFAC_CPU(handle, h_input_string, input_size, h_matched_result );

        }else if ( PFAC_PLATFORM_CPU_OMP == handle->platform){
            char *omp_var_str = getenv( "OMP_NUM_THREADS" ) ;
            if ( NULL == omp_var_str ){
#ifdef DEBUG_MSG
                printf("environment variable OMP_NUM_THREADS is missing, call non-openmp version \n");
#endif
                PFAC_status = PFAC_CPU(handle, h_input_string, input_size, h_matched_result ); 

            }else {
#ifdef DEBUG_MSG
                printf("environment variable OMP_NUM_THREADS = %s, call openmp version \n", omp_var_str );
#endif
                PFAC_status = PFAC_CPU_OMP(handle, h_input_string, input_size, h_matched_result );

            } 
        }
        if ( PFAC_STATUS_SUCCESS != PFAC_status ) { return PFAC_status ; }
        // compresss h_matched_result and construct h_pos
        int zip_idx = 0 ;
        for (int i = 0 ; i < (int)input_size ; i++){
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

    correctTextureMode(handle) ;
     
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
    int state_num = handle->numOfStates ;
    int num_finalState = handle->numOfFinalStates ;
    int initial_state = handle->initial_state ;
    int *patternLen_table = handle->patternLen_table ;
    int *patternID_table = handle->patternID_table ;

    fprintf(fp,"# Transition table: number of states = %d, initial state = %d\n", state_num, initial_state );
    fprintf(fp,"# (current state, input character) -> next state \n");

    for(int state = 0 ; state < state_num ; state++ ){
        for(int j = 0 ; j < (int)(*(handle->table_compact))[state].size(); j++){
            TableEle ele = (*(handle->table_compact))[state][j];
            int ch = ele.ch ;
            int nextState = ele.nextState;
            if ( TRAP_STATE != nextState ){
                if ( (32 <= ch) && (126 >= ch) ){
                    fprintf(fp,"(%4d,%4c) -> %d \n", state, ch, nextState );
                }else{
                    fprintf(fp,"(%4d,%4.2x) -> %d \n", state, ch, nextState );
                }
            }
        }	
    }

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
            //printStringEndNewLine( pos, fp );
            printString( pos, len, fp );
            fprintf(fp, "\n" );
        }else{
            return PFAC_STATUS_INTERNAL_ERROR ;
        }
    }

    return PFAC_STATUS_SUCCESS ;
}



PFAC_status_t  PFAC_memoryUsage( PFAC_handle_t handle )
{
    if ( NULL == handle ){
        return PFAC_STATUS_INVALID_HANDLE ;
    }    

    if ( PFAC_TIME_DRIVEN == handle->perfMode ){
        printf("time-driven: 2-D explicit matrix\n");
        printf("PFAC_table = %d x %d int = %ld bytes\n", 
            handle->numOfStates, CHAR_SET, handle->sizeOfTableInBytes );
        
        double total_bytes = handle->sizeOfTableInBytes ;
        printf("total amount = %7.2f MB\n", total_bytes/1024./1024. );    
            
    }else if ( PFAC_SPACE_DRIVEN == handle->perfMode ){	
        printf("space-driven: hash table\n");
        int size_rowPtr = handle->numOfStates * sizeof(int2);
        printf("hashRowPtr = %d int2 = %d bytes\n", handle->numOfStates, size_rowPtr );
        
        int size_valPtr = handle->sizeOfTableInBytes ;
        printf("hashValPtr = %ld int2 = %d bytes\n", handle->numOfTableEntry, size_valPtr );
        
        int size_tableOfInitialState = CHAR_SET * sizeof(int);
        printf("tableOfInitialState = %d int = %d bytes\n", CHAR_SET, size_tableOfInitialState );
        
        double total_bytes = (double)size_rowPtr + (double)size_valPtr + (double)size_tableOfInitialState ;
        
        printf("total amount = %7.2f MB\n", total_bytes/1024./1024. );
        
        double bytesOf2Dtable = ((double)handle->numOfStates)*((double)CHAR_SET*sizeof(int)) ;
        double ratio = total_bytes / bytesOf2Dtable ;
        printf("(hash table)/(2-D table) = %5.3f\n", ratio);
        
        double hash_eles_div_S = ((double)handle->numOfTableEntry)/((double)(handle->numOfStates));
        printf("|hashValPtr|/S = %5.2f\n", hash_eles_div_S );
        
    }else{
        return PFAC_STATUS_INTERNAL_ERROR ;	
    }
    
    printf("S = number of states (ignore s0) = %d \n", handle->numOfStates-1);
    printf("F = number of final states = %d \n", handle->numOfFinalStates);
    printf("L = number of leaf nodes = %d\n", handle->numOfLeaves);
    double S = handle->numOfStates - 1; // ignore s0
    double F = handle->numOfFinalStates ;
    double L = handle->numOfLeaves ;
    double bound = 1.0 + 71.0 * (F - 1.0)/(S - 1.0);
    printf("1 + 71*(F-1)/(S-1) = %5.2f\n", bound );

    bound = 1.0 + 71.0 * (L - 1.0)/(S - 1.0);
    printf("1 + 71*(L-1)/(S-1) = %5.2f\n", bound );

    bound = 2.0*( 2.0 + 71.0 * (F - 1.0)/(S - 1.0) );
    printf("hash uses int2: 2*(2 + 71*(F-1)/(S-1))= %5.2f\n", bound );

    return PFAC_STATUS_SUCCESS ;
}

