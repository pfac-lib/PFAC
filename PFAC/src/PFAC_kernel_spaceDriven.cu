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
 * almost the same as PFAC_kernel.cu except calling space-driven version
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cuda_runtime.h>
#include <assert.h>
#include <ctype.h>

#include "../include/PFAC_P.h"

//#define DEBUG_MSG

#ifdef __cplusplus
extern "C" {
 
PFAC_status_t  PFAC_kernel_spaceDriven_warpper( 
    PFAC_handle_t handle, char *d_input_string, size_t input_size, int *d_matched_result ) ;
    
} 
#endif // __cplusplus


#define THREAD_BLOCK_EXP   (8)
#define EXTRA_SIZE_PER_TB  (128)
#define THREAD_BLOCK_SIZE  (1 << THREAD_BLOCK_EXP)

#if THREAD_BLOCK_SIZE != 256 
    #error THREAD_BLOCK_SIZE != 256 
#endif

texture < int2, 1, cudaReadModeElementType > tex_hashRowPtr ;
texture < int2, 1, cudaReadModeElementType > tex_hashValPtr ;
texture < int , 1, cudaReadModeElementType > tex_tableOfInitialState ;

/*
 *  Hash table consists of rowPtr and valPtr, similar to CSR format.
 *  valPtr contains all transitions and rowPtr[i] contains offset pointing to valPtr.
 *
 *  Element of rowPtr is int2, which is equivalent to
 *  typedef struct{
 *     int offset ;
 *     int k_sminus1 ;
 *  } 
 *
 *  we encode (k,s-1) by a 32-bit integer, k occupies most significant 16 bits and 
 *  (s-1) occupies Least significant 16 bits.
 *
 *  sj is power of 2 and we need to do modulo s, in order to speedup, we use mask to do 
 *  modulo, say x mod s = x & (s-1)
 *
 *  Element of valPtr is int2, equivalent to
 *  tyepdef struct{
 *     int nextState ;
 *     int ch ;
 *  } 
 *
 */
static __inline__  __device__ int tex_lookup(int state, int inputChar, const int hash_m, const int hash_p )
{ 
    int2 rowEle = tex1Dfetch(tex_hashRowPtr, state); // hashRowPtr[state]
    int offset  = rowEle.x ;
    int nextState = TRAP_STATE ;
    if ( 0 <= offset ){ 
       int k_sminus1 = rowEle.y ;
       int sminus1 = k_sminus1 & HASH_KEY_S_MASK ;
       int k = k_sminus1 >> HASH_KEY_K_MASKBITS ; 
       int x = k * inputChar ;
       int alpha_hat = x >> hash_m ;
       int beta = x - hash_p * alpha_hat ;
       if ( 0 > beta ){ // alpha_hat = (x/p)+1
            beta += hash_p ;
       }      
       int pos = beta & sminus1 ;
       int2 valEle = tex1Dfetch(tex_hashValPtr, offset + pos); // hashValPtr[offset + pos];
       if ( inputChar == valEle.y ){
            nextState = valEle.x ;
       }
    }
    return nextState ;
}


static __inline__  __device__ int notex_lookup(int2* d_hashRowPtr, int2 *d_hashValPtr,
    int state, int inputChar, const int hash_m, const int hash_p )
{ 
    int2 rowEle = d_hashRowPtr[state];
    int offset  = rowEle.x ;
    int nextState = TRAP_STATE ;
    if ( 0 <= offset ){ 
       int k_sminus1 = rowEle.y ;
       int sminus1 = k_sminus1 & HASH_KEY_S_MASK ;
       int k = k_sminus1 >> HASH_KEY_K_MASKBITS ; 
       int x = k * inputChar ;
       int alpha_hat = x >> hash_m ;
       int beta = x - hash_p * alpha_hat ;
       if ( 0 > beta ){ // alpha_hat = (x/p)+1
            beta += hash_p ;
       }      
       int pos = beta & sminus1 ;
       int2 valEle = d_hashValPtr[offset + pos];
       if ( inputChar == valEle.y ){
            nextState = valEle.x ;
       }
    }
    return nextState ;
}

static __inline__  __device__ int tex_loadTableOfInitialState(int ch)
{
    return tex1Dfetch(tex_tableOfInitialState, ch); 
}


template <int BLOCKSIZE, int EXTRA_SIZE_TB, int TEXTURE_ON , int SMEM_ON >
__global__ void PFAC_kernel_spaceDriven(int2 *d_hashRowPtr, int2 *d_hashValPtr, int *d_tableOfInitialState,
    const int hash_m, const int hash_p,
    int *d_input_string, int input_size,
    int n_hat, int num_finalState, int initial_state, int num_blocks_minus1,
    int *d_match_result );
    
//------------------- main function -----------------------

__host__  PFAC_status_t  PFAC_kernel_spaceDriven_warpper( 
    PFAC_handle_t handle, char *d_input_string, size_t input_size,
    int *d_matched_result )
{

#ifdef DEBUG_MSG
    printf("call PFAC_kernel_spaceDriven_warpper \n");
#endif

    cudaError_t cuda_status ;

    int num_finalState = handle->numOfFinalStates;
    int initial_state  = handle->initial_state;
   
   /*
    *  suppose a trhead-block needs to handle a block of 1024 characters, then
    *  thread-block would read EXTRA_SIZE_PER_TB integers to shared memory.
    *  if maxPatternLen >= sizeof(int)*EXTRA_SIZE_PER_TB, then 
    *  there may be out-of-array bound of substring starting from 1023-th character.
    */ 
    bool smem_on = ((4*EXTRA_SIZE_PER_TB-1) >= handle->maxPatternLen) ;
    
    bool texture_on = (PFAC_TEXTURE_ON == handle->textureMode );

    size_t offset ;
    
    /* always bind texture to tex_tableOfInitialState */
    // (3) bind texture to tex_tableOfInitialState
    textureReference *texRefTableOfInitialState ;
    cudaGetTextureReference( (const struct textureReference**)&texRefTableOfInitialState, "tex_tableOfInitialState" );

    cudaChannelFormatDesc channelDesc_tableOfInitialState = cudaCreateChannelDesc<int>();
    // set texture parameters
    tex_tableOfInitialState.addressMode[0] = cudaAddressModeClamp;
    tex_tableOfInitialState.addressMode[1] = cudaAddressModeClamp;
    tex_tableOfInitialState.filterMode     = cudaFilterModePoint;
    tex_tableOfInitialState.normalized     = 0;    
        
    cuda_status = cudaBindTexture( &offset, (const struct textureReference*) texRefTableOfInitialState,
        (const void*) handle->d_tableOfInitialState, (const struct cudaChannelFormatDesc*) &channelDesc_tableOfInitialState, 
        sizeof(int)*CHAR_SET ) ;

    if ( cudaSuccess != cuda_status ){
#ifdef DEBUG_MSG
        printf("Error: cannot bind texture to tableOfInitialState, %s\n", cudaGetErrorString(status) );
#endif            
        return PFAC_STATUS_CUDA_ALLOC_FAILED ;
    }
        
    if ( 0 != offset ){
#ifdef DEBUG_MSG
        printf("Error: offset is not zero\n");
#endif
        return PFAC_STATUS_INTERNAL_ERROR ;
    }
        
    if ( texture_on ){
 
        // (1) bind texture to tex_hashRowPtr
        textureReference *texRefHashRowPtr ;
        cudaGetTextureReference( (const struct textureReference**)&texRefHashRowPtr, "tex_hashRowPtr" );

        cudaChannelFormatDesc channelDesc_hashRowPtr = cudaCreateChannelDesc<int2>();
    
        // set texture parameters
        tex_hashRowPtr.addressMode[0] = cudaAddressModeClamp;
        tex_hashRowPtr.addressMode[1] = cudaAddressModeClamp;
        tex_hashRowPtr.filterMode     = cudaFilterModePoint;
        tex_hashRowPtr.normalized     = 0;
        
        cuda_status = cudaBindTexture( &offset, (const struct textureReference*) texRefHashRowPtr,
            (const void*) handle->d_hashRowPtr, (const struct cudaChannelFormatDesc*) &channelDesc_hashRowPtr, 
            sizeof(int2)*(handle->numOfStates) ) ;
        
        if ( cudaSuccess != cuda_status ){
#ifdef DEBUG_MSG
            printf("Error: cannot bind texture to hashRowPtr, %s\n", cudaGetErrorString(status) );
#endif            
            return PFAC_STATUS_CUDA_ALLOC_FAILED ;
        }
        
        if ( 0 != offset ){
#ifdef DEBUG_MSG
            printf("Error: offset is not zero\n");
#endif
            return PFAC_STATUS_INTERNAL_ERROR ;
        }
        
        // (2) bind texture to tex_hashValPtr
        textureReference *texRefHashValPtr ;
        cudaGetTextureReference( (const struct textureReference**)&texRefHashValPtr, "tex_hashValPtr" );

        cudaChannelFormatDesc channelDesc_hashValPtr = cudaCreateChannelDesc<int2>();        
        // set texture parameters
        tex_hashValPtr.addressMode[0] = cudaAddressModeClamp;
        tex_hashValPtr.addressMode[1] = cudaAddressModeClamp;
        tex_hashValPtr.filterMode     = cudaFilterModePoint;
        tex_hashValPtr.normalized     = 0;

        cuda_status = cudaBindTexture( &offset, (const struct textureReference*) texRefHashValPtr,
            (const void*) handle->d_hashValPtr, (const struct cudaChannelFormatDesc*) &channelDesc_hashValPtr, 
            handle->sizeOfTableInBytes ) ;
        if ( cudaSuccess != cuda_status ){
#ifdef DEBUG_MSG
            printf("Error: cannot bind texture to hashValPtr, %s\n", cudaGetErrorString(status) );
#endif            
            return PFAC_STATUS_CUDA_ALLOC_FAILED ;
        }
        
        if ( 0 != offset ){
#ifdef DEBUG_MSG
            printf("Error: offset is not zero\n");
#endif
            return PFAC_STATUS_INTERNAL_ERROR ;
        }                                            
    }

    // n_hat = number of integers of input string
    int n_hat = (input_size + sizeof(int)-1)/sizeof(int) ;

    // num_blocks = # of thread blocks to cover input stream
    int num_blocks = (n_hat + THREAD_BLOCK_SIZE-1)/THREAD_BLOCK_SIZE ;

    dim3  dimBlock( THREAD_BLOCK_SIZE, 1 ) ;
    dim3  dimGrid ;

    /* 
     *  hardware limitatin of 2-D grid is (65535, 65535), 
     *  1-D grid is not enough to cover large input stream.
     *  For example, input_size = 1G (input stream has 1Gbyte), then 
     *  num_blocks = # of thread blocks = 1G / 1024 = 1M > 65535
     *
     *  However when using 2-D grid, then number of invoke blocks = dimGrid.x * dimGrid.y 
     *  which is bigger than > num_blocks
     *
     *  we need to check this boundary condition inside kernel because
     *  size of d_nnz_per_block is num_blocks
     *
     *  trick: decompose num_blocks = p * 2^15 + q
     */
     
    int p = num_blocks >> 15 ;
    dimGrid.x = num_blocks ;
    if ( p ){
        dimGrid.x = 1<<15 ;
        dimGrid.y = p+1 ;
    }
   
    if (smem_on) {
        if ( texture_on ){
            PFAC_kernel_spaceDriven<THREAD_BLOCK_SIZE, EXTRA_SIZE_PER_TB, 1, 1> <<< dimGrid, dimBlock >>>(
                handle->d_hashRowPtr, handle->d_hashValPtr, handle->d_tableOfInitialState, 
                handle->hash_m, handle->hash_p,
                (int*)d_input_string, input_size, 
                n_hat, num_finalState, initial_state, num_blocks-1, d_matched_result );   
        }else{
            PFAC_kernel_spaceDriven<THREAD_BLOCK_SIZE, EXTRA_SIZE_PER_TB, 0, 1> <<< dimGrid, dimBlock >>>(
                handle->d_hashRowPtr, handle->d_hashValPtr, handle->d_tableOfInitialState,
                handle->hash_m, handle->hash_p,
                (int*)d_input_string, input_size, 
                n_hat, num_finalState, initial_state, num_blocks-1, d_matched_result );           
        }
    }else{
        if ( texture_on ){
            PFAC_kernel_spaceDriven<THREAD_BLOCK_SIZE, EXTRA_SIZE_PER_TB, 1, 0> <<< dimGrid, dimBlock >>>(
                handle->d_hashRowPtr, handle->d_hashValPtr, handle->d_tableOfInitialState,
                handle->hash_m, handle->hash_p,
                (int*)d_input_string, input_size, 
                n_hat, num_finalState, initial_state, num_blocks-1, d_matched_result );   
        }else{
            PFAC_kernel_spaceDriven<THREAD_BLOCK_SIZE, EXTRA_SIZE_PER_TB, 0, 0> <<< dimGrid, dimBlock >>>(
                handle->d_hashRowPtr, handle->d_hashValPtr, handle->d_tableOfInitialState,
                handle->hash_m, handle->hash_p,
                (int*)d_input_string, input_size, 
                n_hat, num_finalState, initial_state, num_blocks-1, d_matched_result );   
        }    
    }

    cuda_status = cudaGetLastError() ;
    if ( cudaSuccess != cuda_status ){
        cudaUnbindTexture(tex_tableOfInitialState);
        if ( texture_on ) { 
            cudaUnbindTexture(tex_hashRowPtr);
            cudaUnbindTexture(tex_hashValPtr);
        }
        return PFAC_STATUS_INTERNAL_ERROR ;
    }

    cudaUnbindTexture(tex_tableOfInitialState);
    if ( texture_on ){
        cudaUnbindTexture(tex_hashRowPtr);
        cudaUnbindTexture(tex_hashValPtr);
    } 

    return PFAC_STATUS_SUCCESS ;
}


/*
 *  (1) transition table of initial state is in the shared memory phi_s02s1
 *      we don't need to look up table in texture tex_PFAC_table
 *
 *  (2) final states are reordered as 1, 2, ..., k
 *      so state number <= k (number of final states) means final state
 */
#define  SUBSEG_MATCH( j, match ) \
    pos = tid + j * BLOCKSIZE ;\
    if ( pos < bdy ){ \
        inputChar = s_char[pos]; \
        state = phi_s02s1[ inputChar ]; \
        if ( TRAP_STATE != state ){ \
            if ( state <= num_finalState ){ \
                match = state;\
            } \
            pos = pos + 1; \
            while ( pos < bdy ) { \
                inputChar = s_char[pos]; \
                state = tex_lookup(state, inputChar, hash_m, hash_p ); \
                if ( TRAP_STATE == state ){ break ;} \
                if ( state <= num_finalState ){ \
                    match = state;\
                }\
                pos = pos + 1;\
            }\
        }\
    }
// end macro

#define  SUBSEG_MATCH_NOTEX( j, match ) \
    pos = tid + j * BLOCKSIZE ;\
    if ( pos < bdy ){ \
        inputChar = s_char[pos]; \
        state = phi_s02s1[ inputChar ]; \
        if ( TRAP_STATE != state ){ \
            if ( state <= num_finalState ){ \
                match = state;\
            } \
            pos = pos + 1; \
            while ( pos < bdy ) { \
                inputChar = s_char[pos]; \
                state = notex_lookup(d_hashRowPtr, d_hashValPtr, state, inputChar, hash_m, hash_p ) ; \
                if ( TRAP_STATE == state ){ break ;} \
                if ( state <= num_finalState ){ \
                    match = state;\
                }\
                pos = pos + 1;\
            }\
        }\
    }
// end macro

#define  SUBSEG_MATCH_NOSMEM( j, match ) \
    pos = ( gbid * BLOCKSIZE * 4 ) + tid + j * BLOCKSIZE ;\
    if ( pos < input_size ){ \
        inputChar = (unsigned char) char_d_input_string[pos]; \
        state = phi_s02s1[ inputChar ]; \
        if ( TRAP_STATE != state ){ \
            if ( state <= num_finalState ){ \
                match = state;\
            } \
            pos = pos + 1; \
            while ( pos < input_size ) { \
                inputChar = (unsigned char) char_d_input_string[pos]; \
                state = tex_lookup(state, inputChar, hash_m, hash_p ); \
                if ( TRAP_STATE == state ){ break ;} \
                if ( state <= num_finalState ){ \
                    match = state;\
                }\
                pos = pos + 1;\
            }\
        }\
    }
// end macro

#define  SUBSEG_MATCH_NOSMEM_NOTEX( j, match ) \
    pos = ( gbid * BLOCKSIZE * 4 ) + tid + j * BLOCKSIZE ;\
    if ( pos < input_size ){ \
        inputChar = (unsigned char) char_d_input_string[pos]; \
        state = phi_s02s1[ inputChar ]; \
        if ( TRAP_STATE != state ){ \
            if ( state <= num_finalState ){ \
                match = state;\
            } \
            pos = pos + 1; \
            while ( pos < input_size ) { \
                inputChar = (unsigned char) char_d_input_string[pos]; \
                state = notex_lookup(d_hashRowPtr, d_hashValPtr, state, inputChar, hash_m, hash_p ) ; \
                if ( TRAP_STATE == state ){ break ;} \
                if ( state <= num_finalState ){ \
                    match = state;\
                }\
                pos = pos + 1;\
            }\
        }\
    }
// end macro


#define MANUAL_EXPAND_2( X )   { X ; X ; }
#define MANUAL_EXPAND_4( X )   { MANUAL_EXPAND_2( MANUAL_EXPAND_2( X ) )  }

/*
 *  occupancy
 *
 *  sm_20:
 *     Used 17 registers, 1024+0 bytes smem, 104 bytes cmem[0] => 1536 threads per SM 
 *      
 *  sm_13:
 *     Used 17 registers, 1096+16 bytes smem, 8 bytes cmem[1]  => 768 threads per SM
 *
 *
 */
template <int BLOCKSIZE, int EXTRA_SIZE_TB, int TEXTURE_ON , int SMEM_ON >
__global__ void PFAC_kernel_spaceDriven(int2 *d_hashRowPtr, int2 *d_hashValPtr, int *d_tableOfInitialState,
    const int hash_m, const int hash_p,
    int *d_input_string, int input_size,
    int n_hat, int num_finalState, int initial_state, int num_blocks_minus1,
    int *d_match_result )
{
    int tid  = threadIdx.x ;
    int gbid = blockIdx.y * gridDim.x + blockIdx.x ;
    int start = gbid * BLOCKSIZE + tid ;
    int pos;     
    int state;
    int inputChar;
    int match[4] = {0,0,0,0};
    __shared__ int s_input[ BLOCKSIZE + EXTRA_SIZE_TB];
    __shared__ int phi_s02s1[ 256 ] ;
    volatile unsigned char *s_char;
    char * char_d_input_string ;

    if ( gbid > num_blocks_minus1 ){
        return ; // whole block is outside input stream
    }

    // load transition table of initial state to shared memory
    // we always bind table of initial state to texture
    
    phi_s02s1[ tid ] = tex_loadTableOfInitialState(tid); // tex_lookup(initial_state, tid); 
    
    if ( SMEM_ON ){      
    
        s_char = (unsigned char *)s_input;

        // read global data to shared memory
        if ( start < n_hat ){
            s_input[tid] = d_input_string[start];
        }
        start += BLOCKSIZE ;
        if ( (start < n_hat) && (tid < EXTRA_SIZE_TB) ){
            s_input[tid+BLOCKSIZE] = d_input_string[start];
        }
    }// if SMEM_ON
    
    __syncthreads(); // important because of phi_s02s1
    
    // bdy = number of legal characters starting at gbid*BLOCKSIZE*4
    int bdy = input_size - ( gbid * BLOCKSIZE * 4 );

    if ( SMEM_ON ){  
        if ( TEXTURE_ON ){
            int j = 0 ;
            MANUAL_EXPAND_4( SUBSEG_MATCH(j, match[j]) ; j++ ; ) 
        }else{
            int j = 0 ;
            MANUAL_EXPAND_4( SUBSEG_MATCH_NOTEX(j, match[j]) ; j++ ;)
        }
    }else{
        char_d_input_string = (char*)d_input_string ; // used only when SMEM_ON = 0
        if ( TEXTURE_ON ){
            int j = 0 ;
            MANUAL_EXPAND_4( SUBSEG_MATCH_NOSMEM(j, match[j]) ; j++ ; ) 
        }else{
            int j = 0 ;
            MANUAL_EXPAND_4( SUBSEG_MATCH_NOSMEM_NOTEX(j, match[j]) ; j++ ;)
        }
    }

    // write 4 results  match[0:3] to global d_match_result[0:input_size)
    // one thread block processes (BLOCKSIZE * 4) substrings
    start = gbid * (BLOCKSIZE * 4) + tid ;

    if ( gbid < num_blocks_minus1 ){
        #pragma unroll
        for (int j = 0 ; j < 4 ; j++ ){
            d_match_result[start] = match[j];
            start += BLOCKSIZE ;
        }
    }else{
        int j = 0 ;
        MANUAL_EXPAND_4( if (start>=input_size) return ; d_match_result[start] = match[j]; \
            j++ ; start += BLOCKSIZE ; )
    }
}

