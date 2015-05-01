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
 *  we know load unbalance is important in pattern matching
 *
 *  so far, kernel has 100% occupancy, 6 thread blocks, 256 threads per block.
 *  resource usage: 14 regs. 2560 Bytes smem
 *
 *  Remark 1: if we want to relax load unbalance, then 8 blocks is an option,
 *            say 8 x 192 = 1536 threads
 *
 *  Remark 2: if we can relax occupancy to 1024 threads /SM, then each thread can
 *            use 32 registers and each block can use 6KB smem.
 *            we can use 256 threads per block and 16 paths per thread or
 *                       128 threads per block and 16 paths per thread
 *
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
 
PFAC_status_t  PFAC_kernel( PFAC_handle_t handle, char *d_input_string, size_t input_size,
    int *d_matched_result) ;
} 
#endif // __cplusplus


#define THREAD_BLOCK_EXP   (8)
#define EXTRA_SIZE_PER_TB  (128)
#define THREAD_BLOCK_SIZE  (1 << THREAD_BLOCK_EXP)

texture < int, 2, cudaReadModeElementType > tex_PFAC_table;

template <int BLOCKSIZE, int EXTRA_SIZE_TB, int TEXTURE_ON , int SMEM_ON >
__global__ void PFAC_kernel_device(int *d_PFAC_table, int *d_input_string, int input_size,
    int n_hat, int num_finalState, int initial_state, int num_blocks_minus1,
    int *d_match_result );


//------------------- main function -----------------------

__host__  PFAC_status_t  PFAC_kernel( PFAC_handle_t handle, char *d_input_string, size_t input_size,
    int *d_matched_result )
{
    cudaError_t cuda_status ;

    int num_finalState = handle->num_finalState;
    int initial_state  = handle->initial_state;
    
    bool smem_on = ((4*EXTRA_SIZE_PER_TB-1) >= handle->maxPatternLen) ;
    
    bool texture_on = (PFAC_TEXTURE_ON == handle->textureMode );

    if ( texture_on ){
        // set texture parameters
        tex_PFAC_table.addressMode[0] = cudaAddressModeClamp;
        tex_PFAC_table.addressMode[1] = cudaAddressModeClamp;
        tex_PFAC_table.filterMode     = cudaFilterModePoint;
        tex_PFAC_table.normalized     = 0;
        // bind the array to the texture
        cuda_status = cudaBindTextureToArray(tex_PFAC_table, handle->d_PFAC_table_array, handle->channelDesc);
        if ( cudaSuccess != cuda_status ){
            return PFAC_STATUS_CUDA_ALLOC_FAILED;
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
            PFAC_kernel_device<THREAD_BLOCK_SIZE, EXTRA_SIZE_PER_TB, 1, 1> <<< dimGrid, dimBlock >>>(
                handle->d_PFAC_table, (int*)d_input_string, input_size, 
                n_hat, num_finalState, initial_state, num_blocks-1, d_matched_result );   
        }else{
            PFAC_kernel_device<THREAD_BLOCK_SIZE, EXTRA_SIZE_PER_TB, 0, 1> <<< dimGrid, dimBlock >>>(
                handle->d_PFAC_table, (int*)d_input_string, input_size, 
                n_hat, num_finalState, initial_state, num_blocks-1, d_matched_result );           
        }
    }else{
        if ( texture_on ){
            PFAC_kernel_device<THREAD_BLOCK_SIZE, EXTRA_SIZE_PER_TB, 1, 0> <<< dimGrid, dimBlock >>>(
                handle->d_PFAC_table, (int*)d_input_string, input_size, 
                n_hat, num_finalState, initial_state, num_blocks-1, d_matched_result );   
        }else{
            PFAC_kernel_device<THREAD_BLOCK_SIZE, EXTRA_SIZE_PER_TB, 0, 0> <<< dimGrid, dimBlock >>>(
                handle->d_PFAC_table, (int*)d_input_string, input_size, 
                n_hat, num_finalState, initial_state, num_blocks-1, d_matched_result );   
        }    
    }

    cuda_status = cudaGetLastError() ;
    if ( cudaSuccess != cuda_status ){
        if ( texture_on ) { cudaUnbindTexture(tex_PFAC_table); }
        return PFAC_STATUS_INTERNAL_ERROR ;
    }

    if ( texture_on ){
        cudaUnbindTexture(tex_PFAC_table);
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
                state = tex2D(tex_PFAC_table, inputChar, state); \
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
                state = *(d_PFAC_table + state*CHAR_SET + inputChar); \
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
                state = tex2D(tex_PFAC_table, inputChar, state); \
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
                state = *(d_PFAC_table + state*CHAR_SET + inputChar); \
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
 *  each thread loads one integer (32-bit) to shared memory s_input[ BLOCKSIZE + EXTRA_SIZE_TB]
 *  also read extra size of this thread block
 *
 *  Example: thread block = 256 and extra size = 128, then each thread read twice
 *      if ( start < n_hat ){
 *          s_input[tid] = d_input_string[start];
 *      }
 *      start += BLOCKSIZE ;
 *      if ( (start < n_hat) && (tid < EXTRA_SIZE_TB) ){
 *           s_input[tid+BLOCKSIZE] = d_input_string[start];
 *      }
 *      __syncthreads();
 *
 *  in general, EXTRA_SIZE_TB may not be BLOCKSIZE/2
 *
 */
template <int BLOCKSIZE, int EXTRA_SIZE_TB, int TEXTURE_ON , int SMEM_ON >
__global__ void PFAC_kernel_device(int *d_PFAC_table, int *d_input_string, int input_size,
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
    if ( TEXTURE_ON ){
        phi_s02s1[ tid ] = tex2D(tex_PFAC_table, tid, initial_state ) ;
    }else{
        phi_s02s1[ tid ] = *(d_PFAC_table + initial_state*CHAR_SET + tid );
    }

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
