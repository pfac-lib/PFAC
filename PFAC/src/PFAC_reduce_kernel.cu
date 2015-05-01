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
 *  F = number of final states, we label final states from s{1}, s{2}, ... s{F}
 *  and initial state is s{F+1}. s{0} is of no use.
 *
 *  if maximum pattern length is less than 512, then we will load transition function
 *  of initial state to shared memory, so we requires BLOCK_SIZE * k = 256 such that
 *  each thread load sevral one transition pairs into shared memory  
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cuda_runtime.h>
#include <assert.h>
#include <ctype.h>

#include "thrust/device_vector.h"
#include "thrust/scan.h"

#include "../include/PFAC_P.h"

//#define DEBUG_MSG


#ifdef __cplusplus
extern "C" {

    PFAC_status_t PFAC_reduce_kernel( PFAC_handle_t handle, int *d_input_string, int input_size,
        int *d_match_result, int *d_pos, int *h_num_matched, int *h_match_result, int *h_pos ); 
        
}
#endif // __cplusplus


#define  BLOCK_EXP             (7)
#define  BLOCK_SIZE            (1 << BLOCK_EXP)
#define  EXTRA_SIZE_PER_TB     (128)
#define  NUM_INTS_PER_THREAD   (2)

#define  BLOCK_SIZE_DIV_256    (2)

#define  NUM_WARPS_PER_BLOCK   (4)

#if  256 != (BLOCK_SIZE_DIV_256 * BLOCK_SIZE) 
    #error 256 != BLOCK_SIZE_DIV_256 * BLOCK_SIZE 
#endif 

#if  BLOCK_SIZE != 32 * NUM_WARPS_PER_BLOCK
    #error BLOCK_SIZE != 32 * NUM_WARPS_PER_BLOCK
#endif

texture < int, 1, cudaReadModeElementType > tex_PFAC_table_reduce;

static __inline__  __device__ int tex_lookup(int state, int inputChar)
{ 
    return  tex1Dfetch(tex_PFAC_table_reduce, state*CHAR_SET + inputChar);
}

template <int TEXTURE_ON , int SMEM_ON >
__global__ void PFAC_reduce_kernel_device(int *d_PFAC_table, int *d_input_string, 
    int input_size, int n_hat, int num_finalState, int initial_state, int num_blocks_minus1,
    int *d_pos, int *d_match_result, int *d_nnz_per_block ) ;


__host__  PFAC_status_t PFAC_reduce_kernel_stage1( PFAC_handle_t handle, 
    int *d_input_string, int input_size,
    int n_hat, int num_blocks, dim3 dimBlock, dim3 dimGrid,
    int *d_match_result, int *d_pos, int *d_nnz_per_block, int *h_num_matched );

__global__ void zip_kernel(int *d_pos, int *d_match_result, int *d_nnz_per_block,
     int num_blocks_minus1, int elements_per_block,
     int *d_pos_zip, int *d_match_result_zip);
     
// ---------------------------- main ----------------------    

/*
 *
 *  Input -
 *      handle  
 *          pointer to a legal PFAC context
 *      d_input_string
 *          input stream in device memory, its size is "input_size" bytes 
 *      input_size
 *          size of input stream
 *
 *  Output -
 *      h_num_matched
 *          pointer to a host memory, it denotes number of matched patterns in the input stream
 *          for example, if device mode is set, and h_num_matched = 5, then 
 *          d_pos[0:4] contains startig position of each matched pattern in the input stream
 *          d_match_result[0:4] contains pattern ID of matched pattern
 *
 *      NOTE: if h_num_matched = 0, then d_pos and d_match_result are not touched, 
 *          their value is at random.
 *          also at this time, (d_pos_zip, d_match_result_zip) is not allocated, so 
 *          space is efficient.
 * 
 *      support 2 mode:
 * 
 *      Device mode: 
 *      (d_pos, d_match_result) pair is device memory
 *      (h_pos, h_match_result) is (NULL,NULL)
 *
 *      1) (d_pos, d_match_result) is used as working space, store local compressed (match,pos)
 *      2) zip (d_pos, d_match_result) to working space (d_pos_zip, d_match_result_zip)
 *      3) copy (d_pos_zip, d_match_result_zip) to (d_pos, d_match_result) via DeviceToDevice
 *
 *      Host mode:
 *      (d_pos, d_match_result) pair is working space
 *      (h_pos, h_match_result) is not (NULL,NULL) 
 *      
 *      1) (d_pos, d_match_result) is used as working space, store local compressed (match,pos)
 *      2) zip (d_pos, d_match_result) to working space (d_pos_zip, d_match_result_zip)
 *      3) copy (d_pos_zip, d_match_result_zip) to (h_pos, h_match_result) via DeviceToHost
 * 
 *  We can combine two modes in a simple way, 
 *      (d_pos, h_pos) is mutually exclusive, so is (d_match_result, h_match_result).
 *  i.e.
 *      if ( h_pos ) then 
 *         h_pos <-- d_pos_zip
 *      else 
 *         d_pos <-- d_pos_zip
 *      end
 *
 *      if ( h_match_result ) then 
 *         h_match_result <-- d_match_result_zip
 *      else
 *         d_match_result <-- d_match_result_zip
 *      end
 *
 */ 
    
__host__  PFAC_status_t PFAC_reduce_kernel( PFAC_handle_t handle, int *d_input_string, int input_size,
    int *d_match_result, int *d_pos, int *h_num_matched, int *h_match_result, int *h_pos )
{
    int *d_nnz_per_block = NULL ;     // working space, d_nnz_per_block[j] = nnz of block j 
    int *d_pos_zip = NULL ;           // working space, compression of initial d_pos
    int *d_match_result_zip = NULL ;  // working space, compression of initial d_match_result  
    cudaError_t cuda_status ;
    PFAC_status_t PFAC_status ;
    
    // n_hat = (input_size + 3)/4 = number of integers of input string
    int n_hat = (input_size + sizeof(int)-1)/sizeof(int) ; 

    // num_blocks = # of thread blocks to cover input stream 
    int num_blocks = (n_hat + BLOCK_SIZE*NUM_INTS_PER_THREAD-1)/(BLOCK_SIZE*NUM_INTS_PER_THREAD) ;

    cuda_status = cudaMalloc((void **)&d_nnz_per_block, num_blocks*sizeof(int) );
    if ( cudaSuccess != cuda_status ){
        return PFAC_STATUS_CUDA_ALLOC_FAILED ;
    }    

    dim3  dimBlock( BLOCK_SIZE, 1 ) ;
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

    PFAC_status = PFAC_reduce_kernel_stage1( handle, d_input_string, input_size,
        n_hat, num_blocks, dimBlock, dimGrid,
        d_match_result, d_pos, d_nnz_per_block, h_num_matched );
     
    if ( PFAC_STATUS_SUCCESS != PFAC_status ){
        cudaFree(d_nnz_per_block);
        return PFAC_STATUS_INTERNAL_ERROR ;
    } 
   
    if ( 0 == *h_num_matched ){
        cudaFree(d_nnz_per_block);
        return PFAC_STATUS_SUCCESS; 
    }
    
    /*
     * stage 3: compression (d_match_result, d_pos) to working space (d_pos_zip, d_match_result_zip)
     *      by information of d_nnz_per_block
     *
     *  after stage 3, d_nnz_per_block is useless
     */
    cudaError_t cuda_status1 = cudaMalloc((void **) &d_pos_zip,          (*h_num_matched)*sizeof(int) );
    cudaError_t cuda_status2 = cudaMalloc((void **) &d_match_result_zip, (*h_num_matched)*sizeof(int) );
    if ( (cudaSuccess != cuda_status1) || (cudaSuccess != cuda_status2) ){
        if ( NULL != d_pos_zip )          { cudaFree(d_pos_zip); }
        if ( NULL != d_match_result_zip ) { cudaFree(d_match_result_zip); }
        cudaFree(d_nnz_per_block);
        return PFAC_STATUS_CUDA_ALLOC_FAILED ;
    }    
    
    int elements_per_block = BLOCK_SIZE * NUM_INTS_PER_THREAD * 4 ;
    zip_kernel<<< dimGrid, dimBlock >>>(d_pos, d_match_result, d_nnz_per_block, 
        num_blocks - 1, elements_per_block,
        d_pos_zip, d_match_result_zip );
        
    cuda_status = cudaGetLastError() ;
    if ( cudaSuccess != cuda_status ){
        cudaFree(d_pos_zip);
        cudaFree(d_match_result_zip);
        cudaFree(d_nnz_per_block); 
        return PFAC_STATUS_INTERNAL_ERROR ;
    }
    
    cudaFree(d_nnz_per_block);    
    
    /*
     *  stage 4: copy data back to d_pos and d_match_result
     *      we can write hand-copy kernel to copy (d_pos_zip, d_match_result)
     *      this should be efficient  
     */
    if ( NULL != h_pos ){
        cuda_status1 = cudaMemcpy(h_pos,          d_pos_zip,          (*h_num_matched)*sizeof(int), cudaMemcpyDeviceToHost);
    }else{ 
        cuda_status1 = cudaMemcpy(d_pos,          d_pos_zip,          (*h_num_matched)*sizeof(int), cudaMemcpyDeviceToDevice);
    }
    if ( NULL != h_match_result ){
        cuda_status2 = cudaMemcpy(h_match_result, d_match_result_zip, (*h_num_matched)*sizeof(int), cudaMemcpyDeviceToHost);
    }else{
        cuda_status2 = cudaMemcpy(d_match_result, d_match_result_zip, (*h_num_matched)*sizeof(int), cudaMemcpyDeviceToDevice);
    }
    
    if ( (cudaSuccess != cuda_status1) || (cudaSuccess != cuda_status2) ){
        cudaFree(d_pos_zip);
        cudaFree(d_match_result_zip);
        return PFAC_STATUS_INTERNAL_ERROR ;
    }    
        
    cudaFree(d_pos_zip);
    cudaFree(d_match_result_zip);
        
    return PFAC_STATUS_SUCCESS;
}

/*
 *  stage 1: perform matching process and zip non-zero (matched thread) into continuous
 *      memory block and keep order. Morever nnz of each thread block is stored in d_nnz_per_block
 *
 *  d_nnz_per_block[j] = nnz of thread block j
 *
 *  since each thread block processes 1024 substrings, so range of d_nnz_per_block[j] is [0,1024] 
 */

__host__  PFAC_status_t PFAC_reduce_kernel_stage1( PFAC_handle_t handle, 
    int *d_input_string, int input_size,
    int n_hat, int num_blocks, dim3 dimBlock, dim3 dimGrid,
    int *d_match_result, int *d_pos, int *d_nnz_per_block, int *h_num_matched )
{
    cudaError_t cuda_status ;

    int num_finalState = handle->numOfFinalStates;
    int initial_state  = handle->initial_state;
    bool smem_on = ((4*EXTRA_SIZE_PER_TB-1) >= handle->maxPatternLen) ;
    bool texture_on = (PFAC_TEXTURE_ON == handle->textureMode );

#ifdef DEBUG_MSG
    if ( texture_on ){
        printf("run PFAC_reduce_kernel (texture ON) \n");
    }else{
        printf("run PFAC_reduce_kernel (texture OFF) \n");
    }

    if (smem_on) {
        printf("run PFAC_reduce_kernel (smem ON ), maxPatternLen = %d\n", handle->maxPatternLen);
    }else{
        printf("run PFAC_reduce_kernel (smem OFF), maxPatternLen = %d\n", handle->maxPatternLen);
    }
#endif

    if ( texture_on ){
        textureReference *texRefTable ;
        cudaGetTextureReference( (const struct textureReference**)&texRefTable, "tex_PFAC_table_reduce" );

        cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<int>();
        
        // set texture parameters
        tex_PFAC_table_reduce.addressMode[0] = cudaAddressModeClamp;
        tex_PFAC_table_reduce.addressMode[1] = cudaAddressModeClamp;
        tex_PFAC_table_reduce.filterMode     = cudaFilterModePoint;
        tex_PFAC_table_reduce.normalized     = 0;
        
        size_t offset ;
        cuda_status = cudaBindTexture( &offset, (const struct textureReference*) texRefTable,
        (const void*) handle->d_PFAC_table, (const struct cudaChannelFormatDesc*) &channelDesc, handle->sizeOfTableInBytes ) ;
        if ( cudaSuccess != cuda_status ){
#ifdef DEBUG_MSG
            printf("Error: cannot bind texture, %s\n", cudaGetErrorString(status) );
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

    if (smem_on) {
        if ( texture_on ){
            PFAC_reduce_kernel_device<1, 1> <<< dimGrid, dimBlock >>>( handle->d_PFAC_table,
                d_input_string, input_size, n_hat, num_finalState, initial_state, num_blocks - 1,
                d_pos, d_match_result, d_nnz_per_block );
        }else{
            PFAC_reduce_kernel_device<0, 1> <<< dimGrid, dimBlock >>>( handle->d_PFAC_table,
                d_input_string, input_size, n_hat, num_finalState, initial_state, num_blocks - 1,
                d_pos, d_match_result, d_nnz_per_block );
        }
    }else{
        if ( texture_on ){
            PFAC_reduce_kernel_device<1, 0> <<< dimGrid, dimBlock >>>( handle->d_PFAC_table,
                d_input_string, input_size, n_hat, num_finalState, initial_state, num_blocks - 1,
                d_pos, d_match_result, d_nnz_per_block );
        }else{
            PFAC_reduce_kernel_device<0, 0> <<< dimGrid, dimBlock >>>( handle->d_PFAC_table,
                d_input_string, input_size, n_hat, num_finalState, initial_state, num_blocks - 1,
                d_pos, d_match_result, d_nnz_per_block );
        }
    }
    
    cuda_status = cudaGetLastError() ;
    if ( cudaSuccess != cuda_status ){
        if ( texture_on ) { 
            cudaUnbindTexture(tex_PFAC_table_reduce); 
        }
        return PFAC_STATUS_INTERNAL_ERROR ;
    }

    if ( texture_on ){
        cudaUnbindTexture(tex_PFAC_table_reduce);
    }

    /*
     *  stage 2: use Thrust to do in-place prefix_sum( d_nnz_per_block[0:num_blocks-1] ) 
     *      
     *  after inclusive_scan, then 
     *
     *  d_nnz_per_block[j] = prefix_sum( d_nnz_per_block[0:j] )
     *
     *  d_nnz_per_block[num_blocks-1] = total number of non-zero = h_num_matched 
     *
     */
    thrust::device_ptr<int> dev_nnz_per_block ( d_nnz_per_block ) ;
    thrust::inclusive_scan(dev_nnz_per_block, dev_nnz_per_block + num_blocks, dev_nnz_per_block );

    cuda_status = cudaMemcpy( h_num_matched, d_nnz_per_block + num_blocks-1, sizeof(int), cudaMemcpyDeviceToHost) ;
    if ( cudaSuccess != cuda_status ){
        return PFAC_STATUS_INTERNAL_ERROR ;
    }

    return PFAC_STATUS_SUCCESS ;
}    



__global__ void zip_kernel(int *d_pos, int *d_match_result, int *d_nnz_per_block,
     int num_blocks_minus1, int elements_per_block,
     int *d_pos_zip, int *d_match_result_zip)
{
    int tid   = threadIdx.x ;
    int gbid  = blockIdx.y * gridDim.x + blockIdx.x ;
    
    if ( gbid > num_blocks_minus1 ){
        return ; // d_nnz_per_block[0:num_blocks-1]
    }
    
    int start = 0 ;
    if ( 0 < gbid ){
        start = d_nnz_per_block[gbid - 1] ;
    }
    int nnz = d_nnz_per_block[gbid] - start ;
    
    int base = gbid * elements_per_block ;
    for( int colIdx = tid ; colIdx < nnz ; colIdx += BLOCK_SIZE ){
        d_pos_zip[ start + colIdx ] = d_pos[ base + colIdx ] ;
        d_match_result_zip[ start + colIdx ] = d_match_result[ base + colIdx ] ;
    }
}
    
    

/*
 *  (1) transition table of initial state is in the shared memory phi_s02s1
 *      we don't need to look up table in texture tex_PFAC_table
 *
 *  (2) final states are reordered as 0, 1, 2, ..., k -1
 *      so state number < k (number of final states) means final state
 */
#define  SUBSEG_MATCH( j, match ) \
    pos = tid + j * BLOCK_SIZE ;\
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
                state = tex_lookup(state, inputChar); \
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
    pos = ( gbid * BLOCK_SIZE * NUM_INTS_PER_THREAD * 4 ) + tid + j * BLOCK_SIZE ;\
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
                state = tex_lookup(state, inputChar); \
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
    pos = tid + j * BLOCK_SIZE ;\
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

#define  SUBSEG_MATCH_NOSMEM_NOTEX( j, match ) \
    pos = ( gbid * BLOCK_SIZE * NUM_INTS_PER_THREAD * 4 ) + tid + j * BLOCK_SIZE ;\
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


/*
 *  caller must reset working space s_Data first
 *
 *  This device function comes from SDK/scan
 *  
 *  original code
 *  [code]
 *    //assuming size <= WARP_SIZE
 *   inline __device__ uint warpScanInclusive(uint idata, uint *s_Data, uint size){
 *       uint pos = 2 * threadIdx.x - (threadIdx.x & (size - 1));
 *       s_Data[pos] = 0;
 *       pos += size;
 *       s_Data[pos] = idata;
 *
 *       for(uint offset = 1; offset < size; offset <<= 1)
 *           s_Data[pos] += s_Data[pos - offset];
 *
 *       return s_Data[pos];
 *   }
 *  [/code]
 *
 *  Question: one may wonder why volatile keyword is missing?
 *  nvcc 3.2 will keep "s_Data[pos] = ..." and cuobjdump shows
 *  [code]
 *       int pos = 2 * id - (id &31);
 *       s_Data[pos] = 0 ;
 *       pos += 32 ;
 *       s_Data[pos] = idata ;
 *       R0 = idata ;
 *       for( int offset = 1 ; offset < 32 ; offset <<= 1 ){
 *           R0 += s_Data[pos - offset];
 *           s_Data[pos] = R0 ;
 *       }
 *       return s_Data[pos];
 *  [/code]
 *
 *  check http://forums.nvidia.com/index.php?showtopic=193730
 *
 */
inline __device__ int warpScanInclusive(int idata, int id, int *s_Data)
{
    int pos = 2 * id - (id &31);
    // s_Data[pos] = 0 ;
    pos += 32 ;
    s_Data[pos] = idata ;
    
    for( int offset = 1 ; offset < 32 ; offset <<= 1 ){
        s_Data[pos] += s_Data[pos - offset];
    }
    return s_Data[pos];
}


#define MANUAL_EXPAND_2( X )   { X ; X ; }
#define MANUAL_EXPAND_4( X )   { MANUAL_EXPAND_2( MANUAL_EXPAND_2( X ) )  }
#define MANUAL_EXPAND_8( X )   { MANUAL_EXPAND_4( MANUAL_EXPAND_4( X ) )  }

/*
 *  resource usage
 *  sm20:
 *      1) use smem
 *          32 regs, 5120B smem, 96B cmem, => 1024 threads / SM
 *      2) no smem
 *          40 regs, 5120B smem, 96B cmem, => 768 threads / SM
 *  sm13:
 *      1) use smem
 *          24 regs, 5200B smem, 52B cmem, => 256 threads / SM
 *      2) no smem
 *          32 regs, 5200B smem, 52B cmem, => 256 threads / SM 
 *
 *  sm11:
 *      1) use smem
 *          24 regs, 5200B smem, 52B cmem, => 256 threads / SM
 *      2) no smem
 *          32 regs, 5200B smem, 8B  cmem, => 256 threads / SM  
 */

template <int TEXTURE_ON , int SMEM_ON >
__global__ void PFAC_reduce_kernel_device(int *d_PFAC_table, int *d_input_string, 
    int input_size, int n_hat, int num_finalState, int initial_state, int num_blocks_minus1,
    int *d_pos, int *d_match_result, int *d_nnz_per_block )
{
    int tid   = threadIdx.x ;
    int gbid  = blockIdx.y * gridDim.x + blockIdx.x ;
    int start ;
    int pos;   
    int state;
    int inputChar;
    int match[4*NUM_INTS_PER_THREAD] ;
    __shared__ int s_input[ BLOCK_SIZE*NUM_INTS_PER_THREAD*4];
    __shared__ int phi_s02s1[ 256 ] ;
    volatile unsigned char *s_char;
    char * char_d_input_string ;
     
    if ( gbid > num_blocks_minus1 ){
        return ; // whole block is outside input stream
    }
        
    #pragma unroll
    for(int i = 0 ; i < 4*NUM_INTS_PER_THREAD ; i++){
        match[i] = 0 ;
    }

    // load transition table of initial state to shared memory
    if ( TEXTURE_ON ){
        #pragma unroll
        for(int i = 0 ; i < BLOCK_SIZE_DIV_256 ; i++){
            phi_s02s1[ tid + i*BLOCK_SIZE ] = tex_lookup(initial_state, tid + i*BLOCK_SIZE); 
        }
    }else{
        #pragma unroll
        for(int i = 0 ; i < BLOCK_SIZE_DIV_256 ; i++){
            phi_s02s1[ tid + i*BLOCK_SIZE ] = *(d_PFAC_table + initial_state*CHAR_SET + (tid + i*BLOCK_SIZE) );
        }    
    }

#if  BLOCK_SIZE < EXTRA_SIZE_PER_TB
    #error BLOCK_SIZE should be bigger than EXTRA_SIZE_PER_TB
#endif

    if ( SMEM_ON ){  
        // legal thread block which contains some input stream
        s_char = (unsigned char *)s_input;

        // read global data to shared memory
        start = gbid * (BLOCK_SIZE*NUM_INTS_PER_THREAD) + tid ;
        #pragma unroll
        for(int i = 0 ; i < NUM_INTS_PER_THREAD ; i++){
            if ( start < n_hat ){
                s_input[tid + i*BLOCK_SIZE] = d_input_string[start];
            }
            start += BLOCK_SIZE ;
        }
        if ( (start < n_hat) && (tid < EXTRA_SIZE_PER_TB) ){
            s_input[tid + NUM_INTS_PER_THREAD*BLOCK_SIZE] = d_input_string[start];
        }
    }// if SMEM_ON
    
    __syncthreads();

    // bdy = number of legal characters starting at gbid*BLOCKSIZE*4
    int bdy = input_size - gbid*(BLOCK_SIZE * NUM_INTS_PER_THREAD * 4);

#if 2 != NUM_INTS_PER_THREAD
    #error  NUM_INTS_PER_THREAD must be 2, or MANUAL_EXPAND_8 is wrong
#endif  
    
    if ( SMEM_ON ){  
        if ( TEXTURE_ON ){
            int j = 0 ;
            MANUAL_EXPAND_8( SUBSEG_MATCH(j, match[j]) ; j++ ; ) 
        }else{
            int j = 0 ;
            MANUAL_EXPAND_8( SUBSEG_MATCH_NOTEX(j, match[j]) ; j++ ;)
        }        
    }else{
        char_d_input_string = (char*)d_input_string ; // used only when SMEM_ON = 0
        if ( TEXTURE_ON ){
            int j = 0 ;
            MANUAL_EXPAND_8( SUBSEG_MATCH_NOSMEM(j, match[j]) ; j++ ; ) 
        }else{
            int j = 0 ;
            MANUAL_EXPAND_8( SUBSEG_MATCH_NOSMEM_NOTEX(j, match[j]) ; j++ ;)
        }            
    
    }
    
    // matching is done, we can re-use shared memory s_input and phi_s02s1 
    // to do inclusive_scan
    // we have 128 thread per block (4 warps per block) and each thread needs to 
    // process 8 (4*NUM_INTS_PER_THREAD) substrings. It is equivalent to say
    // 4 x 8 = 32 warps processing 1024 substrings. 
    // if we concatenate match[j] to a linear array of 1024 entries, then
    // acc_pos[j] of lane_id = number of non-zero of match[j] of thread k, k <= lane_id
    //                       = prefix_sum( match[32*j:32*j+land_id] ) 
    // acc_warp[j] is number of nonzero of match[32*j:32*j+31]
    //                       = prefix_sum( match[32*j:32*j+31] )
    // 
    // stage 1: inclusive scan inside a warp
    int  warp_id = tid >> 5 ;
    int  lane_id = tid & 31 ;
    int  acc_pos[4*NUM_INTS_PER_THREAD] ;
    int *acc_warp = phi_s02s1 ; // alias acc_warp[32] to phi_s02s1
                                // reuse phi_s02s1
#if  32 != (NUM_WARPS_PER_BLOCK * 4*NUM_INTS_PER_THREAD)   
    #error 32 != (NUM_WARPS_PER_BLOCK * 4*NUM_INTS_PER_THREAD)
#endif

    __syncthreads(); // s_input and phi_s02s1 can be re-used
    
#if 200 <= __CUDA_ARCH__

    if ( 0 == warp_id ){
        s_input[lane_id] = 0 ;
    }
    int k = 0 ;
    unsigned int match_pattern ;
    MANUAL_EXPAND_8( match_pattern = __ballot( match[k] > 0 ); \
        match_pattern <<= (31-lane_id); \
        acc_pos[k] = __popc(match_pattern); \
        if ( 31 == lane_id ){ \
            acc_warp[ warp_id + k * NUM_WARPS_PER_BLOCK ] = acc_pos[k] ;\
        }\
        k++ ; )
    __syncthreads();
    
#else

    // clear supplemet area of s_input
    #pragma unroll
    for (int k = 0 ; k < 4 ; k++ ){
        int id = tid + k*BLOCK_SIZE ;
        int pos = 2 * id - (id &31);
        s_input[pos] = 0 ;
    }
    
    __syncthreads();
    int k = 0 ;
    int idata ;
    MANUAL_EXPAND_4( idata = match[k] > 0 ; \
        acc_pos[k] = warpScanInclusive(idata, tid + k*BLOCK_SIZE, s_input); \
        if ( 31 == lane_id ){  \
            acc_warp[ warp_id + k * NUM_WARPS_PER_BLOCK ] = acc_pos[k] ;\
        } \
        k++ ; ) 
    // __syncthreads(); // not necessary     
    k = 0 ;
    MANUAL_EXPAND_4( idata = match[4+k] > 0 ; \
        acc_pos[4+k] = warpScanInclusive(idata, tid + k*BLOCK_SIZE, s_input); \
        if ( 31 == lane_id ){  \
            acc_warp[ warp_id + (4+k) * NUM_WARPS_PER_BLOCK ] = acc_pos[4+k] ;\
        } \
        k++ ; )     
    __syncthreads();

#endif

    // stage 2:  acc_pos[0:7] and acc_warp[0:31] are done, we can re-use s_input again
    // s_input[32+j] = prefix_sum( acc_warp[0:j] )
    // note that s_input[0:31] always keeps zero 
    if ( 0 == warp_id ){
        warpScanInclusive(acc_warp[lane_id], lane_id, s_input);
    }
    
    __syncthreads();

    // stage 3: s_input[32:63] contains information as
    // s_input[32+j+1] - s_input[32+j] = nnz of warp j
    // s_input[63] = prefix_sum( match[0:1023] )
    // correct local position of each matched substring
    // note that position starts from 0, so we need minus 1,
    // for example, suppose acc_pos[0] of warp 0 is
    //           1 1 1 2 3 ...
    // then t0, t3, t4 match, and should write to position of matched result
    //           0 0 0 1 2 ...
    //  d_match_result[ t0, t3, t4, ...]  
    //  d_pos[ 0, 3, 4, ...]
    #pragma unroll
    for (int j = 0 ; j < 4*NUM_INTS_PER_THREAD ; j++ ){
        acc_pos[j] += ( s_input[31 + warp_id + j * NUM_WARPS_PER_BLOCK] - 1 ) ; 
    }
    int nnz = s_input[63];
    
    __syncthreads();

    // stage 4: all data are in acc_pos[] and match[], s_input can be reused again
    // collect non-zero data to s_input, then do coalesced write
    start = gbid * (BLOCK_SIZE * NUM_INTS_PER_THREAD * 4) ;
        
    #pragma unroll    
    for (int j = 0 ; j < 4*NUM_INTS_PER_THREAD ; j++ ){
        if ( match[j] ){
            s_input[ acc_pos[j] ] = match[j];
        }
    }
    __syncthreads();
    
    for (int j = tid ; j < nnz; j+= BLOCK_SIZE ){
        d_match_result[start + j ] = s_input[j] ;
    }
    __syncthreads();
    
    #pragma unroll
    for (int j = 0 ; j < 4*NUM_INTS_PER_THREAD ; j++ ){
        if ( match[j] ){
            s_input[ acc_pos[j] ] = start + tid + j * BLOCK_SIZE ;
        }
    }
    __syncthreads();
    
    for (int j = tid ; j < nnz; j+= BLOCK_SIZE ){
        d_pos[start + j ] = s_input[j] ;
    }
    
    if ( 0 == tid ){
        d_nnz_per_block[ gbid ] = nnz ;
    }       
}

/*
  technical note of PFAC_reduce_kernel_device:

----------------------------------------------------------------------------------------
  
  1) nvcc uses lmem on following code, so 
[code]  
    __syncthreads();
    #pragma unroll
    for (int k = 0 ; k < 4 ; k++ ){
        int idata = match[k] > 0 ;
        acc_pos[k] = warpScanInclusive(idata, tid + k*BLOCK_SIZE, s_input);    
        if ( 31 == lane_id ){
            acc_warp[ warp_id + k * NUM_WARPS_PER_BLOCK ] = acc_pos[k] ;           
        }
    }    
    #pragma unroll
    for (int k = 0 ; k < 4 ; k++ ){
        int idata = match[4+k] > 0 ;
        acc_pos[4+k] = warpScanInclusive(idata, tid + k*BLOCK_SIZE, s_input);    
        if ( 31 == lane_id ){
            acc_warp[ warp_id + (4+k) * NUM_WARPS_PER_BLOCK ] = acc_pos[4+k] ;           
        }
    }
    __syncthreads();
[/code]
    
    is manually unrolled as  

[code]
    __syncthreads();
    int k = 0 ;
    int idata ;
    MANUAL_EXPAND_4( idata = match[k] > 0 ; \
        acc_pos[k] = warpScanInclusive(idata, tid + k*BLOCK_SIZE, s_input); \
        if ( 31 == lane_id ){  \
            acc_warp[ warp_id + k * NUM_WARPS_PER_BLOCK ] = acc_pos[k] ;\
        } \
        k++ ; ) 
    // __syncthreads(); // not necessary     
    k = 0 ;
    MANUAL_EXPAND_4( idata = match[4+k] > 0 ; \
        acc_pos[4+k] = warpScanInclusive(idata, tid + k*BLOCK_SIZE, s_input); \
        if ( 31 == lane_id ){  \
            acc_warp[ warp_id + (4+k) * NUM_WARPS_PER_BLOCK ] = acc_pos[4+k] ;\
        } \
        k++ ; )     
    __syncthreads();
[/code]

---------------------------------------------------------------------------------
  2) simplify following code 

[code]
    if ( TEXTURE_ON ){
        SUBSEG_MATCH(0, match[0]) ;
        SUBSEG_MATCH(1, match[1]) ;
        SUBSEG_MATCH(2, match[2]) ;
        SUBSEG_MATCH(3, match[3]) ;

#if 2 == NUM_INTS_PER_THREAD    
        SUBSEG_MATCH(4, match[4]) ;
        SUBSEG_MATCH(5, match[5]) ;
        SUBSEG_MATCH(6, match[6]) ;
        SUBSEG_MATCH(7, match[7]) ;
#endif

    }else{
     
        SUBSEG_MATCH_NOTEX(0, match[0]) ;
        SUBSEG_MATCH_NOTEX(1, match[1]) ;
        SUBSEG_MATCH_NOTEX(2, match[2]) ;
        SUBSEG_MATCH_NOTEX(3, match[3]) ;
#if 2 == NUM_INTS_PER_THREAD    
        SUBSEG_MATCH_NOTEX(4, match[4]) ;
        SUBSEG_MATCH_NOTEX(5, match[5]) ;
        SUBSEG_MATCH_NOTEX(6, match[6]) ;
        SUBSEG_MATCH_NOTEX(7, match[7]) ;
#endif    
    }
[/code]  

    by compact macro 
    
[code]
    if ( TEXTURE_ON ){
        int j = 0 ;
        MANUAL_EXPAND_8( SUBSEG_MATCH(j, match[j]) ; j++ ; ) 
    }else{
        int j = 0 ;
        MANUAL_EXPAND_8( SUBSEG_MATCH_NOTEX(j, match[j]) ; j++ ;)
    }      
[/code]    

-----------------------------------------------------------------------------------------
 3. optimization on Fermi
 
[code]
    #pragma unroll
    for(int k = 0 ; k < 4*NUM_INTS_PER_THREAD ; k++ ){
        unsigned int match_pattern = __ballot( match[k] > 0 ) ;
        match_pattern <<= (31 - lane_id);
        acc_pos[k] = __popc(match_pattern) ;
        if ( 31 == lane_id ){  
            acc_warp[ warp_id + k * NUM_WARPS_PER_BLOCK ] = acc_pos[k] ;
        } 
    }
[/code]

   becomes

[code]
    int k = 0 ;
    unsigned int match_pattern ;
    MANUAL_EXPAND_8( match_pattern = __ballot( match[k] > 0 ); \
        match_pattern <<= (31-lane_id); \
        acc_pos[k] = __popc(match_pattern); \
        if ( 31 == lane_id ){ \
            acc_warp[ warp_id + k * NUM_WARPS_PER_BLOCK ] = acc_pos[k] ;\
        }\
        k++ ; )
[/code]

-----------------------------------------------------------------------------------------

 */
 
 
