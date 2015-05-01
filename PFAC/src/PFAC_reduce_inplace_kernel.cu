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
 *  two methods can achieve global synchronizations in kernel PFAC_reduce_inplace_kernel
 *  method 1: atomic operation
 *      refer to http://forums.nvidia.com/index.php?showtopic=98444&pid=548609&start=&st=#entry548609
 *  method 2: volatile declaration
 *      this would avoid L1-cache incoherence because memory load directly comes from L2-cache
 *      LD.E.CV appears in assembly code.
 *      .cv cache as volatile (consider cached system memory lines stale, fetch again)
 *
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

    PFAC_status_t PFAC_reduce_inplace_kernel( PFAC_handle_t handle, int *d_input_string, int input_size,
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



__host__  PFAC_status_t PFAC_reduce_space_driven_stage1( PFAC_handle_t handle, 
    int *d_input_string, int input_size,
    int n_hat, int num_blocks, dim3 dimBlock, dim3 dimGrid,
    int *d_match_result, int *d_pos, int *d_nnz_per_block, int *h_num_matched );
    
    
__global__  void  set_semaphore( int *d_w, int num_ones, int size );


__global__ void zip_inplace_kernel(int *d_pos, int *d_matched_result, 
    int *d_nnz_per_block, int *d_atomicBlockID, int *d_semaphore, int *d_spinlock,
    int num_blocks_minus1 );

// ---------------------------- main ----------------------    
 
/*
 *  synchronous call
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
 *      
 *      (d_pos, d_match_result) pair is in device memory
 *
 *      step 1: do matching process and zip matched position and pattern number inside a thread block,
 *              called local compaction. Moreover nnz per block is stored into temporary array d_nnz_per_block
 *
 *      step 2: use Thrust::inclusive_scan to do prefix_sum( d_nnz_per_block )
 *
 *      step 3: do global compaction on the same (d_pos, d_match_result) pair, no additional memory is required.
 * 
 *      support 2 mode: device mode and host mode
 * 
 *      Device mode: 
 *      (d_pos, d_match_result) pair is device memory
 *      (h_pos, h_match_result) is (NULL,NULL)
 *
 *      Host mode:
 *      (d_pos, d_match_result) pair is working space
 *      (h_pos, h_match_result) is not (NULL,NULL) 
 *      
 *
 *  We can combine two modes in a simple way, 
 *      (d_pos, h_pos) is mutually exclusive, so is (d_match_result, h_match_result).
 *  i.e.
 *      if ( h_pos ) then 
 *         h_pos <-- d_pos
 *      end
 *
 *      if ( h_match_result ) then 
 *         h_match_result <-- d_match_result
 *      end
 *
 */     
 
__host__  PFAC_status_t PFAC_reduce_inplace_kernel( PFAC_handle_t handle, int *d_input_string, int input_size,
    int *d_match_result, int *d_pos, int *h_num_matched, int *h_matched_result, int *h_pos )
{

#ifdef DEBUG_MSG
    printf("call PFAC_reduce_inplace_kernel \n");
#endif

    int *d_nnz_per_block = NULL ; // working space, d_nnz_per_block[j] = nnz of block j
    int *d_w = NULL ; // working space, d_w = {d_atomicBlockID, d_semaphore, d_spinlock }
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
    dimGrid.y = 1 ;
    if ( p ){
        dimGrid.x = 1<<15 ;
        dimGrid.y = p+1 ;
    }

    /*
     *  stage 1: do matching process and zip matched position and 
     *      pattern number inside a thread block, called local compaction. 
     *      Moreover nnz per block is stored into temporary array d_nnz_per_block
     *      
     *      use Thrust::inclusive_scan to do prefix_sum( d_nnz_per_block )
     *
     *      suppose B = num_blocks, then 
     *
     *      h_num_matched = d_nnz_per_block[B-1]
     */

    PFAC_status = PFAC_reduce_space_driven_stage1( handle, d_input_string, input_size,
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
     *  stage 2: setup semaphore
     *
     *  suppose B = num_blocks and k = size of d_atomicBlockID, then 
     *      d_w[0:B+k+1) = { d_atomicBlockID[0,k), d_semaphore[0,B), d_spinlock }
     *
     *  set_semaphore kernel uses 1-D grids and 1-D blocks.
     *
     *  typical size of input stream is 1G byte which has 1G substrings.
     *  num_blocks ~ 1M after stage 1 
     *    
     *  block size of set_semaphore kernel is 256, then 
     *
     *  number of grids of set_semaphore ~ 1M/256 = 4K
     *
     */

    int size_d_atomicBlockID = 32 ;
    int num_ones = ( size_d_atomicBlockID + 1 );
    int size_d_w = num_blocks + size_d_atomicBlockID + 1 ;
 
    cuda_status = cudaMalloc((void **) &d_w, size_d_w*sizeof(int) );
    if ( cudaSuccess != cuda_status ){
        return PFAC_STATUS_CUDA_ALLOC_FAILED ;
    }
    
    const int blockSize_semaphore = 256 ;
    
    int num_grids = (size_d_w + (blockSize_semaphore-1))/blockSize_semaphore ;
    if ( num_grids > 65535 ){
        cudaFree( d_w ) ;
        return PFAC_STATUS_INTERNAL_ERROR ;
    }
    
    set_semaphore<<< num_grids, blockSize_semaphore >>>( d_w, num_ones, size_d_w ) ;
    cuda_status = cudaGetLastError() ;
    if ( cudaSuccess != cuda_status ){
        cudaFree( d_w ) ;
        return PFAC_STATUS_INTERNAL_ERROR ;
    }    

    /*
     * stage 3: compression (d_match_result, d_pos) to itself
     *      by information of d_nnz_per_block
     *  
     *  suppose B = num_blocks, then 
     *      d_w[0:B+1] = { d_atomicBlockID, d_semaphore[0,B), d_spinlock }
     *  
     *  NOTE: since block 0 has nothing to do, so d_spinlock = 0 means no spinlock occurs
     *
     *  after stage 3, d_nnz_per_block is useless
     */
    
    int *d_atomicBlockID = d_w ;
    int *d_semaphore = d_w + size_d_atomicBlockID ;
    int *d_spinlock  = d_w + size_d_w - 1 ;
    
    zip_inplace_kernel<<< dimGrid, dimBlock >>>(d_pos, d_match_result, 
        d_nnz_per_block, d_atomicBlockID, d_semaphore, d_spinlock, num_blocks - 1);
 
    cuda_status = cudaGetLastError() ;
    if ( cudaSuccess != cuda_status ){
        cudaFree(d_w); 
        cudaFree(d_nnz_per_block); 
        return PFAC_STATUS_INTERNAL_ERROR ;
    }

    cudaFree(d_w); 
    cudaFree(d_nnz_per_block);    
    
    /*
     *  stage 4: copy data back to h_pos and h_match_result
     */
     
    if ( NULL != h_pos ){
        cuda_status = cudaMemcpy(h_pos, d_pos, (*h_num_matched)*sizeof(int), cudaMemcpyDeviceToHost);
        if (cudaSuccess != cuda_status) {
            return PFAC_STATUS_INTERNAL_ERROR ;
        }
    } 
    if ( NULL != h_matched_result ){
        cuda_status = cudaMemcpy(h_matched_result, d_match_result, (*h_num_matched)*sizeof(int), cudaMemcpyDeviceToHost);
        if (cudaSuccess != cuda_status) {
            return PFAC_STATUS_INTERNAL_ERROR ;
        }        
    } 
     
    return PFAC_STATUS_SUCCESS;
}


/*
 *   d_w[0:size) has "size" elements
 *
 *   suppose k = num_ones = number of ones in d_w
 *
 *   d_w[0:k) = 1
 *   d_w[k:end] = 0 
 *
 *   set_semaphore is 1-D grids, 1-D blocks
 */  
__global__  void  set_semaphore( int *d_w, int num_ones, int size )
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x ;
    int val = 0 ;
    if ( tid < num_ones ){
        val = 1 ;
    }
    if ( tid < size ){
        d_w[tid] = val ;
    }
}

/*
 *  requirement:
 *      initial value of d_atomicBlockID is 1  (must be set by caller)
 *
 *  basic idea: special type of global synchronization
 *      use atomicAdd(d_atomicBlockID,1) to fetch gbid, such that 
 *      if a threads block has gbid k, then for all block j , j < k has 
 *      been processed by some thread blocks.
 *       
 *  Resource usage:
 *
 *  sm20:
 *      29 regs, 4  bytes smem  => 1024 threads per SM
 *  sm13, sm12, sm11:
 *      29 regs, 56+16 bytes smem => 512 threads per SM 
 * 
 */
 
__global__ void zip_inplace_kernel(int *d_pos, int *d_matched_result, 
    int *d_nnz_per_block, int *d_atomicBlockID, int *d_semaphore, int *d_spinlock,
    int num_blocks_minus1 )
{
    int tid = threadIdx.x ;
    __shared__ int gbid_smem;
    
    int match[4*NUM_INTS_PER_THREAD] ;
    int pos[4*NUM_INTS_PER_THREAD] ;
       
    /*
     *  step 1: obtain unprocessed block ID via global synchronization
     *  
     *  1) gbid > 0 because initial value of d_atomicBlockID is 1
     *  2) B = number of blocks to cover input stream, each block has 1024 elements,
     *     then gbid = 1, 2, ..., B-1 is legal.
     *     
     */
    if ( 0 == tid ){
        gbid_smem = atomicAdd( d_atomicBlockID, 1 ) ;
    }
    __syncthreads(); // necessary due to gbid_smem
    
    int gbid = gbid_smem ;  
   
    if ( gbid > num_blocks_minus1 ){
        return ; // d_nnz_per_block[0:num_blocks-1]
    }
    
    int start = d_nnz_per_block[gbid - 1] ;
    int end   = d_nnz_per_block[gbid];
    int nnz   = end - start ;

    // (gbid, nnz) is ready, start to read data

#if  1024 != 4*NUM_INTS_PER_THREAD*BLOCK_SIZE  
    #error   1024 != 4*NUM_INTS_PER_THREAD*BLOCK_SIZE    
#endif

    int w_start = start   >> 10 ;
    int w_end   = (end-1) >> 10 ;

    /*
     *  check two conditions in which no work is required.
     *  we can unlock semaphore as early as possible.
     *
     *  1) if nnz = 0, then no matched data in this block.
     *  2) gbid = w_start, means that all block of id j < gbid has nnz=1024,
     *     we don't need to move data in this block
     *
     */
    if ( (0 == nnz) || (gbid == w_start) ){
        // nothing to read, unlock semaphore 
        if ( 0 == tid ) {
            d_semaphore[gbid] = 1 ; // crucial, if not set, then hangs
                                    // store directly goes to L2-cache, please see PTX document
            //atomicExch( d_semaphore + gbid, 1) ;            
        }                   
        return ;  
    }
    // now, nnz >= 1 or say end >= (start + 1)

    /*
     * step 2: read pair (match, pos) in the block
     */
    const int elements_per_block = BLOCK_SIZE * NUM_INTS_PER_THREAD * 4 ;
    int base = gbid * elements_per_block ;

    #pragma unroll
    for( int j = 0 ; j < 4*NUM_INTS_PER_THREAD ; j++ ){
        int colIdx = tid + j * BLOCK_SIZE ;
        if ( colIdx < nnz ){
            match[j] = d_matched_result[base + colIdx];
            pos[j]   = d_pos[base + colIdx];
        } 
    }
    __syncthreads(); // necessary, make sure all threads have done reading
    
    /*
     * step 3: unlock semaphore after reading
     */
    if ( 0 == tid ) {    
        d_semaphore[gbid] = 1 ; // store directly goes to L2-cache, please see PTX document
        //atomicExch( &d_semaphore[gbid], 1) ;                   
    }
    
    //__syncthreads(); // not necessary
    
   
    /*
     * step 4: global sync point
     *      it can process if block of ID "w_start" and "w_end" complete reading
     *
     *  a thread block processes 1024 substrings, we decompose start and (end-1) by
     *
     *      start   = 1024 * w_start + q1 
     *      (end-1) = 1024 * w_end + q2
     *
     *  1) end > start because we have checked conidtion (0<nnz)
     *  2) either w_end = w_start or w_end = w_start+1 
     *      because nnz < 1024   
     *     
     *  3) gbid > w_start
     *     gbid = w_start if for all block j < gbid, block j has 1024 matched.
     *
     */

   /*
    *   refer to http://forums.nvidia.com/index.php?showtopic=98444&pid=548609&start=&st=#entry548609
    *   here atomic(lock,2,3) is to check value of lock, 2 adn 3 are meaningless because 
    *   lock is either 0 (means lock) or 1 (means unlock) 
    * 
    *   cons: 
    *       1) atomic operation is expensive
    *   pros: 
    *       1) code is simple
    *       2) no need to implement sleep function
    *   
    *   NOTE: atomicCAS can avoid compiler optimization. The following code does not work
    *
    *   [code]
    *       if ( 0 == tid ){
    *           while(0 == d_semaphore[w_start]);
    *       }     
    *       __syncthreads();
    *   [/code]
    *
    *   because compiler would optimize it as
    *   [code]
    *       if ( 0 == tid ){
    *           int reg = d_semaphore[w_start]
    *           while(0 == reg);
    *       }     
    *       __syncthreads();
    *   [/code]
    *   
    *   then deadlock occurs.
    *
    *   use volatile:
    *  [code]
    *  volatile int *w_start_ptr =  d_semaphore+w_start ;
    *  volatile int *w_end_ptr = d_semaphore+w_end ;
    *  if ( 0 == tid ){
    *    while(1){
    *     if ( 2 == (*w_start_ptr + *w_end_ptr) ){
    *         break ;
    *      }
    *    }
    *  }
    *  __syncthreads(); 
    *  [/code]
    * 
    */
    if ( 0 == tid ){
        while(0 == atomicCAS(d_semaphore+w_start, 2, 3));
    }
    __syncthreads();
    if ( (0 == tid ) && (w_start != w_end) ){
        while(0 == atomicCAS(d_semaphore+w_end, 2, 3));
    }
    __syncthreads(); // necessary for global sync

    /*
     * step 5: write data back because of no race condition now
     */
    #pragma unroll
    for( int j = 0 ; j < 4*NUM_INTS_PER_THREAD ; j++ ){
        int colIdx = tid + j * BLOCK_SIZE ;
        if ( colIdx < nnz ){
            d_matched_result[start + colIdx] = match[j] ;
            d_pos[start + colIdx] = pos[j]; 
        } 
    }    
    
}
    
// ------------------- space-driven version of stage 1


texture < int2, 1, cudaReadModeElementType > tex_hashRowPtr_reduce ;
texture < int2, 1, cudaReadModeElementType > tex_hashValPtr_reduce ;
texture < int , 1, cudaReadModeElementType > tex_tableOfInitialState_reduce ;

static __inline__  __device__ int tex_lookup(int state, int inputChar, const int hash_m, const int hash_p )
{ 
    int2 rowEle = tex1Dfetch(tex_hashRowPtr_reduce, state); // hashRowPtr[state]
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
       int2 valEle = tex1Dfetch(tex_hashValPtr_reduce, offset + pos); // hashValPtr[offset + pos];
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
    return tex1Dfetch(tex_tableOfInitialState_reduce, ch); 
}

 

template <int TEXTURE_ON , int SMEM_ON >
__global__ void PFAC_reduce_space_driven_device(
    int2 *d_hashRowPtr, int2 *d_hashValPtr, int *d_tableOfInitialState,
    const int hash_m, const int hash_p,
    int *d_input_string, int input_size, 
    int n_hat, int num_finalState, int initial_state, int num_blocks_minus1,
    int *d_pos, int *d_match_result, int *d_nnz_per_block );
    
/*
 *  stage 1: perform matching process and zip non-zero (matched thread) into continuous
 *      memory block and keep order. Morever nnz of each thread block is stored in d_nnz_per_block
 *
 *  d_nnz_per_block[j] = nnz of thread block j
 *
 *  since each thread block processes 1024 substrings, so range of d_nnz_per_block[j] is [0,1024] 
 */

__host__  PFAC_status_t PFAC_reduce_space_driven_stage1( PFAC_handle_t handle, 
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


    size_t offset ;
    /* always bind texture to tex_tableOfInitialState */
    // (3) bind texture to tex_tableOfInitialState
    textureReference *texRefTableOfInitialState ;
    cudaGetTextureReference( (const struct textureReference**)&texRefTableOfInitialState, "tex_tableOfInitialState_reduce" );

    cudaChannelFormatDesc channelDesc_tableOfInitialState = cudaCreateChannelDesc<int>();
    // set texture parameters
    tex_tableOfInitialState_reduce.addressMode[0] = cudaAddressModeClamp;
    tex_tableOfInitialState_reduce.addressMode[1] = cudaAddressModeClamp;
    tex_tableOfInitialState_reduce.filterMode     = cudaFilterModePoint;
    tex_tableOfInitialState_reduce.normalized     = 0;
    
    cuda_status = cudaBindTexture( &offset, (const struct textureReference*) texRefTableOfInitialState,
        (const void*) handle->d_tableOfInitialState, (const struct cudaChannelFormatDesc*) &channelDesc_tableOfInitialState,
        sizeof(int)*CHAR_SET ) ;

    if ( cudaSuccess != cuda_status ){
#ifdef DEBUG_MSG
        printf("Error: cannot bind texture to tableOfInitialState, %s\n", cudaGetErrorString(cuda_status) );
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
        cudaGetTextureReference( (const struct textureReference**)&texRefHashRowPtr, "tex_hashRowPtr_reduce" );

        cudaChannelFormatDesc channelDesc_hashRowPtr = cudaCreateChannelDesc<int2>();
    
        // set texture parameters
        tex_hashRowPtr_reduce.addressMode[0] = cudaAddressModeClamp;
        tex_hashRowPtr_reduce.addressMode[1] = cudaAddressModeClamp;
        tex_hashRowPtr_reduce.filterMode     = cudaFilterModePoint;
        tex_hashRowPtr_reduce.normalized     = 0;
    
        cuda_status = cudaBindTexture( &offset, (const struct textureReference*) texRefHashRowPtr,
            (const void*) handle->d_hashRowPtr, (const struct cudaChannelFormatDesc*) &channelDesc_hashRowPtr,
            sizeof(int2)*(handle->numOfStates) ) ;

        if ( cudaSuccess != cuda_status ){
#ifdef DEBUG_MSG
            printf("Error: cannot bind texture to hashRowPtr, %s\n", cudaGetErrorString(cuda_status) );
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
        cudaGetTextureReference( (const struct textureReference**)&texRefHashValPtr, "tex_hashValPtr_reduce" );

        cudaChannelFormatDesc channelDesc_hashValPtr = cudaCreateChannelDesc<int2>();
        // set texture parameters
        tex_hashValPtr_reduce.addressMode[0] = cudaAddressModeClamp;
        tex_hashValPtr_reduce.addressMode[1] = cudaAddressModeClamp;
        tex_hashValPtr_reduce.filterMode     = cudaFilterModePoint;
        tex_hashValPtr_reduce.normalized     = 0;
    
        cuda_status = cudaBindTexture( &offset, (const struct textureReference*) texRefHashValPtr,
            (const void*) handle->d_hashValPtr, (const struct cudaChannelFormatDesc*) &channelDesc_hashValPtr,
            handle->sizeOfTableInBytes ) ;
        if ( cudaSuccess != cuda_status ){
#ifdef DEBUG_MSG
            printf("Error: cannot bind texture to hashValPtr, %s\n", cudaGetErrorString(cuda_status) );
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
            PFAC_reduce_space_driven_device<1, 1> <<< dimGrid, dimBlock >>>( 
                handle->d_hashRowPtr, handle->d_hashValPtr, handle->d_tableOfInitialState,
                handle->hash_m, handle->hash_p, 
                d_input_string, input_size, n_hat, num_finalState, initial_state, num_blocks - 1,
                d_pos, d_match_result, d_nnz_per_block );
        }else{
            PFAC_reduce_space_driven_device<0, 1> <<< dimGrid, dimBlock >>>( 
                handle->d_hashRowPtr, handle->d_hashValPtr, handle->d_tableOfInitialState,
                handle->hash_m, handle->hash_p, 
                d_input_string, input_size, n_hat, num_finalState, initial_state, num_blocks - 1,
                d_pos, d_match_result, d_nnz_per_block );
        }
    }else{
        if ( texture_on ){
            PFAC_reduce_space_driven_device<1, 0> <<< dimGrid, dimBlock >>>( 
                handle->d_hashRowPtr, handle->d_hashValPtr, handle->d_tableOfInitialState,
                handle->hash_m, handle->hash_p, 
                d_input_string, input_size, n_hat, num_finalState, initial_state, num_blocks - 1,
                d_pos, d_match_result, d_nnz_per_block );
        }else{
            PFAC_reduce_space_driven_device<0, 0> <<< dimGrid, dimBlock >>>( 
                handle->d_hashRowPtr, handle->d_hashValPtr, handle->d_tableOfInitialState,
                handle->hash_m, handle->hash_p, 
                d_input_string, input_size, n_hat, num_finalState, initial_state, num_blocks - 1,
                d_pos, d_match_result, d_nnz_per_block );
        }
    }
    
    cuda_status = cudaGetLastError() ;
    if ( cudaSuccess != cuda_status ){
        cudaUnbindTexture(tex_tableOfInitialState_reduce);
        if ( texture_on ) { 
            cudaUnbindTexture(tex_hashRowPtr_reduce);
            cudaUnbindTexture(tex_hashValPtr_reduce);
        }
        return PFAC_STATUS_INTERNAL_ERROR ;
    }

    cudaUnbindTexture(tex_tableOfInitialState_reduce);
    if ( texture_on ){
        cudaUnbindTexture(tex_hashRowPtr_reduce);
        cudaUnbindTexture(tex_hashValPtr_reduce);
        
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
__global__ void PFAC_reduce_space_driven_device(
    int2 *d_hashRowPtr, int2 *d_hashValPtr, int *d_tableOfInitialState,
    const int hash_m, const int hash_p,
    int *d_input_string, int input_size, 
    int n_hat, int num_finalState, int initial_state, int num_blocks_minus1,
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
    // we always bind table of initial state to texture
    #pragma unroll
    for(int i = 0 ; i < BLOCK_SIZE_DIV_256 ; i++){
        phi_s02s1[ tid + i*BLOCK_SIZE ] = tex_loadTableOfInitialState(tid + i*BLOCK_SIZE); 
    }    

/*
    if ( TEXTURE_ON ){
        #pragma unroll
        for(int i = 0 ; i < BLOCK_SIZE_DIV_256 ; i++){
            phi_s02s1[ tid + i*BLOCK_SIZE ] = tex_loadTableOfInitialState(tid + i*BLOCK_SIZE); 
        }
    }else{
        #pragma unroll
        for(int i = 0 ; i < BLOCK_SIZE_DIV_256 ; i++){
            phi_s02s1[ tid + i*BLOCK_SIZE ] = d_tableOfInitialState[tid + i*BLOCK_SIZE]; 
        }    
    }
*/

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

