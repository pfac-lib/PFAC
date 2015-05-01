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
#include <cuda_runtime.h>
#include <assert.h>
#include <ctype.h>

#include "../include/PFAC_P.h"

//#define DEBUG_MSG

/*
 *  we have two versions of global synchronizations in kernel PFAC_reduce_inplace_kernel
 *
 *  1) global memory access + sleep function
 *     however it incurs spinlock at random, we set a timeout to detect this phenomenon, 
 *     if spinlock occurs, then kernel return PFAC_STATUS_INTERNAL_ERROR.
 *
 *     SPINLOCK_CHECK will generate this code
 *
 *     we keep this code because it is interesting in spinlock. Assembly code is confirmed that
 *     comoiler nvcc does not do something strange, but at runtime, spinlock occurs.
 *
 *  2) atomic operation, refer to http://forums.nvidia.com/index.php?showtopic=98444&pid=548609&start=&st=#entry548609
 *     so far, it does not hang the app. So it should be good enough.
 *
 *     default setting is 2)
 *
 */
//#define  SPINLOCK_CHECK 

#ifdef __cplusplus
extern "C" {

    PFAC_status_t PFAC_reduce_inplace_kernel( PFAC_handle_t handle, int *d_input_string, int input_size,
        int *d_match_result, int *d_pos, int *h_num_matched, int *h_match_result, int *h_pos ); 
        
}
#endif // __cplusplus

#define  BLOCK_EXP             (7)
#define  BLOCK_SIZE            (1 << BLOCK_EXP)
#define  NUM_INTS_PER_THREAD   (2)


__host__  PFAC_status_t PFAC_reduce_kernel_stage1( PFAC_handle_t handle, 
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
     *  if macro SPINLOCK_CHECK is set, then zip_inplace_kernel is compiled by
     *  a another version which may incur spinlock at random.
     *  if spinlock occurs, then d_spinlock = last block ID which has spinlock.
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
    
#ifdef SPINLOCK_CHECK    
    // check if spinlock occurs
    int h_spinlock ;
    cuda_status = cudaMemcpy( &h_spinlock, d_spinlock, sizeof(int), cudaMemcpyDeviceToHost);
    if (cudaSuccess != cuda_status) {
#ifdef DEBUG_MSG     
        printf("Error: h_spinlock,  %s\n", cudaGetErrorString(cuda_status));
#endif        
        return PFAC_STATUS_INTERNAL_ERROR ;
    }
    if ( 0 !=  h_spinlock ){
#ifdef DEBUG_MSG 
        printf("Error: spinlock occurs at block ID %d \n", h_spinlock );
#endif        
        return PFAC_STATUS_INTERNAL_ERROR ;
    }
#endif    
   
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
 *  spinlock occurs when global synchronization happens.
 *
 *  in our app, this may occur when there are more than one thread blocks in a SM.
 *  (in fact, atomic operation should be the right way to avoid spinlock)
 *
 *  suppose block b100 needs to wait for block b31, 
 *  i.e
 *    if b31 completes reading, then b100 can continue writing data to (d_matched_result, d_pos)
 *
 *  b100 and b31 may be in different SMs. 
 *
 *  suppose there are 8 blocks per SM and each block process 1024 elements, 
 *  i.e. read 1024 (matched,pos) pair.
 *
 *  Question: how many clocks should b100 wait at most if no spinlock occurs ?
 *
 *  one SM processes 1024 x 8 = 2K (matched, pos) pair in coalesced access.
 *  so 2K x 2 / 16 = 256 memory transactions.
 *
 *  assume 256 memory transactions are in the pipeline, then
 *  256 cycles are enough to issue 256 LOAD instructions.
 *
 *  However memory bandwidth is much smaller than performance of ALU.
 * 
 *  bandwidth ~ 100GB/s ~ 25 G LOAD / s
 *  ALU: 515 G instruction/s  on C2050
 *
 *  so perf. of bandwidth is 1/20th of perf. of ALU.
 *
 *  so latency of 256 memory transactions is
 *    500 (lastency of first read) + 256 * 20 < 5K cycles.
 *
 *  There are 14 SMs in Fermi and 30 SMs in Tesla C1060.
 *
 *  maximum time waiting for all thread blocks is 30 * 5K cycles
 *
 *  This is upper bound, so we choose TIMEOUT_BOUND > 150*K.
 *
 *  NOTE: TIMEOUT_BOUND is only used when  macro SPINLOCK_CHECK is set 
 *
 */ 
 
#define  TIMEOUT_BOUND   (1 << 20)


/*
 *  requirement:
 *      initial value of d_atomicBlockID is 1  (must be set by caller)
 *
 *  basic idea: special type of global synchronization
 *      use atomicAdd(d_atomicBlockID,1) to fetch gbid, such that 
 *      if a threads block has gbid k, then for all block j , j < k has 
 *      been processed by some thread blocks.
 *       
 *
 *  Resource usage:
 *
 *  sm20:
 *      SPINLOCK_CHECK is on : 28 regs, 16 bytes smem  => 1024 threads per SM
 *      SPINLOCK_CHECK is off: 29 regs, 4  bytes smem  => 1024 threads per SM
 *  sm13, sm12, sm11:
 *      SPINLOCK_CHECK is on : 28 regs, 64+16 bytes smem => 512 threads per SM
 *      SPINLOCK_CHECK is off: 29 regs, 56+16 bytes smem => 512 threads per SM 
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
            //atomicExch( d_semaphore + gbid, 1) ;            
        }                   
        return ;  
    }
    // now, nnz >= 1 or say end >= (start + 1)
    
#ifdef SPINLOCK_CHECK 
    volatile __shared__ int w_smem[2] ;
    if ( 0 == tid ){
        w_smem[0] = w_start ;
        w_smem[1] = w_end ;
    }
#endif
     

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
        d_semaphore[gbid] = 1 ;
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

#ifdef SPINLOCK_CHECK 

    // this version will incur spinlock, I think that this is problem of cache coherence.
    // how fast that data in L1 cache will be refreshed.
    int timeout = 4 ; 
    if ( 0 == tid ){ 
        while(1){
            // sleeping function to avoid spinlock
            int guard = 1 ;
            for(int j = 0 ; j < timeout ; j++ ){
                guard = guard ^ j ;  
            }
            guard = (guard & 1) + 1 ;
            timeout += guard * timeout ; 
             
            if ( TIMEOUT_BOUND < timeout ){
                *d_spinlock = gbid ; // spinlock occurs, matched result may be wrong           
                break ;
            }
            volatile int s1 = d_semaphore[ w_smem[0] ] ;
            volatile int s2 = d_semaphore[ w_smem[1] ] ;    
            if ( 2 == (s1+s2) ){ 
                break ;
            }
        }// while     
    }
     __syncthreads(); // necessary for global sync

#else

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
    */
    if ( 0 == tid ){
        while(0 == atomicCAS(d_semaphore+w_start, 2, 3));
    }
    __syncthreads();
    if ( (0 == tid ) && (w_start != w_end) ){
        while(0 == atomicCAS(d_semaphore+w_end, 2, 3));
    }
    __syncthreads(); // necessary for global sync

#endif


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
    

 

/*

technical notes of zip_inplace_kernel

-----------------------------------------------------------------------
compiler nvcc removes "sleep function" 

[code]
    __syncthreads();
    int timeout = 2 ;    
    if ( tid < 2 ){
        while(1){
            volatile int s1 = d_semaphore[ w_smem[0] ] ;
            volatile int s2 = d_semaphore[ w_smem[1] ] ;
            if ( 2 == (s1+s2) ){ 
                break ; 
            }
            // sleeping function to avoid spinlock
            int j = 0 ;
            while(1){
                if ( j > timeout){
                    timeout *= 2 ;
                    break ;
                }
                j++ ;
            }
            timeout = (timeout & (TIMEOUT_BOUND-1)) + 2 ;
        }
    }
    __syncthreads(); // necessary for global sync
[code]    

becomes

[code]
    __syncthreads();
    
    if ( tid < 2 ){
        while(1){
            volatile int s1 = d_semaphore[ w_smem[0] ] ;
            volatile int s2 = d_semaphore[ w_smem[1] ] ;
            if ( 2 == (s1+s2) ){ 
                break ; 
            }
        }
    }
    __syncthreads(); // necessary for global sync
[code]    

-----------------------------------------------------------------------------

The following code hangs on Fermi on some benchmarks and is reproducible. 
However from assembly code generated by cuobjdump on sm13, 
compiler does wrong translation even "volatile" is declared.

[code] 
    volatile __shared__ int w_smem[2] ;
    volatile __shared__ int s_smem[2] ;
    
    __syncthreads(); 
    int timeout = 2 ;    
    if ( tid < 2 ){
        int s_idx = w_smem[tid];
        while(1){
            s_smem[tid] = d_semaphore[ s_idx ] ;
            if ( 2 == (s_smem[0] + s_smem[1]) ){ 
                break ; 
            }
            // sleeping function to avoid spinlock
            int j = 0 ;
            while(1){
                if ( j > timeout){
                    timeout *= 2 ;
                    break ;
                }
                j++ ;
            }
            s_smem[tid] = 0 ;
            timeout = (timeout & (TIMEOUT_BOUND-1)) + 2 ;
        }
    }
    __syncthreads(); 
[/code]

becomes 

[code]

    __syncthreads();     
    if ( tid < 2 ){
        int s_idx = w_smem[tid];
        int R0 = d_semaphore[ s_idx ] ;  // wrong translation
                 // because d_semaphore cannot be updated  
        while(1){
            s_smem[tid] = R0 ;
            if ( 2 == (s_smem[0] + s_smem[1]) ){ 
                break ; 
            }
            s_smem[tid] = 0 ;
        }
    }
    __syncthreads(); 

[/code]

-----------------------------------------------------------------------------

possible sleep function
[code]
            // sleeping function to avoid spinlock
            for(int j = 0 ; j < timeout ; j++ ){
                s1 += j ;  
            }
            if ( s1 > timeout ){
                timeout *=2 ;
            }
            timeout = (timeout & (TIMEOUT_BOUND-1)) + 2 ;
[/code]

The following code dispears after optimization
[code]
            int j = 0 ;
            while(1){
                if ( j > timeout){
                    timeout *= 2 ;
                    break ;
                }
                j++ ;
            }
            timeout = (timeout & (TIMEOUT_BOUND-1)) + 2 ;
[/code]

--------------------------------------------------------------------------

This form is not good, if we interchange reading of (s1, s2) and sleep function,
then spinlock is relaxed, why?
[code]
    int timeout = 4 ;
    if ( 0 == tid ){    
        while(1){
            volatile int s1 = d_semaphore[ w_smem[0] ] ;
            volatile int s2 = d_semaphore[ w_smem[1] ] ;
            if ( 2 == (s1+s2) ){ 
                break ; 
            }
            // sleeping function to avoid spinlock
            for(int j = 0 ; j < timeout ; j++ ){
                s1 += j ;  
            }
            if ( s1 > timeout ){
                timeout *= 2 ;
            }
            if ( TIMEOUT_BOUND < timeout ){
                *d_spinlock = gbid ; // spinlock occurs, matched result may be wrong
                break ;
            }
        }
    }
[/code]

*/    


 