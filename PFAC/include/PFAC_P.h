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

#include <cuda_runtime.h>
#include <vector>
using namespace std ;

#ifndef PFAC_P_H_
#define PFAC_P_H_

#include "PFAC.h"

/*
 * debug mode:  PFAC_PRINTF( ... ) printf( __VA_ARGS__ )
 * release mode:  PFAC_PRINTF( ... ) 
 */
//#define PFAC_PRINTF( ... ) printf( __VA_ARGS__ )
//#define PFAC_PRINTF printf
#define PFAC_PRINTF(...) 

#define  FILENAME_LEN    256


#ifdef __cplusplus
extern "C" {
#endif   // __cplusplus

typedef PFAC_status_t (*PFAC_kernel_protoType)( PFAC_handle_t handle, char *d_input_string, size_t input_size,
    int *d_matched_result ) ;

typedef PFAC_status_t (*PFAC_reduce_kernel_protoType)( PFAC_handle_t handle, int *d_input_string, int input_size,
        int *d_match_result, int *d_pos, int *h_num_matched, int *h_match_result, int *h_pos ) ; 
        
#ifdef __cplusplus
}
#endif   // __cplusplus

typedef struct {
    int nextState ; 
    int ch ;
} TableEle ; 

/*
 *  suppose transistion table has S states, labelled as s0, s1, ... s{S-1}
 *  and Bj denotes number of valid transition of s{i}
 *  for each state, we use sj >= Bj^2 locations to contain Bj transistion.
 *  In order to avoid collision, we choose a value k and a prime p such that
 *  (k*x mod p mod sj) != (k*y mod p mod sj) for all characters x, y such that 
 *  (s{j}, x) and (s{j}, y) are valid transitions.
 *  
 *  Hash table consists of rowPtr and valPtr, similar to CSR format.
 *  valPtr contains all transitions and rowPtr[i] contains offset pointing to valPtr.
 *
 *  Element of rowPtr is int2, which is equivalent to
 *  typedef struct{
 *     int offset ;
 *     int k_sminus1 ;
 *  } 
 *
 *  sj is power of 2 and less than 256, and 0 < kj < 256, so we can encode (k,s-1) by a
 *  32-bit integer, k occupies most significant 16 bits and (s-1) occupies Least significant 
 *  16 bits.
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
 *  
 */

#define  HASH_KEY_K_MASK   0xFFFF0000
#define  HASH_KEY_K_MASKBITS   16
#define  HASH_KEY_S_MASK   0x0000FFFF


struct PFAC_context {
    // host
    char **rowPtr ; /* rowPtr[0:k-1] contains k pointer pointing to k patterns which reside in "valPtr"
                     * the order of patterns is sorted by lexicographic, say
                     *     rowPtr[i] < rowPtr[j]
                     *  if either rowPtr[i] = prefix of rowPtr[j] but length(rowPtr[i]) < length(rowPtr[j])
                     *     or \alpha = prefix(rowPtr[i])=prefix(rowPtr[j]) such that
                     *        rowPtr[i] = [\alpha]x[beta]
                     *        rowPtr[j] = [\aloha]y[gamma]
                     *     and x < y
                     *
                     *  pattern ID starts from 1 and follows the order of patterns in input file.
                     *  We record pattern ID in "patternID_table" and legnth of pattern in "patternLen_table".
                     *
                     *  for example, pattern rowPtr[0] = ABC, it has length 3 and ID = 5, then
                     *  patternID_table[0] = 5, and patternLen_table[5] = 3
                     *
                     *  WARNING: pattern ID starts by 1, so patternLen_table[0] is useless, in order to pass
                     *  valgrind, we reset patternLen_table[0] = 0
                     *
                     */
    char *valPtr ;  // contains all patterns, each pattern is terminated by null character '\0'
    int *patternLen_table ;
    int *patternID_table ;

    vector< vector<TableEle> > *table_compact;
    
    int  *h_PFAC_table ; /* explicit 2-D table */

    int2 *h_hashRowPtr ;
    int2 *h_hashValPtr ;
    int  *h_tableOfInitialState ;
    int  hash_p ; // p = 2^m + 1 
    int  hash_m ;

    // device
    int  *d_PFAC_table ; /* explicit 2-D table */

    int2 *d_hashRowPtr ;
    int2 *d_hashValPtr ;
    int  *d_tableOfInitialState ; /* 256 transition function of initial state */

    size_t  numOfTableEntry ; 
    size_t  sizeOfTableEntry ; 
    size_t  sizeOfTableInBytes ; // numOfTableEntry * sizeOfTableEntry
       
    // function pointer of non-reduce kernel under PFAC_TIME_DRIVEN
    PFAC_kernel_protoType  kernel_time_driven_ptr ;
    
    // function pointer of non-reduce kernel under PFAC_SPACE_DRIVEN
    PFAC_kernel_protoType  kernel_space_driven_ptr ;
    
    // function pointer of reduce kernel under PFAC_TIME_DRIVEN
    PFAC_reduce_kernel_protoType  reduce_kernel_ptr ;
    
    // function pointer of reduce kernel under PFAC_SPACE_DRIVEN
    PFAC_reduce_kernel_protoType  reduce_inplace_kernel_ptr ;

    int maxPatternLen ; /* maximum length of all patterns
                         * this number can determine which kernel is proper,
                         * for example, if maximum length is smaller than 512, then
                         * we can call a kernel with smem
                         */
                             
    int  max_numOfStates ; // maximum number of states, this is an estimated number from size of pattern file
    int  numOfPatterns ;  // number of patterns
    int  numOfStates ; // total number of states in the DFA, states are labelled as s0, s1, ..., s{state_num-1}
    int  numOfFinalStates ; // number of final states
    int  initial_state ; // state id of initial state

    int  numOfLeaves ; // number of leaf nodes of transistion table. i.e nodes without fan-out
                       // numOfLeaves <= numOfFinalStates
    
    int  platform ;
    
    int  perfMode ;
    
    int  textureMode ;
    
    bool isPatternsReady ;
    
    int device_no ; // = 10*deviceProp.major + deviceProp.minor ;
    
    char patternFile[FILENAME_LEN] ;
}  ;


#define  CHAR_SET    256
#define  TRAP_STATE  0xFFFFFFFF

/*
 *  output
 *  (1) reordered pattern and corresponding pattern ID
 *  (2) original order of patterns and their pattern length
 */
int dump_reorderPattern( char** rowPtr, int *patternID_table, int *patternLen_table,
    const int pattern_num, char *fileName ) ;


/*
 *  parse pattern file "patternFileName",
 *  (1) store all patterns in "patternPool" and
 *  (2) reorder the patterns according to lexicographic order and store
 *      reordered pointer in "rowPtr"
 *  (3) record original pattern ID in "patternID_table = *patternID_table_ptr"
 *  (4) record pattern length in "patternLen_table = *patternLen_table_ptr"
 *
 *  (5) *pattern_num_ptr = number of patterns
 *  (6) *max_state_num_ptr = estimation (upper bound) of total states in PFAC DFA
 *
 */
PFAC_status_t parsePatternFile( char *patternFileName, char ***rowPtr, char **patternPool,
    int **patternID_table_ptr, int **patternLen_table_ptr, int *max_state_num_ptr, int *pattern_num_ptr ) ;


//void printStringEndNewLine( char *s, FILE* fp = stdout );

void printString( char *s, const int n, FILE* fp );


PFAC_status_t  PFAC_memoryUsage( PFAC_handle_t handle );


#ifdef __cplusplus
extern "C" {
#endif   // __cplusplus
PFAC_status_t PFAC_tex_mutex_lock( void );

PFAC_status_t PFAC_tex_mutex_unlock( void );

#ifdef __cplusplus
}
#endif   // __cplusplus

#endif   // PFAC_P_H_


