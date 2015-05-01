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

#ifndef PFAC_P_H_
#define PFAC_P_H_

#include "PFAC.h"

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
    
    int *PFAC_table ;
    
    int maxPatternLen ; /* maximum length of all patterns
                         * this number can determine which kernel is proper,
                         * for example, if maximum length is smaller than 512, then
                         * we can call a kernel with smem
                         */

    // device
    int *d_PFAC_table   ;
    
    cudaArray *d_PFAC_table_array ;
    cudaChannelFormatDesc channelDesc ;
    
    // function pointer of texture version and non-texture version
    PFAC_kernel_protoType  kernel_ptr ;
    
    // function pointer of reduce function under PFAC_TIME_DRIVEN
    PFAC_reduce_kernel_protoType  reduce_kernel_ptr ;
    
    // function pointer of reduce function under PFAC_SPACE_DRIVEN
    PFAC_reduce_kernel_protoType  reduce_inplace_kernel_ptr ;
    
    int  max_state_num ; // maximum number of states
    int  pattern_num ;   // number of patterns
    int  state_num ;     // total number of states in the DFA
    int  num_finalState ; // number of final states
    int  initial_state ; // state id of initial state

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
PFAC_status_t create_PFACTable_reorder(const char** rowPtr, const int *patternLen_table, const int *patternID_table,
    const int max_state_num, const int pattern_num, const int initial_state, int *state_num_ptr,
    int *PFAC_table ) ;

PFAC_status_t  PFAC_CPU(char *input_string, int input_size,
    int *PFAC_table,
    int num_finalState, int initial_state,
    int *match_result) ;

PFAC_status_t  PFAC_CPU_OMP(char *input_string, int input_size,
    int *PFAC_table,
    int num_finalState, int initial_state,
    int *match_result) ;

void printStringEndNewLine( char *s, FILE* fp = stdout );

#endif   // PFAC_P_H_


