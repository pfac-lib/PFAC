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
#include <sys/time.h>

#include "../include/PFAC_P.h"

#define  PFAC_TABLE_MAP( i , j )   (i)*CHAR_SET + (j)

PFAC_status_t  PFAC_CPU_timeDriven(char *input_string, const int input_size,
    int *PFAC_table,
    int num_finalState, int initial_state,
    int *match_result);
    
PFAC_status_t  PFAC_CPU_spaceDriven(char *input_string, const int input_size,
    int2 *hashRowPtr, int2 *hashValPtr, const int hash_p,
    int num_finalState, int initial_state,
    int *match_result);

/*
 *  Let F denote number of final states, labelled from s1, s2, ..., s{F}
 *  then we requires intial state is s{F+1} or higher.
 *  i.e. state s0 is of no use.
 *
 */
PFAC_status_t  PFAC_CPU( PFAC_handle_t handle, char *h_input_string, const int input_size, int *h_matched_result )
{
    if ( handle->numOfFinalStates >= handle->initial_state ){
        return PFAC_STATUS_INTERNAL_ERROR ;
    }
    	
	  if (PFAC_TIME_DRIVEN == handle->perfMode) {
        return PFAC_CPU_timeDriven(h_input_string, input_size, handle->h_PFAC_table,
            handle->numOfFinalStates, handle->initial_state, h_matched_result) ;   
    }else{
        return PFAC_CPU_spaceDriven(h_input_string, input_size,
            handle->h_hashRowPtr, handle->h_hashValPtr, handle->hash_p,
            handle->numOfFinalStates, handle->initial_state, h_matched_result );
    }
}


PFAC_status_t  PFAC_CPU_timeDriven(char *input_string, const int input_size,
    int *PFAC_table,
    int num_finalState, int initial_state,
    int *match_result)
{
    int start;
    int pos; // position to read input for the thread
    int state;
    int inputChar;
    int match_pattern = 0;

    // initialize match result on CPU
    for (pos = 0; pos < input_size; pos++) {
        match_result[pos] = 0;
    }

    for (start=0; start < input_size; start++) {
    	
        state = initial_state;
        pos = start;
        while ( (pos < input_size) ) {
            // read input character
            inputChar =(unsigned char)input_string[pos];
            
            state = PFAC_table[ PFAC_TABLE_MAP(state,inputChar)];

            if ( TRAP_STATE == state ){ break ; }

            // output match pattern
            if(state <= num_finalState ){
                match_pattern = state;
                match_result[start] = match_pattern;
            }

            pos = pos + 1;
        }
    }

    return PFAC_STATUS_SUCCESS ;

}


PFAC_status_t  PFAC_CPU_spaceDriven(char *input_string, const int input_size,
    int2 *hashRowPtr, int2 *hashValPtr, const int hash_p,
    int num_finalState, int initial_state,
    int *match_result)
{
    int start;
    int pos; // position to read input for the thread
    int state;
    int inputChar;
    int match_pattern = 0;

    // initialize match result on CPU
    for (pos = 0; pos < input_size; pos++) {
        match_result[pos] = 0;
    }

    for (start = 0; start < input_size; start++) {
    	
        state = initial_state;
        pos = start;
        while ( (pos < input_size) ) {
            // read input character
            inputChar =(unsigned char)input_string[pos];
            
            //state = PFAC_table[ PFAC_TABLE_MAP(state,inputChar)];
            // fetch next state in hash table
            int2 rowEle = hashRowPtr[state];
            int offset  = rowEle.x ;
            if ( 0 > offset ){ // offset = -1
                state = TRAP_STATE ;
            }else{
            	  int k_sminus1 = rowEle.y ;
                int sminus1 = k_sminus1 & HASH_KEY_S_MASK ;
                int k = k_sminus1 >> HASH_KEY_K_MASKBITS ; 
                int pos = ( ( k * inputChar ) % hash_p ) &  sminus1 ;
                int2 valEle = hashValPtr[offset + pos];
                int nextState = valEle.x ;
                int ch = valEle.y ;
                if ( inputChar == ch ){
                    state = nextState ;
                }else{
                    state = TRAP_STATE ;	
                }
            }
            // END fetch next state in hash table

            if ( TRAP_STATE == state ){ break ; }

            // output match pattern
            if(state <= num_finalState ){
                match_pattern = state;
                match_result[start] = match_pattern;
            }

            pos = pos + 1;
        }
    }

    return PFAC_STATUS_SUCCESS ;

}

