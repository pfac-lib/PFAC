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
 *  same as simple_example.cpp but running space-efficient PFAC_matchFromHostReduce() 
 *  
 */
 
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <cuda_runtime.h>

#include "../include/PFAC.h"
        
int main(int argc, char **argv)
{
    char dumpTableFile[] = "table.txt" ;	  
    char inputFile[] = "../test/data/example_input" ;
    char patternFile[] = "../test/pattern/example_pattern" ;
    PFAC_handle_t handle ;
    PFAC_status_t PFAC_status ;
    int input_size ;    
    char *h_inputString = NULL ;
    int  *h_matched_result = NULL ;
    int  *h_pos = NULL ;
    int  h_num_matched = 0 ;
   
    // step 1: initialize GPU explicitly
    int deviceID = 0 ;
    cudaError_t cuda_status = cudaSetDevice( deviceID ) ; 
    assert( cudaSuccess == cuda_status );
   
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, deviceID);
    printf("Using Device %d: \"%s\"\n", deviceID, deviceProp.name);
    
    // step 2: create PFAC handle 
    PFAC_status = PFAC_create( &handle ) ;
    assert( PFAC_STATUS_SUCCESS == PFAC_status );
    
    // step 3: read patterns and dump transition table 
    PFAC_status = PFAC_readPatternFromFile( handle, patternFile) ;
    if ( PFAC_STATUS_SUCCESS != PFAC_status ){
        printf("Error: fails to read pattern from file, %s\n", PFAC_getErrorString(PFAC_status) );
        exit(1) ;	
    }
    
    // dump transition table 
    FILE *table_fp = fopen( dumpTableFile, "w") ;
    assert( NULL != table_fp ) ;
    PFAC_status = PFAC_dumpTransitionTable( handle, table_fp );
    fclose( table_fp ) ;
    if ( PFAC_STATUS_SUCCESS != PFAC_status ){
        printf("Error: fails to dump transition table, %s\n", PFAC_getErrorString(PFAC_status) );
        exit(1) ;	
    }
   
    //step 4: prepare input stream
    FILE* fpin = fopen( inputFile, "rb");
    assert ( NULL != fpin ) ;
    
    // obtain file size
    fseek (fpin , 0 , SEEK_END);
    input_size = ftell (fpin);
    rewind (fpin);
    
    // allocate memory to contain the whole file
    h_inputString = (char *) malloc (sizeof(char)*input_size);
    assert( NULL != h_inputString );
 
    h_matched_result = (int *) malloc (sizeof(int)*input_size);
    assert( NULL != h_matched_result );
    memset( h_matched_result, 0, sizeof(int)*input_size ) ;

    h_pos = (int *) malloc (sizeof(int)*input_size);
    assert( NULL != h_pos );
    memset( h_pos, 0, sizeof(int)*input_size ) ;
          
    // copy the file into the buffer
    input_size = fread (h_inputString, 1, input_size, fpin);
    fclose(fpin);    
    
    // step 5: run space-efficient version
    PFAC_status = PFAC_setPerfMode( handle, PFAC_SPACE_DRIVEN );
    assert( PFAC_STATUS_SUCCESS == PFAC_status );
          
    PFAC_status = PFAC_matchFromHostReduce( handle, h_inputString, input_size, h_matched_result, h_pos, &h_num_matched ) ;
    if ( PFAC_STATUS_SUCCESS != PFAC_status ){
        printf("Error: fails to PFAC_matchFromHost, %s\n", PFAC_getErrorString(PFAC_status) );
        exit(1) ;	
    }     

    // step 6: output matched result
    printf("number of matched = %d\n", h_num_matched );
    for (int i = 0; i < h_num_matched; i++) {
        printf("At position %4d, match pattern %d\n", h_pos[i], h_matched_result[i]);
    }
    
    PFAC_status = PFAC_destroy( handle ) ;
    assert( PFAC_STATUS_SUCCESS == PFAC_status );
    
    free(h_inputString);
    free(h_matched_result);
    free(h_pos);
    
    cudaThreadExit(); 
    
    return 0;
}


