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
 * The example shows following operations:
 *
 * 1) Including the header file PFAC.h which resides in directory $(PFAC_LIB_ROOT)/include.
 *    This header file is necessary because it contains declaration of APIs.
 *
 * 2) Initializing the PFAC library by creating a PFAC handle 
 *    (PFAC binds to a GPU context implicitly. If an user wants to bind a specific GPU, 
 *    he must call cudaSetDevice() explicitly before calling PFAC_create() ).
 *
 * 3) Reading patterns from a file and PFAC would create transition table both on the CPU side and the GPU side.
 *
 * 4) Dumping transition table to "table.txt", the content of table is shown in Figure 1 of user guide.
 *
 * 5) Reading an input stream from a file.
 *
 * 6) Performing matching process by calling the PFAC_matchFromHost() function.
 *
 * 7) Showing matched results.
 *
 * 8) Destroying the PFAC handle.
 *
 *
 */
 
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
 
#include <PFAC.h>
        
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

    // step 1: create PFAC handle 
    PFAC_status = PFAC_create( &handle ) ;
    assert( PFAC_STATUS_SUCCESS == PFAC_status );
    
    // step 2: read patterns and dump transition table 
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
   
    //step 3: prepare input stream
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
     
    // copy the file into the buffer
    input_size = fread (h_inputString, 1, input_size, fpin);
    fclose(fpin);    
    
    // step 4: run PFAC on GPU           
    PFAC_status = PFAC_matchFromHost( handle, h_inputString, input_size, h_matched_result ) ;
    if ( PFAC_STATUS_SUCCESS != PFAC_status ){
        printf("Error: fails to PFAC_matchFromHost, %s\n", PFAC_getErrorString(PFAC_status) );
        exit(1) ;	
    }     

    // step 5: output matched result
    for (int i = 0; i < input_size; i++) {
        if (h_matched_result[i] != 0) {
            printf("At position %4d, match pattern %d\n", i, h_matched_result[i]);
        }
    }

    PFAC_status = PFAC_destroy( handle ) ;
    assert( PFAC_STATUS_SUCCESS == PFAC_status );
    
    free(h_inputString);
    free(h_matched_result); 
            
    return 0;
}


