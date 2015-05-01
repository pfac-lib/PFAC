# What is PFAC?
*PFAC* is an open library for exact string matching performed on *GPUs*. PFAC runs on processors that support *CUDA*, including NVIDIA 1.1, 1.2, 1.3, 2.0 and 2.1 architectures. Supporting OS includes ubuntu, Fedora and MAC OS.

PFAC library provides C-style API and users need not have background on GPU computing or parallel computing. PFAC has APIs hiding CUDA stuff. 

# News

 * PFAC r1.0 updated 2011/02/23
 * PFAC r1.0 released 2011/02/21
 * PFAC r1.1 released 2011/04/27
 * PFAC r1.2 released 2012/04/29

# Simple Example

## Example 1: Using PFAC_matchFromHost function
The file "example_pattern" in the directory "../test/pattern/" contains 4 patterns.
```
AB
ABG
BEDE
ED
```
The file "example_input" in the directory "../test/data/"contains a string.
```
ABEDEDABG
```

The following example shows the basic steps to use PFAC library for string matching.
```
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
```

The screen shows the following matched results.
```
At position    0, match pattern 1
At position    1, match pattern 3
At position    2, match pattern 4
At position    4, match pattern 4
At position    6, match pattern 2
``` 

## Example 2: Using PFAC_matchFromDevice function
Prepare the same input file and pattern file. The screen shows the same matched results as  example 1.
```
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <cuda_runtime.h>
#include <PFAC.h>
        
int main(int argc, char **argv)
{
    char dumpTableFile[] = "table.txt" ;	  
    char inputFile[] = "../test/data/example_input" ;
    char patternFile[] = "../test/pattern/example_pattern" ;
    PFAC_handle_t handle ;
    PFAC_status_t PFAC_status ;
    int input_size ;    
    char *h_input_string = NULL ;
    int  *h_matched_result = NULL ;

    char *d_input_string;
    int *d_matched_result;
    cudaError_t status; 

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
    h_input_string = (char *) malloc (sizeof(char)*input_size);
    assert( NULL != h_input_string );
 
    // allocate memory to contain the matched results
    h_matched_result = (int *) malloc (sizeof(int)*input_size);
    assert( NULL != h_matched_result );
    memset( h_matched_result, 0, sizeof(int)*input_size ) ;
     
    // copy the file into the buffer
    input_size = fread (h_input_string, 1, input_size, fpin);
    fclose(fpin); 
        
    // compute the size of for stroing the matched results
    int data_size = sizeof(int) * input_size;   
        
    // allocate GPU memory for input string 
    status = cudaMalloc((void **) &d_input_string, input_size);
    if ( cudaSuccess != status ){
        fprintf(stderr, "Error: %s\n", cudaGetErrorString(status));
        exit(1) ;
    }
    
    // allocate GPU memory for matched result
    status = cudaMalloc((void **) &d_matched_result, data_size);
    if ( cudaSuccess != status ){
        fprintf(stderr, "Error: %s\n", cudaGetErrorString(status));
        exit(1) ;
    }
    
    // copy input string from host to device
    cudaMemcpy(d_input_string, h_input_string, input_size, cudaMemcpyHostToDevice);
          
    // step 4: run PFAC on GPU by calling PFAC_matchFromDevice          
    PFAC_status = PFAC_matchFromDevice( handle, d_input_string, input_size, d_matched_result ) ;
    if ( PFAC_STATUS_SUCCESS != PFAC_status ){
        printf("Error: fails to PFAC_matchFromHost, %s\n", PFAC_getErrorString(PFAC_status) );
        exit(1) ;	
    }     
    
    // copy the result data from device to host
    cudaMemcpy(h_matched_result, d_matched_result, data_size, cudaMemcpyDeviceToHost);

    // step 5: output matched result
    for (int i = 0; i < input_size; i++) {
        if (h_matched_result[i] != 0) {
            printf("At position %4d, match pattern %d\n", i, h_matched_result[i]);
        }
    }

    PFAC_status = PFAC_destroy( handle ) ;
    assert( PFAC_STATUS_SUCCESS == PFAC_status );
    
    free(h_input_string);
    free(h_matched_result); 
    cudaFree(d_input_string);
    cudaFree(d_matched_result);
            
    return 0;
}
```

Refer to the user guide page for further information and examples.

# Publications

1. Cheng-Hung Lin, Chen-Hsiung Liu, Lung-Sheng Chien, Shih-Chieh Chang, "Accelerating Pattern Matching Using a Novel Parallel Algorithm on GPUs," IEEE Transactions on Computers, vol. 62, no. 10, pp. 1906-1916, Oct. 2013, doi:10.1109/TC.2012.254
2. Cheng-Hung Lin, Sheng-Yu Tsai, Chen-Hsiung Liu, Shih-Chieh Chang, and Jyuo-Min Shyu, "Accelerating String Matching Using Multi-threaded Algorithm on GPU,"  in Proc. of IEEE GLOBAL COMMUNICATIONS CONFERENCE (*GLOBECOM*), Miami, Florida, USA, December 6-10, pp.1-5, 2010.

# Contributors
The primary developers of PFAC are *Lung-Sheng Chien*, *Chen-Hsiung Liu*, *Cheng-Hung Lin*, and *Shih-Chieh Chang*.
Website maintainer *Chun-Cheng Huang*
