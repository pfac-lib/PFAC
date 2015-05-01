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
 *  two threads bind to two different GPUs
 *
 *  thread 0 binds GPU 0, thread 1 binds GPU 1 if there are more than two GPUs.
 *
 *  thread 0 processes (pattern 0, input 0)
 *
 *  thread 1 processes (pattern 1, input 1)
 *
 */

 
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <pthread.h>
#include <cuda_runtime.h>

#include <PFAC.h>

#define  FILENAME_MAXLEN   256
#define  NUM_THREADS       2


typedef struct {
    char patternFile[FILENAME_MAXLEN];
    char inputFile[FILENAME_MAXLEN];
    char dumpMatchedFile[FILENAME_MAXLEN] ;
    char dumpTableFile[FILENAME_MAXLEN] ;
    int  tid;
} optsPara ;

void *GPU_match(void *para)
{
    optsPara opts;
    PFAC_handle_t handle ;
    PFAC_status_t PFAC_status ;
    int input_size ;    
    char *h_inputString = NULL ;
    int  *h_matched_result = NULL ;
    int gpu_id = -1;   // GPU ID
    int num_gpus = 0;  // number of CUDA GPUs
    cudaError_t cuda_status ;
        
    opts = *((optsPara*)para) ;
    
    // search for proper GPUs
    cudaGetDeviceCount(&num_gpus);
    if( 1 > num_gpus ) {
        printf("Error: no CUDA capable devices were detected\n");
        exit(1) ;	
    }
    if ( 1 == num_gpus ){
    	  cuda_status = cudaSetDevice(0);
    }else{
        cuda_status = cudaSetDevice(opts.tid);	
    }
    assert( cudaSuccess == cuda_status );    
        
    cuda_status = cudaGetDevice(&gpu_id);
    assert( cudaSuccess == cuda_status );
    
    cudaDeviceProp dprop;
    cudaGetDeviceProperties(&dprop, gpu_id);
    
    char msg[256] ;                
    sprintf(msg, "thread %d uses CUDA device %d: %s, (major,minor)=(%d,%d)\n", 
        opts.tid, gpu_id, dprop.name, dprop.major, dprop.minor );
    printf("%s", msg );    

    // step 1: create PFAC handle 
    PFAC_status = PFAC_create( &handle ) ;
    if ( PFAC_STATUS_SUCCESS != PFAC_status ){
        printf("tid[%d]:  Error: fails to create PFAC handle, %s\n", opts.tid, PFAC_getErrorString(PFAC_status) );
        exit(1) ;	
    }
    
    // step 2: read patterns and dump transition table 
    PFAC_status = PFAC_readPatternFromFile( handle, opts.patternFile) ;
    if ( PFAC_STATUS_SUCCESS != PFAC_status ){
        printf("tid[%d]:  Error: fails to read pattern from file, %s\n", opts.tid, PFAC_getErrorString(PFAC_status) );
        exit(1) ;	
    }
    
    // dump transition table 
    FILE *table_fp = fopen( opts.dumpTableFile, "w") ;
    if ( NULL == table_fp ){
        printf("tid[%d]:  Error: open table file failed\n", opts.tid );
        exit(1) ;	
    }
    PFAC_status = PFAC_dumpTransitionTable( handle, table_fp );
    fclose( table_fp ) ;
    if ( PFAC_STATUS_SUCCESS != PFAC_status ){
        printf("tid[%d]:  Error: fails to dump transition table, %s\n", opts.tid, PFAC_getErrorString(PFAC_status) );
        exit(1) ;	
    }
   
    //step 3: prepare input stream
    FILE* fpin = fopen( opts.inputFile, "rb");
    if ( NULL == fpin ){
        printf("tid[%d]:  Error: open input file failed\n", opts.tid );
        exit(1) ;	
    }
    
    // obtain file size
    fseek (fpin , 0 , SEEK_END);
    input_size = ftell (fpin);
    rewind (fpin);
    
    // allocate memory to contain the whole file
    h_inputString = (char *) malloc (sizeof(char)*input_size);
    if ( NULL == h_inputString ){
        printf("tid[%d]:  Error: memory allocation of input string failed\n", opts.tid );
        exit(1) ;	
    }
 
    h_matched_result = (int *) malloc (sizeof(int)*input_size);
    if ( NULL == h_matched_result ){
        printf("tid[%d]:  Error: memory allocation of matched result failed\n", opts.tid );
        exit(1) ;	
    }
    memset( h_matched_result, 0, sizeof(int)*input_size ) ;
     
    // copy the file into the buffer
    input_size = fread (h_inputString, 1, input_size, fpin);
    fclose(fpin);    
    
    // step 4: run PFAC on GPU           
    PFAC_status = PFAC_matchFromHost( handle, h_inputString, input_size, h_matched_result ) ;
    if ( PFAC_STATUS_SUCCESS != PFAC_status ){
        printf("tid[%d]:  Error: fails to PFAC_matchFromHost, %s\n", opts.tid, PFAC_getErrorString(PFAC_status) );
        exit(1) ;	
    }     

    // step 5: output matched result
    FILE *output_fp = fopen( opts.dumpMatchedFile, "w") ;
    if ( NULL == output_fp ){
        printf("tid[%d]:  Error: open output file failed\n", opts.tid );
        exit(1) ;	
    }
    for (int i = 0; i < input_size; i++) {
        if (h_matched_result[i] != 0) {
            fprintf(output_fp, "At position %4d, match pattern %d\n", i, h_matched_result[i]);
        }
    }

    PFAC_status = PFAC_destroy( handle ) ;
    if ( PFAC_STATUS_SUCCESS != PFAC_status ){
        printf("tid[%d]:  Error: fails to destroy PFAC handle, %s\n", opts.tid, PFAC_getErrorString(PFAC_status) );
        exit(1) ;	
    }
    
    free(h_inputString);
    free(h_matched_result); 
            
    pthread_exit((void*)para);
}


int main(int argc, char **argv)
{
    pthread_t threads[NUM_THREADS];
    int rc;
    void *status;
    optsPara opts[NUM_THREADS] ;
    
    // initialize options
    memset( opts, 0, sizeof(opts) ) ;
    
    // setting optiolns
    sprintf( opts[0].patternFile, "../test/pattern/example_pattern" ) ;
    sprintf( opts[1].patternFile, "../test/pattern/example_pattern2" ) ;
    sprintf( opts[0].inputFile,   "../test/data/example_input" ) ;
    sprintf( opts[1].inputFile,   "../test/data/example_input2" ) ;
    sprintf( opts[0].dumpMatchedFile, "match1" ) ;
    sprintf( opts[1].dumpMatchedFile, "match2" ) ;
    sprintf( opts[0].dumpTableFile, "table1" ) ;
    sprintf( opts[1].dumpTableFile, "table2" ) ;
    opts[0].tid = 0 ;
    opts[1].tid = 1 ;
    
    // create threads to run GPU match
    for(int i = 0; i < NUM_THREADS; i++){
        rc = pthread_create(&threads[i], NULL, GPU_match, (void *)(&opts[i]));
        if (rc) {
            printf("ERROR; return code from pthread_create() is %d\n", rc);
            exit(-1);
        }
    }
    
    // merge threads
    for(int i = 0; i < NUM_THREADS; i++){
        rc = pthread_join(threads[i], (void **)(&status));
        if (rc) {
            printf("ERROR; return code from pthread_join() is %d\n", rc);
            exit(-1);
        }
    }
    
    for(int i = 0; i < NUM_THREADS; i++){
        printf("thread %d processes \n", i);
        printf("\t pattern file \"%s\" \n", opts[i].patternFile );	
        printf("\t input file \"%s\" \n", opts[i].inputFile );	
        printf("\t dump matched result to file \"%s\" \n", opts[i].dumpMatchedFile );	
        printf("\t dump transition table to file \"%s\" \n", opts[i].dumpTableFile );	
    }
    
    return 0;
}



