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
 *  problem formulation:
 *  deploy m avaiable GPUs to process a huge input stream.
 *  m threads are created, each thread has a private PFAC context 
 *  which binds to a specific GPU.
 *
 *  each thread processes a sgement of input stream, the chunk size 
 *  of input stream is determined by 
 *        chunk size = min{device memory of all GPUs}/8 
 *  then all GPUs can work.
 *
 *  job scheduling is very simple, static with a fixed chunk size.
 *  
 *  WARNINH: LD_LIBRARY_PATH must contain dynamic module of pthread library,
 *      or segmentation fault occurs at run-time
 *      for example, /usr/lib64 on Fedora x86_64
 *
 */ 
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cuda_runtime.h>
#include <assert.h>
#include <ctype.h>
#include <vector>

using namespace std ;

#include "../include/PFAC.h"

#define FILENAME_MAXLEN     256

typedef struct {
    char patternFile[FILENAME_MAXLEN];
    char inputFile[FILENAME_MAXLEN];
    bool isTextureOn ;
    int  debug ;
} optsPara ;


void showHelp(void)
{
    printf("useage: [bin] -P [pattern] -I [input file] -T -D\n");
    printf("-T : texture mode\n");
    printf("-D : debug mode, print any mismatch between different GPUs\n");
}

void processCommandOption( int argc, char** argv, optsPara *opts)
{
    if ( 2 > argc ){
        showHelp();
        exit(1) ;
    }
    argc-- ;
    argv++ ;
    for( ; argc > 0 ; argc-- , argv++){
        if ( '-' != argv[0][0] ){ continue ; }
            switch( argv[0][1] ){
            case 'I' : // input file
                if ( '\0' != argv[0][2] ){
                    fprintf(stderr, "Error: %s is not -I\n", argv[0]);
                    exit(1);
                }
                argc-- ;
                argv++ ;
                if ( 0 >= argc ){
                    fprintf(stderr, "Error: miss input file after option -I\n");
                    exit(1) ;
                }

                if ( FILENAME_MAXLEN > strlen(argv[0]) ){
                    strcpy( opts->inputFile, argv[0]) ;
                }else{
                    fprintf(stderr, "Error: length of input file name is greater than %d \n",
                        FILENAME_MAXLEN );
                    exit(1) ;
                }
                break ;
            case 'P' :
                if ( '\0' != argv[0][2] ){
                    fprintf(stderr, "Error: %s is not -P\n", argv[0]);
                    exit(1);
                }
                argc-- ;
                argv++ ;
                if ( 0 >= argc ){
                    fprintf(stderr, "Error: miss pattern file after option -P\n");
                    exit(1) ;
                }
                if ( FILENAME_MAXLEN > strlen(argv[0]) ){
                    strcpy( opts->patternFile, argv[0]) ;
                }else{
                    fprintf(stderr, "Error: length of pattern file name is greater than %d \n",
                        FILENAME_MAXLEN );
                    exit(1) ;
                }
                break ;
            case 'T' :
                opts->isTextureOn = 1 ;
                break ;
            case 'D' :
                opts->debug = 1 ;
                break ;
            default:
                fprintf(stderr, "Error: unknown options %s\n", argv[0] );
                showHelp();
                exit(1);
        }// switch
    }

    if ( 0 == strlen( opts->patternFile) ){
        fprintf(stderr, "Error: patttern file must be specified \n");
        exit(1) ;
    }
    if ( 0 == strlen( opts->inputFile) ){
        fprintf(stderr, "Error: input file must be specified \n");
        exit(1) ;
    }
}

#define   MAX_THREADS    8
#define   imin( a , b ) ((a)<(b))?(a):(b)
#define   imax( a , b ) ((a)>(b))?(a):(b)

int  parsePatterns( char *patternfilename, int *num_patterns, int *max_patternLen );
    
int main(int argc, char **argv)
{
    optsPara opts ;
    FILE *fpin = NULL;
    int input_size ;
    char *h_inputString = NULL ;
    int  *h_matched_result ;
    int  *ref_matched_result ;
    double  totalGlobalMem[MAX_THREADS] ;

    memset( &opts, 0, sizeof(optsPara) ) ;
    processCommandOption( argc, argv, &opts) ;

   /*  check available GPUs
    *  
    *  suppose there are m GPUs, then m threads are created, 
    *  each thread has a private PFAC context and binds to a GPU.
    * 
    *  To activate all GPUs, we must check available device memory of 
    *  each GPU and determine a maximum size of input stream such that
    *  all GPUs can work well.
    *
    */
    int num_gpus = 0;     
    cudaGetDeviceCount(&num_gpus);
    if( 1 > num_gpus ) {
        printf("Error: no CUDA capable devices were detected\n");
        exit(1);
    }

    printf("number of host CPUs:\t%d\n", omp_get_num_procs());
    printf("number of CUDA devices:\t%d\n", num_gpus );
    for(int i = 0; i < num_gpus; i++){
        cudaDeviceProp dprop;
        cudaGetDeviceProperties(&dprop, i);
        totalGlobalMem[i] = (double)dprop.totalGlobalMem ;
        printf("   %d: %s, (major,minor)=(%d,%d), totalGloalMem = %6.0f MB\n", 
            i, dprop.name, dprop.major, dprop.minor, totalGlobalMem[i]/1024./1024. );
    }
    printf("---------------------------\n");

    // launch num_gpus threads, each thread binds to one GPU
    num_gpus = imin( num_gpus , MAX_THREADS );
    printf("@@@@ Launch %d threads to %d GPUs\n", num_gpus, num_gpus );
   
    /*
     *  suppose M = minimum device memory of all GPUs, and each GPU should compute 
     *  input stream of N bytes and output matched result of N*4 bytes, then 
     *  we require
     *         5*N < M 
     *  Here we choose N = basic_unit = M / 8 because some graphic card may 
     *  connect to a display. 
     * 
     */
    // minGlobalMem = minimum of device memory of all GPUs
    double minGlobalMem = totalGlobalMem[0] ;
    for(int i = 1; i < num_gpus; i++){
    	  if ( minGlobalMem > totalGlobalMem[i] ){
    	  	  minGlobalMem = totalGlobalMem[i] ;
    	  }
    }
    int basic_unit = ((int)minGlobalMem) >> 3 ;
    printf("minGlobalMem = %6.0f MB, basic unit of input stream = %d bytes\n", 
        minGlobalMem/1024./1024. , basic_unit );   
       
   /*
    *  prepare input stream on host memory, this is global visible for all threads.
    *  each thread will process a segment of input stream.
    */
    fpin = fopen( opts.inputFile, "rb");
    assert( NULL != fpin ) ;
    
    fseek (fpin , 0 , SEEK_END); 
    input_size = ftell (fpin); // obtain file size
    rewind (fpin);
      
    // allocate memory to contain the whole file
    h_inputString = (char *) malloc (sizeof(char)*input_size);
    assert ( NULL != h_inputString );

    // copy the file into the buffer
    input_size = fread (h_inputString, 1, input_size, fpin);
    fclose(fpin);

    printf("size of input stream is %d bytes (%6.0f MB)\n", input_size, 
        ((double)input_size)/1024. /1024. );    

   /*
    *  find maximum length of patterns 
    *  This step is crucial because each thread only cover a segment of input stream,
    *  and additional tail should append to this segment or boundary effect cannot be 
    *  resolved.
    */
    printf("\n ---------- parse pattern file --------------- \n");
    int max_patternLen ;
    int num_patterns ;
    parsePatterns( opts.patternFile, &num_patterns, &max_patternLen ) ;
    
    printf("num_patterns = %d, maximum pattern length = %d\n", num_patterns,  max_patternLen );    
    
   /* 
    *  prepare global matched result, h_matched_result[]
    *  each thread processes a sgement of input stream and then 
    *  write partial result into a segment of h_matched_result[]
    */
    h_matched_result = (int *) malloc(sizeof(int)*input_size);
    assert( NULL != h_matched_result );
   
   
    printf("\n ------ run multi-GPU (threads = %d)---------- \n", num_gpus );
    
    omp_set_num_threads(num_gpus);   // create as many CPU threads as there are CUDA devices
   
    #pragma omp parallel shared(num_gpus, opts, h_inputString, input_size, max_patternLen )
    {
        PFAC_handle_t handle ;
        PFAC_status_t PFAC_status ;
    	  cudaError_t cuda_status ;
        int   max_size_per_input ; 
        char *d_input_string = NULL ;
        int  *d_matched_result = NULL ;
        char msg[256]; /* collect output data into buffer "msg", then send to output buffer by 
                        * printf().  
                        */
         	  
        unsigned int tid = omp_get_thread_num();
        unsigned int num_threads = omp_get_num_threads();
        
       /*
        *  OpenMP will create "num_gpus" threads, thread ID starts from 0.
        *
        *  bind a thread with ID "tid" to a GPU with ID "tid" 
        *
        */
        assert( tid < num_gpus ) ;
        
        cuda_status = cudaSetDevice(tid);
        assert( cudaSuccess == cuda_status );
        
        sprintf(msg,"thread %d (of %d) binds CUDA device\n", tid, num_threads);
        printf("%s", msg);
        
       /* 
        *  each thread of ID "tid" creates a private PFAC context which binds to GPU "tid"
        *  
        *  WARNING: if command line option -T is specified, then texture mode is configured.
        *  however if PFAC library cannot bind texture memory, then it returns error code.
        *  at this time, the thread dies and final result is wrong. 
        */
        PFAC_status = PFAC_create( &handle ) ;
        assert(PFAC_STATUS_SUCCESS == PFAC_status);

        PFAC_status = PFAC_setPlatform(handle, PFAC_PLATFORM_GPU ) ;
        assert(PFAC_STATUS_SUCCESS == PFAC_status);

        if (opts.isTextureOn ){
            PFAC_status = PFAC_setTextureMode( handle, PFAC_TEXTURE_ON ) ;
        }else{
            PFAC_status = PFAC_setTextureMode( handle, PFAC_TEXTURE_OFF ) ;
        }
        if ( PFAC_STATUS_SUCCESS != PFAC_status ){
            sprintf(msg, "Error: tid %d fails to setTextureMode, %s\n", tid, PFAC_getErrorString(PFAC_status));
            printf("%s", msg);
            exit(1) ;
        }
        
        /*
         *  all threads read the same pattern file 
         */
        PFAC_status = PFAC_readPatternFromFile(handle, opts.patternFile) ;
        assert(PFAC_STATUS_SUCCESS == PFAC_status);
        
        /*
         *  every GPU can accept an input stream of size "basic_unit" bytes,
         *  each thread deals with one small input stream of size "basic_unit" bytes,
         *  however extra "max_patternLen + 1" bytes must be appended in the tail 
         *  in order to cover overlapping area.
         *  Hence a thread needs to process "basic_unit + max_patternLen + 1"  characters
         *  
         */
        max_size_per_input = basic_unit + max_patternLen + 1 ;
        
        cuda_status = cudaMalloc((void **) &d_input_string, max_size_per_input );
        if ( cudaSuccess != cuda_status ){
            sprintf(msg, "Error: tid %d cannot allocate d_input_string\n", tid );
            printf("%s", msg);
            exit(1) ;
        }        
        cuda_status = cudaMalloc((void **) &d_matched_result, max_size_per_input*sizeof(int) );
        if ( cudaSuccess != cuda_status ){
            sprintf(msg, "Error: tid %d cannot allocate d_matched_result\n", tid );
            printf("%s", msg);
            exit(1) ;
        }
        
        /*
         *  adopt static scheduling
         *  suppose input stream has N bytes, and B = basic_unit, then input stream is 
         *  divided into M = (N+B-1)/B segments. 
         *  suppose number of threads is 4, then static scheduling means
         *  thread 0 processes segments 0, 4,  8, 12, ...
         *  thread 1 processes segments 1, 5,  9, 13, ...
         *  thread 2 processes segments 2, 6, 10, 14, ...
         *  thread 2 processes segments 3, 7, 11, 15, ...
         *
         *  formula:  tid processes segment (tid + k * num_threads) for k = 0, 1, 2, ...
         */       
        for( int start = tid * basic_unit ; start < input_size ; start += num_threads * basic_unit ){
        	  int end   = start + basic_unit ;
        	  int guard = end + max_patternLen + 1 ;
            end   = imin(  end, input_size );
            guard = imin(guard, input_size );
            
            sprintf(msg, "tid %d processes input[%d, %d) \n", tid, start, end );
            printf("%s", msg);
                
            int num_chars = guard - start ;
            // copy input string from host to device
            cuda_status = cudaMemcpy(d_input_string, h_inputString + start, num_chars, cudaMemcpyHostToDevice);
            if ( cudaSuccess != cuda_status ){
                sprintf(msg, "Error: tid %d h_input_string --> d_input_string fails \n", tid );
                printf("%s", msg);
                exit(1) ;
            }   
            
            PFAC_status = PFAC_matchFromDevice( handle, d_input_string, num_chars, d_matched_result ) ;
            if ( PFAC_STATUS_SUCCESS != PFAC_status ){
                sprintf(msg, "Error: tid %d PFAC_matchFromDevice fails, %s\n", tid, PFAC_getErrorString(PFAC_status));
                printf("%s", msg);
                exit(1) ;
            }            
            
            // copy the result data from device to host
            cuda_status = cudaMemcpy(h_matched_result + start, d_matched_result, (end-start)*sizeof(int), cudaMemcpyDeviceToHost);
            if ( cudaSuccess != cuda_status ){
                sprintf(msg, "Error: tid %d d_matched_result --> h_matched_result fails, %s \n", tid, cudaGetErrorString(cuda_status) );
                printf("%s", msg);
                exit(1);
            }                
        }

        sprintf(msg, "tid %d is done\n", tid );
        printf("%s", msg);
        
        cudaFree(d_input_string);
        cudaFree(d_matched_result);
                
        PFAC_status = PFAC_destroy( handle) ;
        assert(PFAC_STATUS_SUCCESS == PFAC_status);
        
    }// end omp parallel

    // reference model
    PFAC_handle_t handle ;
    PFAC_status_t PFAC_status ;
            
    PFAC_status = PFAC_create( &handle ) ;
    assert(PFAC_STATUS_SUCCESS == PFAC_status);

    PFAC_status = PFAC_setPlatform(handle, PFAC_PLATFORM_GPU ) ;
    assert(PFAC_STATUS_SUCCESS == PFAC_status);
    
    PFAC_status = PFAC_setTextureMode( handle, PFAC_TEXTURE_OFF ) ;
    assert(PFAC_STATUS_SUCCESS == PFAC_status);
    
    PFAC_status = PFAC_readPatternFromFile(handle, opts.patternFile) ;
    assert(PFAC_STATUS_SUCCESS == PFAC_status);
    
    ref_matched_result = (int *) malloc(sizeof(int)*input_size);
    assert( NULL != ref_matched_result );
        
    PFAC_status = PFAC_matchFromHost( handle, h_inputString, input_size, ref_matched_result ) ;       
    assert( NULL != ref_matched_result ); 
    
    printf("\n ------ compare matched result to reference model ------\n");
     
    int num_mismatched = 0 ;
    int num_matched = 0 ;
    for ( int i = 0 ; i < input_size ; i++){
        int a = ref_matched_result[i] ;
        int b = h_matched_result[i] ;
        if ( a != b ){
            if ( opts.debug ){
                fprintf(stdout, "At position %4d, ref_matched_result(%d) != h_matched_result(%d)\n", i, a, b );
            }
            num_mismatched++ ;
        }
        if ( 0 < a ){
            num_matched++ ;
        }
    }    
    if ( 0 == num_mismatched ){
        printf("\t pass: number of matched = %d\n", num_matched );
    }else{
        printf("\t Error: mismatch %d \n", num_mismatched );
    }    
 
    // free resource
    free(h_inputString);
    free(h_matched_result);
    free(ref_matched_result);
   
    cudaThreadExit();

    return 0;
}

static void printStringEndNewLine( char *s, FILE* fp )
{
    if ( '\n' == *s ) { return ; }
    fprintf(fp,"%c", '\"');
    while( 1 ){
        int ch = (unsigned char) *s++ ;
        if ( '\n' == ch ){ break ;}
        if ( (32 <= ch) && (126 >= ch) ){
            fprintf(fp,"%c", ch );
        }else{
            fprintf(fp,"%2.2x", ch );
        }
    }
    fprintf(fp,"%c", '\"');
}

int  parsePatterns( char *patternfilename, 
    int *num_patterns, int *max_patternLen )
{
    assert( NULL != num_patterns ) ;
    assert( NULL != max_patternLen );
	  
    FILE* fpin = NULL ;
    int file_size ;
    char *buffer = NULL ;
    vector<char*> rowIdxArray ;
    vector<int>   patternLenArray ;
    int len ;
    	  
    fpin = fopen(patternfilename, "rb");
    assert (NULL != fpin ) ;

    // step 1: find size of the file
    fseek (fpin , 0 , SEEK_END);
    file_size = ftell (fpin);
    rewind (fpin);

    // step 2: allocate a buffer to contains all patterns
    buffer = (char*)malloc(sizeof(char)*file_size ) ;
    assert ( NULL != buffer );

    // copy the file into the buffer
    file_size = fread (buffer, 1, file_size, fpin);
    fclose(fpin);

    rowIdxArray.push_back( buffer ) ;
    len = 0 ;
    for( int i = 0 ; i < file_size ; i++){
        if ( '\n' == buffer[i] ){
            if ( i > 0 && '\n' != buffer[i-1] ){ // non-empty line
                patternLenArray.push_back( len ) ;
 
                printStringEndNewLine( rowIdxArray.back(), stdout );
                printf(" ,length = %d\n", len );
 
                rowIdxArray.push_back( buffer + i + 1) ;
            }
            len = 0 ;
        }else{
            len++ ;
        }
    }
    
    *num_patterns = rowIdxArray.size()-1 ;
    
    *max_patternLen = 0 ;	
    for(int i = 0 ; i < *num_patterns ; i++){
        *max_patternLen = imax(*max_patternLen, patternLenArray[i]) ;
    }	

    free( buffer );
    return 0 ;
}



