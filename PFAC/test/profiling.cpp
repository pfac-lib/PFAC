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
 *  report timing, 4 combinations chosen from command line options
 *
 *  1) PFAC_matchFromDevice() + texture ON 
 *  2) PFAC_matchFromDevice() + texture OFF
 *  3) PFAC_matchFromHost() + texture ON 
 *  4) PFAC_matchFromHost() + texture OFF
 * 
 *  Note: time-efficient version is used.
 *  
 */
 
 
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cuda_runtime.h>
#include <assert.h>
#include <ctype.h>

#include "../include/PFAC.h"
 
#define FILENAME_MAXLEN     256


template< int HOST_ON >
void profile_PFAC( PFAC_handle_t handle, char *h_inputString, int input_size,
    int *h_matched_result, bool isTextureOn );

typedef struct {
    char patternFile[FILENAME_MAXLEN];
    char inputFile[FILENAME_MAXLEN];
    int  deviceID ;
    bool isTextureOn ;
    int  timeOnHost ;
} optsPara ;


void showHelp(void)
{
    printf("useage: [bin] -P [pattern] -I [input file] -G[GPU id] -t -TH\n");
    printf("default of [GPU id] is 0 \n");
    printf("-t : texture mode\n");
    printf("-TH : timing on host, including data transfer \n");
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
            case 'G':
                opts->deviceID = argv[0][2] - '0';
                break ;
            case 't' :
                opts->isTextureOn = 1 ;
                break ;
            case 'T' :
                if ( 'D' == argv[0][2] ){
                    opts->timeOnHost = 0 ;
                }else if ( 'H' == argv[0][2] ){
                    opts->timeOnHost = 1 ;
                }else{
                    fprintf(stderr, "Error: %s is not -TH or -TD\n", argv[0]);
                    exit(1);
                }
                break ;
            default:
                fprintf(stderr, "Error: unknown options %s\n", argv[0] );
                showHelp();
                exit(1);
        } // switch
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

int main(int argc, char **argv)
{
    cudaError_t cuda_status;
    int input_size ;
    char *h_inputString = NULL ;
    int  *h_matched_result = NULL ;

    PFAC_handle_t handle ;
    PFAC_status_t PFAC_status ;

    FILE *fpin = NULL;
    optsPara opts ;

    memset( &opts, 0, sizeof(optsPara) ) ;
    processCommandOption( argc, argv, &opts) ;

    // step 1: initialize GPU explicitly
    cuda_status = cudaSetDevice( opts.deviceID ) ;
    assert( cudaSuccess == cuda_status);

    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, opts.deviceID);
    printf("Using Device %d: \"%s\"\n", opts.deviceID, deviceProp.name);
    printf("major = %d, minor = %d\n", deviceProp.major, deviceProp.minor );

    PFAC_status = PFAC_create( &handle ) ;
    assert(PFAC_STATUS_SUCCESS == PFAC_status);

    PFAC_status = PFAC_setPlatform( handle, PFAC_PLATFORM_GPU ) ;
    assert(PFAC_STATUS_SUCCESS == PFAC_status);

    PFAC_status = PFAC_readPatternFromFile( handle, opts.patternFile) ;
    if (PFAC_STATUS_SUCCESS != PFAC_status){
        printf("Error: fails to read pattern from file, %s\n", PFAC_getErrorString(PFAC_status) );
        exit(1) ;
    }
    
    // step 2: read input data
    fpin = fopen( opts.inputFile, "rb");
    assert(NULL != fpin);
    // obtain file size
    fseek (fpin , 0 , SEEK_END);
    input_size = ftell (fpin);
    rewind (fpin);

    // allocate memory to contain the whole file
    h_inputString = (char *) malloc (sizeof(char)*input_size);
    assert(NULL != h_inputString);

    h_matched_result = (int *) malloc (sizeof(int)*input_size);
    assert(NULL != h_matched_result);

    // copy the file into the buffer
    input_size = fread (h_inputString, 1, input_size, fpin);
    fclose(fpin);

    // step 3: run on GPU
    memset( h_matched_result, 0, sizeof(int)*input_size ) ;
   
    if ( opts.timeOnHost ){
        profile_PFAC<1>( handle, h_inputString, input_size, h_matched_result, opts.isTextureOn );
    }else{
        profile_PFAC<0>( handle, h_inputString, input_size, h_matched_result, opts.isTextureOn );
    }

    PFAC_status = PFAC_destroy( handle) ;
    assert(PFAC_STATUS_SUCCESS == PFAC_status);

    free( h_inputString ) ;
    free( h_matched_result ) ;

    cudaThreadExit();

    return 0;
}


template< int HOST_ON >
void profile_PFAC( PFAC_handle_t handle, char *h_inputString, int input_size,
    int *h_matched_result, bool isTextureOn )
{
    PFAC_status_t  PFAC_status ;
    cudaError_t cuda_status ;
    int n_hat ;
    char *d_input_string = NULL ;
    int  *d_matched_result = NULL ;
    char msg[256];
    
    if (HOST_ON){
    	  strcpy( msg, "PFAC_matchFromHost + " ) ; 
    }else{
    	  strcpy( msg, "PFAC_matchFromDevice + ") ;
    }
    if ( isTextureOn ){
    	 strcat( msg, "texture ON") ;
    }else{
    	 strcat( msg, "texture OFF") ;
    }
    
    printf("\n@@@@ profile %s \n", msg );	
    
    if (isTextureOn ){
        PFAC_status = PFAC_setTextureMode( handle, PFAC_TEXTURE_ON ) ;
    }else{
        PFAC_status = PFAC_setTextureMode( handle, PFAC_TEXTURE_OFF ) ;     
    }
    if ( PFAC_STATUS_SUCCESS != PFAC_status ){
        printf("Error: fails to PFAC_setTextureMode, %s\n", PFAC_getErrorString(PFAC_status));
        exit(1) ;
    }

    if ( !HOST_ON ){
        // allocate memory for input string and result
        // basic unit of d_input_string is integer
        // n_hat = number of integers of input string
        n_hat = (input_size + sizeof(int)-1)/sizeof(int) ;
        cuda_status = cudaMalloc((void **) &d_input_string, n_hat*sizeof(int) );
        if ( cudaSuccess != cuda_status ){
            printf("Error: cannot allocate d_input_string\n");
            exit(1) ;
        }
        
        cuda_status = cudaMalloc((void **) &d_matched_result, input_size*sizeof(int) );
        if ( cudaSuccess != cuda_status ){
            printf("Error: cannot allocate d_matched_result\n");
            exit(1) ;
        }
        
        // copy input string from host to device
        cuda_status = cudaMemcpy(d_input_string, h_inputString, input_size, cudaMemcpyHostToDevice);
        if ( cudaSuccess != cuda_status ){
            printf("Error: h_input_string --> d_input_string fails \n");
            exit(1) ;
        }
    }
	  
    // record time setting
    cudaEvent_t start, stop;
    float time;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);

    if (HOST_ON){
        PFAC_status = PFAC_matchFromHost( handle, h_inputString, input_size,
            h_matched_result ) ;
    }else{
        PFAC_status = PFAC_matchFromDevice( handle, d_input_string, input_size,
            d_matched_result ) ;
    }
    
    if ( PFAC_STATUS_SUCCESS != PFAC_status ){
        printf("Error: %s\n", PFAC_getErrorString(PFAC_status) );
        exit(1) ;
    }
    
    // record time setting
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time, start, stop);
    
    printf("The elapsed time is %f ms\n", time );
    printf("The input size is %d bytes\n", input_size );
    printf("The throughput is %f Gbps\n", (float)(input_size*8)/(time*1000000) );

    if (!HOST_ON){
        // copy the result data from device to host
        cuda_status = cudaMemcpy(h_matched_result, d_matched_result, input_size*sizeof(int), cudaMemcpyDeviceToHost);
        if ( cudaSuccess != cuda_status ){
            printf("Error: d_matched_result --> h_matched_result fails, %s \n", cudaGetErrorString(cuda_status) );
            exit(1);
        }
    }

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    // report number of matched
    int num_matched = 0 ;
    for(int i = 0 ; i < input_size ; i++){
        if ( 0 < h_matched_result[i] ){
            num_matched++ ;	
        }	
    }
    printf("The number of matched is %d \n", num_matched );

    if (!HOST_ON){
        // show memory usage of GPU
        size_t free_byte ;
        size_t total_byte ;
        cuda_status = cudaMemGetInfo( &free_byte, &total_byte ) ;
        if ( cudaSuccess != cuda_status ){
            printf("Error: cudaMemGetInfo fails, %s \n", cudaGetErrorString(cuda_status) );
            exit(1);
        }
        double free_db  = (int)free_byte ;
        double total_db = (int)total_byte ;
        double used_db  = total_db - free_db ;
        
        printf("\n@@@@ report GPU memory usage \n");
        printf("\t total = %6.0f MB, used = %6.0f MB, free = %6.0f MB\n",
            total_db/1024.0/1024.0, used_db/1024.0/1024.0, free_db/1024.0/1024.0 );

        cudaFree(d_input_string);
        cudaFree(d_matched_result);
    }
	
}
