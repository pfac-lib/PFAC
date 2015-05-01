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

#ifndef PFAC_H_
#define PFAC_H_

#ifdef __cplusplus
extern "C" {
#endif   // __cplusplus


typedef enum {
    PFAC_PLATFORM_GPU = 0,  // default
    PFAC_PLATFORM_CPU = 1,
    PFAC_PLATFORM_CPU_OMP = 2
}PFAC_platform_t ;

typedef enum {
    PFAC_AUTOMATIC   = 0,  // default
    PFAC_TEXTURE_ON  = 1,
    PFAC_TEXTURE_OFF = 2
}PFAC_textureMode_t ;

typedef enum {
    PFAC_TIME_DRIVEN = 0, // default
    PFAC_SPACE_DRIVEN = 1 
}PFAC_perfMode_t;

/*
 *  The purpose of PFAC_STATUS_BASE is to separate CUDA error code and PFAC error code 
 *  but align PFAC_STATUS_SUCCESS to cudaSuccess.
 *
 *  cudaError_enum is defined in /usr/local/cuda/include/cuda.h
 *  The last one is 
 *      CUDA_ERROR_UNKNOWN                        = 999
 *
 *  That is why PFAC_STATUS_BASE = 10000 > 999
 *
 *  However now we regard all CUDA non-allocation error as PFAC_STATUS_INTERNAL_ERROR,
 *  PFAC_STATUS_BASE may be removed in the future 
 */
typedef enum {
    PFAC_STATUS_SUCCESS = 0 ,
    PFAC_STATUS_BASE = 10000, 
    PFAC_STATUS_ALLOC_FAILED,
    PFAC_STATUS_CUDA_ALLOC_FAILED,    
    PFAC_STATUS_INVALID_HANDLE,
    PFAC_STATUS_INVALID_PARAMETER, 
    PFAC_STATUS_PATTERNS_NOT_READY,
    PFAC_STATUS_FILE_OPEN_ERROR,
    PFAC_STATUS_LIB_NOT_EXIST,   
    PFAC_STATUS_ARCH_MISMATCH,
    PFAC_STATUS_INTERNAL_ERROR 
} PFAC_status_t ;

struct PFAC_context ;

typedef struct PFAC_context* PFAC_handle_t ;

/*
 *  return
 *  ------
 *  PFAC_STATUS_SUCCESS          if context is allocated successfully
 *  PFAC_STATUS_ALLOC_FAILED     if PFAC_context cannot be allocated, please check host memory usage
 *  PFAC_STATUS_ARCH_MISMATCH    if compute capability is 1.0, PFAC would use 32-bit atomic operation, 
 *                               which does not support sm1.0  
 *  PFAC_STATUS_LIB_NOT_EXIST    if environment variable LD_LIBRARY_PATH does not contain $(PFAC_LIB_ROOT)/lib
 *  PFAC_STATUS_INTERNAL_ERROR   please report bugs
 *
 */
PFAC_status_t  PFAC_create( PFAC_handle_t *handle ) ;

/*
 *  return
 *  ------
 *  PFAC_STATUS_SUCCESS          if context is de-allocated successfully
 *  PFAC_STATUS_INVALID_HANDLE   if "handle" is a NULL pointer
 *  
 */
PFAC_status_t  PFAC_destroy( PFAC_handle_t handle ) ;

/*
 *  return
 *  ------
 *  PFAC_STATUS_SUCCESS            if operation is successful
 *  PFAC_STATUS_INVALID_HANDLE     if "handle" is a NULL pointer
 *  PFAC_STATUS_INVALID_PARAMETER  if "platform" is not PFAC_GPU, PFAC_CPU or PFAC_CPU_OMP
 *  
 */
PFAC_status_t  PFAC_setPlatform( PFAC_handle_t handle, PFAC_platform_t platform ) ;

/*
 *  return
 *  ------
 *  PFAC_STATUS_SUCCESS            if operation is successful
 *  PFAC_STATUS_INVALID_HANDLE     if "handle" is a NULL pointer
 *  PFAC_STATUS_INVALID_PARAMETER  if "textureModeSel" is not PFAC_AUTOMATIC or PFAC_TEXTURE_ON or PFAC_TEXTURE_OFF
 *  PFAC_STATUS_CUDA_ALLOC_FAILED  if either texture binding fails or allocation of linear memory fails,
 *                                 please check resource of GPU by CUDA API cudaMemGetInfo() 
 *  PFAC_STATUS_INTERNAL_ERROR     please report bugs
 *
 */
PFAC_status_t  PFAC_setTextureMode( PFAC_handle_t handle, PFAC_textureMode_t textureModeSel ) ;


/*
 *  return
 *  ------
 *  PFAC_STATUS_SUCCESS            if operation is successful
 *  PFAC_STATUS_INVALID_HANDLE     if "handle" is a NULL pointer
 *  PFAC_STATUS_INVALID_PARAMETER  if "perfModeSel" is not PFAC_TIME_DRIVEN or PFAC_SPACE_DRIVEN
 *
 */
PFAC_status_t  PFAC_setPerfMode( PFAC_handle_t handle, PFAC_perfMode_t perfModeSel ) ;


/*
 *  return
 *  ------
 *  char * pointer to a NULL-terminated string. This is string literal, do not overwrite it.
 *
 */
const char* PFAC_getErrorString( PFAC_status_t status ) ;

/*
 *  return
 *  ------
 *  PFAC_STATUS_SUCCESS            if operation is successful
 *  PFAC_STATUS_INTERNAL_ERROR     please report bugs
 *
 */
PFAC_status_t  PFAC_dumpTransitionTable( PFAC_handle_t handle, FILE *fp ) ;

/*
 *  return
 *  ------
 *  PFAC_STATUS_SUCCESS             if operation is successful
 *  PFAC_STATUS_INVALID_HANDLE      if "handle" is a NULL pointer,
 *                                  please call PFAC_create() to create a legal handle
 *  PFAC_STATUS_INVALID_PARAMETER   if "filename" is a NULL pointer. 
 *                                  The library does not support patterns from standard input
 *  PFAC_STATUS_FILE_OPEN_ERROR     if file "filename" does not exist
 *  PFAC_STATUS_ALLOC_FAILED         
 *  PFAC_STATUS_CUDA_ALLOC_FAILED   if host (device) memory is not enough to parse pattern file.
 *                                  The pattern file is too large to allocate host(device) memory.
 *                                  Please split the pattern file into smaller and try again
 *  PFAC_STATUS_INTERNAL_ERROR      please report bugs
 *  
 */
PFAC_status_t  PFAC_readPatternFromFile( PFAC_handle_t handle, char *filename ) ;

/*
 *  return
 *  ------
 *  PFAC_STATUS_SUCCESS             if operation is successful
 *  PFAC_STATUS_INVALID_HANDLE      if "handle" is a NULL pointer
 *  PFAC_STATUS_INVALID_PARAMETER   if "d_inputString" or "d_matched_result" is a NULL pointer
 *  PFAC_STATUS_PATTERNS_NOT_READY  if patterns are not loaded first.
 *                                  Please use PFAC_readPatternFromFile() first
 *  PFAC_STATUS_INTERNAL_ERROR      please report bugs
 *  
 */
PFAC_status_t  PFAC_matchFromDevice( PFAC_handle_t handle, char *d_inputString, size_t size,
    int *d_matched_result ) ;


/*
 *  return
 *  ------
 *  PFAC_STATUS_SUCCESS             if operation is successful
 *  PFAC_STATUS_INVALID_HANDLE      if "handle" is a NULL pointer
 *  PFAC_STATUS_INVALID_PARAMETER   if "h_inputString" or "h_matched_result" is a NULL pointer
 *  PFAC_STATUS_PATTERNS_NOT_READY  if patterns are not loaded first.
 *                                  Please use PFAC_readPatternFromFile() first
 *  PFAC_STATUS_CUDA_ALLOC_FAILED   if device memory is not enough to allocate d_input string or d_matched result.
 *                                  Users must check memory usage of GPU. If the memory is not enough, 
 *                                  users must divide the input stream into small ones and call PFAC_matchFromHost()
 *                                  multiple times
 *  PFAC_STATUS_INTERNAL_ERROR      please report bugs
 *  
 */
PFAC_status_t  PFAC_matchFromHost( PFAC_handle_t handle, char *h_inputString, size_t size,
    int *h_matched_result ) ;

/*
 *  return
 *  ------
 *  The same as PFAC_matchFromDevice()
 */
PFAC_status_t  PFAC_matchFromDeviceReduce( PFAC_handle_t handle, char *d_inputString, size_t size,
    int *d_matched_result, int *d_pos, int *h_num_matched ) ;

/*
 *  return
 *  ------
 *  The same as PFAC_matchFromHost()
 */    
PFAC_status_t  PFAC_matchFromHostReduce( PFAC_handle_t handle, char *h_inputString, size_t size,
    int *h_matched_result, int *h_pos, int *h_num_matched ) ;



#ifdef __cplusplus
}
#endif   // __cplusplus


#endif   // PFAC_H_

