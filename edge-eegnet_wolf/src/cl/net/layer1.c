/**
 * @file layer1.c
 * @author Tibor Schneider
 * @date 2020/01/29
 * @brief This file contains the Implementation for the first layer
 */

/*
 * Copyright (C) 2020 ETH Zurich. All rights reserved.
 *
 * Author: Tibor Schneider, ETH Zurich
 *
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the License); you may
 * not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an AS IS BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */


#include "rt/rt_api.h"
#include "layers.h"
#include "net.h"
#include "../func/functional.h"

#ifdef PARALLEL

#ifndef NUM_WORKERS
#define NUM_WORKERS 8
#endif

typedef struct
{
    int8_t* p_data;    // pointer to entire data vector on L1
    int8_t* p_weight;  // pointer to entire weight vector on L1
    int32_t* p_factor; // pointer to all factors on L1
    int32_t* p_offset; // pointer to all offsets on L1
    int8_t* p_thread_result; // pointer to thread local data
    int8_t* p_result;  // pointer to result on L2
} _net_layer1_kernel_t;

/**
 * @brief Layer1 kernel convolves all output channels
 */
void _net_layer1_kernel(void* args) {

    // get core id
    unsigned int core_id = rt_core_id();

    // extract parameters
    int8_t* _p_data = ((_net_layer1_kernel_t*)args)->p_data;
    int8_t* _p_weight = ((_net_layer1_kernel_t*)args)->p_weight;
    int32_t* _p_factor = ((_net_layer1_kernel_t*)args)->p_factor;
    int32_t* _p_offset = ((_net_layer1_kernel_t*)args)->p_offset;
    int8_t* _p_result = ((_net_layer1_kernel_t*)args)->p_result;

    int8_t* _p_data_iter;
    int8_t* _p_weight_iter;
    int8_t* _p_result_filter;
    
    int32_t _factor;
    int32_t _offset;

    unsigned int _filter = core_id;
    unsigned int _k, _ch;
    
    int32_t _elem;


    // loop until all filters are computed
    while (_filter < NET_F1) {

        _p_weight_iter = _p_weight + _filter * NET_C_ALIGN;
        _factor = _p_factor[_filter];
        _offset = _p_offset[_filter];

        _p_result_filter = _p_result + _filter * NET_T_ALIGN;
        
        for(int _T = 0; _T < NET_T; _T++) {  // _T stands for the current time sample
            _p_data_iter = _p_data + _T * NET_C_ALIGN; // always reuse data for different filter
            
            _elem = func_dotp(_p_data_iter, _p_weight_iter, NET_C_ALIGN);
            _elem = (_elem + _offset) / _factor;

            // clamp
            _elem = __CLIP_R(_elem, 127);
            *(_p_result_filter + _T) = (int8_t)_elem;

            // go to the next element
        }

        _filter += NUM_WORKERS; // update filter
    }

    // wait for all workers to finish
    rt_team_barrier();
}
#endif // PARALLEL


/**
 * @brief Execute the 1st layer
 * 
 * This layer does the following operation on the data:
 * 1. Convolution in time, with NET_F1 different filters of length 2, applied on all T sammples equally.
 * 2. Apply Batch Normalization
 *
 * @warning p_result must already be allocated on L2!
 *
 * @param p_data Pointer to the input data, of shape [NET_T, NET_C], aligned to [NET_T, NET_C_ALIGN]
 * @param p_result Pointer to the output data of shape [NET_F1, NET_T] aligned to [NET_F1, NET_T_ALIGN].
 */
void net_layer1(const int8_t* p_data, int8_t* p_result) {

#ifdef PARALLEL
    // allocate memory
    int8_t* _p_data_loc = rt_alloc(RT_ALLOC_CL_DATA, sizeof(int8_t) * NET_T * NET_C_ALIGN);
    int8_t* _p_weight_loc = rt_alloc(RT_ALLOC_CL_DATA, sizeof(int8_t) * NET_F1 * NET_C_ALIGN);
    int8_t * _p_result_loc = rt_alloc(RT_ALLOC_CL_DATA, sizeof(int8_t) * NET_F1 * NET_T_ALIGN);
    int32_t* _p_factor_loc = rt_alloc(RT_ALLOC_CL_DATA, sizeof(int32_t) * NET_F1);
    int32_t* _p_offset_loc = rt_alloc(RT_ALLOC_CL_DATA, sizeof(int32_t) * NET_F1);

    if (_p_offset_loc == NULL) {
        printf("Not Enough space on L1 memory\n");
        return;
    }
    
    rt_dma_copy_t _copy;
    rt_dma_memcpy((unsigned int)p_data,
                  (unsigned int)_p_data_loc,
                  sizeof(int8_t) * NET_T * NET_C_ALIGN,
                  RT_DMA_DIR_EXT2LOC, 0, &_copy);
    rt_dma_memcpy((unsigned int)net_l1_weight_align,
                  (unsigned int)_p_weight_loc,
                  sizeof(int8_t) * NET_F1 * NET_C_ALIGN,
                  RT_DMA_DIR_EXT2LOC, 1, &_copy);
    rt_dma_memcpy((unsigned int)net_l1_factor,
                  (unsigned int)_p_factor_loc,
                  sizeof(int32_t) * NET_F1,
                  RT_DMA_DIR_EXT2LOC, 1, &_copy);
    rt_dma_memcpy((unsigned int)net_l1_offset,
                  (unsigned int)_p_offset_loc,
                  sizeof(int32_t) * NET_F1,
                  RT_DMA_DIR_EXT2LOC, 1, &_copy);
    // wait until all dma transfers of the input data is complete
    rt_dma_wait(&_copy);

    // prepare the arguments for the cluster
    _net_layer1_kernel_t args;
    args.p_data = _p_data_loc;
    args.p_weight = _p_weight_loc;
    args.p_factor = _p_factor_loc;
    args.p_offset = _p_offset_loc;
    args.p_result = _p_result_loc;

    // call the cluster
    rt_team_fork(NUM_WORKERS, _net_layer1_kernel, (void*)(&args));

     // copy back the results
    rt_dma_memcpy((unsigned int)p_result,
                  (unsigned int)_p_result_loc,
                  sizeof(int8_t) * NET_F1 * NET_T_ALIGN,
                  RT_DMA_DIR_LOC2EXT, 0, &_copy);
    rt_dma_wait(&_copy);

    // free up the memory
    rt_free(RT_ALLOC_CL_DATA, _p_data_loc, sizeof(int8_t) * NET_T * NET_C_ALIGN);
    rt_free(RT_ALLOC_CL_DATA, _p_weight_loc, sizeof(int8_t) * NET_F1 * NET_C_ALIGN);
    rt_free(RT_ALLOC_CL_DATA, _p_result_loc, sizeof(int8_t) * NET_F1 * NET_T_ALIGN);
    rt_free(RT_ALLOC_CL_DATA, _p_factor_loc, sizeof(int32_t) * NET_F1);
    rt_free(RT_ALLOC_CL_DATA, _p_offset_loc, sizeof(int32_t) * NET_F1);

#else//PARALLEL
    // allocate memory
    int8_t* _p_data_loc = rt_alloc(RT_ALLOC_CL_DATA, sizeof(int8_t) * NET_T * NET_C_ALIGN);
    int8_t* _p_weight_loc = rt_alloc(RT_ALLOC_CL_DATA, sizeof(int8_t) * NET_F1 * NET_C_ALIGN);
    int8_t * _p_result_loc = rt_alloc(RT_ALLOC_CL_DATA, sizeof(int8_t) * NET_F1 * NET_T_ALIGN);
    int32_t* _p_factor_loc = rt_alloc(RT_ALLOC_CL_DATA, sizeof(int32_t) * NET_F1);
    int32_t* _p_offset_loc = rt_alloc(RT_ALLOC_CL_DATA, sizeof(int32_t) * NET_F1);

    if (_p_offset_loc == NULL) {
        printf("Not Enough space on L1 memory\n");
        return;
    }
    
    rt_dma_copy_t _copy;
    rt_dma_memcpy((unsigned int)p_data,
                  (unsigned int)_p_data_loc,
                  sizeof(int8_t) * NET_T * NET_C_ALIGN,
                  RT_DMA_DIR_EXT2LOC, 0, &_copy);
    rt_dma_memcpy((unsigned int)net_l1_weight_align,
                  (unsigned int)_p_weight_loc,
                  sizeof(int8_t) * NET_F1 * NET_C_ALIGN,
                  RT_DMA_DIR_EXT2LOC, 1, &_copy);
    rt_dma_memcpy((unsigned int)net_l1_factor,
                  (unsigned int)_p_factor_loc,
                  sizeof(int32_t) * NET_F1,
                  RT_DMA_DIR_EXT2LOC, 1, &_copy);
    rt_dma_memcpy((unsigned int)net_l1_offset,
                  (unsigned int)_p_offset_loc,
                  sizeof(int32_t) * NET_F1,
                  RT_DMA_DIR_EXT2LOC, 1, &_copy);
    // wait until all dma transfers of the input data is complete
    rt_dma_wait(&_copy);


    
    int32_t _factor;
    int32_t _offset;

    unsigned int _k, _ch;
    unsigned int _filter = 0;
    
    int8_t* _p_data_iter;
    int8_t* _p_weight_iter;
    int8_t* _p_result_filter;
    
    int32_t _elem;

    // loop until all filters are computed
    while (_filter < NET_F1) {
        _p_weight_iter = _p_weight_loc + _filter * NET_C_ALIGN;
        _p_result_filter = _p_result_loc + _filter * NET_T_ALIGN;
        
        _factor = _p_factor_loc[_filter];
        _offset = _p_offset_loc[_filter];

        
        for(int _T = 0; _T < NET_T; _T++) {  // _T stands for the current time sample
            _p_data_iter = _p_data_loc + _T * NET_C_ALIGN; // always reuse data for different filter
            
            _elem = func_dotp(_p_data_iter, _p_weight_iter, NET_C_ALIGN);
            _elem = (_elem + _offset) / _factor;

            // clamp
            _elem = __CLIP_R(_elem, 127);
            *(_p_result_filter + _T) = (int8_t)_elem;

            // go to the next element
        }

        _filter += 1; // update filter
    }

     // copy back the results
    rt_dma_memcpy((unsigned int)p_result,
                  (unsigned int)_p_result_loc,
                  sizeof(int8_t) * NET_F1 * NET_T_ALIGN,
                  RT_DMA_DIR_LOC2EXT, 0, &_copy);
    rt_dma_wait(&_copy);

    // free up the memory
    rt_free(RT_ALLOC_CL_DATA, _p_data_loc, sizeof(int8_t) * NET_T * NET_C_ALIGN);
    rt_free(RT_ALLOC_CL_DATA, _p_weight_loc, sizeof(int8_t) * NET_F1 * NET_C_ALIGN);
    rt_free(RT_ALLOC_CL_DATA, _p_result_loc, sizeof(int8_t) * NET_F1 * NET_T_ALIGN);
    rt_free(RT_ALLOC_CL_DATA, _p_factor_loc, sizeof(int32_t) * NET_F1);
    rt_free(RT_ALLOC_CL_DATA, _p_offset_loc, sizeof(int32_t) * NET_F1);
#endif//PARALLEL
}