/**
 * @file layer3.c
 * @author Tibor Schneider
 * @date 2020/01/31
 * @brief This file contains the Implementation for the third layer
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

typedef struct {
    int8_t* p_data;
    int8_t* p_result;
    int8_t* p_weight;
} _net_layer3_kernel_t;

/**
 * @brief Kernel doing the layer3 work
 */
void _net_layer3_kernel(void* args) {

    unsigned int _core_id = rt_core_id();

    // get values from args
    _net_layer3_kernel_t* _args = args;

    int8_t* _p_data_iter = _args->p_data;
    int8_t* _p_result_iter = _args->p_result;
    int8_t* _p_weight_iter = _args->p_weight;

    // go to the correct position for the thread
    _p_weight_iter += _core_id * NET_L3_WEIGHT_LEN;
    _p_data_iter += _core_id * NET_L3_PAD_INPUT_LEN_ALIGN;
    _p_result_iter += _core_id * NET_T8_ALIGN;

    unsigned int _k = _core_id;

    while (_k < NET_F2) {

        // do the computation
        func_conv_scale(_p_data_iter, NET_L3_PAD_INPUT_LEN, _p_weight_iter, NET_L3_WEIGHT_LEN, NET_L3_FACTOR, 0, _p_result_iter);

        // go to the next _k (for this core)
        _k += NUM_WORKERS;
        _p_weight_iter += NUM_WORKERS * NET_L3_WEIGHT_LEN;
        _p_data_iter += NUM_WORKERS * NET_L3_PAD_INPUT_LEN_ALIGN;
        _p_result_iter += NUM_WORKERS * NET_T8_ALIGN;

    }

    // wait for all cores to finish
    rt_team_barrier();

}

#endif //PARALLEL

/**
 * @brief Execute the 3rd layer
 *
 * This layer does the following operation on the data:
 * 1. Depthwise convolution in time, with 1 filter per NET_F2 of length 16.
 *
 * @warning p_result must already be allocated on L2!
 *
 * @param p_data Pointer to the input data, of shape [NET_F2, NET_T8], aligned to [NET_F2, NET_T8_ALIGN]
 * @param p_result Pointer to the output data of shape [NET_F2, NET_T8] aligned to [NET_F2, NET_T8_ALIGN]
 */
void net_layer3(const int8_t* p_data, int8_t * p_result) {

#ifdef PARALLEL

    const int8_t* _p_data_iter = p_data;          // iterator over the current input vector
    const int8_t* _p_result_iter = p_result;      // iterator over the current output vector

    rt_dma_copy_t _copy;

    // allocate local memory
    int8_t* _p_data_loc = rt_alloc(RT_ALLOC_CL_DATA, sizeof(int8_t) * NET_F2 * NET_L3_PAD_INPUT_LEN_ALIGN);
    int8_t* _p_result_loc = rt_alloc(RT_ALLOC_CL_DATA, sizeof(int8_t) * NET_F2 * NET_T8_ALIGN);
    int8_t* _p_weight_loc = rt_alloc(RT_ALLOC_CL_DATA, sizeof(int8_t) * NET_F2 * NET_L3_WEIGHT_LEN);

    // copy all the weights at once, because we get less overhead
    rt_dma_memcpy((unsigned int)net_l3_weight,
                  (unsigned int)_p_weight_loc,
                  sizeof(int8_t) * NET_F2 * NET_L3_WEIGHT_LEN,
                  RT_DMA_DIR_EXT2LOC, 0, &_copy);

    // copy all input vectors
    int8_t* _p_data_loc_iter = _p_data_loc;
    for (int _k = 0; _k < NET_F2; _k++) {

        // initialize input to have zero padding
        *((int32_t*)(_p_data_loc_iter + 0)) = 0;
        *((int32_t*)(_p_data_loc_iter + 4)) = 0;
        *((int32_t*)(_p_data_loc_iter + NET_L3_PAD_INPUT_LEN_ALIGN - 4)) = 0;
        *((int32_t*)(_p_data_loc_iter + NET_L3_PAD_INPUT_LEN_ALIGN - 8)) = 0;
        *((int32_t*)(_p_data_loc_iter + NET_L3_PAD_INPUT_LEN_ALIGN - 12)) = 0;

        rt_dma_memcpy((unsigned int)_p_data_iter,
                      (unsigned int)_p_data_loc_iter + NET_L3_PAD_START,
                      sizeof(int8_t) * NET_T8,
                      RT_DMA_DIR_EXT2LOC, 1, &_copy);

        // go to the next element
        _p_data_iter += NET_T8_ALIGN;
        _p_data_loc_iter += NET_L3_PAD_INPUT_LEN_ALIGN;
    }

    //wait for all copies to finish
    rt_dma_wait(&_copy);

    // prepare the arguments
    _net_layer3_kernel_t _args;
    _args.p_data = _p_data_loc;
    _args.p_result = _p_result_loc;
    _args.p_weight = _p_weight_loc;

    rt_team_fork(NUM_WORKERS, _net_layer3_kernel, &_args);

    // copy back the results
    rt_dma_memcpy((unsigned int)p_result,
                  (unsigned int)_p_result_loc,
                  sizeof(int8_t) * NET_F2 * NET_T8,
                  RT_DMA_DIR_LOC2EXT, 0, &_copy);
    rt_dma_wait(&_copy);

    rt_free(RT_ALLOC_CL_DATA, _p_data_loc, sizeof(int8_t) * NET_F2 * NET_L3_PAD_INPUT_LEN_ALIGN);
    rt_free(RT_ALLOC_CL_DATA, _p_result_loc, sizeof(int8_t) * NET_F2 * NET_T8_ALIGN);
    rt_free(RT_ALLOC_CL_DATA, _p_weight_loc, sizeof(int8_t) * NET_F2 * NET_L3_WEIGHT_LEN);

#else //PARALLEL

    /*
     * Depthwise Convoluton, compute every channel separately
     */

    const int8_t* _p_data_iter = p_data;          // iterator over the current input vector
    const int8_t* _p_result_iter = p_result;      // iterator over the current output vector

    rt_dma_copy_t _copy;

    // allocate local memory
    int8_t* _p_data_loc = rt_alloc(RT_ALLOC_CL_DATA, sizeof(int8_t) * NET_L3_PAD_INPUT_LEN_ALIGN);
    int32_t* _p_tmp_result_loc = rt_alloc(RT_ALLOC_CL_DATA, sizeof(int32_t) * NET_T8);
    int8_t* _p_result_loc = rt_alloc(RT_ALLOC_CL_DATA, sizeof(int8_t) * NET_T8_ALIGN);
    int8_t* _p_weight_loc = rt_alloc(RT_ALLOC_CL_DATA, sizeof(int8_t) * NET_F2 * NET_L3_WEIGHT_LEN);

    // initialize input to have zero padding
    *((int32_t*)(_p_data_loc + 0)) = 0;
    *((int32_t*)(_p_data_loc + 4)) = 0;
    *((int32_t*)(_p_data_loc + NET_L3_PAD_INPUT_LEN_ALIGN - 4)) = 0;
    *((int32_t*)(_p_data_loc + NET_L3_PAD_INPUT_LEN_ALIGN - 8)) = 0;
    *((int32_t*)(_p_data_loc + NET_L3_PAD_INPUT_LEN_ALIGN - 12)) = 0;

    // copy all the weights at once, because we get less overhead
    rt_dma_memcpy((unsigned int)net_l3_weight,
                  (unsigned int)_p_weight_loc,
                  sizeof(int8_t) * NET_F2 * NET_L3_WEIGHT_LEN,
                  RT_DMA_DIR_EXT2LOC, 0, &_copy);
    rt_dma_wait(&_copy);

    int8_t* _p_weight_loc_iter = _p_weight_loc;  // iterator over the current weights (filter)

    // loop over all channels
    for (unsigned int _k = 0; _k < NET_F2; _k++) {

        // copy the corresponding input data to local memory, keeping the padding
        rt_dma_memcpy((unsigned int)_p_data_iter,
                      (unsigned int)_p_data_loc + NET_L3_PAD_START,
                      sizeof(int8_t) * NET_T8,
                      RT_DMA_DIR_EXT2LOC, 0, &_copy);
        rt_dma_wait(&_copy);

        // do the convolution
        func_conv(_p_data_loc, NET_L3_PAD_INPUT_LEN, _p_weight_loc_iter, NET_L3_WEIGHT_LEN, _p_tmp_result_loc);

        // scale the values
        func_transform_32to8(_p_tmp_result_loc, NET_T8, NET_L3_FACTOR, 1, _p_result_loc);

        // copy the results back
        rt_dma_memcpy((unsigned int)_p_result_iter,
                      (unsigned int)_p_result_loc,
                      sizeof(int8_t) * NET_T8,
                      RT_DMA_DIR_LOC2EXT, 0, &_copy);
        rt_dma_wait(&_copy);

        // move the iterators to the next elements
        _p_data_iter += NET_T8_ALIGN;
        _p_result_iter += NET_T8_ALIGN;
        _p_weight_loc_iter += NET_L3_WEIGHT_LEN;

    }

    rt_free(RT_ALLOC_CL_DATA, _p_data_loc, sizeof(int8_t) * NET_L3_PAD_INPUT_LEN_ALIGN);
    rt_free(RT_ALLOC_CL_DATA, _p_tmp_result_loc, sizeof(int32_t) * NET_T8);
    rt_free(RT_ALLOC_CL_DATA, _p_result_loc, sizeof(int8_t) * NET_T8_ALIGN);
    rt_free(RT_ALLOC_CL_DATA, _p_weight_loc, sizeof(int8_t) * NET_F2 * NET_L3_WEIGHT_LEN);


#endif //PARALLEL

}


/**
 * @brief Flip the F2 and T//8 dimension inplace after layer 3, before layer 4
 * p_data will be of shape [NET_T8_ALIGN, NET_F2] afterwards.
 *
 * @warning p_result must already be allocated on L2!
 *
 * @param p_data Pointer to the input data, of shape [NET_F2, NET_T8_ALIGN], aligned to [NET_T8_ALIGN, NET_F2]
 */
void net_layer3_flip_inplace(int8_t* p_data) {

    // Data is small enough that we can just copy everything to L1, transform and store it back

    int8_t* _p_data_loc = rt_alloc(RT_ALLOC_CL_DATA, sizeof(int8_t) * NET_F2 * NET_T8_ALIGN);
    int8_t* _p_result_loc = rt_alloc(RT_ALLOC_CL_DATA, sizeof(int8_t) * NET_T8_ALIGN * NET_F2);

    rt_dma_copy_t _copy;

    // copy everything
    rt_dma_memcpy((unsigned int)p_data,
                  (unsigned int)_p_data_loc,
                  sizeof(int8_t) * NET_F2 * NET_T8_ALIGN,
                  RT_DMA_DIR_EXT2LOC, 0, &_copy);
    rt_dma_wait(&_copy);

    // flip everything
    func_flip_2d_axis(_p_data_loc, NET_F2, NET_T8, _p_result_loc);

    // copy everything back
    rt_dma_memcpy((unsigned int)p_data,
                  (unsigned int)_p_result_loc,
                  sizeof(int8_t) * NET_T8 * NET_F2,
                  RT_DMA_DIR_LOC2EXT, 0, &_copy);
    rt_dma_wait(&_copy);

}