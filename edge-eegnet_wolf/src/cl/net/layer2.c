/**
 * @file layer2.c
 * @author Tibor Schneider
 * @date 2020/01/31
 * @brief This file contains the Implementation for the second layer
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

#ifdef REORDER_BN

typedef struct
{
    int8_t* p_data;    // pointer to current input image (L1 memory)
    int8_t* p_weight;  // pointer to current weight vector (L1 memory)
    int32_t* p_offset;  // BN offset
    int32_t* p_factor;  // BN factor
    int8_t* p_result; // pointer to result vector (L1 memory)
    int32_t* p_thread; // pointer to result vector (L1 memory)
} _net_layer2_kernel_t;

/**
 * @brief Layer2 kernel
 */
void _net_layer2_kernel(void* args) {
    unsigned int _filter = rt_core_id();

    int8_t* _p_data = ((_net_layer2_kernel_t*)args)->p_data; // base pointer to data
    int8_t* _p_weight = ((_net_layer2_kernel_t*)args)->p_weight;
    int32_t* _p_offset = ((_net_layer2_kernel_t*)args)->p_offset;
    int32_t* _p_factor = ((_net_layer2_kernel_t*)args)->p_factor;
    int8_t* _p_result = ((_net_layer2_kernel_t*)args)->p_result;
    int32_t* p_thread = ((_net_layer2_kernel_t*)args)->p_thread;

    int8_t* _p_data_filter;
    int32_t* _p_data_thread;
    int32_t* _p_data_thread_iter;
    int8_t* _p_weight_filter;
    int8_t* _p_result_filter;
    int32_t _factor;
    int32_t _offset;

    int32_t _sum;  // stores the sum for the pooling

    _p_data_filter = _p_data + _filter * NET_L2_PAD_INPUT_LEN_ALIGN;
    _p_weight_filter = _p_weight + _filter * NET_L2_WEIGHT_LEN_ALIGN;
    _p_result_filter = _p_result +  _filter * NET_T8;
    
    _factor = _p_factor[_filter];
    _offset = _p_offset[_filter];

    int32_t _threshold = -(_offset >> 3);

    _p_data_thread = p_thread + _filter * NET_L2_PAD_INPUT_LEN_ALIGN;

    // convolve and scale the data (always the correct parts)
#ifdef CROSS_CORRELATE
    func_xcorr(_p_data_filter, NET_L2_PAD_INPUT_LEN,
               _p_weight_filter, NET_L2_WEIGHT_LEN,
               _p_data_thread);
#else //CROSS_CORRELATE
    func_conv(_p_data_filter, NET_L2_PAD_INPUT_LEN,
              _p_weight_filter, NET_L2_WEIGHT_LEN,
              _p_data_thread);
#endif //CROSS_CORRELATE

    _p_data_thread_iter = _p_data_thread;
    
    for (unsigned int _t_out = 0; _t_out < NET_T8; _t_out++) {
        // reset the sum
        _sum = 0;
        // loop over all 8 elements in the local neighborhood
        for (unsigned int _t_pool = 0; _t_pool < 8; _t_pool++) {
            // do relu on each element
            _sum += __MAX(*(_p_data_thread_iter++), _threshold);
        }

        // do avg pooling division
        _sum = _sum + _offset;
        _sum = _sum / _factor;


        // clamp
        _sum = __CLIP_R(_sum, 127);

        *(_p_result_filter+_t_out) = (int8_t)_sum;
    }
        
    // wait for all workers to finish
    rt_team_barrier();  
}

#else//REORDER_BN

typedef struct
{
    int8_t* p_data;    // pointer to current input image (L1 memory)
    int8_t* p_weight;  // pointer to current weight vector (L1 memory)
    int32_t* p_offset;  // BN offset
    int32_t* p_factor;  // BN factor
    int8_t* p_result; // pointer to result vector (L1 memory)
    int8_t* p_thread; // pointer to result vector (L1 memory)
} _net_layer2_kernel_t;

/**
 * @brief Layer2 kernel
 */
void _net_layer2_kernel(void* args) {
    unsigned int core_id = rt_core_id();
    unsigned int _filter = core_id;

    int8_t* _p_data = ((_net_layer2_kernel_t*)args)->p_data; // base pointer to data
    int8_t* _p_weight = ((_net_layer2_kernel_t*)args)->p_weight;
    int32_t* _p_offset = ((_net_layer2_kernel_t*)args)->p_offset;
    int32_t* _p_factor = ((_net_layer2_kernel_t*)args)->p_factor;
    int8_t* _p_result = ((_net_layer2_kernel_t*)args)->p_result;
    int8_t* p_thread = ((_net_layer2_kernel_t*)args)->p_thread;

    int8_t* _p_data_filter;
    int8_t* _p_data_thread;
    int8_t* _p_data_thread_iter;
    int8_t* _p_weight_filter;
    int8_t* _p_result_filter;
    int32_t _factor;
    int32_t _offset;

    int32_t _sum;  // stores the sum for the pooling

    while (_filter < NET_F2) {
        _p_data_filter = _p_data + _filter * NET_L2_PAD_INPUT_LEN_ALIGN;
        _p_weight_filter = _p_weight + _filter * NET_L2_WEIGHT_LEN_ALIGN;
        _p_result_filter = _p_result +  _filter * NET_T8;
        
        _factor = _p_factor[_filter];
        _offset = _p_offset[_filter];

        _factor = _factor >> 3;
        _offset = _offset >> 3;

        _p_data_thread = p_thread + core_id * NET_L2_PAD_INPUT_LEN_ALIGN;

        // convolve and scale the data (always the correct parts)
#ifdef CROSS_CORRELATE
        func_xcorr_scale(_p_data_filter, NET_L2_PAD_INPUT_LEN,
                         _p_weight_filter, NET_L2_WEIGHT_LEN,
                         _factor, _offset, _p_data_thread);
#else //CROSS_CORRELATE
        func_conv_scale(_p_data_filter, NET_L2_PAD_INPUT_LEN,
                        _p_weight_filter, NET_L2_WEIGHT_LEN,
                        _factor, _offset, _p_data_thread);
#endif //CROSS_CORRELATE

        _p_data_thread_iter = _p_data_thread;
        
        for (unsigned int _t_out = 0; _t_out < NET_T8; _t_out++) {
            // reset the sum
            _sum = 0;
            // loop over all 8 elements in the local neighborhood
            for (unsigned int _t_pool = 0; _t_pool < 8; _t_pool++) {
                // do relu on each element
                _sum += __MAX(*(_p_data_thread_iter++), 0);
            }

            // do avg pooling division
            _sum = _sum >> 3;

            // clamp
            _sum = __CLIP_R(_sum, 127);

            *(_p_result_filter+_t_out) = (int8_t)_sum;
        }

        _filter += NUM_WORKERS; // update filter
        
        // wait for all workers to finish
        rt_team_barrier();
    }
    
}
#endif//REORDER_BN
#endif //PARALLEL

/**
 * @brief Execute the 2nd layer
 * 
 * This layer does the following operation on the data:
 * 2. Depthwise convolution in space, with NET_D filters per NET_F1, the same filter for each time sample
 * 4. Apply Batch Normalization
 * 5. Apply ReLU
 * 6. Apply Avg Pooling with kernel (1, 8)
 *
 * @warning p_result must already be allocated on L2!
 *
 * @param p_data Pointer to the input data, of shape [NET_F1, NET_T, NET_C], aligned to [NET_F1, NET_T, NET_C_ALIGN]
 * @param p_result Pointer to the output data of shape [NET_F2, NET_T8] aligned to [NET_F2, NET_T8_ALIGN]
 */
void net_layer2(const int8_t* p_data, int8_t * p_result) {

#ifdef PARALLEL  
#ifdef REORDER_BN
    int8_t* _p_data_loc = rt_alloc(RT_ALLOC_CL_DATA, sizeof(int8_t) * NUM_WORKERS * NET_L2_PAD_INPUT_LEN_ALIGN);
    int8_t* _p_weight_loc = rt_alloc(RT_ALLOC_CL_DATA, sizeof(int8_t) * NUM_WORKERS * NET_L2_WEIGHT_LEN_ALIGN);
    int8_t * _p_result_loc = rt_alloc(RT_ALLOC_CL_DATA, sizeof(int8_t) * NUM_WORKERS * NET_T8_ALIGN);
    int32_t* _p_factor_loc = rt_alloc(RT_ALLOC_CL_DATA, sizeof(int32_t) * NUM_WORKERS);
    int32_t* _p_offset_loc = rt_alloc(RT_ALLOC_CL_DATA, sizeof(int32_t) * NUM_WORKERS);
    int32_t* _p_thread = rt_alloc(RT_ALLOC_CL_DATA, sizeof(int32_t) * NUM_WORKERS * NET_L2_PAD_INPUT_LEN_ALIGN);

    if (_p_thread == NULL) {
        printf("Not Enough space on L1 memory\n");
        return;
    }

    
    rt_dma_copy_t _copy;
    for (int _BATCH = 0; _BATCH < 2; _BATCH++){
        
        const int8_t* _p_data_iter = p_data  + _BATCH  * NUM_WORKERS * NET_T_ALIGN;
        int8_t* _p_data_loc_iter = _p_data_loc;

        
        // iterators over data
        for (int _filter = 0; _filter < NUM_WORKERS; _filter++) {
        
            // pre data pad
            int32_t* _p_pad_iter = (int32_t*)_p_data_loc_iter;
            for (int _i = 0; _i < (NET_L2_PAD_START + 3) / 4; _i++) {
                *(_p_pad_iter++) = 0;
            }

            // post data pad
            _p_pad_iter = (int32_t*)(_p_data_loc_iter + NET_L2_PAD_INPUT_LEN_ALIGN - 4);
            for (int _i = 0; _i < (NET_L2_PAD_END + 3) / 4 + (NET_L2_PAD_INPUT_LEN % 4 + 3) / 4; _i++) {
                *(_p_pad_iter--) = 0;
            }

            // start the DMA transfer
            int merge = _filter == 0 ? 0 : 1;
            rt_dma_memcpy((unsigned int)_p_data_iter,
                          (unsigned int)(_p_data_loc_iter + NET_L2_PAD_START),
                          sizeof(int8_t) * NET_T,
                          RT_DMA_DIR_EXT2LOC, merge, &_copy);

            // move to the next filter
            _p_data_iter += NET_T_ALIGN;
            _p_data_loc_iter += NET_L2_PAD_INPUT_LEN_ALIGN;
        }

#ifdef CROSS_CORRELATE
        rt_dma_memcpy((unsigned int)net_l2_weight_reverse + _BATCH  * NUM_WORKERS * NET_L2_WEIGHT_LEN_ALIGN,
                      (unsigned int)_p_weight_loc,
                      sizeof(int32_t) * NUM_WORKERS * NET_L2_WEIGHT_LEN_ALIGN,
                      RT_DMA_DIR_EXT2LOC, 1, &_copy);
#else//CROSS_CORRELATE
        rt_dma_memcpy((unsigned int)net_l2_weight + _BATCH  * NUM_WORKERS * NET_L2_WEIGHT_LEN_ALIGN,
                      (unsigned int)_p_weight_loc,
                      sizeof(int32_t) * NUM_WORKERS * NET_L2_WEIGHT_LEN_ALIGN,
                      RT_DMA_DIR_EXT2LOC, 1, &_copy);
#endif//CROSS_CORRELATE
        rt_dma_memcpy((unsigned int)net_l2_factor + _BATCH  * NUM_WORKERS * sizeof(int32_t),
                      (unsigned int)_p_factor_loc,
                      sizeof(int32_t) * NUM_WORKERS,
                      RT_DMA_DIR_EXT2LOC, 1, &_copy);
        rt_dma_memcpy((unsigned int)net_l2_offset + _BATCH  * NUM_WORKERS * sizeof(int32_t),
                      (unsigned int)_p_offset_loc,
                      sizeof(int32_t) * NUM_WORKERS,
                      RT_DMA_DIR_EXT2LOC, 1, &_copy);

        // wait until all dma transfers of the input data is complete
        rt_dma_wait(&_copy);
        
        _net_layer2_kernel_t args;
        args.p_data = _p_data_loc;
        args.p_weight = _p_weight_loc;
        args.p_factor = _p_factor_loc;
        args.p_offset = _p_offset_loc;
        args.p_result = _p_result_loc;
        args.p_thread = _p_thread;
        
        rt_team_fork(NUM_WORKERS, _net_layer2_kernel, (void*)(&args));

        // copy back the results
        rt_dma_memcpy((unsigned int)p_result + _BATCH  * NUM_WORKERS * NET_T8_ALIGN,
                      (unsigned int)_p_result_loc,
                      sizeof(int8_t) * NUM_WORKERS * NET_T8_ALIGN,
                      RT_DMA_DIR_LOC2EXT, 0, &_copy);
        rt_dma_wait(&_copy);
    }
    
    // free up the memory
    rt_free(RT_ALLOC_CL_DATA, _p_data_loc, sizeof(int8_t) * NUM_WORKERS * NET_L2_PAD_INPUT_LEN);
    rt_free(RT_ALLOC_CL_DATA, _p_weight_loc, sizeof(int8_t) * NUM_WORKERS * NET_L2_WEIGHT_LEN_ALIGN);
    rt_free(RT_ALLOC_CL_DATA, _p_result_loc, sizeof(int8_t) * NUM_WORKERS * NET_T8_ALIGN);
    rt_free(RT_ALLOC_CL_DATA, _p_factor_loc, sizeof(int32_t) * NET_F2);
    rt_free(RT_ALLOC_CL_DATA, _p_offset_loc, sizeof(int32_t) * NET_F2);
    rt_free(RT_ALLOC_CL_DATA, _p_thread, sizeof(int8_t) * NUM_WORKERS * NET_L2_PAD_INPUT_LEN);

#else//REORDER_BN

    int8_t* _p_data_loc = rt_alloc(RT_ALLOC_CL_DATA, sizeof(int8_t) * NET_F1 * NET_L2_PAD_INPUT_LEN_ALIGN);
    int8_t* _p_weight_loc = rt_alloc(RT_ALLOC_CL_DATA, sizeof(int8_t) * NET_F2 * NET_L2_WEIGHT_LEN_ALIGN);
    int8_t * _p_result_loc = rt_alloc(RT_ALLOC_CL_DATA, sizeof(int8_t) * NET_F2 * NET_T8_ALIGN);
    int32_t* _p_factor_loc = rt_alloc(RT_ALLOC_CL_DATA, sizeof(int32_t) * NET_F2);
    int32_t* _p_offset_loc = rt_alloc(RT_ALLOC_CL_DATA, sizeof(int32_t) * NET_F2);
    int8_t* _p_thread = rt_alloc(RT_ALLOC_CL_DATA, sizeof(int8_t) * NUM_WORKERS * NET_L2_PAD_INPUT_LEN_ALIGN);

    if (_p_thread == NULL) {
        printf("Not Enough space on L1 memory\n");
        return;
    }

    rt_dma_copy_t _copy;
    
    // iterators over data
    const int8_t* _p_data_iter = p_data;
    int8_t* _p_data_loc_iter = _p_data_loc;
    
    for (int _filter = 0; _filter < NET_F2; _filter++) {
        
        // pre data pad
        int32_t* _p_pad_iter = (int32_t*)_p_data_loc_iter;
        for (int _i = 0; _i < (NET_L2_PAD_START + 3) / 4; _i++) {
            *(_p_pad_iter++) = 0;
        }

        // post data pad
        _p_pad_iter = (int32_t*)(_p_data_loc_iter + NET_L2_PAD_INPUT_LEN_ALIGN - 4);
        for (int _i = 0; _i < (NET_L2_PAD_END + 3) / 4 + (NET_L2_PAD_INPUT_LEN % 4 + 3) / 4; _i++) {
            *(_p_pad_iter--) = 0;
        }

        // start the DMA transfer
        int merge = _filter == 0 ? 0 : 1;
        rt_dma_memcpy((unsigned int)_p_data_iter,
                      (unsigned int)(_p_data_loc_iter + NET_L2_PAD_START),
                      sizeof(int8_t) * NET_T,
                      RT_DMA_DIR_EXT2LOC, merge, &_copy);

        // move to the next filter
        _p_data_iter += NET_T_ALIGN;
        _p_data_loc_iter += NET_L2_PAD_INPUT_LEN_ALIGN;
    }

#ifdef CROSS_CORRELATE
    rt_dma_memcpy((unsigned int)net_l2_weight_reverse,
                  (unsigned int)_p_weight_loc,
                  sizeof(int32_t) * NET_F2 * NET_L2_WEIGHT_LEN_ALIGN,
                  RT_DMA_DIR_EXT2LOC, 1, &_copy);
#else//CROSS_CORRELATE
    rt_dma_memcpy((unsigned int)net_l2_weight,
                  (unsigned int)_p_weight_loc,
                  sizeof(int32_t) * NET_F2 * NET_L2_WEIGHT_LEN_ALIGN,
                  RT_DMA_DIR_EXT2LOC, 1, &_copy);
#endif//CROSS_CORRELATE
    rt_dma_memcpy((unsigned int)net_l2_factor,
                  (unsigned int)_p_factor_loc,
                  sizeof(int32_t) * NET_F2,
                  RT_DMA_DIR_EXT2LOC, 1, &_copy);
    rt_dma_memcpy((unsigned int)net_l2_offset,
                  (unsigned int)_p_offset_loc,
                  sizeof(int32_t) * NET_F2,
                  RT_DMA_DIR_EXT2LOC, 1, &_copy);

    // wait until all dma transfers of the input data is complete
    rt_dma_wait(&_copy);
    
    _net_layer2_kernel_t args;
    args.p_data = _p_data_loc;
    args.p_weight = _p_weight_loc;
    args.p_factor = _p_factor_loc;
    args.p_offset = _p_offset_loc;
    args.p_result = _p_result_loc;
    args.p_thread = _p_thread;

    rt_team_fork(NUM_WORKERS, _net_layer2_kernel, (void*)(&args));

    // copy back the results
    rt_dma_memcpy((unsigned int)p_result,
                  (unsigned int)_p_result_loc,
                  sizeof(int8_t) * NET_F2 * NET_T8_ALIGN,
                  RT_DMA_DIR_LOC2EXT, 0, &_copy);
    rt_dma_wait(&_copy);

    // free up the memory
    rt_free(RT_ALLOC_CL_DATA, _p_data_loc, sizeof(int8_t) * NET_F1 * NET_L2_PAD_INPUT_LEN);
    rt_free(RT_ALLOC_CL_DATA, _p_weight_loc, sizeof(int8_t) * NET_F2 * NET_L2_WEIGHT_LEN_ALIGN);
    rt_free(RT_ALLOC_CL_DATA, _p_result_loc, sizeof(int8_t) * NET_F2 * NET_T8_ALIGN);
    rt_free(RT_ALLOC_CL_DATA, _p_factor_loc, sizeof(int32_t) * NET_F2);
    rt_free(RT_ALLOC_CL_DATA, _p_offset_loc, sizeof(int32_t) * NET_F2);
    rt_free(RT_ALLOC_CL_DATA, _p_thread, sizeof(int8_t) * NUM_WORKERS * NET_L2_PAD_INPUT_LEN);

#endif//REORDER_BN
#else//parallel

#ifdef REORDER_BN
    int8_t* _p_data_loc = rt_alloc(RT_ALLOC_CL_DATA, sizeof(int8_t) * NET_F1 * NET_L2_PAD_INPUT_LEN_ALIGN);
    int8_t* _p_weight_loc = rt_alloc(RT_ALLOC_CL_DATA, sizeof(int8_t) * NET_F2 * NET_L2_WEIGHT_LEN_ALIGN);
    int8_t * _p_result_loc = rt_alloc(RT_ALLOC_CL_DATA, sizeof(int8_t) * NET_F2 * NET_T8_ALIGN);
    int32_t* _p_factor_loc = rt_alloc(RT_ALLOC_CL_DATA, sizeof(int32_t) * NET_F2);
    int32_t* _p_offset_loc = rt_alloc(RT_ALLOC_CL_DATA, sizeof(int32_t) * NET_F2);
    int32_t* _p_thread = rt_alloc(RT_ALLOC_CL_DATA, sizeof(int32_t)* NET_L2_PAD_INPUT_LEN_ALIGN);

    if (_p_offset_loc == NULL) {
        printf("Not Enough space on L1 memory\n");
        return;
    }

    rt_dma_copy_t _copy;
    
    // iterators over data
    const int8_t* _p_data_iter = p_data;
    int8_t* _p_data_loc_pad = _p_data_loc;
    
    for (int _filter = 0; _filter < NET_F2; _filter++) {
        
        // pre data pad
        int32_t* _p_pad_iter = (int32_t*)_p_data_loc_pad;
        for (int _i = 0; _i < (NET_L2_PAD_START + 3) / 4; _i++) {
            *(_p_pad_iter++) = 0;
        }

        // post data pad
        _p_pad_iter = (int32_t*)(_p_data_loc_pad + NET_L2_PAD_INPUT_LEN_ALIGN - 4);
        for (int _i = 0; _i < (NET_L2_PAD_END + 3) / 4 + (NET_L2_PAD_INPUT_LEN % 4 + 3) / 4; _i++) {
            *(_p_pad_iter--) = 0;
        }

        // start the DMA transfer
        int merge = _filter == 0 ? 0 : 1;
        rt_dma_memcpy((unsigned int)_p_data_iter,
                      (unsigned int)(_p_data_loc_pad + NET_L2_PAD_START),
                      sizeof(int8_t) * NET_T,
                      RT_DMA_DIR_EXT2LOC, merge, &_copy);

        // move to the next filter
        _p_data_iter += NET_T_ALIGN;
        _p_data_loc_pad += NET_L2_PAD_INPUT_LEN_ALIGN;
    }

#ifdef CROSS_CORRELATE
    rt_dma_memcpy((unsigned int)net_l2_weight_reverse,
                  (unsigned int)_p_weight_loc,
                  sizeof(int32_t) * NET_F2 * NET_L2_WEIGHT_LEN_ALIGN,
                  RT_DMA_DIR_EXT2LOC, 1, &_copy);
#else//CROSS_CORRELATE
    rt_dma_memcpy((unsigned int)net_l2_weight,
                  (unsigned int)_p_weight_loc,
                  sizeof(int32_t) * NET_F2 * NET_L2_WEIGHT_LEN_ALIGN,
                  RT_DMA_DIR_EXT2LOC, 1, &_copy);
#endif//CROSS_CORRELATE
    rt_dma_memcpy((unsigned int)net_l2_factor,
                  (unsigned int)_p_factor_loc,
                  sizeof(int32_t) * NET_F2,
                  RT_DMA_DIR_EXT2LOC, 1, &_copy);
    rt_dma_memcpy((unsigned int)net_l2_offset,
                  (unsigned int)_p_offset_loc,
                  sizeof(int32_t) * NET_F2,
                  RT_DMA_DIR_EXT2LOC, 1, &_copy);

    // wait until all dma transfers of the input data is complete
    rt_dma_wait(&_copy);

    int32_t _factor;
    int32_t _offset;
    int32_t _sum;  // stores the sum for the pooling
    
    int32_t* _p_data_thread_iter;
    int8_t* _p_data_loc_iter = _p_data_loc;
    int8_t* _p_weight_loc_iter = _p_weight_loc;
    int8_t* _p_result_loc_iter = _p_result_loc;

    unsigned int _filter = 0;
    while (_filter < NET_F2) {
        _p_data_loc_iter = _p_data_loc + _filter * NET_L2_PAD_INPUT_LEN_ALIGN;
        _p_weight_loc_iter = _p_weight_loc + _filter * NET_L2_WEIGHT_LEN_ALIGN;
        _p_result_loc_iter = _p_result_loc+ _filter * NET_T8;
        _p_data_thread_iter = _p_thread;
        
        _factor = _p_factor_loc[_filter];
        _offset = _p_offset_loc[_filter];
        
        int32_t _threshold = -(_offset >> 3);

#ifdef CROSS_CORRELATE
        func_xcorr(_p_data_loc_iter, NET_L2_PAD_INPUT_LEN,
                         _p_weight_loc_iter, NET_L2_WEIGHT_LEN,
                         _p_thread);
#else//CROSS_CORRELATE
        func_conv(_p_data_loc_iter, NET_L2_PAD_INPUT_LEN,
                        _p_weight_loc_iter, NET_L2_WEIGHT_LEN,
                        _p_thread);
#endif//CROSS_CORRELATE

        for (unsigned int _t_out = 0; _t_out < NET_T8; _t_out++) {
            // reset the sum
            _sum = 0;
            // loop over all 8 elements in the local neighborhood
            for (unsigned int _t_pool = 0; _t_pool < 8; _t_pool++) {
                // do relu on each element
                _sum += __MAX(*(_p_data_thread_iter++), _threshold);
            }
            // do BN
            _sum = _sum + _offset;
            _sum = _sum / _factor;

            // clamp
            _sum = __CLIP_R(_sum, 127);

            *(_p_result_loc_iter+_t_out) = (int8_t)_sum;
        }

        // update filter
        _filter += 1; 
    }

    // copy back the results
    rt_dma_memcpy((unsigned int)p_result,
                  (unsigned int)_p_result_loc,
                  sizeof(int8_t) * NET_F2 * NET_T8_ALIGN,
                  RT_DMA_DIR_LOC2EXT, 0, &_copy);
    rt_dma_wait(&_copy);

    // free up the memory
    rt_free(RT_ALLOC_CL_DATA, _p_data_loc, sizeof(int8_t) * NET_F1 * NET_L2_PAD_INPUT_LEN);
    rt_free(RT_ALLOC_CL_DATA, _p_weight_loc, sizeof(int8_t) * NET_F2 * NET_L2_WEIGHT_LEN_ALIGN);
    rt_free(RT_ALLOC_CL_DATA, _p_result_loc, sizeof(int8_t) * NET_F2 * NET_T8_ALIGN);
    rt_free(RT_ALLOC_CL_DATA, _p_factor_loc, sizeof(int32_t) * NET_F2);
    rt_free(RT_ALLOC_CL_DATA, _p_offset_loc, sizeof(int32_t) * NET_F2);
    rt_free(RT_ALLOC_CL_DATA, _p_thread, sizeof(int8_t) * NET_L2_PAD_INPUT_LEN);

#else//REORDER_BN

    int8_t* _p_data_loc = rt_alloc(RT_ALLOC_CL_DATA, sizeof(int8_t) * NET_F1 * NET_L2_PAD_INPUT_LEN_ALIGN);
    int8_t* _p_weight_loc = rt_alloc(RT_ALLOC_CL_DATA, sizeof(int8_t) * NET_F2 * NET_L2_WEIGHT_LEN_ALIGN);
    int8_t * _p_result_loc = rt_alloc(RT_ALLOC_CL_DATA, sizeof(int8_t) * NET_F2 * NET_T8_ALIGN);
    int32_t* _p_factor_loc = rt_alloc(RT_ALLOC_CL_DATA, sizeof(int32_t) * NET_F2);
    int32_t* _p_offset_loc = rt_alloc(RT_ALLOC_CL_DATA, sizeof(int32_t) * NET_F2);
    int8_t* _p_thread = rt_alloc(RT_ALLOC_CL_DATA, sizeof(int8_t)* NET_L2_PAD_INPUT_LEN_ALIGN);


    if (_p_offset_loc == NULL) {
        printf("Not Enough space on L1 memory\n");
        return;
    }

    rt_dma_copy_t _copy;
    
    // iterators over data
    const int8_t* _p_data_iter = p_data;
    int8_t* _p_data_loc_pad = _p_data_loc;
    
    for (int _filter = 0; _filter < NET_F2; _filter++) {
        
        // pre data pad
        int32_t* _p_pad_iter = (int32_t*)_p_data_loc_pad;
        for (int _i = 0; _i < (NET_L2_PAD_START + 3) / 4; _i++) {
            *(_p_pad_iter++) = 0;
        }

        // post data pad
        _p_pad_iter = (int32_t*)(_p_data_loc_pad + NET_L2_PAD_INPUT_LEN_ALIGN - 4);
        for (int _i = 0; _i < (NET_L2_PAD_END + 3) / 4 + (NET_L2_PAD_INPUT_LEN % 4 + 3) / 4; _i++) {
            *(_p_pad_iter--) = 0;
        }

        // start the DMA transfer
        int merge = _filter == 0 ? 0 : 1;
        rt_dma_memcpy((unsigned int)_p_data_iter,
                      (unsigned int)(_p_data_loc_pad + NET_L2_PAD_START),
                      sizeof(int8_t) * NET_T,
                      RT_DMA_DIR_EXT2LOC, merge, &_copy);

        // move to the next filter
        _p_data_iter += NET_T_ALIGN;
        _p_data_loc_pad += NET_L2_PAD_INPUT_LEN_ALIGN;
    }

#ifdef CROSS_CORRELATE
    rt_dma_memcpy((unsigned int)net_l2_weight_reverse,
                  (unsigned int)_p_weight_loc,
                  sizeof(int32_t) * NET_F2 * NET_L2_WEIGHT_LEN_ALIGN,
                  RT_DMA_DIR_EXT2LOC, 1, &_copy);
#else//CROSS_CORRELATE
    rt_dma_memcpy((unsigned int)net_l2_weight,
                  (unsigned int)_p_weight_loc,
                  sizeof(int32_t) * NET_F2 * NET_L2_WEIGHT_LEN_ALIGN,
                  RT_DMA_DIR_EXT2LOC, 1, &_copy);
#endif//CROSS_CORRELATE
    rt_dma_memcpy((unsigned int)net_l2_factor,
                  (unsigned int)_p_factor_loc,
                  sizeof(int32_t) * NET_F2,
                  RT_DMA_DIR_EXT2LOC, 1, &_copy);
    rt_dma_memcpy((unsigned int)net_l2_offset,
                  (unsigned int)_p_offset_loc,
                  sizeof(int32_t) * NET_F2,
                  RT_DMA_DIR_EXT2LOC, 1, &_copy);

    // wait until all dma transfers of the input data is complete
    rt_dma_wait(&_copy);

    int32_t _factor;
    int32_t _offset;
    int32_t _sum;  // stores the sum for the pooling
    
    int8_t* _p_data_thread_iter;
    int8_t* _p_data_loc_iter = _p_data_loc;
    int8_t* _p_weight_loc_iter = _p_weight_loc;
    int8_t* _p_result_loc_iter = _p_result_loc;

    unsigned int _filter = 0;
    while (_filter < NET_F2) {
        _p_data_loc_iter = _p_data_loc + _filter * NET_L2_PAD_INPUT_LEN_ALIGN;
        _p_weight_loc_iter = _p_weight_loc + _filter * NET_L2_WEIGHT_LEN_ALIGN;
        _p_result_loc_iter = _p_result_loc+ _filter * NET_T8;
        _p_data_thread_iter = _p_thread;
        
        _factor = _p_factor_loc[_filter];
        _offset = _p_offset_loc[_filter];

        _factor = _factor >> 3;
        _offset = _offset >> 3;
#ifdef CROSS_CORRELATE
        func_xcorr_scale(_p_data_loc_iter, NET_L2_PAD_INPUT_LEN,
                         _p_weight_loc_iter, NET_L2_WEIGHT_LEN,
                         _factor, _offset, _p_thread);
#else //CROSS_CORRELATE
        func_conv_scale(_p_data_loc_iter, NET_L2_PAD_INPUT_LEN,
                        _p_weight_loc_iter, NET_L2_WEIGHT_LEN,
                        _factor, _offset, _p_thread);
#endif //CROSS_CORRELATE

        for (unsigned int _t_out = 0; _t_out < NET_T8; _t_out++) {
            // reset the sum
            _sum = 0;
            // loop over all 8 elements in the local neighborhood
            for (unsigned int _t_pool = 0; _t_pool < 8; _t_pool++) {

                // do relu on each element
                _sum += __MAX(*(_p_data_thread_iter++), 0);

            }

            // do avg pooling division
            _sum = _sum >> 3;

            // clamp
            _sum = __CLIP_R(_sum, 127);

            *(_p_result_loc_iter+_t_out) = (int8_t)_sum;
        }

        // update filter
        _filter += 1; 
    }

    // copy back the results
    rt_dma_memcpy((unsigned int)p_result,
                  (unsigned int)_p_result_loc,
                  sizeof(int8_t) * NET_F2 * NET_T8_ALIGN,
                  RT_DMA_DIR_LOC2EXT, 0, &_copy);
    rt_dma_wait(&_copy);

    // free up the memory
    rt_free(RT_ALLOC_CL_DATA, _p_data_loc, sizeof(int8_t) * NET_F1 * NET_L2_PAD_INPUT_LEN);
    rt_free(RT_ALLOC_CL_DATA, _p_weight_loc, sizeof(int8_t) * NET_F2 * NET_L2_WEIGHT_LEN_ALIGN);
    rt_free(RT_ALLOC_CL_DATA, _p_result_loc, sizeof(int8_t) * NET_F2 * NET_T8_ALIGN);
    rt_free(RT_ALLOC_CL_DATA, _p_factor_loc, sizeof(int32_t) * NET_F2);
    rt_free(RT_ALLOC_CL_DATA, _p_offset_loc, sizeof(int32_t) * NET_F2);
    rt_free(RT_ALLOC_CL_DATA, _p_thread, sizeof(int8_t) * NET_L2_PAD_INPUT_LEN);

#endif//REORDER_BN
#endif//parallel
}
