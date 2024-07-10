/**
 * @file model.c
 * @author Tibor Schneider
 * @date 2020/02/01
 * @brief This file contains the implementation for the main model function
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
#include "model.h"
#include "layers.h"
#include "net.h"

/**
 * @brief computes the output of the entire model
 *
 * @warning p_output must already be allocated on L2 memory
 *
 * @param p_data Pointer to the input data, of shape [NET_T, NET_C], aligned to [NET_T, NET_C_ALIGN]
 * @param p_output Pointer to output data, allocated on L2 memory, of shape [NET_N]
 */
void net_model_compute(const int8_t* p_data, int8_t* p_output) {
    /*
     * Layer 1
     */

    // allocate data for result
    int8_t * p_l1_output = rt_alloc(RT_ALLOC_L2_CL_DATA, sizeof(int8_t) * NET_F1 * NET_T_ALIGN);

    // compute layer 1
    net_layer1(p_data, p_l1_output);

    /*
     * Layer 2
     */

    // allocate memory
    int8_t* p_l2_output = rt_alloc(RT_ALLOC_L2_CL_DATA, sizeof(int8_t) * NET_F2 * NET_T8_ALIGN);
    
    // // compute layer 2
    net_layer2(p_l1_output, p_l2_output);

    // free l1 memory
    rt_free(RT_ALLOC_L2_CL_DATA, (void*)p_l1_output, sizeof(int8_t) * NET_F1 * NET_T_ALIGN);

    /*
     * Layer 3
     */

    // allocate memory
    int8_t * p_l3_output = rt_alloc(RT_ALLOC_L2_CL_DATA, sizeof(int8_t) * NET_F2 * NET_T8_ALIGN);

    // compute layer 3
    net_layer3(p_l2_output, p_l3_output);
    
    // free l2 memory
    rt_free(RT_ALLOC_L2_CL_DATA, (void*)p_l2_output, sizeof(int8_t) * NET_F2 * NET_T8_ALIGN);
    
#ifdef FLIP_LAYERS
    // flip the dimension
    net_layer3_flip_inplace(p_l3_output);
#endif //FLIP_LAYERS

    /*
     * Layer 4
     */

    // allocate memory
    int8_t * p_l4_output = rt_alloc(RT_ALLOC_L2_CL_DATA, sizeof(int8_t) * NET_F2 * NET_T64_ALIGN);

    // compute layer 4
    net_layer4(p_l3_output, p_l4_output);

    // free l3 memory
    rt_free(RT_ALLOC_L2_CL_DATA, (void*)p_l3_output, sizeof(int8_t) * NET_F2 * NET_T8_ALIGN);

    /*
     * Layer 5
     */

    // compute layer 5
    net_layer5(p_l4_output, p_output);

    // free l4 memory
    rt_free(RT_ALLOC_L2_CL_DATA, (void*)p_l4_output, sizeof(int8_t) * NET_F2 * NET_T64_ALIGN);
}
