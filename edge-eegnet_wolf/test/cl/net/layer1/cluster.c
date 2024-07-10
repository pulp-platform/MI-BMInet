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


#include "stdio.h"
#include "rt/rt_api.h"
#include "test_stimuli.h"
#include "../../../../src/cl/func/functional.h"
#include "../../../../src/cl/net/net.h"
#include "../../../../src/cl/net/layers.h"

int do_bench(rt_perf_t* perf, int events) {

    // allocate result memory
    int8_t * p_output = rt_alloc(RT_ALLOC_FC_DATA, sizeof(int8_t) * NET_F1 * NET_L2_PAD_INPUT_LEN);
    
    //setup performance measurement
    rt_perf_conf(perf, events);
    
    // start performance measurement
    rt_perf_reset(perf);
    rt_perf_start(perf);
    
    net_layer1(x_vec, p_output);

    rt_perf_stop(perf);

    // for (int i = 0; i < NET_F1 * NET_L2_PAD_INPUT_LEN; i++){
    //     printf("%d, ", *(p_output + i));
    // }

    int num_err = 0;
    int8_t* p_output_tmp;
    
    for (int k = 0; k < NET_F1; k++) {
        for (int ch = 0; ch < NET_T; ch++) {
            const int8_t* p_exp_tmp;
            p_output_tmp = p_output + (k * NET_T_ALIGN) + ch;
            p_exp_tmp = y_exp_vec + (k * NET_T_ALIGN) + ch;
            if (*(p_output_tmp) != *(p_exp_tmp)) {
                // printf("got: %d (%d) at posistion: [%d,%d]\n", *(p_output_tmp), *(p_exp_tmp), k, ch);
                num_err += 1;
            }
        }
    }

    // free memory
    rt_free(RT_ALLOC_L2_CL_DATA, (void*) p_output, sizeof(int8_t) * NET_F1 * NET_C * NET_T_ALIGN);

    return num_err;
}

void cluster_entry(void* arg) {

    // setup performance measurement
    rt_perf_t perf;
    rt_perf_init(&perf);

    int result;

    result = do_bench(&perf, (1<<RT_PERF_CYCLES | 1<<RT_PERF_INSTR));

    // print the results
    if (result == 0) {
        printf("## 1: result: OK\n");
    } else {
        printf("## 1: result: FAIL\n");
    }
    printf("## 1: errors: %d\n", result);
    printf("## 1: cycles: %d\n", rt_perf_read(RT_PERF_CYCLES));
    printf("## 1: instructions: %d\n", rt_perf_read(RT_PERF_INSTR));
}
