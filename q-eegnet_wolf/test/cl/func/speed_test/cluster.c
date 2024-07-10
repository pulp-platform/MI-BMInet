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

RT_CL_DATA static int32_t* _vecA;
RT_CL_DATA static int32_t* _vecB;
RT_CL_DATA static int32_t* _res;

void cluster_entry(void* arg) {
    _vecA = rt_alloc(RT_ALLOC_CL_DATA, sizeof(vecA));
    _vecB = rt_alloc(RT_ALLOC_CL_DATA, sizeof(vecB));
    _res = rt_alloc(RT_ALLOC_CL_DATA, sizeof(vecB));

    rt_dma_copy_t copy;
    rt_dma_memcpy((unsigned int)vecA, (unsigned int)_vecA, sizeof(vecA), RT_DMA_DIR_EXT2LOC, 0, &copy);
    rt_dma_memcpy((unsigned int)vecB, (unsigned int)_vecB, sizeof(vecB), RT_DMA_DIR_EXT2LOC, 1, &copy);
    rt_dma_wait(&copy);


    // setup performance measurement
    rt_perf_t perf;
    rt_perf_init(&perf);
    rt_perf_conf(&perf, (1<<RT_PERF_CYCLES | 1<<RT_PERF_INSTR));

    // start performance measurement
    int result;
    rt_perf_reset(&perf);
    rt_perf_start(&perf);

#ifdef MULT
    for (int i = 0; i < LENGTH_VEC; i++){
        _res[i] = _vecA[i] * _vecB[i];
    }
#else//MULT
    for (int i = 0; i < LENGTH_VEC; i++){
        _res[i] = _vecA[i] / _vecB[i];
    }
#endif//MULT
    
    rt_perf_stop(&perf);

    printf("## 1: result: OK\n");
    printf("## 1: cycles: %d\n", rt_perf_read(RT_PERF_CYCLES));
    printf("## 1: instructions: %d\n", rt_perf_read(RT_PERF_INSTR));
}
