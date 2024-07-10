"""
This file will test the convolution implementation
"""

__author__ = "Tibor Schneider"
__email__ = "sctibor@student.ethz.ch"
__version__ = "1.0"
__license__ = "Apache 2.0"
__copyright__ = """
    Copyright (C) 2020 ETH Zurich. All rights reserved.

    Author: Tibor Schneider, ETH Zurich

    SPDX-License-Identifier: Apache-2.0

    Licensed under the Apache License, Version 2.0 (the License); you may
    not use this file except in compliance with the License.
    You may obtain a copy of the License at

    www.apache.org/licenses/LICENSE-2.0

    Unless required by applicable law or agreed to in writing, software
    distributed under the License is distributed on an AS IS BASIS, WITHOUT
    WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
    See the License for the specific language governing permissions and
    limitations under the License.
"""


import random
import os
import numpy as np
from test_utils import parse_output, TestLogger
from header_file import HeaderFile, HeaderConstant, HeaderArray
from makefile import Makefile

TESTNAME = "cl::func::conv"
RESULT_FILE = "result.out"


def gen_stimuli(size = 100, do_mult=False):
    """
    This function generates the stimuli (input and output) for the test
    """
    if do_mult:
        vecA = [random.randint(-32768, 32768) for _ in range(size)]
        vecB = [random.randint(-32768, 32768) for _ in range(size)]
        result = np.multiply(vecA, vecB)
        return vecA, vecB, result
    else:
        vecA = [random.randint(-1073741824, 1073741824) for _ in range(size)]
        vecB = [random.randint(-32768, 32768) for _ in range(size)]
        
        vecMax = np.maximum(vecA, vecB)
        vecMin = np.minimum(vecA, vecB)
        
        result = np.fix(np.divide(vecMax, vecMin)).astype(int)
    
        return vecMax, vecMin, result


def test():
    """
    Execute the tests
    Returns: (n_total, n_success)
    """

    logger = TestLogger(TESTNAME)

    for do_mult in [False, True]:
        for n in [50, 100, 500, 1000]:
            # generate makefile
            mkf = Makefile()
            mkf.add_fc_test_source("test.c")
            mkf.add_cl_test_source("cluster.c")
            
            if do_mult:
                mkf.add_define("MULT", do_mult)
            
            mkf.write()

            # generate the stimuli
            vecA, vecB, res = gen_stimuli(n, do_mult)

            # prepare header file
            header = HeaderFile("test_stimuli.h")

            header.add(HeaderConstant("LENGTH_VEC", len(vecA)))
            header.add(HeaderArray("vecA", "int32_t", vecA))
            header.add(HeaderArray("vecB", "int32_t", vecB))
            header.add(HeaderArray("result", "int32_t", res))

            header.write()

            # compile and run
            os.system("make clean all run > {}".format(RESULT_FILE))

            # parse output
            result = parse_output(RESULT_FILE)

            if do_mult:
                case = "Mult"
            else:
                case = "Div"
            casename = "Speed Test: {} {} runs".format(case, n)

            # log the result
            logger.show_subcase_result(casename, result)

    # return summary
    return logger.summary()
