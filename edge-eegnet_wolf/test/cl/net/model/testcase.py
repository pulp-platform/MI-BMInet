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
from header_file import HeaderFile, HeaderConstant, HeaderArray, align_array, align_array_size
from makefile import Makefile
from golden_model import GoldenModel
import functional as F

TESTNAME = "cl::net::model"
RESULT_FILE = "result.out"

INPUT_FILENAME = "../../../../data/verification.npz"
NET_FILENAME = "../../../../data/net.npz"
CONFIG_FILENAME = "../../../../data/config.json"


def gen_stimuli(random_input=False, reorder_bn=True):
    """
    This function generates the stimuli (input and output) for the test
    """
    model = GoldenModel(CONFIG_FILENAME, NET_FILENAME, clip_balanced=False, reorder_bn=reorder_bn)
    if random_input:
        x = np.random.randint(-60, 60, (model.C, model.T))
    else:
        x = np.load(INPUT_FILENAME)["input"][0, :, :]
        x = F.quantize_to_int(x, model.input_scale)
 
    y_exp = model(x)
    y_exp_align = align_array(y_exp)

    x = np.transpose(x, (1, 0)) # store flipped data for better preformance
    x_align = align_array(x)
 
    return x, x_align, y_exp, y_exp_align

def test():
    """
    Execute the tests
    Returns: (n_total, n_success)
    """

    logger = TestLogger(TESTNAME)

    for simd, parallel, xcorr, reorder in [
        (False, False, False, False),
        (True, False, False, False),
        (True, True, False, False),
        (True, True, True, False),
        (True, True, True, True),
    ]:
        # generate makefile
        mkf = Makefile()
        mkf.add_fc_test_source("test.c")
        mkf.add_cl_test_source("cluster.c")
        mkf.add_cl_prog_source("net/model.c")
        mkf.add_cl_prog_source("net/layer1.c")
        mkf.add_cl_prog_source("net/layer2.c")
        mkf.add_cl_prog_source("net/layer3.c")
        mkf.add_cl_prog_source("net/layer4.c")
        mkf.add_cl_prog_source("net/layer5.c")
        mkf.add_cl_prog_source("net/net.c")
        mkf.add_cl_prog_source("func/transform.c")
        mkf.add_cl_prog_source("func/dotp.c")
        mkf.add_cl_prog_source("func/conv.c")
        mkf.add_cl_prog_source("func/flip.c")
        mkf.add_cl_prog_source("func/xcorr.c")

        if not simd:
            mkf.add_define("NO_SIMD")
        if parallel:
            mkf.add_define("PARALLEL")
        if xcorr:
            mkf.add_define("CROSS_CORRELATE")
        if reorder:
            mkf.add_define("REORDER_BN")
        mkf.add_define("FLIP_LAYERS")
        

        mkf.write()

        # generate the stimuli
        _, x_align, _, y_exp_align = gen_stimuli(reorder_bn=reorder)

        # prepare header file
        header = HeaderFile("test_stimuli.h")
        header.add(HeaderArray("x_vec", "int8_t", x_align.ravel()))
        header.add(HeaderArray("y_exp_vec", "int8_t", y_exp_align.ravel()))
        header.write()

        # compile and run
        os.system("make clean all run > {}".format(RESULT_FILE))

        # parse output
        result = parse_output(RESULT_FILE)

        # prepare the case name
        subcase_name = "naive"
        if simd:
            subcase_name = "+ SIMD"
        if parallel:
            subcase_name = "+ parallel"
        if xcorr:
            subcase_name = "+ cross correlations"
        if reorder:
            subcase_name = "+ reorder BN"

        # log the result
        logger.show_subcase_result(subcase_name, result)

    # return summary
    return logger.summary()
