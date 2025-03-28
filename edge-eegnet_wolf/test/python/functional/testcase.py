"""
Test the golden model
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


import numpy as np
from test_utils import TestLogger
import convert_torch_format as convert
import functional as F

# TODO add a meaningful name for the test
TESTNAME = "python::Functional"
NET_FILENAME = "../../../data/net.npz"
DATA_FILENAME = "../../../data/verification.npz"
CONFIG_FILENAME = "../../../data/config.json"

EPSILON = 1e-5

def test():
    """
    Execute the tests
    Returns: (n_total, n_success)
    """
    logger = TestLogger(TESTNAME)
    
    # load net and the data
    data = dict(np.load(DATA_FILENAME))
    net = dict(np.load(NET_FILENAME))

    logger.show_subcase_result("Quantize", test_quantize(net, data))
    logger.show_subcase_result("Batch Norm", test_batch_norm(net, data, layer=1))
    logger.show_subcase_result("Average Pool", test_avg_pool(net, data))
    logger.show_subcase_result("Conv in Space", test_depthwise_conv_space(net, data))
    logger.show_subcase_result("Conv in Time 1", test_conv_time(net, data))
    logger.show_subcase_result("Conv in Time 2", test_depthwise_conv_time(net, data))
    logger.show_subcase_result("Conv pointwise", test_pointwise_conv(net, data))
    logger.show_subcase_result("Linear layer", test_linear(net, data))

    # return summary
    return logger.summary()


def test_quantize(net, data, epsilon=EPSILON):
    x_list = [data["input"][0, :, :],
              data["layer1_bn_out"][0, :, :, :],
              data["layer2_pool_out"][0, :, 0, :],
              data["layer3_conv_out"][0, :, 0, :],
              data["layer4_pool_out"][0, :, 0, :]]
    y_exp_list = [data["input_quant"][0, 0, :, :],
                  data["layer1_activ"][0, :, :, :],
                  data["layer2_activ"][0, :, 0, :],
                  data["layer3_activ"][0, :, 0, :],
                  data["layer4_activ"][0, :, 0, :]]
    scale_list = [convert.ste_quant(net, "quant1"),
                  convert.ste_quant(net, "quant2"),
                  convert.ste_quant(net, "quant3"),
                  convert.ste_quant(net, "quant4"),
                  convert.ste_quant(net, "quant5")]
    casename_list = ["input", "layer1", "layer2", "layer3", "layer4"]

    ret = {}
    for casename, x, y_exp, scale in zip(casename_list, x_list, y_exp_list, scale_list):

        y = F.quantize(x, scale)

        l1_error = np.abs(y - y_exp).mean()
        success = l1_error < epsilon

        ret[casename] = {"result": success, "l1_error": l1_error}

    return ret


def test_batch_norm(net, data, layer=2, epsilon=EPSILON):
    if layer == 1:
        x = data["layer1_conv_out"][0,:,:,:]
        y_exp = data["layer1_bn_out"][0,:,:,:]
        scale, offset = convert.batch_norm(net, "batch_norm1")
    elif layer == 2:
        x = data["layer2_conv_out"][0,:,0,:]
        y_exp = data["layer2_bn_out"][0,:,0,:]
        scale, offset = convert.batch_norm(net, "batch_norm2")
    elif layer == 4:
        x = data["layer4_conv_out"][0,:,0,:]
        y_exp = data["layer4_bn_out"][0,:,0,:]
        scale, offset = convert.batch_norm(net, "batch_norm3")
    else:
        raise TypeError("Layer must be either 1, 2 or 4")

    y = F.batch_norm(x, scale, offset)

    l1_error = np.abs(y - y_exp).mean()
    success = l1_error < epsilon

    return {"Batch Norm": {"result": success, "l1_error": l1_error}}


def test_avg_pool(net, data, epsilon=EPSILON):
    x = data["layer2_relu_out"][0,:,0,:]
    y_exp = data["layer2_pool_out"][0,:,0,:]

    y = F.pool(x, (1, 8), "mean")

    l1_error = np.abs(y - y_exp).mean()
    success = l1_error < epsilon

    return {"Batch Norm": {"result": success, "l1_error": l1_error}}


def test_conv_time(net, data, epsilon=EPSILON):
    x = data["layer1_activ"][0, :,:,:]
    y_exp = data["layer2_conv_out"][0,:,0,:]
    weight = net["conv2.weightFrozen"][:,0,0,:]
    weight = np.flip(weight, 1)

    y = F.conv_time(x, weight)

    l1_error = np.abs(y - y_exp).mean()
    success = l1_error < epsilon

    return {"Batch Norm": {"result": success, "l1_error": l1_error}}

def test_depthwise_conv_space(net, data, epsilon=EPSILON):
    x = data["input_quant"][0,0,:,:]
    y_exp = data["layer1_conv_out"][0,:,:,:]
    weight = net["conv1.weightFrozen"][:,0,:,0]
    weight = np.flip(weight, 1)

    y = F.depthwise_conv_space(x, weight)

    l1_error = np.abs(y - y_exp).mean()
    success = l1_error < epsilon

    return {"Batch Norm": {"result": success, "l1_error": l1_error}}


def test_depthwise_conv_time(net, data, epsilon=EPSILON):
    x = data["layer2_activ"][0,:,0,:]
    y_exp = data["layer3_conv_out"][0,:,0,:]
    weight = net["sep_conv1.weightFrozen"][:,0,0,:]
    weight = np.flip(weight, 1)

    y = F.depthwise_conv_time(x, weight)

    l1_error = np.abs(y - y_exp).mean()
    success = l1_error < epsilon

    return {"Batch Norm": {"result": success, "l1_error": l1_error}}


def test_pointwise_conv(net, data, epsilon=EPSILON):
    x = data["layer3_activ"][0,:,0,:]
    y_exp = data["layer4_conv_out"][0,:,0,:]
    weight = net["sep_conv2.weightFrozen"][:,:,0,0]

    y = F.pointwise_conv(x, weight)

    l1_error = np.abs(y - y_exp).mean()
    success = l1_error < epsilon

    return {"Batch Norm": {"result": success, "l1_error": l1_error}}


def test_linear(net, data, epsilon=EPSILON):
    x = data["layer4_activ"].ravel()
    y_exp = data["output"][0,:]
    weight = net["fc.weightFrozen"][:,:]
    bias = net["fc.bias"][:]

    y = F.linear(x, weight, bias)

    l1_error = np.abs(y - y_exp).mean()
    success = l1_error < epsilon

    return {"Batch Norm": {"result": success, "l1_error": l1_error}}

