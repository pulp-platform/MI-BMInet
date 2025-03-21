"""
This File generates the header files defining the trained and quantized network.
The following files are required
- [project_root]/data/config.json containing the QuantLab configuration how the network was trained
- [project_root]/data/net.npz, containing the entire network
"""

__author__ = "Tibor Schneider"
__email__ = "sctibor@student.ethz.ch"
__version__ = "0.0.2"
__date__ = "2020/01/28"
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


import argparse
import json
import numpy as np

from header_file import HeaderFile, HeaderConstant, HeaderScalar, HeaderArray, HeaderComment
from header_file import align_array, align_array_size
import convert_torch_format as convert

DEFAULT_HEADER_NAME = "../src/cl/net/net.h"
DEFAULT_CONFIG_JSON = "config.json"
DEFAULT_NET_NPZ = "net.npz"

WEIGHT_L2_PAD = 4 * 0


def gen_net_header(net_file, config_file, output_file):

    # load network
    net = np.load(net_file)

    # load configuration file
    with open(config_file, "r") as _f:
        config = json.load(_f)
    # we only need the network parameters
    net_params = config["indiv"]["net"]["params"]

    # only allow nets with 255 levels
    assert net_params["weightInqNumLevels"] == 255
    assert net_params["actSTENumLevels"] == 255
    assert net_params["F2"] % 4 == 0
    assert net_params["N"] == 4

    # prepare params
    if net_params["F2"] is None:
        net_params["F2"] = net_params["F1"] * net_params["D"]

    # only allow F2 = F1 * D
    assert net_params["F2"] == net_params["F1"] * net_params["D"]

    # start the header file
    header = HeaderFile(output_file, "__NET_NET_H__", with_c=True)

    # add network dimensions
    header.add(HeaderComment("Network Dimensions", blank_line=False))
    header.add(HeaderConstant("NET_F1", net_params["F1"], blank_line=False))
    header.add(HeaderConstant("NET_F2", net_params["F2"], blank_line=False))
    header.add(HeaderConstant("NET_D", net_params["D"], blank_line=False))
    header.add(HeaderConstant("NET_C", net_params["C"], blank_line=False))
    header.add(HeaderConstant("NET_C_ALIGN", align_array_size(net_params["C"]), blank_line=False))
    header.add(HeaderConstant("NET_T", net_params["T"], blank_line=False))
    header.add(HeaderConstant("NET_T_ALIGN", align_array_size(net_params["T"]), blank_line=False))
    header.add(HeaderConstant("NET_T8", net_params["T"] // 8, blank_line=False))
    header.add(HeaderConstant("NET_T8_ALIGN", align_array_size(net_params["T"] // 8), blank_line=False))
    header.add(HeaderConstant("NET_T64", (net_params["T"] // 8) // 8, blank_line=False))
    header.add(HeaderConstant("NET_T64_ALIGN", align_array_size((net_params["T"] // 8) // 8), blank_line=False))
    header.add(HeaderConstant("NET_N", net_params["N"], blank_line=True))

    # Layer 1
    input_scale = convert.ste_quant(net, "quant1")
    weight, weight_scale = convert.inq_conv2d(net, "conv1", store_reversed=True)
    bn_scale, bn_offset = convert.batch_norm(net, "batch_norm1")
    output_scale = convert.ste_quant(net, "quant2")
    factor, offset = convert.div_factor_batch_norm(input_scale, weight_scale, output_scale, bn_scale, bn_offset)
    weight = weight.reshape(net_params["F2"], net_params["C"])
    weight_align = align_array(weight)
    
    # weight = np.flip(weight.reshape(net_params["F2"], net_params["C"]),axis=1)
   
    header.add(HeaderComment("Layer 1\n"
                             "=======\n"
                             "Convolution + BN\n\n"
                             "Input:  [C, T]\n"
                             "Weight: [C, 1]\n"
                             "Output: [F2, 1, T]",
                             mode="/*"))
    header.add(HeaderArray("net_l1_factor", "int32_t", factor.ravel()))
    header.add(HeaderArray("net_l1_offset", "int32_t", offset.ravel()))
    header.add(HeaderConstant("NET_L1_WEIGHT_LEN", weight.shape[-1]))
    header.add(HeaderConstant("NET_L1_WEIGHT_LEN_ALIGN", align_array_size(weight.shape[-1])))
    header.add(HeaderArray("net_l1_weight", "int8_t", weight.ravel()))
    header.add(HeaderArray("net_l1_weight_align", "int8_t", weight_align.ravel()))
    header.add(HeaderArray("net_l1_weight_32", "int32_t", weight.ravel()))

    header.write()
    
    # layer2
    input_scale = convert.ste_quant(net, "quant2")
    weight, weight_scale = convert.inq_conv2d(net, "conv2")
    weight = weight.reshape(net_params["F2"], 64)
    weight = align_array(weight)
    weight_reverse, _ = convert.inq_conv2d(net, "conv2", store_reversed=True)
    weight_reverse = weight_reverse.reshape(net_params["F2"], 64)
    weight_reverse = align_array(weight_reverse)
    bn_scale, bn_offset = convert.batch_norm(net, "batch_norm2")
    output_scale = convert.ste_quant(net, "quant3")
    factor, offset = convert.div_factor_batch_norm(input_scale, weight_scale, output_scale, bn_scale, bn_offset, pool=8)

    # add padding to the weight vector of 4
    if WEIGHT_L2_PAD > 0:
        weight_reverse_pad = np.zeros((net_params["F2"], 64 + WEIGHT_L2_PAD))
        weight_reverse_pad[:, :-WEIGHT_L2_PAD] = weight_reverse
    else:
        weight_reverse_pad = weight_reverse

    header.add(HeaderComment("Layer 2\n"
                             "=======\n"
                             "Convolution + BN + ReLU + Pooling\n\n"
                             "Input:  [F2, 1, T]\n"
                             "Weight: [F2, 1, 64]\n"
                             "Output: [F2, T // 8]",
                             mode="/*"))
    header.add(HeaderConstant("NET_L2_PAD_START", 31))
    header.add(HeaderConstant("NET_L2_PAD_END", 32))
    header.add(HeaderConstant("NET_L2_PAD_INPUT_LEN", net_params["T"] + 31 + 32))
    header.add(HeaderConstant("NET_L2_PAD_INPUT_LEN_ALIGN", align_array_size(net_params["T"] + 31 + 32)))
    header.add(HeaderArray("net_l2_factor", "int32_t", factor.ravel()))
    header.add(HeaderArray("net_l2_offset", "int32_t", offset.ravel()))
    header.add(HeaderConstant("NET_L2_WEIGHT_LEN", weight.shape[-1]))
    header.add(HeaderConstant("NET_L2_WEIGHT_LEN_ALIGN", weight_reverse_pad.shape[-1]))
    header.add(HeaderArray("net_l2_weight", "int8_t", weight.ravel()))
    header.add(HeaderArray("net_l2_weight_reverse", "int8_t", weight_reverse.ravel()))
    header.add(HeaderArray("net_l2_weight_reverse_pad", "int8_t", weight_reverse_pad.ravel()))

    # layer3
    input_scale = convert.ste_quant(net, "quant3")
    weight, weight_scale = convert.inq_conv2d(net, "sep_conv1")
    output_scale = convert.ste_quant(net, "quant4")
    factor = convert.div_factor(input_scale, weight_scale, output_scale)
    weight = weight.reshape(net_params["F2"], 16)

    header.add(HeaderComment("Layer 3\n"
                             "=======\n"
                             "Convolution\n\n"
                             "Input:  [F2, T // 8]\n"
                             "Weight: [F2, 16]\n"
                             "Output: [F2, T // 8]",
                             mode="/*", blank_line=False))
    header.add(HeaderConstant("NET_L3_PAD_START", 7))
    header.add(HeaderConstant("NET_L3_PAD_END", 8))
    header.add(HeaderConstant("NET_L3_PAD_INPUT_LEN", net_params["T"] // 8 + 7 + 8))
    header.add(HeaderConstant("NET_L3_PAD_INPUT_LEN_ALIGN", align_array_size(net_params["T"] // 8 + 7 + 8)))
    header.add(HeaderConstant("NET_L3_FACTOR", factor))
    header.add(HeaderConstant("NET_L3_WEIGHT_LEN", weight.shape[-1]))
    header.add(HeaderArray("net_l3_weight", "int8_t", weight.ravel()))

    # layer4
    input_scale = convert.ste_quant(net, "quant4")
    weight, weight_scale = convert.inq_conv2d(net, "sep_conv2")
    output_scale = convert.ste_quant(net, "quant5")
    bn_scale, bn_offset = convert.batch_norm(net, "batch_norm3")
    factor, offset = convert.div_factor_batch_norm(input_scale, weight_scale, output_scale, bn_scale, bn_offset, pool=8)
    weight = weight.reshape(net_params["F2"], net_params["F2"])

    header.add(HeaderComment("Layer 4\n"
                             "=======\n"
                             "Convolution + BN + ReLU + Pooling\n\n"
                             "Input:  [F2, T // 8]\n"
                             "Weight: [F2, F2]\n"
                             "Output: [F2, T // 64]",
                             mode="/*"))
    header.add(HeaderArray("net_l4_factor", "int32_t", factor.ravel()))
    header.add(HeaderArray("net_l4_offset", "int32_t", offset.ravel()))
    header.add(HeaderConstant("NET_L4_WEIGHT_LEN", weight.shape[-1]))
    header.add(HeaderArray("net_l4_weight", "int8_t", weight.ravel()))

    # layer5
    input_scale = convert.ste_quant(net, "quant5")
    output_scale = convert.ste_quant(net, "quant6")
    weight, bias, weight_scale = convert.inq_linear(net, "fc")
    weight = weight.reshape(net_params["N"], net_params["F2"] * (net_params["T"] // 64))
    #weight = align_array(weight)
    # we want to align, not for the product F2*T//64, but for T//64 itself.
    t64 = net_params["T"] // 64
    t64_align = align_array_size(t64)
    weight_align = np.zeros((net_params["N"], net_params["F2"] * t64_align), dtype=int)
    for i in range(net_params["F2"]):
        weight_align[:, i * t64_align: i * t64_align + t64] = weight[:, i * t64: (i + 1) * t64]
    factor = convert.div_factor(input_scale, weight_scale, output_scale)

    header.add(HeaderComment("Layer 5\n"
                             "=======\n"
                             "Linear Layer (without scaling in the end)\n\n"
                             "Input:  [F2, T // 64]\n"
                             "Weight: [N, F2 * (T // 64)]\n"
                             "Bias:   [N]\n"
                             "Output: [N]",
                             mode="/*"))
    header.add(HeaderConstant("NET_L5_FACTOR", factor))
    header.add(HeaderArray("net_l5_bias", "int8_t", bias.ravel()))
    header.add(HeaderConstant("NET_L5_WEIGHT_LEN", weight_align.shape[-1]))
    header.add(HeaderArray("net_l5_weight", "int8_t", weight_align.ravel()))

    # store the header file
    # exit()
    header.write()


if __name__ == "__main__":

    parser = argparse.ArgumentParser("Generates the header file defining the trained EEGNet")
    parser.add_argument("-o", "--output", help="Export header file name", default=DEFAULT_HEADER_NAME)
    parser.add_argument("-n", "--net",    help="numpy file containing the network", default=DEFAULT_NET_NPZ)
    parser.add_argument("-c", "--config", help="configuration file name", default=DEFAULT_CONFIG_JSON)
    args = parser.parse_args()

    gen_net_header(args.net, args.config, args.output)
