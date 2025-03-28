"""
This File generates the header files defining the input data.
The following files are required
- [project_root]/data/input.npz containing the data
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
from header_file import align_array
import convert_torch_format as convert
import functional as F

DEFAULT_HEADER_NAME = "../src/cl/input.h"
# DEFAULT_HEADER_NAME = "headers/input.h"
DEFAULT_CONFIG_JSON = "config.json"
DEFAULT_NET_NPZ = "net.npz"
DEFAULT_INPUT_NPZ = "input.npz"


def gen_input_header_from_file(net_file, config_file, input_file, output_file):
    # load network
    net = np.load(net_file)
    data = np.load(input_file)

    # load configuration file
    with open(config_file, "r") as _f:
        config = json.load(_f)
    # we only need the network parameters
    net_params = config["indiv"]["net"]["params"]

    gen_input_header(net, net_params, data["input"], output_file)


def gen_input_header(net, net_params, data, output_file):

    # only allow nets with 255 levels
    assert net_params["weightInqNumLevels"] == 255
    assert net_params["actSTENumLevels"] == 255

    # extract and prepare the input data
    scale_factor = convert.ste_quant(net, "quant1")
    input_quant = F.quantize_to_int(data, scale_factor)    
    input_quant_trans = np.transpose(input_quant, (0, 2, 1))
    input_quant_trans_align = align_array(input_quant_trans)

    # generate the header file
    header = HeaderFile(output_file, "__INPUT_H__", with_c=True)
    header.add(HeaderArray("input_data", "int8_t", input_quant_trans.ravel()))
    header.add(HeaderArray("input_data_align", "int8_t", input_quant_trans_align.ravel()))
    header.write()


if __name__ == "__main__":

    parser = argparse.ArgumentParser("Generates the header file defining the trained EEGNet")
    parser.add_argument("-o", "--output", help="Export header file name", default=DEFAULT_HEADER_NAME)
    parser.add_argument("-n", "--net",    help="numpy file containing the network", default=DEFAULT_NET_NPZ)
    parser.add_argument("-c", "--config", help="configuration file name", default=DEFAULT_CONFIG_JSON)
    parser.add_argument("-i", "--input", help="numpy file containing the input", default=DEFAULT_INPUT_NPZ)
    args = parser.parse_args()

    gen_input_header_from_file(args.net, args.config, args.input, args.output)
