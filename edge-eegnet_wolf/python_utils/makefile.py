"""
Class to generate a Makefile
"""

__author__ = "Tibor Schneider"
__email__ = "sctibor@student.ethz.ch"
__version__ = "0.1.0"
__date__ = "2020/01/29"
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


import os

FILES_IN_ROOT = {"src", "test", "doc", "data", "Makefile"}


class Makefile:
    """ Makefile generation """
    def __init__(self, project_root=None, use_dsp=True, opt_level=3):
        self.fc_sources = []
        self.cl_sources = []
        self.defines = []
        self.use_dsp = use_dsp
        self.project_root = project_root
        self.opt_level=opt_level
        if self.project_root is None:
            # search the project root
            self.project_root = os.getcwd()
            current_files = set(os.listdir(self.project_root))
            while not FILES_IN_ROOT <= current_files:
                # go in parent directory
                self.project_root = os.path.realpath(os.path.join(self.project_root, ".."))
                current_files = set(os.listdir(self.project_root))
                assert self.project_root != "/"
        else:
            current_files = set(os.listdir(self.project_root))
            assert FILES_IN_ROOT <= current_files

    def add_fc_test_source(self, name):
        """ add test source file, located in current directory """
        self.fc_sources.append(name)

    def add_cl_test_source(self, name):
        """ add test source file, located in current directory """
        self.cl_sources.append(name)

    def add_fc_prog_source(self, name):
        """ add source file from the actual program, starting at root/src/fc/ """
        source_file = os.path.join(self.project_root, "src/fc", name)
        assert os.path.exists(source_file)
        assert source_file.endswith(".c")
        self.cl_sources.append(source_file)

    def add_cl_prog_source(self, name):
        """ add source file from the actual program, starting at root/src/cl/ """
        source_file = os.path.join(self.project_root, "src/cl", name)
        assert os.path.exists(source_file)
        assert source_file.endswith(".c")
        self.cl_sources.append(source_file)

    def add_define(self, name, value=None):
        """ Those defines will be passed to gcc with -Dname=value flag """
        assert name.isupper()
        if value is None:
            self.defines.append(name)
        else:
            self.defines.append("{}={}".format(name, value))

    def __str__(self):
        ret = ""
        ret += "PULP_APP = test\n\n"

        # add cl sources
        if self.cl_sources:
            ret += "PULP_APP_CL_SRCS = \\\n"
            ret += "\n".join(["    {} \\".format(name) for name in self.cl_sources])
            ret += "\n\n"

        # add fc sources
        if self.fc_sources:
            ret += "PULP_APP_FC_SRCS = \\\n"
            ret += "\n".join(["    {} \\".format(name) for name in self.fc_sources])
            ret += "\n\n"

        # link dsp library
        if self.use_dsp:
            ret += "PULP_LDFLAGS += -lplpdsp\n"

        ret += "PULP_CFLAGS = -O{} -g \n\n".format(self.opt_level)

        # add compiler flags
        ret += "\n".join(["PULP_CFLAGS += -D{}".format(define) for define in self.defines])
        ret += "\n\n"

        # include the pulp sdk
        ret += "include $(PULP_SDK_HOME)/install/rules/pulp_rt.mk\n"
        return ret

    def write(self):
        """ write Makefile, while deleting the existing one """
        filename = "Makefile"
        if os.path.exists(filename):
            os.remove(filename)

        with open(filename, "w") as _f:
            _f.write(str(self))
