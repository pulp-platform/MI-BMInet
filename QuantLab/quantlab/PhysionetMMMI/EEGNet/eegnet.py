# Copyright (c) 2019 Tibor Schneider

import numpy as np
import torch as t
import torch.nn.functional as F

from quantlab.indiv.inq_ops import INQController, INQLinear, INQConv2d
from quantlab.indiv.ste_ops import STEActivation, STEController


class EEGNet(t.nn.Module):
    """
    Quantized EEGNet
    """

    def __init__(self, F1=8, D=2, F2=None, C=22, T=1125, N=4, p_dropout=0.5,
                 dropout_type='TimeDropout2d', quantWeight=True, quantAct=True,
                 weightInqSchedule=None, weightInqNumLevels=255, weightInqStrategy="matnitude",
                 weightInqInitMethod="uniform", actSTENumLevels=255, actSTEStartEpoch=2,
                 floorToZero=False, actFirstLayerNumLevels=None, weightFirstLayerNumLevels=None,
                 first_layer_only=False):
        """
        F1:           Number of spectral filters
        D:            Number of spacial filters (per spectral filter), F2 = F1 * D
        F2:           Number or None. If None, then F2 = F1 * D
        C:            Number of EEG channels
        T:            Number of time samples
        N:            Number of classes
        p_dropout:    Dropout Probability
        dropout_type: string, either 'dropout', 'SpatialDropout2d' or 'TimeDropout2D'
        floorToZero:  STE rounding is done by floor towards zero
        """
        super(EEGNet, self).__init__()

        if weightInqSchedule is None:
            raise TypeError("Parameter weightInqSchedule is not set")
        if weightFirstLayerNumLevels is None:
            weightFirstLayerNumLevels = weightInqNumLevels
        if actFirstLayerNumLevels is None:
            actFirstLayerNumLevels = actSTENumLevels

        weightInqSchedule = {int(k): v for k, v in weightInqSchedule.items()}

        # prepare network constants
        if F2 is None:
            F2 = F1 * D

        # Prepare Dropout Type
        if dropout_type.lower() == 'dropout':
            dropout = t.nn.Dropout
        elif dropout_type.lower() == 'spatialdropout2d':
            dropout = t.nn.Dropout2d
        elif dropout_type.lower() == 'timedropout2d':
            dropout = TimeDropout2d
        else:
            raise ValueError("dropout_type must be one of SpatialDropout2d, Dropout or "
                             "WrongDropout2d")

        # store local values
        self.F1, self.D, self.F2, self.C, self.T, self.N = (F1, D, F2, C, T, N)
        self.p_dropout = p_dropout

        # Number of input neurons to the final fully connected layer
        n_features = (T // 8) // 8

        # prepare helper functions to easily declare activation, convolution and linear unit
        def activ():
            return t.nn.ReLU(inplace=True)

        def quantize(numLevels=None, first=False):
            start = actSTEStartEpoch
            monitor = start - 1
            if numLevels is None or (not first and first_layer_only):
                numLevels = actSTENumLevels
            if quantAct:
                return STEActivation(startEpoch=start, monitorEpoch=monitor,
                                     numLevels=numLevels, floorToZero=floorToZero)
            else:
                return t.nn.Identity()

        def linear(name, n_in, n_out, bias=True, first=False):
            if quantWeight and not (not first and first_layer_only):
                return INQLinear(n_in, n_out, bias=bias, numLevels=weightInqNumLevels,
                                 strategy=weightInqStrategy, quantInitMethod=weightInqInitMethod)
            else:
                return t.nn.Linear(n_in, n_out, bias=bias)

        def conv2d(name, in_channels, out_channels, kernel_size, numLevels=None, first=False, **argv):
            if quantWeight and not (not first and first_layer_only):
                if numLevels is None:
                    numLevels = weightInqNumLevels
                return INQConv2d(in_channels, out_channels, kernel_size,
                                 numLevels=numLevels, strategy=weightInqStrategy,
                                 quantInitMethod=weightInqInitMethod, **argv)
            else:
                return t.nn.Conv2d(in_channels, out_channels, kernel_size, **argv)

        # Block 1
        self.quant1 = quantize(actFirstLayerNumLevels, first=True)
        self.conv1_pad = t.nn.ZeroPad2d((63, 64, 0, 0))
        self.conv1 = conv2d("conv1", 1, F1, (1, 128), bias=False,
                            numLevels=weightFirstLayerNumLevels, first=True)
        self.batch_norm1 = t.nn.BatchNorm2d(F1, momentum=0.01, eps=0.001)
        self.quant2 = quantize()
        self.conv2 = conv2d("conv2", F1, D * F1, (C, 1), groups=F1, bias=False)
        self.batch_norm2 = t.nn.BatchNorm2d(D * F1, momentum=0.01, eps=0.001)
        self.activation1 = activ()
        self.pool1 = t.nn.AvgPool2d((1, 8))
        self.quant3 = quantize()
        # self.dropout1 = dropout(p=p_dropout)
        self.dropout1 = t.nn.Dropout(p=p_dropout)

        # Block 2
        self.sep_conv_pad = t.nn.ZeroPad2d((7, 8, 0, 0))
        self.sep_conv1 = conv2d("sep_conv1", D * F1, D * F1, (1, 16), groups=D * F1, bias=False)
        self.quant4 = quantize()
        self.sep_conv2 = conv2d("sep_conv2", D * F1, F2, (1, 1), bias=False)
        self.batch_norm3 = t.nn.BatchNorm2d(F2, momentum=0.01, eps=0.001)
        self.activation2 = activ()
        self.pool2 = t.nn.AvgPool2d((1, 8))
        self.quant5 = quantize()
        self.dropout2 = dropout(p=p_dropout)

        # Fully connected layer (classifier)
        self.flatten = Flatten()
        self.fc = linear("fc", F2 * n_features, N, bias=True)
        self.quant6 = quantize(255)

        self.inqController = INQController(INQController.getInqModules(self), weightInqSchedule,
                                           clearOptimStateOnStep=True)
        self.steController = STEController(STEController.getSteModules(self),
                                           clearOptimStateOnStart=True)

        # initialize weights
        # self._initialize_params()

    def forward(self, x, with_stats=False):

        # input dimensions: (s, 1, C, T)
        x = self.quant1(x)

        # Block 1
        x = self.conv1_pad(x)
        x = self.conv1(x)            # output dim: (s, F1, C, T-1)
        x = self.batch_norm1(x)
        x = self.quant2(x)
        x = self.conv2(x)            # output dim: (s, D * F1, 1, T-1)
        x = self.batch_norm2(x)
        x = self.activation1(x)
        x = self.pool1(x)            # output dim: (s, D * F1, 1, T // 8)
        x = self.quant3(x)
        x = self.dropout1(x)

        # Block2
        x = self.sep_conv_pad(x)
        x = self.sep_conv1(x)        # output dim: (s, D * F1, 1, T // 8 - 1)
        x = self.quant4(x)
        x = self.sep_conv2(x)        # output dim: (s, F2, 1, T // 8 - 1)
        x = self.batch_norm3(x)
        x = self.activation2(x)
        x = self.pool2(x)            # output dim: (s, F2, 1, T // 64)
        x = self.quant5(x)
        x = self.dropout2(x)

        # Classification
        x = self.flatten(x)          # output dim: (s, F2 * (T // 64))
        x = self.fc(x)               # output dim: (s, N)
        x = self.quant6(x)

        if with_stats:
            stats = [('conv1_w', self.conv1.weight.data),
                     ('conv2_w', self.conv2.weight.data),
                     ('sep_conv1_w', self.sep_conv1.weight.data),
                     ('sep_conv2_w', self.sep_conv2.weight.data),
                     ('fc_w', self.fc.weight.data),
                     ('fc_b', self.fc.bias.data)]
            return stats, x
        return x

    def forward_with_tensor_stats(self, x):
        return self.forward(x, with_stats=True)

    def _initialize_params(self, weight_init=t.nn.init.xavier_uniform_, bias_init=t.nn.init.zeros_):
        """
        Initializes all the parameters of the model

        Parameters:
         - weight_init: t.nn.init inplace function
         - bias_init:   t.nn.init inplace function

        """
        def init_weight(m):
            if isinstance(m, t.nn.Conv2d) or isinstance(m, t.nn.Linear):
                weight_init(m.weight)
            if isinstance(m, t.nn.Linear):
                bias_init(m.bias)

        self.apply(init_weight)


class Flatten(t.nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)


class TimeDropout2d(t.nn.Dropout2d):
    """
    Dropout layer, where the last dimension is treated as channels
    """
    def __init__(self, p=0.5, inplace=False):
        """
        See t.nn.Dropout2d for parameters
        """
        super(TimeDropout2d, self).__init__(p=p, inplace=inplace)

    def forward(self, input):
        if self.training:
            input = input.permute(0, 3, 1, 2)
            input = F.dropout2d(input, self.p, True, self.inplace)
            input = input.permute(0, 2, 3, 1)
        return input
