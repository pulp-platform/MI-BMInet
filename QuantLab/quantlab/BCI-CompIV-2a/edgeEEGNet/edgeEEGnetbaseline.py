# Copyright (c) 2019 Tibor Schneider

import numpy as np
import torch as t
import torch.nn.functional as F


class edgeEEGNetBaseline(t.nn.Module):
    """
    edgeEEGNet

    In order for the baseline to be launched with the same logic as the quantized models, an empty
    quantization scheme and an empty thermostat schedule needs to be configured.
    Use the following templates for the 'net' and 'thermostat' configurations (for the "net" object,
    all params can be omitted to use the default ones):

    "net": {
      "class": "edgeEEGNetBaseline",
      "params": {
        "F1": 8,
        "D": 2,
        "F2": 16,
        "C": 22,
        "T": 1125,
        "N": 4,
        "p_dropout": 0.5,
        "activation": "relu",
        "dropout_type": "TimeDropout2D",
      },
      "pretrained": null,
      "loss_fn": {
        "class": "CrossEntropyLoss",
        "params": {}
      }
    }

    "thermostat": {
      "class": "edgeEEGNetBaseline",
      "params": {
        "noise_schemes": {},
        "bindings": []
      }
    }
    """

    def __init__(self, F1=8, D=2, F2=None, C=22, T=1125, N=4, Nf=64, p_dropout=0.5, activation='relu',
                 dropout_type='TimeDropout2D'):
        """
        F1:           Number of spectral filters
        D:            Number of spacial filters (per spectral filter), F2 = F1 * D
        F2:           Number or None. If None, then F2 = F1 * D
        C:            Number of EEG channels
        T:            Number of time samples
        N:            Number of classes
        p_dropout:    Dropout Probability
        activation:   string, either 'elu' or 'relu'
        dropout_type: string, either 'dropout', 'SpatialDropout2d' or 'TimeDropout2D'
        """
        super(edgeEEGNetBaseline, self).__init__()

        # prepare network constants
        if F2 is None:
            F2 = F1 * D

        # check the activation input
        activation = activation.lower()
        assert activation in ['elu', 'relu']

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
        self.p_dropout, self.activation = (p_dropout, activation)

        # Number of input neurons to the final fully connected layer
        n_features = (T // 8) // 8

        # Block 1
        self.conv1 = t.nn.Conv2d(1, F1, (C, 1), bias=False)
        self.batch_norm1 = t.nn.BatchNorm2d(F1, momentum=0.01, eps=0.001)
        self.conv2_pad = t.nn.ZeroPad2d((31, 32, 0, 0))
        self.conv2 = t.nn.Conv2d(F1, D * F1, (1, Nf), groups=F1, bias=False)
        self.batch_norm2 = t.nn.BatchNorm2d(D * F1, momentum=0.01, eps=0.001)
        self.activation1 = t.nn.ELU(inplace=True) if activation == 'elu' else t.nn.ReLU(inplace=True)
        self.pool1 = t.nn.AvgPool2d((1, 8))
        # self.dropout1 = dropout(p=p_dropout)
        self.dropout1 = t.nn.Dropout(p=p_dropout)

        # Block 2
        self.sep_conv_pad = t.nn.ZeroPad2d((7, 8, 0, 0))
        self.sep_conv1 = t.nn.Conv2d(D * F1, D * F1, (1, 16), groups=D * F1, bias=False)
        self.sep_conv2 = t.nn.Conv2d(D * F1, F2, (1, 1), bias=False)
        self.batch_norm3 = t.nn.BatchNorm2d(F2, momentum=0.01, eps=0.001)
        self.activation2 = t.nn.ELU(inplace=True) if activation == 'elu' else t.nn.ReLU(inplace=True)
        self.pool2 = t.nn.AvgPool2d((1, 8))
        self.dropout2 = dropout(p=p_dropout)

        # Fully connected layer (classifier)
        self.flatten = Flatten()
        self.fc = t.nn.Linear(F2 * n_features, N, bias=True)

        # initialize weights
        self._initialize_params()

    def forward(self, x, with_stats=False):

        # input dimensions: (s, 1, C, T)

        # Block 1
        x = self.conv1(x)            # output dim: (s, F1, C, T-1)
        x = self.batch_norm1(x)
        x = self.conv2_pad(x)
        x = self.conv2(x)            # output dim: (s, D * F1, 1, T-1)
        x = self.batch_norm2(x)
        x = self.activation1(x)
        x = self.pool1(x)            # output dim: (s, D * F1, 1, T // 8)
        x = self.dropout1(x)

        # Block2
        x = self.sep_conv_pad(x)
        x = self.sep_conv1(x)        # output dim: (s, D * F1, 1, T // 8 - 1)
        x = self.sep_conv2(x)        # output dim: (s, F2, 1, T // 8 - 1)
        x = self.batch_norm3(x)
        x = self.activation2(x)
        x = self.pool2(x)            # output dim: (s, F2, 1, T // 64)
        x = self.dropout2(x)
        #print(x.shape)
        # Classification
        x = self.flatten(x)          # output dim: (s, F2 * (T // 64))
        x = self.fc(x)               # output dim: (s, N)

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


class Flatten(t.nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)
