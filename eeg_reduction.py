#!/usr/bin/env python3

#*----------------------------------------------------------------------------*
#* Copyright (C) 2020 ETH Zurich, Switzerland                                 *
#* SPDX-License-Identifier: Apache-2.0                                        *
#*                                                                            *
#* Licensed under the Apache License, Version 2.0 (the "License");            *
#* you may not use this file except in compliance with the License.           *
#* You may obtain a copy of the License at                                    *
#*                                                                            *
#* http://www.apache.org/licenses/LICENSE-2.0                                 *
#*                                                                            *
#* Unless required by applicable law or agreed to in writing, software        *
#* distributed under the License is distributed on an "AS IS" BASIS,          *
#* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.   *
#* See the License for the specific language governing permissions and        *
#* limitations under the License.                                             *
#*                                                                            *
#* Authors: Xiaying Wang, Michael Hersche, Batuhan Toemekce, Burak Kaya       *
#*----------------------------------------------------------------------------*

__author__ = "Xiaying Wang, Michael Hersche"
__email__ = "xiaywang@iis.ee.ethz.ch, herschmi@iis.ee.ethz.ch"

import numpy as np
import scipy.signal as scp
from keras.models import load_model
from numpy import linalg as la
import pdb
from sklearn.preprocessing import minmax_scale, scale, normalize

def get_channels(x, n_ch=64, rows='dist'):

    channels = None

    if x.shape[1] == n_ch:
        channels = np.arange(0,n_ch)

    # distributed over the head
    elif rows is 'dist':
        if n_ch == 38:
            channels = np.array([0,2,4,6,8,10,12,14,16,18,20,21,22,23,24,26,28,29,31,33,35,37,40,41,42,43,46,48,50,52,54,55,57,59,60,61,62,63])
        elif n_ch == 27:
            channels = np.array([0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,38,39,40,41,44,45])
        elif n_ch == 19:
            channels = np.array([8,10,12,21,23,29,31,33,35,37,40,41,46,48,50,52,54,60,62])
        elif n_ch ==8:
            channels = np.array([8,10,12,25,27,48,52,57])

    # headset configuration with channels over sensorimotor area
    elif rows.lower() == 'crow':
        if n_ch == 2: 
            channels = np.array([8,12]) # C3, C4
        elif n_ch == 3: 
            channels = np.array([8,10,12]) # C3, CZ, C4
        elif n_ch == 5:
            channels = np.array([7,8,10,12,13]) # C5, C3, CZ, C4, C6
        elif n_ch == 7:
            channels = np.array([7,8,9,10,11,12,13]) # C5, C3, C1, CZ, C2, C4, C6
        elif n_ch == 9:
            channels = np.array([40,7,8,9,10,11,12,13,41]) # T7,C5, C3, C1, CZ, C2, C4, C6,T8
        elif n_ch == 11:
            channels = np.array([42,40,7,8,9,10,11,12,13,41,43]) # T9,T7,C5,C3,C1,CZ,C2,C4,C6,T8,T10
    elif rows.lower() == 'ccfrows':
        if n_ch ==4: 
            channels = np.array([8,10,1,5]) # C3, C4 // FC3, FC4
        elif n_ch == 6:
            channels = np.array([8,10,12,1,3,5]) # C3, CZ, C4, // FC3, FCZ, FC4
        elif n_ch == 10:
            channels = np.array([7,8,10,12,13,0,1,3,5,6]) # C5, C3, CZ, C4, C6 // FC5,FC3, FCZ, FC4,FC6
        elif n_ch == 14:
            channels = np.array([7,8,9,10,11,12,13,0,1,2,3,4,5,6]) # C5, C3, C1, CZ, C2, C4, C6//FC5, FC3, FC1, FCZ, FC2, FC4, FC6
        elif n_ch == 18:
            channels = np.array([40,7,8,9,10,11,12,13,41,38,0,1,2,3,4,5,6,39]) # T7,C5, C3, C1, CZ, C2, C4, C6,T8// FT7, FC5, FC3, FC1, FCZ, FC2, FC4, FC6, FT8
        elif n_ch == 20:
            channels = np.array([42,40,7,8,9,10,11,12,13,41,43,38,0,1,2,3,4,5,6,39])# T9,T7,C5,C3,C1,CZ,C2,C4,C6,T8,T10// FT7, FC5, FC3, FC1, FCZ, FC2, FC4, FC6, FT8
    elif rows.lower() == 'ccprows':
        if n_ch ==4: 
            channels = np.array([8,10,15,19]) # C3, C4 // CP3, CP4
        elif n_ch == 6:
            channels = np.array([8,10,12,15,17,19]) # C3, CZ, C4, // CP3, CPZ, CP4
        elif n_ch == 10:
            channels = np.array([7,8,10,12,13,14,15,17,19,20]) # C5, C3, CZ, C4, C6 // CP5, CP3, CPZ, CP4, CP6
        elif n_ch == 14:
            channels = np.array([7,8,9,10,11,12,13,14,15,16,17,18,19,20]) # C5, C3, C1, CZ, C2, C4, C6// CP5, CP3, CP1, CPZ, CP2, CP4, CP6
        elif n_ch == 18:
            channels = np.array([40,7,8,9,10,11,12,13,41,44,14,15,16,17,18,19,20,45]) # T7,C5, C3, C1, CZ, C2, C4, C6,T8// TP7, CP5, CP3, CP1, CPZ, CP2, CP4, CP6, TP8
        elif n_ch == 20:
            channels = np.array([42,40,7,8,9,10,11,12,13,41,43,44,14,15,16,17,18,19,20,45])# T9,T7,C5,C3,C1,CZ,C2,C4,C6,T8,T10// TP7, CP5, CP3, CP1, CPZ, CP2, CP4, CP6, TP8
    elif rows.lower() == 'auto':
        channels=None

    return channels


def eeg_reduction(x, n_ds = 1, n_ch = 64, T = 3, fs = 160, rows='dist'):
    '''
    Inputs
    ------
    x : np array ()
        input array
    n_ds: int
        downsampling factor
    n_ch: int
        number of channels
    T: float
        time [s] to classify
    fs: int
        samlping frequency [Hz]

    Outputs
    -------
    '''

    channels = get_channels(x, n_ch, rows)

    if T != 0:
        n_s_orig = int(T*fs)
        n_s = int(np.ceil(T*fs/n_ds)) # number of time samples
        n_trial = x.shape[0]

    # channel selection
    if n_ds >1:
        x = x[:,channels]
        y = np.zeros((n_trial, n_ch,n_s))
        for trial in range(n_trial):
            for chan in range(n_ch):
                # downsampling
                #pdb.set_trace()
                y[trial,chan] = scp.decimate(x[trial,chan,:n_s_orig],n_ds) if len(x.shape)==3 else scp.decimate(x[trial,chan,:n_s_orig,0],n_ds)
    else:
        y = x[:,channels]
        y = y[:,:,:n_s_orig] if T!=0 else y[:,:,:]

    return y


def net_weights_channel_selection(X, n_ch = 64, net_path='./results/your-global-experiment/model/global_class_4_ds1_nch64_T3_split_0.h5', modelname='EEGNet', do_normalize=False):
    '''
    Inputs
    ------
    X : np array ()
        input array
    n_ch: int
        number of channels
    net_path: string array
        path to the network from which to get the weights
    modelname: string array
        EEGNet or edgeEEGNet

    Outputs
    -------
    X_red: np array
        np array to be given as input to the network with n_ch channels with highest L2 norm of the weights of the spatial convolution

    '''

    if n_ch == 64:
        return X

    model = load_model(net_path)
    #print(X.shape) # (8820, 64, 480, 1)

    # for layer in model.layers:
    #     try:
    #         print(layer, layer.get_weights()[0].shape)
    #     except:
    #         print(layer, layer.get_weights())
    if modelname.lower()=='eegnet':
        w = model.layers[3].get_weights()[0]
    elif modelname.lower()=='edgeeegnet':
        w = model.layers[1].get_weights()[0]
    #print(w.shape) # (64, 1, 8, 2) for EEGNet, (64, 1, 1, 16) for edgeEEGNet
    if do_normalize=='minmax':
        print("do_normalize minmax")
        for i in range(w.shape[2]):
            for j in range(w.shape[3]):
                w[:, 0, i, j] = minmax_scale(w[:, 0, i, j])
                #print("shape ", w[:, 0, i, j].shape, w[:, 0, i, j], minmax_scale(w[:, 0, i, j]), w[:, 0, i, j].min(), w[:, 0, i, j].max())
    if do_normalize=='univar':
        print("do_normalize mean 0 univariance")
        # Warning: the standard deviation
        # of the data is probably bery close to 0!
        for i in range(w.shape[2]):
            for j in range(w.shape[3]):
                w[:, 0, i, j] = scale(w[:, 0, i, j])
    if do_normalize=='unitnorm':
        print("do_normalize unitnorm")
        for i in range(w.shape[2]):
            for j in range(w.shape[3]):
                w[:, :, i, j] = normalize(w[:, :, i, j], axis=0)
                #print("shape ", w[:, :, i, j].shape, w[:, :, i, j].flatten(), normalize(w[:, :, i, j], axis=0).flatten(), w[:, :, i, j].min(), w[:, :, i, j].max())
    wl2 = la.norm(w, axis=(2,3))
    #print(wl2.shape) # (64, 1)
    sort_i = np.argsort(wl2, axis=0)[::-1][:,0]
    channels = sort_i[:n_ch]
    #print(channels, channels.shape) # (38)
    X_red = X[:,channels]
    #print(X_red.shape) # (8820, 38, 480, 1)

    return X_red


def net_weights_channel_selection_folds_avg(X, n_ch = 64, num_splits=5, net_path='./results/your-global-experiment/model/global_class_4_ds1_nch64_T3_split_0.h5', modelname='EEGNet'):
    '''
    Inputs
    ------
    X : np array ()
        input array
    n_ch: int
        number of channels
    net_path: string array
        path to the network from which to get the weights
    modelname: string array
        EEGNet or edgeEEGNet

    Outputs
    -------
    X_red: np array
        np array to be given as input to the network with n_ch channels with highest L2 norm of the weights of the spatial convolution

    '''
    if modelname != 'EEGNet' and modelname != 'edgeEEGNet':
        raise ValueError("only EEGNet and edgeEEGNet supported\n")

    if n_ch == 64:
        return X

    for split_ctr in range(num_splits):

        net_path_fold = net_path[:-4]+str(split_ctr)+'.h5'

        model = load_model(net_path_fold)

        if modelname=='EEGNet':
            w = model.layers[3].get_weights()[0]
        elif modelname=='edgeEEGNet':
            w = model.layers[1].get_weights()[0]

        if split_ctr == 0:
            w_sum = w
        else:
            w_sum = w_sum + w

    wl2 = la.norm(w_sum, axis=(2,3))
    sort_i = np.argsort(wl2, axis=0)[::-1][:,0]
    channels = sort_i[:n_ch]
    X_red = X[:,channels]

    return X_red
