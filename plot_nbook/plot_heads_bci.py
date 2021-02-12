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
#* Authors: Xiaying Wang       *
#*----------------------------------------------------------------------------*

__author__ = "Xiaying Wang"
__email__ = "xiaywang@iis.ee.ethz.ch"

import os
import numpy as np
import scipy.signal as scp
import os
from keras.models import load_model
from numpy import linalg as la

import mne
from matplotlib import pyplot as plt
from mne.defaults import HEAD_SIZE_DEFAULT
from mne.channels._standard_montage_utils import _read_theta_phi_in_degrees



def net_weights_cs(n_ch = 64, net_path='./results/your-global-experiment/model/bci_class_4_ds1_nch64_T3_split_0.h5', modelname='EEGNet'):
    '''
    Inputs
    ------
    x : np array ()
        input array
    n_ch: int
        number of channels
    fs: int
        samlping frequency [Hz]
    net_path: string array
        path to the network from which to get the weights


    Outputs
    -------

    '''

    model = load_model(net_path)
    #print(X.shape) # (8820, 64, 480, 1)

    # for layer in model.layers:
    #     try:
    #         print(layer, layer.get_weights()[0].shape)
    #     except:
    #         print(layer, layer.get_weights())
    if modelname=='EEGNet':
        w = model.layers[3].get_weights()[0]
    elif modelname=='edgeEEGNet':
        w = model.layers[1].get_weights()[0]
    #print(w.shape) # (64, 1, 8, 2) for EEGNet, (64, 1, 1, 16) for edgeEEGNet
    wl2 = la.norm(w, axis=(2,3))
    #print(wl2.shape) # (64, 1)
    sort_i = np.argsort(wl2, axis=0)[::-1][:,0]
    channels = sort_i[:n_ch]
    #print(channels, channels.shape) # (38)
    #print(X_red.shape) # (8820, 38, 480, 1)

    return wl2, channels


def net_weights_folds_avg_cs(n_ch = 64, num_splits=5, net_path='./results/your-global-experiment/model/bci_class_4_ds1_nch64_T3_split_0.h5', modelname='EEGNet'):
    '''
    Inputs
    ------
    n_ch: int
        number of channels
    net_path: string array
        path to the network from which to get the weights
    modelname: string array
        EEGNet or edgeEEGNet

    Outputs
    -------
    wl2: array
        L2 norm of weights
    channels: array of integers
        n_ch channel numbers which have highest L2 norm of the weights of the spatial convolution

    '''

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

    return wl2, channels


def plot_my_topomap(data, channels, montage_info, my_biosemi_montage, fname='./plots/channels_head_plots/topomap.png'):

    fake_evoked = mne.EvokedArray(data, montage_info)
    fake_evoked.set_montage(my_biosemi_montage)

    channels_bool = np.zeros((wl2.shape[0], 1), dtype=bool)
    channels_bool[channels] = True


    # Plot the EEG sensors positions

    # fig, ax = plt.subplots(ncols=1, figsize=(6, 6), gridspec_kw=dict(top=0.9),
    #                        sharex=True, sharey=True)

    # # we plot the channel positions with default sphere - the mne way
    # fake_evoked.plot_sensors(axes=ax, show=False, show_names=True)

    # # add titles
    # fig.texts[0].remove()
    # ax.set_title('MNE channel projection', fontweight='bold')
    # plt.show()


    # Topomaps (topoplots)

    fig, ax = plt.subplots(ncols=1, figsize=(6, 6), gridspec_kw=dict(top=0.9),
                           sharex=True, sharey=True)

    maskParams = dict(marker='o', markerfacecolor='g', markeredgecolor='g', linewidth=0, markersize=10, markeredgewidth=2)

    topo, _=mne.viz.plot_topomap(fake_evoked.data[:, 0], fake_evoked.info, axes=ax,
                                 show=False, show_names=True, names=fake_evoked.ch_names,
                                 vmin=min(data), vmax=max(data), mask=channels_bool,
                                 mask_params=maskParams, extrapolate='head', contours=0, image_interp='lanczos')
                                 #cmap='RdBu_r')

    # add titles
    ax.set_title('MNE', fontweight='bold')

    fig.colorbar(topo, fraction=0.046, pad=0.04)

    #plt.show()
    plt.savefig(fname)
    print("figure saved in", fname)



# HYPERPARAMETER TO SET
num_classes_list = [4] # list of number of classes to test {2,3,4}
n_ds = 1 # downsamlping factor {1,2,3}
n_ch_list = [2]#, 3, 5, 7, 8, 9, 11, 4, 6, 10, 14, 18, 19, 20, 16, 22] #[16, 24, 32] #[2, 3, 5, 7, 9, 11, 4, 6, 10, 14, 18, 20]#[8, 19, 38, 64] # number of channels {8,19,27,38,64}
net_weights_red = True
T_list = [0] # duration to classify {1,2,3}

num_splits = 9

# for channel selection using network weights
net_dirpath = '../results/bci-cubeEEGNet-weights-same-subj/model/'
same_folds = True
figsavepath= './plots/channels_head_plots/bci-cubeEEGNet-test'

modelname = 'EEGNet'

os.makedirs(figsavepath, exist_ok=True)


# set up the EEG montage for plotting
fname = 'physionet_mmmi22chs.tsv'

my_biosemi_montage = _read_theta_phi_in_degrees(fname=fname, head_size=HEAD_SIZE_DEFAULT,
                                     fid_names=['Nz', 'LPA', 'RPA'],
                                     add_fiducials=False)

#print(my_biosemi_montage)

# <DigMontage | 0 extras (headshape), 0 HPIs, 3 fiducials, 64 channels>


n_channels = len(my_biosemi_montage.ch_names)
#print(my_biosemi_montage.ch_names)
montage_info = mne.create_info(ch_names=my_biosemi_montage.ch_names, sfreq=250.,
                            ch_types='eeg')

#rng = np.random.RandomState(0)
#data = rng.normal(size=(n_channels, 1)) * 1e-6


for num_classes in num_classes_list:
    for n_ch in n_ch_list:
        for T in T_list:

            plt.close('all')

            if same_folds:
                net_path = os.path.join(net_dirpath, f'bci_class_{num_classes}_ds{n_ds}_nch64cs_T{T}_split_0.h5')

            else:
                net_path = os.path.join(net_dirpath, f'bci_class_{num_classes}_ds{n_ds}_nch64_T{T}_split_0.h5')

            wl2, channels = net_weights_folds_avg_cs(n_ch=n_ch, num_splits=num_splits, net_path=net_path, modelname=modelname)

            if same_folds:
                fname = os.path.join(figsavepath, modelname+f'_bci_class_{num_classes}_ds{n_ds}_nch{n_ch}cs_T{T}_avg.png')

            else:
                fname = os.path.join(figsavepath, modelname+f'_bci_class_{num_classes}_ds{n_ds}_nch{n_ch}_T{T}_avg.png')

            plot_my_topomap(wl2, channels, montage_info, my_biosemi_montage, fname=fname)

            for split_ctr in range(num_splits):

                if same_folds:
                    net_path = os.path.join(net_dirpath, f'bci_class_{num_classes}_ds{n_ds}_nch64cs_T{T}_split_{split_ctr}.h5')

                else:
                    net_path = os.path.join(net_dirpath, f'bci_class_{num_classes}_ds{n_ds}_nch64_T{T}_split_{split_ctr}.h5')

                wl2, channels = net_weights_cs(n_ch=n_ch, net_path=net_path, modelname=modelname)

                if same_folds:
                    fname = os.path.join(figsavepath, modelname+f'_bci_class_{num_classes}_ds{n_ds}_nch{n_ch}cs_T{T}_subj_{split_ctr}.png')

                else:
                    fname = os.path.join(figsavepath, modelname+f'_bci_class_{num_classes}_ds{n_ds}_nch{n_ch}_T{T}_subj_{split_ctr}.png')

                plot_my_topomap(wl2, channels, montage_info, my_biosemi_montage, fname=fname)

