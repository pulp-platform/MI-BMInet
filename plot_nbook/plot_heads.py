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
from tqdm import tqdm
import models as models

import mne
from matplotlib import pyplot as plt
from mne.defaults import HEAD_SIZE_DEFAULT
from mne.channels._standard_montage_utils import _read_theta_phi_in_degrees

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


# HYPERPARAMETER TO SET

DATASET = 'bci'
modelname = 'edgeEEGNet'
expname = 'exp3_bak'

num_classes_list = [4,3,2] # list of number of classes to test {2,3,4}
n_ds = 1 # downsamlping factor {1,2,3}
n_ch_list = [22] #[16, 24, 32] #[2, 3, 5, 7, 9, 11, 4, 6, 10, 14, 18, 20]#[8, 19, 38, 64] # number of channels {8,19,27,38,64}
net_weights_red = True

if DATASET == 'physionet':
    num_splits = 5
    T_list = [3] # duration to classify {1,2,3}
    net_dirpath = f'../results/global-experiment-cube{modelname}'+'-weights-same-folds-{}/model/'
elif DATASET == 'bci':
    num_splits = 9
    T_list = [0] # duration to classify {1,2,3}
    net_dirpath = f'../logs/{DATASET}/{modelname}/{expname}'+'/model/{}'

# for channel selection using network weights
#SINGLERUN = True
RUN = range(25) #[0,1,2,3,4] 
same_folds = True
PLOT_SINGLE_RUN = False

# if SINGLERUN:
#     net_dirpath = f'../results/global-experiment-cubeedgeEEGNet-weights-same-folds-{RUN}/model/'
#     #net_dirpath = f'../results/bci-cubeEEGNet-weights-same-subj-{RUN}/model/'
#     figsavepath= f'./plots/channels_head_plots/{DATASET}-cubeedgeEEGNet_{RUN}'
# else:
#     net_dirpath = '../results/global-experiment-cubeedgeEEGNet-weights-same-folds-{}/model/'
#     # net_dirpath = f'../results/bci-cubeEEGNet-weights-same-subj-{}/model/'
#     figsavepath= './plots/channels_head_plots/{DATASET}-cubeedgeEEGNet_avgruns'

figsavepath= './plots/channels_head_plots/{}'+f'-cube{modelname}'+'_{}'

FOR_QUANT = True # plot green dots on topomap for the selected channels


def net_weights_cs(n_ch = 64, net_path='./results/your-global-experiment/model/global_class_4_ds1_nch64_T3_split_0.h5', modelname='EEGNet'):
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

    model = load_model(net_path, custom_objects={'TimeDropout2D':models.TimeDropout2D})
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


def net_weights_folds_avg_cs(n_ch = 64, num_splits=5, net_path='./results/your-global-experiment/model/global_class_4_ds1_nch64_T3_split_0.h5', modelname='EEGNet'):
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

        model = load_model(net_path_fold, custom_objects={'TimeDropout2D':models.TimeDropout2D})

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


def plot_my_topomap(data, channels, montage_info, my_biosemi_montage, fname='./plots/channels_head_plots/topomap.svg'):

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

    if np.asarray(channels).any():
        maskParams = dict(marker='o', markerfacecolor='g', markeredgecolor='g', linewidth=0, markersize=10, markeredgewidth=2)
        show_names=True
    else:
        # make electrodes maker thicker
        channels_bool[:] = True
        maskParams = dict(marker='o', markerfacecolor='k', markeredgecolor='k', linewidth=0, markersize=2, markeredgewidth=0)
        show_names=False

    topo, _=mne.viz.plot_topomap(fake_evoked.data[:, 0], fake_evoked.info, axes=ax,
                                 show=False, show_names=show_names, names=fake_evoked.ch_names,
                                 vmin=min(data), vmax=max(data), mask=channels_bool,
                                 mask_params=maskParams, extrapolate='head', contours=6,
                                 image_interp='bicubic', outlines='head')
                                 #cmap='RdBu_r')

    # add titles
    #ax.set_title('MNE', fontweight='bold')

    fig.colorbar(topo, fraction=0.046, pad=0.04)

    #plt.show()
    plt.savefig(fname)
    print("figure saved in", fname)


#set up the EEG montage for plotting
if DATASET == 'physionet':
    fname = 'physionet_mmmi64chs.tsv'
elif DATASET == 'bci':
    fname = 'physionet_mmmi22chs.tsv'

my_biosemi_montage = _read_theta_phi_in_degrees(fname=fname, head_size=HEAD_SIZE_DEFAULT,
                                     fid_names=['Nz', 'LPA', 'RPA'],
                                     add_fiducials=False)

#print(my_biosemi_montage)

# <DigMontage | 0 extras (headshape), 0 HPIs, 3 fiducials, 64 channels>


n_channels = len(my_biosemi_montage.ch_names)
#print(my_biosemi_montage.ch_names)
montage_info = mne.create_info(ch_names=my_biosemi_montage.ch_names, sfreq=160.,
                            ch_types='eeg')

#rng = np.random.RandomState(0)
#data = rng.normal(size=(n_channels, 1)) * 1e-6

if DATASET == 'physionet':
    modelfname = 'global_class_{}_ds{}_nch64{}_T{}_split_{}.h5'
    savefname = '_global_class_{}_ds{}_nch{}{}_T{}_{}.svg'
elif DATASET == 'bci':
    modelfname = '_class{}_nch22_split{}.h5'
    savefname = '_bci_class_{}_ds{}_nch{}{}_T{}_{}.svg'


for num_classes in num_classes_list:
    for n_ch in n_ch_list:
        for T in T_list:

            if DATASET == 'bci':
                wl2_avg = np.zeros((len(RUN), 22, 1))
                wl2_folds = np.zeros((len(RUN), num_splits, 22, 1))
            elif DATASET == 'physionet':
                wl2_avg = np.zeros((len(RUN), 64, 1))
                wl2_folds = np.zeros((len(RUN), num_splits, 64, 1))

            with tqdm(desc='Runs', total=len(RUN), ascii=True) as bar:
                for i, r in enumerate(RUN):

                    if PLOT_SINGLE_RUN:
                        os.makedirs(figsavepath.format(DATASET, r), exist_ok=True)

                    plt.close('all')

                    if same_folds:
                        if DATASET == 'physionet':
                            net_path = os.path.join(net_dirpath.format(r), modelfname.format(num_classes, n_ds, 'cs', T, 0))
                        else:
                            net_path = os.path.join(net_dirpath.format(r), modelfname.format(num_classes, 0))

                    else:
                        net_path = os.path.join(net_dirpath.format(r), modelfname.format(num_classes, n_ds, '', T, 0))

                    wl2, channels = net_weights_folds_avg_cs(n_ch=n_ch, num_splits=num_splits, net_path=net_path, modelname=modelname)
                    wl2_avg[i] = wl2

                    if PLOT_SINGLE_RUN:

                        if same_folds:
                            fname = os.path.join(figsavepath.format(DATASET, r), modelname+savefname.format(num_classes, n_ds, n_ch, 'cs', T, 'avg'))

                        else:
                            fname = os.path.join(figsavepath.format(DATASET, r), modelname+savefname.format(num_classes, n_ds, n_ch, '', T, 'avg'))

                        plot_my_topomap(wl2, channels, montage_info, my_biosemi_montage, fname=fname)

                    for split_ctr in range(num_splits):

                        if same_folds:
                            if DATASET == 'physionet':
                                net_path = os.path.join(net_dirpath.format(r), modelfname.format(num_classes, n_ds, 'cs', T, split_ctr))
                            else:
                                net_path = os.path.join(net_dirpath.format(r), modelfname.format(num_classes, split_ctr))

                        else:
                            net_path = os.path.join(net_dirpath.format(r), modelfname.format(num_classes, n_ds, '', T, split_ctr))

                        wl2, channels = net_weights_cs(n_ch=n_ch, net_path=net_path, modelname=modelname)
                        wl2_folds[i, split_ctr] = wl2

                        if PLOT_SINGLE_RUN:

                            if same_folds:
                                fname = os.path.join(figsavepath.format(DATASET, r), modelname+savefname.format(num_classes, n_ds, n_ch, 'cs', T, f'split_{split_ctr}'))

                            else:
                                fname = os.path.join(figsavepath.format(DATASET, r), modelname+savefname.format(num_classes, n_ds, n_ch, 'cs', T, f'split_{split_ctr}'))

                            plot_my_topomap(wl2, channels, montage_info, my_biosemi_montage, fname=fname)

                    bar.update()

            os.makedirs(figsavepath.format(DATASET, 'avgruns'), exist_ok=True)
            # average over different runs
            wl2 = np.mean(wl2_avg, axis=0)
            fname = os.path.join(figsavepath.format(DATASET, 'avgruns'), modelname+savefname.format(num_classes, n_ds, n_ch, 'cs', T, 'avg'))
            if FOR_QUANT:
                sort_i = np.argsort(wl2, axis=0)[::-1][:,0]
                channels = sort_i[:n_ch]
                plot_my_topomap(wl2, channels, montage_info, my_biosemi_montage, fname=fname)
            else:
                plot_my_topomap(wl2, [], montage_info, my_biosemi_montage, fname=fname)

            for split_ctr in range(num_splits):
                wl2 = np.mean(wl2_folds[:,split_ctr,:,:], axis=0)
                fname = os.path.join(figsavepath.format(DATASET, 'avgruns'), modelname+savefname.format(num_classes, n_ds, n_ch, 'cs', T, f'split_{split_ctr}'))
                plot_my_topomap(wl2, [], montage_info, my_biosemi_montage, fname=fname)

