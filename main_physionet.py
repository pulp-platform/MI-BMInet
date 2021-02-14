#!/usr/bin/env python3

#*----------------------------------------------------------------------------*
#* Copyright (C) 2021 ETH Zurich, Switzerland                                 *
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

__author__ = "Xiaying Wang"
__email__ = " xiaywang@ethz.ch"


import os
import sys, argparse
from tqdm import tqdm
from contextlib import redirect_stdout, redirect_stderr
import json
import numpy as np
from tensorflow.keras import utils as np_utils
from keras.callbacks import LearningRateScheduler, ReduceLROnPlateau
from keras.optimizers import Adam
from keras import backend as K
from sklearn.model_selection import KFold

import get_data as get
import models as models
from eeg_reduction import *

import warnings
# warnings.filterwarnings("ignore")
# warnings.filterwarnings('ignore', category=DeprecationWarning)
# warnings.filterwarnings('ignore', category=FutureWarning)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 

# import time
# start = time.time()

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

TOPOLOGY = "edgeEEGNet"
EXP_ID = 0
LOG_NOTE = 'cs with net weights from same folds'

BENCHMARK = True
N_ITER = 5
N_FOLDS = 5

DATAPATH  = "/usr/scratch/sassauna4/xiaywang/Projects/BCI/physionet/"
LOG_FOLDER = 'logs/physionet'
EXP_FOLDER = os.path.join(LOG_FOLDER, TOPOLOGY, f'exp{EXP_ID}')

STATS_FNAME = '{}_class{}_nch{}_split{}'
DATAFILE = os.path.join(EXP_FOLDER, '{}', STATS_FNAME+'{}')

SAVE_ONLY_64CH_MODEL = False
if BENCHMARK:
    SAVE_ONLY_64CH_MODEL = True

REDUCE_MANUAL = True # channels overall the brain ('dist') or over the sensorimotor area ('headset')
REDUCE_NETW = False # channel reduction using network weights (spatial conv)
do_normalize='' # for channel reduction using net weights
#HEADSETMODE='headset' # 'dist' or 'headset' or ''
RMODE='dist' # 'crow', 'ccfrows', 'ccprows' or 'dist' or 'auto'
if REDUCE_MANUAL and REDUCE_NETW:
    raise ValueError('choose either model reduction by manual selection or by network weights, but not both at the same time')
if REDUCE_MANUAL and RMODE == '':
    raise ValueError('choose the method of manually reducing the data: crow, ccfrows, ccprows, dist, auto')
if REDUCE_NETW:
    RMODE='auto'
    #HEADSETMODE=''

os.makedirs(os.path.join(EXP_FOLDER, 'stats'), exist_ok=True)
os.makedirs(os.path.join(EXP_FOLDER, 'model'), exist_ok=True)
os.makedirs(os.path.join(EXP_FOLDER, 'plots'), exist_ok=True)

N_EPOCHS = 100 # number of epochs for training

num_classes_list = [2, 3, 4] # list of number of classes to test {2,3,4}
n_ch_list = [2, 3, 5, 7, 8, 9, 11, 4, 6, 10, 14, 18, 19, 20, 16, 24, 32, 38, 64] #[8, 19, 38, 64] #[16, 24, 32, 64] #[2, 3, 5, 7, 9, 11, 4, 6, 10, 14, 18, 20, 64] # number of channels
# {8,19,27,38,64}
# remember to put 64 if you have net_weights_same_folds = True because it has
# to train the 64 full model within each fold, then select the channels and
# train within the same fold
T=3
n_ds=1

if RMODE != 'auto':
    if RMODE == 'dist':
        manual_ch = [8, 19, 27, 38]
    elif RMODE == 'crow':
        manual_ch = [2, 3, 5, 7, 9, 11]
    elif RMODE in ('ccprows', 'ccfrows'):
        manual_ch = [4, 6, 10, 14, 18, 20]
    n_ch_set = set(n_ch_list)
    intersection = n_ch_set.intersection(manual_ch)
    n_ch_list = list(intersection)


def step_decay(epoch):
    '''Multi step learning rate scheduler (for EEGNet on Physionet)'''
    if(epoch < 20):
        lr = 0.01
    elif(epoch < 50):
        lr = 0.001
    else:
        lr = 0.0001
    return lr


def step_decay2(epoch):
    '''Multi step learning rate scheduler (for edgeEEGNet on Physionet)'''
    if(epoch < 40):
        lr = 0.01
    elif(epoch < 80):
        lr = 0.001
    else:
        lr = 0.0001
    return lr


def train_validate(X_train, y_train, X_test, y_test, n_classes=4, n_channels=64, dropoutRate=0.2):

    # check the activation input
    topology = TOPOLOGY.lower()
    assert topology in ['eegnet', 'edgeeegnet']

    if topology == 'edgeeegnet':
        lrate = LearningRateScheduler(step_decay2)
        model = models.cubeedgeEEGNetCF1(nb_classes = n_classes, Chans=n_channels, Samples=X_train.shape[2], dropoutRate=dropoutRate)
    elif topology == 'eegnet':
        lrate = LearningRateScheduler(step_decay)
        model = models.cubeEEGNet(nb_classes = n_classes, Chans=n_channels, Samples=X_train.shape[2], dropoutRate=dropoutRate)

    adam_alpha = Adam()
    model.compile(loss='categorical_crossentropy', optimizer=adam_alpha, metrics = ['accuracy'])

    # do training
    history = model.fit(X_train, y_train, 
                        validation_data=(X_test, y_test),
                        batch_size = 16, epochs = N_EPOCHS, callbacks=[lrate],
                        shuffle=True, verbose = 0)

    return model, history

def run(tbar=None, bench=''):

    if REDUCE_NETW or REDUCE_MANUAL:
        # delete 64 from n_ch_list if it's there
        try:
            n_ch_list.remove(64)
        except:
            pass

        stats = np.zeros((len(num_classes_list), len(n_ch_list)+1, N_FOLDS, 4))
    else:
        stats = np.zeros((len(num_classes_list), len(n_ch_list), N_FOLDS, 4))

    for i, num_classes in enumerate(num_classes_list):

        # Load data
        print('loading data')
        try:
            npzfile = np.load(DATAPATH+f'{num_classes}class.npz')
            try:
                X_orig, y_orig = npzfile['X'], npzfile['y']
            except:
                X_orig, y_orig = npzfile['X_Train'], npzfile['y_Train']
        except:
            X_orig, y_orig = get.get_data(DATAPATH, n_classes=num_classes)
            np.savez(DATAPATH+f'{num_classes}class', X = X_orig, y = y_orig)

        # Expand dimensions to match expected EEGNet input
        X_orig = (np.expand_dims(X_orig, axis=-1))
        # convert labels to one-hot encodings.
        y_cat = np_utils.to_categorical(y_orig)

        # using 5 folds
        kf = KFold(n_splits = N_FOLDS)

        with tqdm(desc=f'{N_FOLDS}-fold CV on {len(n_ch_list)} n_ch', total=len(n_ch_list), ascii=True) as bar1:
            for j, n_ch in enumerate(n_ch_list):
                fstats = np.zeros((N_FOLDS, 4))
                #with tqdm(desc=f'{N_FOLDS} fold cross validation', total=N_FOLDS, ascii=True) as bar:
                for split_ctr, (train, test) in enumerate(kf.split(X_orig, y_cat)):

                    if REDUCE_NETW or REDUCE_MANUAL:
                        # check if the baseline model with all the channels
                        # is already trained.
                        if os.path.isfile(DATAFILE.format('model', '', num_classes, 64, split_ctr, '.h5')):
                            #print('{} exists'.format(DATAFILE.format('model', num_classes, 64, split_ctr, '.h5')))
                            pass

                        else:
                            #print("Channel reduction. Prepare the baseline model with full number of channels, i.e. 64")

                            model, history = train_validate(X_orig[train], y_cat[train],
                                                            X_orig[test], y_cat[test],
                                                            n_classes=num_classes, n_channels=64)

                            # if BENCHMARK:

                            #     np.savez(DATAFILE.format(f'stats/bench{bench}', num_classes, 64,
                            #                              split_ctr, '.npz'), **history.history)
                            #     model.save(DATAFILE.format(f'model/bench{bench}', num_classes, 64, split_ctr, '.h5'))
                            # else:
                            np.savez(DATAFILE.format(f'stats/{bench}', '', num_classes, 64,
                                                     split_ctr, '.npz'), **history.history)
                            model.save(DATAFILE.format(f'model/{bench}', '', num_classes, 64, split_ctr, '.h5'))


                            fstats = loadfstats(fstats, history.history, split_ctr)
                            stats[i, len(n_ch_list), :] = fstats
                            #print(stats[i, len(n_ch_list)])

                            #Clear Models
                            K.clear_session()

                        #bar.update()
                        if tbar is not None:
                            tbar.update()

                    if REDUCE_NETW:
                        X = net_weights_channel_selection(X_orig, n_ch=n_ch, net_path=DATAFILE.format(f'model/{bench}', '', num_classes, 64, split_ctr, '.h5'), modelname=TOPOLOGY, do_normalize=do_normalize)
                    else:
                        # reduce EEG data (downsample, number of channels, time window)
                        # manual selection of channels
                        X = eeg_reduction(X_orig, n_ch = n_ch, T=T, n_ds=n_ds, rows=RMODE)

                    if len(X.shape) == 5: # if the required n_ch is not
                        # available in this configuration, channels in
                        # eeg_reduction is None, which makes the shape from
                        # (n_trials, n_ch, n_samples, 1) to (n_trials, 1, n_ch,
                        # n_samples, 1)
                        #print('here?', n_ch, X.shape)
                        warnings.warn('the desired number of n_ch is not available for this configuration. Check eeg_reduction.py for available configurations')
                        break

                    model, history = train_validate(X[train], y_cat[train], X[test], y_cat[test],
                                                        n_classes=num_classes, n_channels=n_ch)

                    np.savez(DATAFILE.format(f'stats/{bench}', RMODE, num_classes, n_ch, split_ctr, '.npz'),
                             **history.history)
                    if not SAVE_ONLY_64CH_MODEL:
                        model.save(DATAFILE.format(f'model/{bench}', RMODE, num_classes, n_ch, split_ctr, '.h5'))

                    fstats = loadfstats(fstats, history.history, split_ctr)

                    #Clear Models
                    K.clear_session()

                    #bar.update()
                    if tbar is not None:
                        tbar.update()

                #mean_fstats = np.mean(fstats, axis=0)
                #std_fstats = np.std(fstats, axis=0)
                #print(f'CV acc {mean_fstats[1]:.4f} +- {std_fstats[1]:.4f}')
                stats[i,j,:] = fstats

                bar1.update()

    return stats


def loadfstats(fstats, history, fold):
    #history.keys = dict_keys(['val_loss', 'val_acc', 'loss', 'acc', 'lr']) 
    fstats[fold, 0] = history['acc'][-1]
    fstats[fold, 1] = history['val_acc'][-1]
    fstats[fold, 2] = history['loss'][-1]
    fstats[fold, 3] = history['val_loss'][-1]
    return fstats

if __name__ == '__main__':

    if BENCHMARK:

        if REDUCE_NETW or REDUCE_MANUAL:
            if 64 in n_ch_list:
                bstats = np.zeros((N_ITER, len(num_classes_list), len(n_ch_list), N_FOLDS, 4))
            else:
                bstats = np.zeros((N_ITER, len(num_classes_list), len(n_ch_list)+1, N_FOLDS, 4))
        else:
            bstats = np.zeros((N_ITER, len(num_classes_list), len(n_ch_list), N_FOLDS, 4))

        with tqdm(desc=f'Benchmarking Measurement {EXP_ID:02}', total=np.prod(bstats.shape)/4, ascii=True) as bar:
            for b in range(N_ITER):
                os.makedirs(os.path.join(EXP_FOLDER, f'model/{b}'), exist_ok=True)
                os.makedirs(os.path.join(EXP_FOLDER, f'stats/{b}'), exist_ok=True)

                # for num_classes in num_classes_list:
                #     for split_ctr in range(N_FOLDS):
                #         if REDUCE_NETW or REDUCE_MANUAL:
                #             # check if the baseline model with all the channels
                #             # is already trained.
                #             if os.path.isfile(DATAFILE.format('model', num_classes, 64, split_ctr, '.h5')):
                #                 #print('{} exists'.format(DATAFILE.format('model', num_classes,
                #                 #64, split_ctr, '.h5')))
                #                 os.makedirs(os.path.join(EXP_FOLDER, f'model/bench{-1}'), exist_ok=True)
                #                 os.system('mv {} {}'.format(DATAFILE.format('model', num_classes, 64, split_ctr, '.h5'), DATAFILE.format(f'model/bench{b-1}', num_classes, 64, split_ctr, '.h5')))
                with open(os.devnull, 'w') as devnull, redirect_stderr(devnull), redirect_stdout(devnull):
                    bstats[b] = run(tbar=bar, bench=b)

                # in order to not lose the benchmarks, save every time it
                # finishes a run
                np.savez(os.path.join(EXP_FOLDER, 'stats', f'{RMODE}_bench.npz'),
                         train_acc=bstats[:,:,:,:,0], val_acc=bstats[:,:,:,:,1],
                         train_loss=bstats[:,:,:,:,2], val_loss=bstats[:,:,:,:,3])

                # os.makedirs(os.path.join(EXP_FOLDER, f'model/bench{b-1}'), exist_ok=True)
                # os.system('mv {} {}'.format(DATAFILE.format('model', num_classes, 64, split_ctr, '.h5'), DATAFILE.format(f'model/bench{b-1}', num_classes, 64, split_ctr, '.h5')))

    else:
        run()
