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

from sklearn.metrics import cohen_kappa_score

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
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

TOPOLOGY = "edgeEEGNet"
EXP_ID = 3
LOG_NOTE = 'cs with net weights from same folds'

BENCHMARK = True
N_ITER = 25
N_SUBJECTS = 9

DATAPATH  = "/usr/scratch/bismantova/xiaywang/Projects/BCI/datasets/BCI-CompIV-2a/QuantLab/BCI-CompIV-2a/data/"
LOG_FOLDER = 'logs/bci'
EXP_FOLDER = os.path.join(LOG_FOLDER, TOPOLOGY, f'exp{EXP_ID}')

STATS_FNAME = '{}_class{}_nch{}_split{}'
DATAFILE = os.path.join(EXP_FOLDER, '{}', STATS_FNAME+'{}')

SAVE_ONLY_22CH_MODEL = False
if BENCHMARK:
    SAVE_ONLY_22CH_MODEL = True

REDUCE_MANUAL = False # channels overall the brain ('dist') or over the sensorimotor area ('headset')
REDUCE_NETW = True # channel reduction using network weights (spatial conv)
do_normalize='' # for channel reduction using net weights
#HEADSETMODE='headset' # 'dist' or 'headset' or ''
RMODE='auto' # 'crow', 'ccfrows', 'ccprows' or 'dist' or 'auto'
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

N_EPOCHS = 500 # number of epochs for training

num_classes_list = [2,3,4] # list of number of classes to test {2,3,4}
n_ch_list = [2, 3, 4, 5, 6, 7, 8, 9, 22] #, 11, 14, 16, 18, 19, 20, 22]
# remember to put 64 if you have net_weights_same_folds = True because it has
# to train the 64 full model within each fold, then select the channels and
# train within the same fold
T=0
n_ds=1

KAPPA_SCORE = True

if RMODE != 'auto' and REDUCE_MANUAL:
    if RMODE == 'dist':
        raise ValueError("ValueError no dist configuration for bci comp iv 2a dataset")
    elif RMODE == 'crow':
        manual_ch = [2, 3, 5, 7]
    elif RMODE in ('ccprows', 'ccfrows'):
        manual_ch = [4, 6]
    n_ch_set = set(n_ch_list)
    intersection = n_ch_set.intersection(manual_ch)
    n_ch_list = list(intersection)


def train_validate(X_train, y_train, X_test, y_test, n_classes=4, n_channels=64, dropoutRate=0.2):

    # check the activation input
    topology = TOPOLOGY.lower()
    assert topology in ['eegnet', 'edgeeegnet']

    if topology == 'edgeeegnet':
        model = models.cubeedgeEEGNetCF1(nb_classes = n_classes, Chans=n_channels, Samples=X_train.shape[2], dropoutRate=dropoutRate, dropoutType='TimeDropout2D', kernLength=64, numFilters=32)
    elif topology == 'eegnet':
        model = models.cubeEEGNet(nb_classes = n_classes, Chans=n_channels, Samples=X_train.shape[2], dropoutRate=dropoutRate, dropoutType='TimeDropout2D', kernLength=64, numFilters=32)

    adam_alpha = Adam(lr=(0.001))
    model.compile(loss='categorical_crossentropy', optimizer=adam_alpha, metrics = ['accuracy'])

    # do training
    history = model.fit(X_train, y_train,
                        validation_data=(X_test, y_test),
                        batch_size = 32, epochs = N_EPOCHS, verbose = 0)

    return model, history

def run(tbar=None, bench=''):

    if REDUCE_NETW or REDUCE_MANUAL:
        # delete 64 from n_ch_list if it's there
        try:
            n_ch_list.remove(22)
        except:
            pass

        stats = np.zeros((len(num_classes_list), len(n_ch_list)+1, N_SUBJECTS, 4))
        if KAPPA_SCORE:
            kappascore = np.zeros((len(num_classes_list), len(n_ch_list)+1, N_SUBJECTS))
    else:
        stats = np.zeros((len(num_classes_list), len(n_ch_list), N_SUBJECTS, 4))
        if KAPPA_SCORE:
            kappascore = np.zeros((len(num_classes_list), len(n_ch_list), N_SUBJECTS))

    for i, num_classes in enumerate(num_classes_list):

        with tqdm(desc=f'{N_SUBJECTS} on {len(n_ch_list)} n_ch', total=len(n_ch_list)*N_SUBJECTS, ascii=True) as bar1:
            for subject in range(N_SUBJECTS):

                # Load data
                print('loading data')
                X_train, y_train = get.get_data_bci(subject+1, training=True, path=DATAPATH, n_classes=num_classes)
                X_test, y_test = get.get_data_bci(subject+1, training=False, path=DATAPATH, n_classes=num_classes)

                # Expand dimensions to match expected EEGNet input
                X_train_orig = (np.expand_dims(X_train, axis=-1))
                X_test_orig = (np.expand_dims(X_test, axis=-1))
                # number of temporal sample per trial
                n_samples = np.shape(X_train_orig)[2]

                # convert labels to one-hot encodings.
                y_train_cat = np_utils.to_categorical(y_train)
                y_test_cat = np_utils.to_categorical(y_test)

                if REDUCE_NETW or REDUCE_MANUAL:
                    # check if the baseline model with all the channels
                    # is already trained.
                    if os.path.isfile(DATAFILE.format(f'model/{bench}', '', num_classes, 22, subject, '.h5')):
                        #print('{} exists'.format(DATAFILE.format(f'model/{bench}', '', num_classes, 22, subject, '.h5')))
                        pass

                    else:
                        #print("Channel reduction. Prepare the baseline model with full number of channels, i.e. 22")

                        model, history = train_validate(X_train_orig, y_train_cat,
                                                        X_test_orig, y_test_cat,
                                                        n_classes=num_classes, n_channels=22,
                                                        dropoutRate=0.5)

                        # if BENCHMARK:

                        #     np.savez(DATAFILE.format(f'stats/bench{bench}', num_classes, 64,
                        #                              subject, '.npz'), **history.history)
                        #     model.save(DATAFILE.format(f'model/bench{bench}', num_classes, 64, subject, '.h5'))
                        # else:
                        np.savez(DATAFILE.format(f'stats/{bench}', '', num_classes, 22,
                                                 subject, '.npz'), **history.history)
                        model.save(DATAFILE.format(f'model/{bench}', '', num_classes, 22, subject, '.h5'))

                        if KAPPA_SCORE:
                            preds = model.predict(X_test_orig, batch_size=32, verbose=1)
                            y_pred=np.argmax(preds, axis=1)
                            print(cohen_kappa_score(y_test.astype(int), y_pred))
                            kappascore[i, len(n_ch_list), subject] = cohen_kappa_score(y_test.astype(int), y_pred)
                            #print(np.sum(y_pred==y_test.astype(int))/len(y_test))
                            #print(history.history['val_acc']-np.sum(y_pred==y_test.astype(int))/len(y_test))
                            #np.savez(DATAFILE.format(f'stats/{bench}', '', num_classes, 22,
                            #                    subject, '_kappa.npz'), kappascore=kappascore)

                        stats[i, len(n_ch_list), subject, :] = load1fstats(history.history)
                        # save full model stats to the last index
                        #print(stats[i, len(n_ch_list)])

                        #Clear Models
                        K.clear_session()

                        #bar.update()
                        if tbar is not None:
                            tbar.update()

                for j, n_ch in enumerate(n_ch_list):

                    if REDUCE_NETW:
                        X_train = net_weights_channel_selection_bci(X_train_orig, n_ch=n_ch, net_path=DATAFILE.format(f'model/{bench}', '', num_classes, 22, subject, '.h5'), modelname=TOPOLOGY, do_normalize=do_normalize)
                        X_test = net_weights_channel_selection_bci(X_test_orig, n_ch=n_ch, net_path=DATAFILE.format(f'model/{bench}', '', num_classes, 22, subject, '.h5'), modelname=TOPOLOGY, do_normalize=do_normalize)
                    else:
                        # reduce EEG data (downsample, number of channels, time window)
                        # manual selection of channels
                        X_train = eeg_reduction(X_train_orig, n_ch = n_ch, T=T, n_ds=n_ds, rows=RMODE)
                        X_test = eeg_reduction(X_test_orig, n_ch = n_ch, T=T, n_ds=n_ds, rows=RMODE)

                    if len(X_train.shape) == 5: # if the required n_ch is not
                        # available in this configuration, channels in
                        # eeg_reduction is None, which makes the shape from
                        # (n_trials, n_ch, n_samples, 1) to (n_trials, 1, n_ch,
                        # n_samples, 1)
                        #print('here?', n_ch, X.shape)
                        warnings.warn('the desired number of n_ch is not available for this configuration. Check eeg_reduction.py for available configurations')
                        break

                    model, history = train_validate(X_train, y_train_cat, X_test, y_test_cat,
                                                    n_classes=num_classes, n_channels=n_ch,
                                                    dropoutRate=0.5)
                    print('trained')
                    np.savez(DATAFILE.format(f'stats/{bench}', RMODE, num_classes, n_ch, subject, '.npz'),
                             **history.history)
                    if not SAVE_ONLY_22CH_MODEL:
                        model.save(DATAFILE.format(f'model/{bench}', RMODE, num_classes, n_ch, subject, '.h5'))

                    if KAPPA_SCORE:
                        preds = model.predict(X_test, batch_size=32, verbose=1)
                        y_pred=np.argmax(preds, axis=1)
                        print(cohen_kappa_score(y_test.astype(int), y_pred))
                        kappascore[i, j, subject] = cohen_kappa_score(y_test.astype(int), y_pred)
                        #print(np.sum(y_pred==y_test.astype(int))/len(y_test))
                        #print(history.history['val_acc']-np.sum(y_pred==y_test.astype(int))/len(y_test))
                        #np.savez(DATAFILE.format(f'stats/{bench}', '', num_classes, n_ch,
                        #                         subject, '_kappa.npz'), kappascore=kappascore)

                    fstats = load1fstats(history.history)

                    #Clear Models
                    K.clear_session()

                    #bar.update()
                    if tbar is not None:
                        tbar.update()

                    #mean_fstats = np.mean(fstats, axis=0)
                    #std_fstats = np.std(fstats, axis=0)
                    #print(f'CV acc {mean_fstats[1]:.4f} +- {std_fstats[1]:.4f}')
                    stats[i,j,subject,:] = fstats

                bar1.update()

    return stats, kappascore


def load1fstats(history):
    #history.keys = dict_keys(['val_loss', 'val_acc', 'loss', 'acc', 'lr'])
    fstats = np.zeros((1,4))
    fstats[0, 0] = history['acc'][-1]
    fstats[0, 1] = history['val_acc'][-1]
    fstats[0, 2] = history['loss'][-1]
    fstats[0, 3] = history['val_loss'][-1]
    return fstats

if __name__ == '__main__':

    if BENCHMARK:

        if REDUCE_NETW or REDUCE_MANUAL:
            if 22 in n_ch_list:
                bstats = np.zeros((N_ITER, len(num_classes_list), len(n_ch_list), N_SUBJECTS, 4))
                if KAPPA_SCORE:
                    kappascore = np.zeros((N_ITER, len(num_classes_list), len(n_ch_list), N_SUBJECTS))
            else:
                bstats = np.zeros((N_ITER, len(num_classes_list), len(n_ch_list)+1, N_SUBJECTS, 4))
                if KAPPA_SCORE:
                    kappascore = np.zeros((N_ITER, len(num_classes_list), len(n_ch_list)+1, N_SUBJECTS))
        else:
            bstats = np.zeros((N_ITER, len(num_classes_list), len(n_ch_list), N_SUBJECTS, 4))
            if KAPPA_SCORE:
                kappascore = np.zeros((N_ITER, len(num_classes_list), len(n_ch_list), N_SUBJECTS))

        with tqdm(desc=f'Benchmarking Measurement {EXP_ID:02}', total=np.prod(bstats.shape)/4, ascii=True) as bar:
            for b in range(N_ITER):
                os.makedirs(os.path.join(EXP_FOLDER, f'model/{b}'), exist_ok=True)
                os.makedirs(os.path.join(EXP_FOLDER, f'stats/{b}'), exist_ok=True)

                #with open(os.devnull, 'w') as devnull, redirect_stderr(devnull), redirect_stdout(devnull):
                if KAPPA_SCORE:
                    bstats[b], kappascore[b] = run(tbar=bar, bench=b)
                else:
                    bstats[b], _ = run(tbar=bar, bench=b)

                # in order to not lose the benchmarks, save every time it
                # finishes a run
                np.savez(os.path.join(EXP_FOLDER, 'stats', f'{RMODE}_bench.npz'),
                         train_acc=bstats[:,:,:,:,0], val_acc=bstats[:,:,:,:,1],
                         train_loss=bstats[:,:,:,:,2], val_loss=bstats[:,:,:,:,3])
                np.savez(os.path.join(EXP_FOLDER, 'stats', f'{RMODE}_bench_kappascore.npz'),
                         kappascore=kappascore)

                # os.makedirs(os.path.join(EXP_FOLDER, f'model/bench{b-1}'), exist_ok=True)
                # os.system('mv {} {}'.format(DATAFILE.format('model', num_classes, 64, subject, '.h5'), DATAFILE.format(f'model/bench{b-1}', num_classes, 64, subject, '.h5')))

    else:
        run()
