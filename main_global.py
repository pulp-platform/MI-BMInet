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

#################################################
#
# Global model training and validation 
#
#################################################


import numpy as np
import os
import get_data as get
from tensorflow.keras import utils as np_utils
from keras.callbacks import LearningRateScheduler, ReduceLROnPlateau
from keras.optimizers import Adam
from keras import backend as K
from sklearn.model_selection import KFold

# EEGNet models
import models as models
# Channel reduction, downsampling, time window
from eeg_reduction import *

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


#################################################
#
# Learning Rate Constant Scheduling
#
#################################################
def step_decay(epoch):
    if(epoch < 20):
        lr = 0.01
    elif(epoch < 50):
        lr = 0.001
    else:
        lr = 0.0001
    return lr
lrate = LearningRateScheduler(step_decay)

def step_decay2(epoch):
    if(epoch < 40):
        lr = 0.01
    elif(epoch < 80):
        lr = 0.001
    else:
        lr = 0.0001
    return lr
lrate2 = LearningRateScheduler(step_decay2)


#################################################
#
# Save results
#
#################################################
def save_results(history,num_classes,n_ds,n_ch,T,split_ctr):

    # Save metrics  
    results = np.zeros((4,len(history.history['acc'])))
    results[0] = history.history['acc']
    results[1] = history.history['val_acc']
    results[2] = history.history['loss']
    results[3] = history.history['val_loss']
    results_str = f'{results_dir}{experiment_name}/stats/global_class_{num_classes}_ds{n_ds}_nch{n_ch}_T{T}_split_{split_ctr}.csv'
                 
    np.savetxt(results_str, np.transpose(results))
    return results[0:2,-1]



# CHANGE EXPERIMENT NAME FOR DIFFERENT TESTS!!
experiment_name = 'cubeai-global-EEGNet'
modelname = 'EEGNet'

datapath = "/usr/scratch/sassauna4/xiaywang/Projects/BCI/PhysionetMMMI/"
#datapath = "/usr/scratch/xavier/herschmi/EEG_data/physionet/"
results_dir=f'results/'
#os.makedirs(results_dir, exist_ok=True)
os.makedirs(f'{results_dir}{experiment_name}/stats', exist_ok=True)
os.makedirs(f'{results_dir}{experiment_name}/model', exist_ok=True)
os.makedirs(f'{results_dir}{experiment_name}/plots', exist_ok=True)

# for channel selection using network weights
net_dirpath = 'results/your-global-experiment-edgeC2F1_100/model/'

# HYPERPARAMETER TO SET 
num_classes_list = [4] # list of number of classes to test {2,3,4}
n_epochs = 100 # number of epochs for training
n_ds = 1 # downsamlping factor {1,2,3}
n_ch_list = [8, 19, 38, 64] # number of channels {8,19,27,38,64}

net_weights_red = False # channel reduction using network weights (spatial conv)
net_weights_avg_folds = False # average the 5 folds weights

T_list = [3] # duration to classify {1,2,3}

if modelname == 'edgeEEGNet':
    lrate = LearningRateScheduler(step_decay2)
elif modelname == 'EEGNet':
    lrate = LearningRateScheduler(step_decay)


# model settings 
kernLength = int(np.ceil(128/n_ds))
poolLength = int(np.ceil(8/n_ds))
num_splits = 5
acc = np.zeros((num_splits,2))


for num_classes in num_classes_list:
    for n_ch in n_ch_list:
        for T in T_list:

            # Load data
            #X, y = get.get_data(datapath, n_classes=num_classes)

            ######## If you want to save the data after loading once from .edf (faster)
            #np.savez(datapath+f'{num_classes}class',X = X, y = y)
            npzfile = np.load(datapath+f'{num_classes}class.npz')
            X, y = npzfile['X'], npzfile['y']

            if net_weights_red:
                # reduce EEG channels based on network weights L2 norm later
                X = eeg_reduction(X,n_ds = n_ds, n_ch = 64, T = T)
            else:
                # reduce EEG data (downsample, number of channels, time window)
                # manual selection of channels
                X = eeg_reduction(X,n_ds = n_ds, n_ch = n_ch, T = T)

            # Expand dimensions to match expected EEGNet input
            X_orig = (np.expand_dims(X, axis=-1))
            # number of temporal sample per trial
            n_samples = np.shape(X_orig)[2]

            # convert labels to one-hot encodings.
            y_cat = np_utils.to_categorical(y)

            # using 5 folds
            kf = KFold(n_splits = num_splits)

            if net_weights_red and net_weights_avg_folds:

                net_path = os.path.join(net_dirpath, f'global_class_{num_classes}_ds{n_ds}_nch64_T{T}_split_0.h5')

                X = net_weights_channel_selection_folds_avg(X_orig, n_ch=n_ch, net_path=net_path, modelname=modelname)

            for split_ctr, (train, test) in enumerate(kf.split(X_orig, y)):

                if net_weights_red:

                    if not net_weights_avg_folds:

                        net_path = os.path.join(net_dirpath, f'global_class_{num_classes}_ds{n_ds}_nch64_T{T}_split_{split_ctr}.h5')

                        X = net_weights_channel_selection(X_orig, n_ch=n_ch, net_path=net_path, modelname=modelname)
                else:
                    X = X_orig

                # init model
                model = models.cubeedgeEEGNetCF1(nb_classes = num_classes, Chans=n_ch, Samples=n_samples, regRate=0.25,
                                dropoutRate=0.2, kernLength=kernLength, poolLength=poolLength, numFilters=8, 
                                dropoutType='Dropout')


                # Set Learning Rate
                adam_alpha = Adam(lr=(0.0001))
                model.compile(loss='categorical_crossentropy', optimizer=adam_alpha, metrics = ['accuracy'])
                np.random.seed(42*(split_ctr+1))
                np.random.shuffle(train)
                # do training
                history = model.fit(X[train], y_cat[train], 
                        validation_data=(X[test], y_cat[test]),
                        batch_size = 16, epochs = n_epochs, callbacks=[lrate], verbose = 2)

                acc[split_ctr] = save_results(history,num_classes,n_ds,n_ch,T,split_ctr)
                
                print('Fold {:}\t{:.4f}\t{:.4f}'.format(split_ctr,acc[split_ctr,0], acc[split_ctr,1]))

                #Save model
                model.save(f'{results_dir}{experiment_name}/model/global_class_{num_classes}_ds{n_ds}_nch{n_ch}_T{T}_split_{split_ctr}.h5')

                #Clear Models
                K.clear_session()

            print('AVG \t {:.4f}\t{:.4f}'.format(acc[:,0].mean(), acc[:,1].mean()))


           


