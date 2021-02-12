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

import time

start = time.time()

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

class_names = ['Left hand', 'Right hand',
			   'Both Feet', 'Tongue']


#################################################
#
# Save results
#
#################################################
def save_results(results_str, history,num_classes,n_ds,n_ch,T,subject):

    # Save metrics  
    results = np.zeros((4,len(history.history['acc'])))
    results[0] = history.history['acc']
    results[1] = history.history['val_acc']
    results[2] = history.history['loss']
    results[3] = history.history['val_loss']
    #results_str = f'{results_dir}{experiment_name}/stats/bci_class_{num_classes}_ds{n_ds}_nch{n_ch}_T{T}_split_{subject}.csv'
    np.savetxt(results_str, np.transpose(results))

    return results[0:2,-1]

# Change experiment name, modelname, model declaration, net weights bool,
# net_dirpath, do_normalize

# CHANGE EXPERIMENT NAME FOR DIFFERENT TESTS!!
experiment_name = 'bci-cubeedgeEEGNet-weights-same-subj-3'
modelname = 'edgeEEGNet'

#datapath = "/usr/scratch/bismantova/xiaywang/Projects/BCI/datasets/PhysionetMMMI/QuantLab/PhysionetMMMI/data/" # there is only 4class there
#datapath = "/usr/scratch/sassauna4/xiaywang/Projects/BCI/physionet/"
datapath = "/usr/scratch/bismantova/xiaywang/Projects/BCI/datasets/BCI-CompIV-2a/QuantLab/BCI-CompIV-2a/data/"
#datapath = "/usr/scratch/xavier/herschmi/EEG_data/physionet/"
results_dir=f'results/'
#os.makedirs(results_dir, exist_ok=True)
os.makedirs(f'{results_dir}{experiment_name}/stats', exist_ok=True)
os.makedirs(f'{results_dir}{experiment_name}/model', exist_ok=True)
os.makedirs(f'{results_dir}{experiment_name}/plots', exist_ok=True)

# for channel selection using network weights pretrained
net_dirpath = 'results/bci-cubeedgeEEGNet/model/' # -edgeC2F1_100

# HYPERPARAMETER TO SET 
num_classes_list = [4] # list of number of classes to test {2,3,4}
n_epochs = 500 # number of epochs for training
n_ds = 1 # downsamlping factor {1,2,3}
n_ch_list = [2, 3, 5, 7, 8, 9, 11, 4, 6, 10, 14, 18, 19, 20, 16, 22]#[2, 3, 5, 7, 8, 9, 11, 4, 6, 10, 14, 18, 19, 20, 16, 24, 32, 38, 64] #[8, 19, 38, 64] #[16, 24, 32, 64] #[2, 3, 5, 7, 9, 11, 4, 6, 10, 14, 18, 20, 64] # number of channels
# {8,19,27,38,64}
# remember to put 64 (physionet) or 22 (BCI CompIV2a) if you have net_weights_same_subj = True because it has
# to train the 64 full model within each fold, then select the channels and
# train within the same fold
net_weights_red = True # channel reduction using network weights (spatial conv)
net_weights_same_subj = True # such that the channels selected never see the
do_normalize=''

T_list = [0] # duration to classify {1,2,3}

# if modelname == 'edgeEEGNet':
#     lrate = LearningRateScheduler(step_decay2)
# elif modelname == 'EEGNet':
#     lrate = LearningRateScheduler(step_decay)

# model settings 
kernLength = int(np.ceil(128/n_ds))
poolLength = int(np.ceil(8/n_ds))
n_subjects = 9
acc = np.zeros((n_subjects, 2))


for num_classes in num_classes_list:
    for T in T_list:
        for n_ch in n_ch_list:
            for subject in range(n_subjects):

                if net_weights_same_subj and n_ch != 22:
                    print("net_weights_same_subj {}, n_ch {}".format(net_weights_same_subj, n_ch))
                    continue
                print("net_weights_same_subj {}, n_ch {}, training full channels model".format(net_weights_same_subj, n_ch))

                # Load data
                X_train, y_train = get.get_data_bci(subject+1, training=True, path=datapath, n_classes=num_classes)
                X_test, y_test = get.get_data_bci(subject+1, training=False, path=datapath, n_classes=num_classes)

                if net_weights_red:
                    # reduce EEG channels based on network weights L2 norm later
                    X_train = eeg_reduction(X_train, n_ds = n_ds, n_ch = 22, T = T) # (273,22,1125)
                    X_test = eeg_reduction(X_test, n_ds = n_ds, n_ch = 22, T = T) # (281,22,1125)
                else:
                    # reduce EEG data (downsample, number of channels, time window)
                    # manual selection of channels
                    X_train = eeg_reduction(X_train, n_ds = n_ds, n_ch = n_ch, T = T)
                    X_test = eeg_reduction(X_test, n_ds = n_ds, n_ch = n_ch, T = T)


                # Expand dimensions to match expected EEGNet input
                X_train_orig = (np.expand_dims(X_train, axis=-1))
                X_test_orig = (np.expand_dims(X_test, axis=-1))
                # number of temporal sample per trial
                n_samples = np.shape(X_train_orig)[2]

                # convert labels to one-hot encodings.
                y_train_cat = np_utils.to_categorical(y_train)
                y_test_cat = np_utils.to_categorical(y_test)


                print('Subject {}'.format(subject))

                if net_weights_red:

                    if net_weights_same_subj:

                        X_train = X_train_orig
                        X_test = X_test_orig
                        print('net_weights_same_subj {}, X_train shape {}, X_test shape {}'.format(net_weights_same_subj, X_train.shape, X_test.shape))

                    else:

                        net_path = os.path.join(net_dirpath, f'bci_class_{num_classes}_ds{n_ds}_nch22_T{T}_subj_{subject}.h5')

                        X_train = net_weights_channel_selection(X_train_orig, n_ch=n_ch, net_path=net_path, modelname=modelname, do_normalize=do_normalize)
                        X_test = net_weights_channel_selection(X_test_orig, n_ch=n_ch, net_path=net_path, modelname=modelname, do_normalize=do_normalize)
                        print('net_weights_same_subj {}, X_train shape {}, X_test shape {}'.format(net_weights_same_subj, X_train.shape, X_test.shape))

                else:
                    X_train = X_train_orig
                    X_test = X_test_orig
                    print('net_weights_red {}, X_train shape {}, X_test shape {}'.format(net_weights_red, X_train.shape, X_test.shape))

                #exit()


                if os.path.isfile(f'{results_dir}{experiment_name}/stats/bci_class_{num_classes}_ds{n_ds}_nch22cs_T{T}_subj_{subject}.csv'):
                    print(f'File {results_dir}{experiment_name}/stats/bci_class_{num_classes}_ds{n_ds}_nch22cs_T{T}_subj_{subject}.csv exists')
                else:
                    print (f'File {results_dir}{experiment_name}/stats/bci_class_{num_classes}_ds{n_ds}_nch22cs_T{T}_subj_{subject}.csv doesn\'t exists')

                    # init model
                    model = models.cubeedgeEEGNetCF1(nb_classes = num_classes, Chans=n_ch, Samples=n_samples, regRate=0.25,
                                    dropoutRate=0.2, kernLength=kernLength, poolLength=poolLength, numFilters=8, 
                                    dropoutType='Dropout')


                    print("shapes ", X_train.shape, y_train_cat.shape, X_test.shape, y_test_cat.shape)
                    # Set Learning Rate
                    adam_alpha = Adam(lr=(0.001))
                    model.compile(loss='categorical_crossentropy', optimizer=adam_alpha, metrics = ['accuracy'])
                    np.random.seed(42*(subject+1))
                    # do training
                    history = model.fit(X_train, y_train_cat,
                            validation_data=(X_test, y_test_cat),
                            batch_size = 32, epochs = n_epochs, verbose = 2)

                    if net_weights_same_subj:

                        acc[subject] = save_results(f'{results_dir}{experiment_name}/stats/bci_class_{num_classes}_ds{n_ds}_nch22cs_T{T}_split_{subject}.csv', history,num_classes,n_ds,n_ch,T,subject)

                        #Save model
                        model.save(f'{results_dir}{experiment_name}/model/bci_class_{num_classes}_ds{n_ds}_nch22cs_T{T}_split_{subject}.h5')

                        print('net_weights_same_subj {}, save in {}'.format(net_weights_same_subj, f'{results_dir}{experiment_name}/model/bci_class_{num_classes}_ds{n_ds}_nch22cs_T{T}_split_{subject}'))

                    else:

                        acc[subject] = save_results(f'{results_dir}{experiment_name}/stats/bci_class_{num_classes}_ds{n_ds}_nch{n_ch}_T{T}_split_{subject}.csv', history,num_classes,n_ds,n_ch,T,subject)

                        #Save model
                        model.save(f'{results_dir}{experiment_name}/model/bci_class_{num_classes}_ds{n_ds}_nch{n_ch}_T{T}_split_{subject}.h5')

                        print('net_weights_same_subj {}, save in {}'.format(net_weights_same_subj, f'{results_dir}{experiment_name}/model/bci_class_{num_classes}_ds{n_ds}_nch{n_ch}_T{T}_split_{subject}'))

                    print('Subject {:}\t{:.4f}\t{:.4f}'.format(subject, acc[subject, 0], acc[subject, 1]))


                #Clear Models
                K.clear_session()

                if net_weights_same_subj:

                    for n_ch_cs in n_ch_list:

                        if n_ch_cs == 22:
                            print('net_weights_same_subj {}, n_ch_cs {}. No need to run channel selection for 64 channels'.format(net_weights_same_subj, n_ch_cs))
                            continue
                        print('net_weights_same_subj {}, n_ch_cs {}'.format(net_weights_same_subj, n_ch_cs))

                        net_path = os.path.join(f'{results_dir}{experiment_name}/model/bci_class_{num_classes}_ds{n_ds}_nch22cs_T{T}_split_{subject}.h5')

                        X_train = net_weights_channel_selection(X_train_orig, n_ch=n_ch_cs, net_path=net_path, modelname=modelname, do_normalize=do_normalize)
                        X_test = net_weights_channel_selection(X_test_orig, n_ch=n_ch_cs, net_path=net_path, modelname=modelname, do_normalize=do_normalize)
                        print('read net_path from {} for weights and train a new model with selected channels'.format(net_path))

                        # init model
                        model_cs = models.cubeedgeEEGNetCF1(nb_classes = num_classes, Chans=n_ch_cs, Samples=n_samples, regRate=0.25,
                                    dropoutRate=0.2, kernLength=kernLength, poolLength=poolLength, numFilters=8, 
                                    dropoutType='Dropout')

                        # Set Learning Rate
                        adam_alpha = Adam(lr=(0.001))
                        model_cs.compile(loss='categorical_crossentropy', optimizer=adam_alpha, metrics = ['accuracy'])

                        np.random.seed(42*(subject+1))
                        # do training
                        history = model_cs.fit(X_train, y_train_cat,
                                validation_data=(X_test, y_test_cat),
                                batch_size = 32, epochs = n_epochs, verbose = 2)

                        acc[subject] = save_results(f'{results_dir}{experiment_name}/stats/bci_class_{num_classes}_ds{n_ds}_nch{n_ch_cs}cs_T{T}_split_{subject}.csv', history,num_classes,n_ds,n_ch_cs,T,subject)

                        #Save model
                        model_cs.save(f'{results_dir}{experiment_name}/model/bci_class_{num_classes}_ds{n_ds}_nch{n_ch_cs}cs_T{T}_split_{subject}.h5')

                        print('net_weights_same_subj {}, save in {}'.format(net_weights_same_subj, f'{results_dir}{experiment_name}/model/bci_class_{num_classes}_ds{n_ds}_nch{n_ch_cs}cs_T{T}_split_{subject}.h5'))

                        print('Subject {:}\t{:.4f}\t{:.4f}'.format(subject, acc[subject,0], acc[subject,1]))

                        #Clear Models
                        K.clear_session()


            print('AVG \t {:.4f}\t{:.4f}'.format(acc[:,0].mean(), acc[:,1].mean()))

        break


end = time.time()
print("time used: ", end - start)
