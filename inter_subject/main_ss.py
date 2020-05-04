#!/usr/bin/env python3

__author__ = "Batuhan Tomekce, Burak Alp Kaya, Tianhong Gan"
__email__ = "tbatuhan@ethz.ch, bukaya@ethz.ch, tianhonggan@outlook.com"

import os
import pdb
import numpy as np

# our functions to get data
import pyedflib
import get_data as get

# tensorflow part
from tensorflow.keras import utils as np_utils
from keras.callbacks import ModelCheckpoint
from keras.callbacks import EarlyStopping
from keras.callbacks import LearningRateScheduler
from keras.optimizers import Adam
from keras.models import load_model
from keras import backend as K

# EEGNet models
import models as models
from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold

# for channel selection
from channel_selection import channel_selection_eegweights_fromglobal

# layer selection
from layer_freeze import freeze_layers

# plot graphs
import matplotlib.pyplot as plt
from plot_graph import plot_subject_avg, plot_avg, plot_model_avg

# Select GPU
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

#################################################
# Version 1

# 64-channel global model trained,
# selected N channels based on EEGNet weights,
# train from scratch N-channel global model,
# use N-channel global model to retrain final epoch SS.
#
# 5 global models, one for each fold are used
# and channels selected for each.

# Finally, results within, and across
# subjects are averaged and plotted.

#################################################

# Remove excluded subjects from subjects list
def exclude_subjects(all_subjects=range(1,110), excluded_subjects=[88,92,100,104]):
    subjects = [x for x in all_subjects if (x not in excluded_subjects)]
    return subjects

subjects = exclude_subjects()

# For channel selection
NO_channels = 64 # total number of EEG channels
n_ch_vec = [8,16,19,24,38] # number of selected channels
num_classes_list = [4] # specify number of classses for input data

# For freezing layers
no_layers_unfrozen = 3 # 1: fc trainable, 2: sep_conv and fc trainable, 3: depth_conv, sep_conv and fc trainable
if no_layers_unfrozen < 4 and no_layers_unfrozen > 0:
    freeze_training = True
else:
    freeze_training = False

# For training
n_epochs = 10
lr = 1e-3
verbose = 2 # verbosity for data loader and keras: 0 minimum

# Set data path
PATH = "/usr/scratch/badile01/sem20f12/files"

for NO_selected_channels in n_ch_vec:
    # Make necessary directories for files
    if not freeze_training:
        results_dir=f'ss/{NO_selected_channels}ch'
    else:
        results_dir=f'ss/freeze{no_layers_unfrozen}/{NO_selected_channels}ch'
    os.makedirs(f'{results_dir}/stats', exist_ok=True)
    os.makedirs(f'{results_dir}/model', exist_ok=True)
    os.makedirs(f'{results_dir}/plots', exist_ok=True)
    os.makedirs(f'{results_dir}/stats/avg', exist_ok=True)
    os.makedirs(f'{results_dir}/plots/avg', exist_ok=True)

    for num_classes in num_classes_list:
        # Using 5 folds
        num_splits = 5
        kf_global = KFold(n_splits = num_splits)
        split_ctr = 0

        for train_global, test_global in kf_global.split(subjects):
            # Select channels for this fold
            selected_channels = channel_selection_eegweights_fromglobal(NO_channels, NO_selected_channels, num_classes, split_ctr)

            for sub_idx in test_global:
                subject = subjects[sub_idx]
                X_sub, y_sub = get.get_data(PATH, n_classes=num_classes, subjects_list=[subject])
                X_sub = np.expand_dims(X_sub, axis=1)
                y_sub_cat = np_utils.to_categorical(y_sub)
                SAMPLE_SIZE = np.shape(X_sub)[3]
                kf_subject = StratifiedKFold(n_splits=4, shuffle=True, random_state=42)
                X_sub = X_sub[:,:,selected_channels,:]
                sub_split_ctr = 0

                for train_sub, test_sub in kf_subject.split(X_sub, y_sub):
                    print(f'N_Classes:{num_classes}, Model: {split_ctr} \n Subject: {subject:03d}, Split: {sub_split_ctr}')
                    model = load_model(f'global/model/global_class_{num_classes}_ds1_nch{NO_selected_channels}_T3_split_{split_ctr}_v1.h5') # SS-TL from global model

                    # For layer freezing
                    if freeze_training:
                        model = freeze_layers(model, no_layers_unfrozen, verbose = True)

                    # First evaluation of model
                    first_eval = model.evaluate(X_sub[test_sub], y_sub_cat[test_sub], batch_size=16)

                    train_accu = np.array([])
                    valid_accu = np.array([])
                    train_loss = np.array([])
                    valid_loss = np.array([])

                    # The first elements of the arrays from evaluation
                    train_accu = np.append(train_accu, first_eval[1])
                    valid_accu = np.append(valid_accu, first_eval[1])
                    train_loss = np.append(train_loss, first_eval[0])
                    valid_loss = np.append(valid_loss, first_eval[0])

                    # Set Learning Rate
                    adam_alpha = Adam(lr=lr)
                    model.compile(loss='categorical_crossentropy', optimizer=adam_alpha, metrics = ['accuracy'])

                    # Creating a history object
                    history = model.fit(X_sub[train_sub], y_sub_cat[train_sub],
                            validation_data=(X_sub[test_sub], y_sub_cat[test_sub]),
                            batch_size = 16, epochs = n_epochs, verbose = verbose)

                    train_accu = np.append(train_accu, history.history['acc'])
                    valid_accu = np.append(valid_accu, history.history['val_acc'])
                    train_loss = np.append(train_loss, history.history['loss'])
                    valid_loss = np.append(valid_loss, history.history['val_loss'])

                    sub_str = '{0:03d}'.format(subject)

                    # Save metrics
                    train_accu_str = f'{results_dir}/stats/train_accu_v1_class_{num_classes}_subject_{sub_str}_fold_{sub_split_ctr}.csv'
                    valid_accu_str = f'{results_dir}/stats/valid_accu_v1_class_{num_classes}_subject_{sub_str}_fold_{sub_split_ctr}.csv'
                    train_loss_str = f'{results_dir}/stats/train_loss_v1_class_{num_classes}_subject_{sub_str}_fold_{sub_split_ctr}.csv'
                    valid_loss_str = f'{results_dir}/stats/valid_loss_v1_class_{num_classes}_subject_{sub_str}_fold_{sub_split_ctr}.csv'

                    np.savetxt(train_accu_str, train_accu)
                    np.savetxt(valid_accu_str, valid_accu)
                    np.savetxt(train_loss_str, train_loss)
                    np.savetxt(valid_loss_str, valid_loss)

                    # Save model
                    # print('Saving model...')
                    # model.save(f'{results_dir}/model/subject{sub_str}_fold{sub_split_ctr}.h5')

                    K.clear_session()
                    sub_split_ctr = sub_split_ctr + 1
                    
            split_ctr = split_ctr + 1

    # Get average for each subject for all splits and plot
    plot_subject_avg(num_classes_list,results_dir,subjects,n_epochs,lr)

    # Get average for everything and plot
    plot_avg(num_classes_list,results_dir,subjects,NO_selected_channels,n_epochs,lr)

    # Get average of each model for all subjects and plot
    plot_model_avg(num_classes_list,results_dir,subjects,NO_selected_channels,n_epochs,lr)
