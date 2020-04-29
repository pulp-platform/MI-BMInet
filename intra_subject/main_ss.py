#!/usr/bin/env python3

__author__ = "Batuhan Tomekce, Burak Alp Kaya, Tianhong Gan"
__email__ = "tbatuhan@ethz.ch, bukaya@ethz.ch, tianhonggan@outlook.com"

import numpy as np
import os
import pdb

# our functions to get and present data
import pyedflib
import get_data as get
import matplotlib.pyplot as plt

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
from eeg_reduction import eeg_reduction_cs

# layer selection
from layer_freeze import freeze_layers, print_model

# Select GPU
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

#################################################
#
# Remove excluded subjects from subjects list
#
#################################################

def exclude_subjects(all_subjects=range(1,110), excluded_subjects=[88,92,100,104]):
    subjects = [x for x in all_subjects if (x not in excluded_subjects)]
    return subjects

#################################################
#
# Version 1
#
# 64-channel global model trained,
# selected N channels based on EEGNet weights,
# train from scratch N-channel global model,
# use N-channel global model to retrain final epoch SS.
#
# 5 global models, one for each fold are used
# and channels selected for each.
#
# Finally, results within, and across
# subjects are averaged and plotted.
#
#################################################

# Exclude subjects whose data we do not use
subjects = exclude_subjects()

# For channel selection
NO_channels = 64 # total number of EEG channels
n_ch_vec = [8] # number of selected channels
num_classes_list = [4] # specify number of classses for input data

# layers_to_freeze = np.array([0,1])
# no_layers_unfrozen = 5-len(layers_to_freeze)
no_layers_unfrozen = 3

# Retraining parameters
if no_layers_unfrozen > 3:
    n_epochs = 10
else:
    n_epochs = 2
lr = 1e-3
verbose = 2 # verbosity for data loader and keras: 0 minimum

# Set data path
PATH = "/usr/scratch/badile01/sem20f12/files"

for NO_selected_channels in n_ch_vec:
    # Make necessary directories for files
    if no_layers_unfrozen > 3:
        results_dir=f'ss/{NO_selected_channels}ch'
    else:
        results_dir=f'ss/freeze{no_layers_unfrozen}/{NO_selected_channels}ch'
    os.makedirs(f'{results_dir}/stats', exist_ok=True)
    os.makedirs(f'{results_dir}/model', exist_ok=True)
    os.makedirs(f'{results_dir}/plots', exist_ok=True)
    os.makedirs(f'{results_dir}/stats/avg', exist_ok=True)
    os.makedirs(f'{results_dir}/plots/avg', exist_ok=True)

    for num_classes in num_classes_list:
        for subject in subjects:
            # subject = subjects[sub_idx]
            X_sub, y_sub = get.get_data(PATH, n_classes=num_classes, subjects_list=[subject])
            X_sub = np.expand_dims(X_sub, axis=1)
            y_sub_cat = np_utils.to_categorical(y_sub)
            SAMPLE_SIZE = np.shape(X_sub)[3]
            kf_subject = KFold(n_splits = 3)
            sub_split_ctr = 0

            for train_sub, test_sub in kf_subject.split(X_sub, y_sub):
                # Select channels for this fold
                # selected_channels = channel_selection_eegweights_fromglobal(NO_channels, NO_selected_channels, num_classes, sub_split_ctr)
                # X_sub_cs = X_sub[:,:,selected_channels,:]
                X_sub_cs = eeg_reduction_cs(X_sub, sub_split_ctr, n_ds = 1, n_ch = NO_selected_channels, T = 3, fs = 160, num_classes = num_classes)

                print(f'N_Classes:{num_classes}, Model: {sub_split_ctr} \n Subject: {subject:03d}, Split: {sub_split_ctr}')

                model = load_model(f'global/model/global_class_{num_classes}_ds1_nch{NO_selected_channels}_T3_split_{sub_split_ctr}_v1.h5') # SS-TL from global model

                if no_layers_unfrozen < 4 && no_layers_unfrozen > 0:
                    freeze_layers(model, no_layers_unfrozen)
                    print_model(model, no_layers_unfrozen)

                first_eval = model.evaluate(X_sub_cs[test_sub], y_sub_cat[test_sub], batch_size=16)

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
                history = model.fit(X_sub_cs[train_sub], y_sub_cat[train_sub],
                        validation_data=(X_sub_cs[test_sub], y_sub_cat[test_sub]),
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

    # Get average for each subject and plot
    for num_classes in num_classes_list:
        os.makedirs(f'{results_dir}/stats/{num_classes}_class', exist_ok=True)
        os.makedirs(f'{results_dir}/plots/{num_classes}_class', exist_ok=True)
        for subject in subjects:
            train_accu = np.zeros(n_epochs+1)
            valid_accu = np.zeros(n_epochs+1)
            train_loss = np.zeros(n_epochs+1)
            valid_loss = np.zeros(n_epochs+1)
            sub_str = '{0:03d}'.format(subject)
            for sub_split_ctr in range(0,3):
                # Save metrics
                train_accu_step = np.loadtxt(f'{results_dir}/stats/train_accu_v1_class_{num_classes}_subject_{sub_str}_fold_{sub_split_ctr}.csv')
                valid_accu_step = np.loadtxt(f'{results_dir}/stats/valid_accu_v1_class_{num_classes}_subject_{sub_str}_fold_{sub_split_ctr}.csv')
                train_loss_step = np.loadtxt(f'{results_dir}/stats/train_loss_v1_class_{num_classes}_subject_{sub_str}_fold_{sub_split_ctr}.csv')
                valid_loss_step = np.loadtxt(f'{results_dir}/stats/valid_loss_v1_class_{num_classes}_subject_{sub_str}_fold_{sub_split_ctr}.csv')

                train_accu += train_accu_step
                valid_accu += valid_accu_step
                train_loss += train_loss_step
                valid_loss += valid_loss_step

            train_accu = train_accu/3
            valid_accu = valid_accu/3
            train_loss = train_loss/3
            valid_loss = valid_loss/3

            np.savetxt(f'{results_dir}/stats/{num_classes}_class/train_accu_v1_class_{num_classes}_subject_{sub_str}_avg.csv', train_accu)
            np.savetxt(f'{results_dir}/stats/{num_classes}_class/valid_accu_v1_class_{num_classes}_subject_{sub_str}_avg.csv', valid_accu)
            np.savetxt(f'{results_dir}/stats/{num_classes}_class/train_loss_v1_class_{num_classes}_subject_{sub_str}_avg.csv', train_loss)
            np.savetxt(f'{results_dir}/stats/{num_classes}_class/valid_loss_v1_class_{num_classes}_subject_{sub_str}_avg.csv', valid_loss)

            # Plot Accuracy
            plt.plot(train_accu, label='Training')
            plt.plot(valid_accu, label='Validation')
            plt.title(f'S:{sub_str} C:{num_classes} Accuracy')
            plt.xlabel('Epochs')
            plt.ylabel('Accuracy')
            plt.legend()
            plt.savefig(f'{results_dir}/plots/{num_classes}_class/accu_avg_{num_classes}_c_{sub_str}.pdf')
            plt.clf()

            # Plot Loss
            plt.plot(train_loss, label='Training')
            plt.plot(valid_loss, label='Validation')
            plt.title(f'S:{sub_str} C:{num_classes} Loss')
            plt.xlabel('Epochs')
            plt.ylabel('Loss')
            plt.legend()
            plt.savefig(f'{results_dir}/plots/{num_classes}_class/loss_avg_{num_classes}_c_{sub_str}.pdf')
            plt.clf()

    # Get average for everything and plot
    for num_classes in num_classes_list:
        train_accu = np.zeros(n_epochs+1)
        valid_accu = np.zeros(n_epochs+1)
        train_loss = np.zeros(n_epochs+1)
        valid_loss = np.zeros(n_epochs+1)
        for subject in subjects:
            sub_str = '{0:03d}'.format(subject)
            train_accu_step = np.loadtxt(f'{results_dir}/stats/{num_classes}_class/train_accu_v1_class_{num_classes}_subject_{sub_str}_avg.csv')
            valid_accu_step = np.loadtxt(f'{results_dir}/stats/{num_classes}_class/valid_accu_v1_class_{num_classes}_subject_{sub_str}_avg.csv')
            train_loss_step = np.loadtxt(f'{results_dir}/stats/{num_classes}_class/train_loss_v1_class_{num_classes}_subject_{sub_str}_avg.csv')
            valid_loss_step = np.loadtxt(f'{results_dir}/stats/{num_classes}_class/valid_loss_v1_class_{num_classes}_subject_{sub_str}_avg.csv')

            train_accu += train_accu_step
            valid_accu += valid_accu_step
            train_loss += train_loss_step
            valid_loss += valid_loss_step

        train_accu = train_accu/len(subjects)
        valid_accu = valid_accu/len(subjects)
        train_loss = train_loss/len(subjects)
        valid_loss = valid_loss/len(subjects)

        print("SS Validation Accuracy {:.4f}".format(valid_accu[-1]))

        np.savetxt(f'{results_dir}/stats/avg/train_accu_v1_class_{num_classes}_ss_retrained_avg.csv', train_accu)
        np.savetxt(f'{results_dir}/stats/avg/valid_accu_v1_class_{num_classes}_ss_retrained_avg.csv', valid_accu)
        np.savetxt(f'{results_dir}/stats/avg/train_loss_v1_class_{num_classes}_ss_retrained_avg.csv', train_loss)
        np.savetxt(f'{results_dir}/stats/avg/valid_loss_v1_class_{num_classes}_ss_retrained_avg.csv', valid_loss)

        # Plot Accuracy
        plt.plot(train_accu, label='Training')
        plt.plot(valid_accu, label='Validation')
        plt.title(f'SS Retraining Accuracy C:{num_classes} LR: {lr} CH: {NO_selected_channels}')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.savefig(f'{results_dir}/plots/avg/accu_avg_{num_classes}_c.pdf')
        plt.clf()

        # Plot Loss
        plt.plot(train_loss, label='Training')
        plt.plot(valid_loss, label='Validation')
        plt.title(f'SS Retraining Loss C:{num_classes} LR: {lr} CH: {NO_selected_channels}')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.savefig(f'{results_dir}/plots/avg/loss_avg_{num_classes}_c.pdf')
        plt.clf()

    # Get average of each model and plot
    for num_classes in num_classes_list:
        for sub_split_ctr in range(0,3):
            train_accu = np.zeros(n_epochs+1)
            valid_accu = np.zeros(n_epochs+1)
            train_loss = np.zeros(n_epochs+1)
            valid_loss = np.zeros(n_epochs+1)
            for subject in subjects:
                sub_str = '{0:03d}'.format(subject)

                train_accu_step = np.loadtxt(f'{results_dir}/stats/train_accu_v1_class_{num_classes}_subject_{sub_str}_fold_{sub_split_ctr}.csv')
                valid_accu_step = np.loadtxt(f'{results_dir}/stats/valid_accu_v1_class_{num_classes}_subject_{sub_str}_fold_{sub_split_ctr}.csv')
                train_loss_step = np.loadtxt(f'{results_dir}/stats/train_loss_v1_class_{num_classes}_subject_{sub_str}_fold_{sub_split_ctr}.csv')
                valid_loss_step = np.loadtxt(f'{results_dir}/stats/valid_loss_v1_class_{num_classes}_subject_{sub_str}_fold_{sub_split_ctr}.csv')

                train_accu += train_accu_step
                valid_accu += valid_accu_step
                train_loss += train_loss_step
                valid_loss += valid_loss_step

            train_accu = train_accu/len(subjects)
            valid_accu = valid_accu/len(subjects)
            train_loss = train_loss/len(subjects)
            valid_loss = valid_loss/len(subjects)

            np.savetxt(f'{results_dir}/stats/avg/train_accu_v1_class_{num_classes}_ss_retrained_model_{sub_split_ctr}_avg.csv', train_accu)
            np.savetxt(f'{results_dir}/stats/avg/valid_accu_v1_class_{num_classes}_ss_retrained_model_{sub_split_ctr}_avg.csv', valid_accu)
            np.savetxt(f'{results_dir}/stats/avg/train_loss_v1_class_{num_classes}_ss_retrained_model_{sub_split_ctr}_avg.csv', train_loss)
            np.savetxt(f'{results_dir}/stats/avg/valid_loss_v1_class_{num_classes}_ss_retrained_model_{sub_split_ctr}_avg.csv', valid_loss)

            # Plot Accuracy
            plt.plot(train_accu, label='Training')
            plt.plot(valid_accu, label='Validation')
            plt.title(f'SS Retraining Accuracy C:{num_classes} M:{sub_split_ctr} LR: {lr} CH: {NO_selected_channels}')
            plt.xlabel('Epochs')
            plt.ylabel('Accuracy')
            plt.legend()
            plt.savefig(f'{results_dir}/plots/avg/accu_avg_{num_classes}_c_model_{sub_split_ctr}.pdf')
            plt.clf()

            # Plot Loss
            plt.plot(train_loss, label='Training')
            plt.plot(valid_loss, label='Validation')
            plt.title(f'SS Retraining Loss C:{num_classes} M:{sub_split_ctr} LR: {lr} CH: {NO_selected_channels}')
            plt.xlabel('Epochs')
            plt.ylabel('Loss')
            plt.legend()
            plt.savefig(f'{results_dir}/plots/avg/loss_avg_{num_classes}_c_model_{sub_split_ctr}.pdf')
            plt.clf()
