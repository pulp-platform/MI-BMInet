#!/usr/bin/env python3

# 2. 64 channels global model trained,
#    64 channels final epoch SS,
#    *then select 8 channels based on SS EEGNet weights,
#    *train last epochs 8 channels SS

__author__ = "Batuhan Tomekce and Burak Alp Kaya"
__email__ = "tbatuhan@ethz.ch, bukaya@ethz.ch"

import numpy as np
import os
from datetime import datetime
import pdb
# pyEDFlib is a python library to read/write EDF+/BDF+ files based on EDFlib.
import pyedflib
# our functions to test and load data
import get_data as get
#import data_tester as test
# tensorflow part
from tensorflow.keras import utils as np_utils
from keras.callbacks import ModelCheckpoint
from keras.callbacks import EarlyStopping
from keras.callbacks import LearningRateScheduler
#from tensorflow.keras import optimizers
from keras.optimizers import Adam
from keras.models import load_model
from keras import backend as K
# EEGNet models
import models as models
from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold
import matplotlib.pyplot as plt
# tools for plotting confusion matrices
#from matplotlib import pyplot as plt
#from conf_matrix import plot_confusion_matrix
# for csp
from channel_selection import CS_Model, channel_selection_csp, channel_selection_eegweights, channel_selection_eegweights_fromss

# Select GPU
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# Remove excluded subjects from subjects list
def exclude_subjects(all_subjects=range(1,110), excluded_subjects=[88,92,100,104]):
    subjects = [x for x in all_subjects if (x not in excluded_subjects)]
    return subjects

#################################################
#
# Learning Rate Sparse Scheduling
# used on Subject Specific Retraining
#
# 5 Global models have alredy been trained
# now, these models are used, and further
# (subject-specifically) retrained.
#
# Finally, results within, and across
# subjects are averaged and plotted.
#
#################################################

# Number of epochs to use with 10^[-3,-4,-5] Learning Rate
epochs = [2,3,5]
lrates = [-3,-4,-5]

# Set data path
PATH = "/usr/scratch/badile01/sem20f12/files"
# Make necessary directories for files
results_dir=f'SS-TL'
os.makedirs(f'{results_dir}/stats', exist_ok=True)
os.makedirs(f'{results_dir}/model', exist_ok=True)
os.makedirs(f'{results_dir}/plots', exist_ok=True)
os.makedirs(f'{results_dir}/stats/avg', exist_ok=True)
os.makedirs(f'{results_dir}/plots/avg', exist_ok=True)

# Specify number of classses for input data
num_classes_list = [4]
# Exclude subjects whose data we do not use
subjects = exclude_subjects()
# For channel selection
cs_model = CS_Model()
cs_csp = False
cs_eeg = True

for num_classes in num_classes_list:
    # using 5 folds
    num_splits = 5
    kf_global = KFold(n_splits = num_splits)
    n_epochs = 10

    split_ctr = 0
    for train_global, test_global in kf_global.split(subjects):
        for sub_idx in test_global:
            subject = subjects[sub_idx]
            X_sub, y_sub = get.get_data(PATH, n_classes=num_classes, subjects_list=[subject])
            print(X_sub.shape)

            # csp select channels for specific subject
            if cs_csp:
                selected_channels = channel_selection_csp(X_sub, y_sub, cs_model.NO_csp, cs_model.filter_bank, cs_model.time_windows, cs_model.NO_classes, cs_model.NO_channels, cs_model.NO_selected_channels, cs_model.channel_selection_method)
                X_sub = X_sub[:,selected_channels,:]

            X_sub = np.expand_dims(X_sub, axis=1)
            y_sub_cat = np_utils.to_categorical(y_sub)
            SAMPLE_SIZE = np.shape(X_sub)[3]
            kf_subject = StratifiedKFold(n_splits=4, shuffle=True, random_state=42)
            sub_split_ctr = 0

            # eegnet weight select channels for specific subject
            if cs_eeg:
                selected_channels = channel_selection_eegweights_fromss(subject, cs_model.NO_channels, cs_model.NO_selected_channels)
                X_sub = X_sub[:,:,selected_channels,:]

            print(X_sub.shape)

            for train_sub, test_sub in kf_subject.split(X_sub, y_sub):
                print(f'N_Classes:{num_classes}, Model: {split_ctr} \n Subject: {subject:03d}, Split: {sub_split_ctr}')

                if not cs_csp and not cs_eeg:
                    model = load_model(f'Global/model/global_class_{num_classes}_ds1_nch64_T3_split_{split_ctr}_v1.h5')
                else:
                    model_global = load_model(f'Global/model/global_class_{num_classes}_ds1_nch64_T3_split_{split_ctr}_v1.h5')
                    print(model_global.summary())
                    model = models.EEGNet(nb_classes = cs_model.NO_classes, Chans=cs_model.NO_selected_channels, Samples=480, regRate=0.25,
                                    dropoutRate=0.2, kernLength=128, poolLength=8, numFilters=8, dropoutType='Dropout')
                    print(model.summary())

                    adam_alpha = Adam(lr=(0.0001))
                    model.compile(loss='categorical_crossentropy', optimizer=adam_alpha, metrics = ['accuracy'])
                    # pdb.set_trace()
                    model.layers[1].set_weights(model_global.layers[1].get_weights())
                    model.layers[2].set_weights(model_global.layers[2].get_weights())
                    model.layers[3].set_weights([model_global.layers[3].get_weights()[0][selected_channels,:]])
                    model.layers[4].set_weights(model_global.layers[4].get_weights())

                print(X_sub[test_sub].shape)
                # pdb.set_trace()
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
                for j, lrate in enumerate(lrates):
                    # Set Learning Rate
                    adam_alpha = Adam(lr=(10**lrate))
                    model.compile(loss='categorical_crossentropy', optimizer=adam_alpha, metrics = ['accuracy'])
                    # creating a history object
                    history = model.fit(X_sub[train_sub], y_sub_cat[train_sub],
                            validation_data=(X_sub[test_sub], y_sub_cat[test_sub]),
                            batch_size = 16, epochs = epochs[j], verbose = 2)
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

                #Save model
                #print('Saving model...')
                # model.save(f'{results_dir}/model/subject{sub_str}_fold{sub_split_ctr}.h5')

                K.clear_session()
                sub_split_ctr = sub_split_ctr + 1
        split_ctr = split_ctr + 1

for num_classes in num_classes_list:
    os.makedirs(f'{results_dir}/stats/{num_classes}_class', exist_ok=True)
    os.makedirs(f'{results_dir}/plots/{num_classes}_class', exist_ok=True)
    for subject in subjects:
        train_accu = np.zeros(n_epochs+1)
        valid_accu = np.zeros(n_epochs+1)
        train_loss = np.zeros(n_epochs+1)
        valid_loss = np.zeros(n_epochs+1)
        sub_str = '{0:03d}'.format(subject)
        for sub_split_ctr in range(0,4):
            # Save metrics
            train_accu_step = np.loadtxt(f'{results_dir}/stats/train_accu_v1_class_{num_classes}_subject_{sub_str}_fold_{sub_split_ctr}.csv')
            valid_accu_step = np.loadtxt(f'{results_dir}/stats/valid_accu_v1_class_{num_classes}_subject_{sub_str}_fold_{sub_split_ctr}.csv')
            train_loss_step = np.loadtxt(f'{results_dir}/stats/train_loss_v1_class_{num_classes}_subject_{sub_str}_fold_{sub_split_ctr}.csv')
            valid_loss_step = np.loadtxt(f'{results_dir}/stats/valid_loss_v1_class_{num_classes}_subject_{sub_str}_fold_{sub_split_ctr}.csv')

            train_accu += train_accu_step
            valid_accu += valid_accu_step
            train_loss += train_loss_step
            valid_loss += valid_loss_step

        train_accu = train_accu/4
        valid_accu = valid_accu/4
        train_loss = train_loss/4
        valid_loss = valid_loss/4

        np.savetxt(f'{results_dir}/stats/{num_classes}_class/train_accu_v1_class_{num_classes}_subject_{sub_str}_avg.csv', train_accu)
        np.savetxt(f'{results_dir}/stats/{num_classes}_class/valid_accu_v1_class_{num_classes}_subject_{sub_str}_avg.csv', valid_accu)
        np.savetxt(f'{results_dir}/stats/{num_classes}_class/train_loss_v1_class_{num_classes}_subject_{sub_str}_avg.csv', train_loss)
        np.savetxt(f'{results_dir}/stats/{num_classes}_class/valid_loss_v1_class_{num_classes}_subject_{sub_str}_avg.csv', valid_loss)
        # Plot Accuracy
        plt.plot(train_accu, label='Training')
        plt.plot(valid_accu, label='Validation')
        plt.title(f'S:{sub_str} C:{num_classes} Acc.: LR: 2-3-5, DR=0.2')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.savefig(f'{results_dir}/plots/{num_classes}_class/accu_avg_{num_classes}_c_{sub_str}.pdf')
        plt.clf()
        # Plot Loss
        plt.plot(train_loss, label='Training')
        plt.plot(valid_loss, label='Validation')
        plt.title(f'S:{sub_str} C:{num_classes} Loss: LR: 2-3-5, DR=0.2')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.savefig(f'{results_dir}/plots/{num_classes}_class/loss_avg_{num_classes}_c_{sub_str}.pdf')
        plt.clf()

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
    plt.title(f'SS Retraining C:{num_classes} Acc.: LR: 2-3-5, DR=0.2')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig(f'{results_dir}/plots/avg/accu_avg_{num_classes}_c.pdf')
    plt.clf()
    # Plot Loss
    plt.plot(train_loss, label='Training')
    plt.plot(valid_loss, label='Validation')
    plt.title(f'SS Retraining C:{num_classes} Loss: LR: 2-3-5, DR=0.2')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(f'{results_dir}/plots/avg/loss_avg_{num_classes}_c.pdf')
    plt.clf()

for num_classes in num_classes_list:
    num_splits = 5
    kf_global = KFold(n_splits = num_splits)
    n_epochs = 10

    split_ctr = 0
    for train_global, test_global in kf_global.split(subjects):
        train_accu = np.zeros(n_epochs+1)
        valid_accu = np.zeros(n_epochs+1)
        train_loss = np.zeros(n_epochs+1)
        valid_loss = np.zeros(n_epochs+1)
        for sub_idx in test_global:
            subject = subjects[sub_idx]
            sub_str = '{0:03d}'.format(subject)
            train_accu_step = np.loadtxt(f'{results_dir}/stats/{num_classes}_class/train_accu_v1_class_{num_classes}_subject_{sub_str}_avg.csv')
            valid_accu_step = np.loadtxt(f'{results_dir}/stats/{num_classes}_class/valid_accu_v1_class_{num_classes}_subject_{sub_str}_avg.csv')
            train_loss_step = np.loadtxt(f'{results_dir}/stats/{num_classes}_class/train_loss_v1_class_{num_classes}_subject_{sub_str}_avg.csv')
            valid_loss_step = np.loadtxt(f'{results_dir}/stats/{num_classes}_class/valid_loss_v1_class_{num_classes}_subject_{sub_str}_avg.csv')

            train_accu += train_accu_step
            valid_accu += valid_accu_step
            train_loss += train_loss_step
            valid_loss += valid_loss_step

        train_accu = train_accu/len(test_global)
        valid_accu = valid_accu/len(test_global)
        train_loss = train_loss/len(test_global)
        valid_loss = valid_loss/len(test_global)

        np.savetxt(f'{results_dir}/stats/avg/train_accu_v1_class_{num_classes}_ss_retrained_model_{split_ctr}_avg.csv', train_accu)
        np.savetxt(f'{results_dir}/stats/avg/valid_accu_v1_class_{num_classes}_ss_retrained_model_{split_ctr}_avg.csv', valid_accu)
        np.savetxt(f'{results_dir}/stats/avg/train_loss_v1_class_{num_classes}_ss_retrained_model_{split_ctr}_avg.csv', train_loss)
        np.savetxt(f'{results_dir}/stats/avg/valid_loss_v1_class_{num_classes}_ss_retrained_model_{split_ctr}_avg.csv', valid_loss)

        # Plot Accuracy
        plt.plot(train_accu, label='Training')
        plt.plot(valid_accu, label='Validation')
        plt.title(f'SS Retraining C:{num_classes} M:{split_ctr} Acc.: LR: 2-3-5, DR=0.2')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.savefig(f'{results_dir}/plots/avg/accu_avg_{num_classes}_c_model_{split_ctr}.pdf')
        plt.clf()
        # Plot Loss
        plt.plot(train_loss, label='Training')
        plt.plot(valid_loss, label='Validation')
        plt.title(f'SS Retraining C:{num_classes} M:{split_ctr} Loss: LR: 2-3-5, DR=0.2')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.savefig(f'{results_dir}/plots/avg/loss_avg_{num_classes}_c_model_{split_ctr}.pdf')
        plt.clf()

        split_ctr = split_ctr + 1
