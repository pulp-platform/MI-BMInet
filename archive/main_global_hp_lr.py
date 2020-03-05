#!/usr/bin/env python3

__author__ = "Batuhan Tomekce and Burak Alp Kaya"
__email__ = "tbatuhan@ethz.ch, bukaya@ethz.ch"

import numpy as np
import os
from datetime import datetime
# pyEDFlib is a python library to read/write EDF+/BDF+ files based on EDFlib.
import pyedflib
# our functions to test and load data
import get_data as get
#import data_tester as test
# tensorflow part
from tensorflow.keras import utils as np_utils
from keras.callbacks import ModelCheckpoint
from keras.callbacks import EarlyStopping
#from tensorflow.keras import optimizers
from keras.optimizers import Adam
from keras.models import load_model
from keras import backend as K
# EEGNet models
import models as models
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
        
# tools for plotting confusion matrices
#from matplotlib import pyplot as plt
#from conf_matrix import plot_confusion_matrix

# FIND OPTIMAL Learning Rate
# This search is done for 4 classes
# Using the first 84 subjects as training data
# Leaving the rest 21 untouched as test data
# Using 4-fold Cross-Validation
# Search LR from 10^-1 to 10^-5

# Set data parameters
PATH = "../files/"

current_time = datetime.now()
results_dir=f'global_trainer_hp_lr'
#os.makedirs(results_dir, exist_ok=True)
os.makedirs(f'{results_dir}/stats', exist_ok=True)
os.makedirs(f'{results_dir}/model', exist_ok=True)
os.makedirs(f'{results_dir}/plots', exist_ok=True)

# specify number of classses for input data
num_classes = 4       

# Load data
X_Train, y_Train = get.get_data(PATH, n_classes=num_classes, subjects_list=range(1,85))

# Expand dimensions to match expected EEGNet input
X_Train_real = (np.expand_dims(X_Train, axis=1))

# use sample size
SAMPLE_SIZE = np.shape(X_Train_real)[3]

# convert labels to one-hot encodings.
y_Train_cat = np_utils.to_categorical(y_Train)

# using 4 folds
num_splits = 4
kf = KFold(n_splits = num_splits)

# DELETE IF NO ERROR
# create a 2D array for fold creation. # 640 is here the sample size.
#x_train_aux = np.reshape(X_Train_real, (np.shape(X_Train_real)[0], 64*SAMPLE_SIZE))

lr_exps = [(-x) for x in range(1,6)]

n_epochs = 500

for lr_exp in lr_exps:
    #np.random.seed(100)
    #np.random.shuffle(train)
    split_ctr = 0
    for train, test in kf.split(X_Train_real, y_Train):
        
        print(f'Learning Rate = 10^{lr_exp}, Split = {split_ctr}')
        model = models.EEGNet(nb_classes = num_classes, Chans=64, Samples=SAMPLE_SIZE, regRate=0.25,
                        dropoutRate=0.1, kernLength=128, numFilters=8, dropoutType='Dropout')
        # Set Learning Rate
        adam_alpha = Adam(lr=(10**lr_exp))
        model.compile(loss='categorical_crossentropy', optimizer=adam_alpha, metrics = ['accuracy'])
        # creating a history object
        history = model.fit(X_Train_real[train], y_Train_cat[train], 
                validation_data=(X_Train_real[test], y_Train_cat[test]),
                batch_size = 16, epochs = n_epochs , verbose = 2)

        # Save metrics
        train_accu_str = f'{results_dir}/stats/train_accu_split_{split_ctr}_lr_1e-{lr_exp}.csv'
        valid_accu_str = f'{results_dir}/stats/valid_accu_split_{split_ctr}_lr_1e-{lr_exp}.csv'
        train_loss_str = f'{results_dir}/stats/train_loss_split_{split_ctr}_lr_1e-{lr_exp}.csv'
        valid_loss_str = f'{results_dir}/stats/valid_loss_split_{split_ctr}_lr_1e-{lr_exp}.csv'
         
        np.savetxt(train_accu_str, history.history['acc'])
        np.savetxt(valid_accu_str, history.history['val_acc'])
        np.savetxt(train_loss_str, history.history['loss'])
        np.savetxt(valid_loss_str, history.history['val_loss'])

        #Save model
        print('Saving model...')
        model.save(f'{results_dir}/model/global_class_{num_classes}_split_{split_ctr}_lr_1e-{lr_exp}.h5')

        #Clear Models
        K.clear_session()
        split_ctr = split_ctr + 1

# Once all CV folds are done, calculate averages, plot, and save
for lr_exp in lr_exps:
    train_accu = np.zeros(n_epochs)
    valid_accu = np.zeros(n_epochs)
    train_loss = np.zeros(n_epochs)
    valid_loss = np.zeros(n_epochs)
    split_ctr = 0
    for train, test in kf.split(X_Train_real, y_Train):
        train_accu_step = np.loadtxt(f'{results_dir}/stats/train_accu_split_{split_ctr}_lr_1e-{lr_exp}.csv')
        valid_accu_step = np.loadtxt(f'{results_dir}/stats/valid_accu_split_{split_ctr}_lr_1e-{lr_exp}.csv')
        train_loss_step = np.loadtxt(f'{results_dir}/stats/train_loss_split_{split_ctr}_lr_1e-{lr_exp}.csv')
        valid_loss_step = np.loadtxt(f'{results_dir}/stats/valid_loss_split_{split_ctr}_lr_1e-{lr_exp}.csv')
        
        train_accu += train_accu_step
        valid_accu += valid_accu_step
        train_loss += train_loss_step
        valid_loss += valid_loss_step

        split_ctr = split_ctr + 1
    
    train_accu = train_accu/num_splits
    valid_accu = valid_accu/num_splits
    train_loss = train_loss/num_splits
    valid_loss = valid_loss/num_splits
   
    np.savetxt(f'{results_dir}/stats/train_accu_lr_1e-{lr_exp}_avg.csv', train_accu)
    np.savetxt(f'{results_dir}/stats/valid_accu_lr_1e-{lr_exp}_avg.csv', valid_accu)
    np.savetxt(f'{results_dir}/stats/train_loss_lr_1e-{lr_exp}_avg.csv', train_loss)
    np.savetxt(f'{results_dir}/stats/valid_loss_lr_1e-{lr_exp}_avg.csv', valid_loss)
    
    # Plot Accuracy 
    plt.plot(train_accu, label='Training')
    plt.plot(valid_accu, label='Validation')
    plt.title(f'Accuracy: LR=10^{lr_exp}, DR=0.1')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig(f'{results_dir}/plots/accu_lr_1e-{lr_exp}_avg.pdf')
    plt.clf()
    # Plot Loss
    plt.plot(train_loss, label='Training')
    plt.plot(valid_loss, label='Validation')
    plt.title(f'Loss: LR=10^{lr_exp}, DR=0.1')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(f'{results_dir}/plots/loss_lr_1e-{lr_exp}_avg.pdf')
    plt.clf()


