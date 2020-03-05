#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Loads '.edf' data format in the data shape of EEGNet
"""

import numpy as np

# pyEDFlib is a python library to read/write EDF+/BDF+ files based on EDFlib.
import pyedflib as edf
import os

# to calculate mean and the standard deviation
import statistics as stats
import random


__author__ = "Batuhan Tomekce and Burak Alp Kaya"
__email__ = "tbatuhan@ethz.ch, bukaya@ethz.ch"

def get_data(PATH, long = False, normalization = 0,subjects_list=range(1,110)):
    '''
    Keyword arguments:
    PATH: path to directory under which the test data lies.
    normalization -- [0 (default)]: no normalization, 1: normalized per channel, 2: normalized per all trials
    long -- If True: Trials of length 6s returned. If False: Trials of length 3s returned
    subjects_list -- [range(1,110) (default)]: array of subject numbers in range [1, .. , 109] 
            ,used if not all subjects are to be taken
    
    Return: data_return     numpy matrix     size = NO_events x 64 x 656
            class_return    numpy matrix     size = NO_events
            X_Train: Training Trials
            y_Train: Training labels
            mean: (if normalization != 0) mean of normalized Training set
            std_dev: (if normalization != 0) std_dev of normalized Training set
    '''
    # Define subjects whose data is not taken, for details see data tester
    excluded_subjects = [88,92,100,104,106]
    # Define subjects whose data is taken, namely from 1 to 109 excluding excluded_subjects
    subjects = [x for x in subjects_list if (x not in excluded_subjects)]
    # Define baseline and MI runs for Training Data
    baseline_run = [1]
    mi_runs = [4, 6, 8, 10]
    # Read Rest(0) data from baseline
    X_0, y_0 = read_data(subjects = subjects, runs = baseline_run, PATH = PATH, long=long)
    # Read L/R/F data from MI runs
    X_LRF, y_LRF = read_data(subjects = subjects, runs = mi_runs, PATH = PATH, long=long)
    
    # Merge Rest(0) data from baseline run with the L/R/F data from the MI runs
    X_Train = np.concatenate((X_0, X_LRF))
    y_Train = np.append(y_0, y_LRF)
    
    # Shuffle Data reproducably
    np.random.seed(42)
    np.random.shuffle(X_Train)
    np.random.seed(42)
    np.random.shuffle(y_Train)
    
    if(normalization == 0):
        return X_Train, y_Train
    # do normalization
    if(normalization == 1):
        #TODO: declare std_dev, mean arrays to return
        for i in range(X_Train.shape[0]):
            for ii in range(X_Train.shape[1]):
                std_dev = stats.stdev(X_Train[i,ii])
                mean = stats.mean(X_Train[i,ii])
                X_Train[i,ii] = (X_Train[i,ii] - mean) / std_dev
        
        return X_Train, y_Train, mean, std_dev
    
    if(normalization == 2):
        #TODO: implement second type of normalization
        mean = std_dev = 0
        
        return X_Train, y_Train, mean, std_dev
    
    
def read_data(subjects , runs, PATH, long=False):
    '''
    Keyword arguments:
    subject -- array of subject numbers in range [1, .. , 109] (integer)
    runs -- array of the numbers of the runs in range [1, .. , 14] (integer)
    normalization -- 0(default): no normalization, 1: normalized per channel, 2: normalized per all trials
    
    Return: data_return     numpy matrix     size = NO_events x 64 x 656
            class_return    numpy matrix     size = NO_events
            X: Trials
            y: labels
    '''
    """
    DATA EXPLANATION:
        
        LABELS:
        both first_set and second_set
            T0: rest
        first_set (real motion in runs 3, 7, and 11; imagined motion in runs 4, 8, and 12)
            T1: the left fist 
            T2: the right fist
        second_set (real motion in runs 5, 9, and 13; imagined motion in runs 6, 10, and 14)
            T1: both fists
            T2: both feet
        
        This Program:
            Here, we get data from the first_set (rest, left fist, right fist), 
            and also data from the second_set (rest, both feet).
            We ignore data for T1 from the second_set and thus return data for 
            four classes/categories of events: Rest, Left Fist, Right Fist, Both Feet.
    """
    base_file_name = 'S{:03d}R{:02d}.edf'
    base_subject_directory = 'S{:03d}'
    
    # Define runs where the two different sets of tasks were performed
    baseline = np.array([1])
    first_set = np.array([4,8,12])
    second_set = np.array([6,10,14])
    
    # Number of EEG channels
    NO_channels = 64
    # Number of Trials extracted per Run
    NO_trials = 14
    
    '''
    # Define Sample size per Trial 
    if not long:
        Window_Length = int(160 * 3) # 3s Trials: 480 samples
    else:
        Window_Length = int(160 * 6) # 6s Trials: 960 samples 
    '''
    

    
    if not long:
        Window_Length = int(160 * 3) # 3s Trials: 480 samples
        # initialize empty arrays to concatanate with itself later
        X = np.empty((0,NO_channels,Window_Length))
        y = np.empty(0)
        for subject in subjects:
            #For each subject, a certain number of trials from each class should be extracted
            counter_0 = 0
            counter_L = 0
            counter_R = 0
            counter_F = 0
            for run in runs:
                # Create file name variable to access edf file
                filename = base_file_name.format(subject,run)
                directory = base_subject_directory.format(subject)
                file_name = os.path.join(PATH,directory,filename)
                # Read file
                f = edf.EdfReader(file_name)
                # Signal Parameters - measurement frequency
                freq = f.getSampleFrequency(0)
                # Number of eeg channels = number of signals in file
                n = f.signals_in_file
                # These are the eeg channel/electrode names
                #signal_labels = f.getSignalLabels()
                # Initiate eg.: 64*20000 matrix to hold all datapoints
                sigbufs = np.zeros((n, f.getNSamples()[0]))
                # Here: n=64 arange(n) creates array ([0, 1, ..., n-2, n-1])
                for i in np.arange(n):
                    # Save the read data (vectorwise) in a matrix
                    # Fill the matrix with all datapoints from each channel
                    sigbufs[i, :] = f.readSignal(i)
                
                # Get Label information
                annotations = f.readAnnotations()
                
                # close the file
                f.close()
                
                # Get the specific label information
                labels = annotations[2]
                points = freq*annotations[0]
                
                labels_int = np.empty(0)
                data_step = np.empty((0,NO_channels, Window_Length))             
                
                if run in second_set:
                    for ii in range(0,np.size(labels)):
                        if(labels[ii] == 'T0' and counter_0 < NO_trials):
                            continue
                            counter_0 += 1
                            labels_int = np.append(labels_int,[0])
                            
                        elif(labels[ii] == 'T2' and counter_F < NO_trials):
                            counter_F += 1
                            labels_int = np.append(labels_int,[3])
                            # change data shape and seperate events
                            data_step = np.vstack((data_step, np.array(sigbufs[:,int(points[ii]):int(points[ii])+Window_Length])[None]))        
                    
                elif run in first_set:
                    for ii in range(0,np.size(labels)):
                        if(labels[ii] == 'T0' and counter_0 < NO_trials):
                            continue
                            counter_0 += 1
                            labels_int = np.append(labels_int, [0])
                            
                        elif(labels[ii] == 'T1' and counter_L < NO_trials):
                            counter_L += 1
                            labels_int = np.append(labels_int, [1])
                            data_step = np.vstack((data_step, np.array(sigbufs[:,int(points[ii]):int(points[ii])+Window_Length])[None]))
                            
                        elif(labels[ii] == 'T2' and counter_R < NO_trials):
                            counter_R += 1
                            labels_int = np.append(labels_int, [2])
                            data_step = np.vstack((data_step, np.array(sigbufs[:,int(points[ii]):int(points[ii])+Window_Length])[None]))
                            
                elif run in baseline:
                    for ii in range(0,NO_trials):
                        if(counter_0 < NO_trials):  
                            counter_0 += 1
                            labels_int = np.append(labels_int, [0])
                            data_step = np.vstack((data_step, np.array(sigbufs[:,(ii*Window_Length):((ii+1)*Window_Length)])[None]))

            # concatenate arrays in order to get the whole data in one input array    
            X = np.concatenate((X,data_step))
            y = np.concatenate((y,labels_int))

    else:
        Window_Length = int(160) # 1s Trials: 480 samples
        # initialize empty arrays to concatanate with itself later
        X = np.empty((0,NO_channels,Window_Length*6))
        y = np.empty(0)
        for subject in subjects:
            #For each subject, a certain number of trials from each class should be extracted
            counter_0 = 0
            counter_L = 0
            counter_R = 0
            counter_F = 0
            for run in runs:
                # Create file name variable to access edf file
                filename = base_file_name.format(subject,run)
                directory = base_subject_directory.format(subject)
                file_name = os.path.join(PATH,directory,filename)
                # Read file
                f = edf.EdfReader(file_name)
                # Signal Parameters - measurement frequency
                freq = f.getSampleFrequency(0)
                # Number of eeg channels = number of signals in file
                n = f.signals_in_file
                # These are the eeg channel/electrode names
                #signal_labels = f.getSignalLabels()
                # Initiate eg.: 64*20000 matrix to hold all datapoints
                sigbufs = np.zeros((n, f.getNSamples()[0]))
                # Here: n=64 arange(n) creates array ([0, 1, ..., n-2, n-1])
                for i in np.arange(n):
                    # Save the read data (vectorwise) in a matrix
                    # Fill the matrix with all datapoints from each channel
                    sigbufs[i, :] = f.readSignal(i)
                
                # Get Label information
                annotations = f.readAnnotations()
                
                length = f.file_duration
                
                # close the file
                f.close()
                
                # Get the specific label information
                labels = annotations[2]
                points = freq*annotations[0]
                
                labels_int = np.empty(0)
                data_step = np.empty((0,NO_channels, Window_Length*6))             
                
                if run in second_set:
                    for ii in range(0,np.size(labels)):
                        if(int(points[ii]) + 5*Window_Length < length*160):
                                
                            if(labels[ii] == 'T0' and counter_0 < NO_trials):
                                continue
                                counter_0 += 1
                                labels_int = np.append(labels_int,[0])
                                
                            elif(labels[ii] == 'T2' and counter_F < NO_trials):
                                counter_F += 1
                                labels_int = np.append(labels_int,[3])
                                # change data shape and seperate events
                                data_step = np.vstack((data_step, np.array(sigbufs[:,int(points[ii])-Window_Length:int(points[ii])+ 5*Window_Length])[None]))        
                        else:
                            print(f'out of bounds: {points[ii]} + {5*Window_Length} > {length*160}')
                elif run in first_set:
                    for ii in range(0,np.size(labels)):
                        if(int(points[ii]) + 5*Window_Length < length*160):
                            if(labels[ii] == 'T0' and counter_0 < NO_trials):
                                continue
                                counter_0 += 1
                                labels_int = np.append(labels_int, [0])
                                
                            elif(labels[ii] == 'T1' and counter_L < NO_trials):
                                counter_L += 1
                                labels_int = np.append(labels_int, [1])
                                data_step = np.vstack((data_step, np.array(sigbufs[:,int(points[ii])-Window_Length:int(points[ii])+Window_Length*5])[None]))
                                
                            elif(labels[ii] == 'T2' and counter_R < NO_trials):
                                counter_R += 1
                                labels_int = np.append(labels_int, [2])
                                data_step = np.vstack((data_step, np.array(sigbufs[:,int(points[ii])-Window_Length:int(points[ii])+Window_Length*5])[None]))
                        else:
                            print(f'out of bounds: {points[ii]} + {5*Window_Length} > {length*160}')          
                elif run in baseline:
                    for ii in range(0,NO_trials):
                        index_begin = random.randint(0,54)
                        if(counter_0 < NO_trials):  
                            counter_0 += 1
                            labels_int = np.append(labels_int, [0])
                            data_step = np.vstack((data_step, np.array(sigbufs[:,(index_begin*Window_Length):((index_begin+6)*Window_Length)])[None]))

            # concatenate arrays in order to get the whole data in one input array    
            X = np.concatenate((X,data_step))
            y = np.concatenate((y,labels_int))
        
    return X, y
