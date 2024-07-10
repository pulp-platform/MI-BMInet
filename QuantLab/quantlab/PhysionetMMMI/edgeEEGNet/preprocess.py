# Copyright (c) 2019 UniMoRe, Matteo Spallanzani, Tibor Schneider, Xiaying Wang

from os import path

import numpy as np
import scipy.io as sio
from scipy.signal import butter, sosfilt
import numpy as np
import torch as t
from torchvision.transforms import ToTensor, Normalize, Compose

from quantlab.treat.data.split import transform_random_split

#from .get_data import get_data


"""
In order to use this preprocessing module, use the following 'data' configuration

"data": {
  "subject": 1
  "fs": 250,
  "f1_fraction": 1.5,
  "f2_fraction": 6.0,
  "filter": {
    # SEE BELOW
  }
  "valid_fraction": 0.1,
  "bs_train": 32,
  "bs_valid": 32,
  "use_test_as_valid": false
}

For using no filter, you can leave out the "data"."filter" object, or set the "data".filter"."type"
to "none".

For using highpass, use the following filter
"filter": {
  "type": "highpass",
  "fc": 4.0,
  "order": 4
}

For using bandpass, use the following filter
"filter": {
  "type": "bandpass",
  "fc_low": 4.0,
  "fc_high": 40.0,
  "order": 5
}
"""


class PhysionetMMMI(t.utils.data.Dataset):

    def __init__(self, datapath, num_classes=4, subj_indexes=range(1,110), transform=None, train=None, fold=None, cs=[]):
        print(datapath)
        self.datapath = datapath
        self.transform = transform
        self.num_classes = num_classes
        if train is None:
            if not path.isfile(path.join(datapath, f'{num_classes}class.npz')):
                print(f'{num_classes}.npz file not existing. Load .edf and save data in npz files for faster loading of data next time.')
                X, y = get_data(datapath, n_classes=num_classes, subjects_list=subj_indexes)
                np.savez(path.join(datapath,f'{num_classes}class'), X = X, y = y)
            npzfile = np.load(path.join(datapath, f'{num_classes}class.npz'))
            X, y = npzfile['X'], npzfile['y']
        else:
            if fold is None:
                raise ValueError("fold is None")
            if train:
                if not path.isfile(path.join(datapath, f'{num_classes}class_train_f{fold}.npz')):
                    print(f'{num_classes}class_train_f{fold}.npz file not existing. Load .edf and save data in npz files for faster loading of data next time.')
                    X, y = get_data(datapath, n_classes=num_classes, subjects_list=subj_indexes)
                    np.savez(path.join(datapath,f'{num_classes}class_train_f{fold}'), X = X, y = y)
                npzfile = np.load(path.join(datapath, f'{num_classes}class_train_f{fold}.npz'))
                X, y = npzfile['X'], npzfile['y']
            else:
                if not path.isfile(path.join(datapath, f'{num_classes}class_test_f{fold}.npz')):
                    print(f'{num_classes}class_test_f{fold}.npz file not existing. Load .edf and save data in npz files for faster loading of data next time.')
                    X, y = get_data(datapath, n_classes=num_classes, subjects_list=subj_indexes)
                    np.savez(path.join(datapath,f'{num_classes}class_test_f{fold}'), X = X, y = y)
                npzfile = np.load(path.join(datapath, f'{num_classes}class_test_f{fold}.npz'))
                X, y = npzfile['X'], npzfile['y']
        
        if cs:
            X = X[:,cs,:]
        print(X.shape) #(3528, 64, 480)
        self.samples = t.Tensor(X).to(dtype=t.float)
        self.labels = t.Tensor(y).to(dtype=t.long)


    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx, :, :]
        #print(sample.shape)
        label = self.labels[idx]
        if self.transform:
            sample = self.transform(sample)
        return sample, label


class HighpassFilter(object):

    def __init__(self, fs, fc, order):
        nyq = 0.5 * fs
        norm_fc = fc / nyq
        self.sos = butter(order, norm_fc, btype='highpass', output='sos')

    def __call__(self, sample):
        for ch in sample.shape[0]:
            sample[ch, :] = sosfilt(self.sos, sample[ch, :])
        return sample


class BandpassFilter(object):

    def __init__(self, fs, fc_low, fc_high, order):
        nyq = 0.5 * fs
        norm_fc_low = fc_low / nyq
        norm_fc_high = fc_high / nyq
        self.sos = butter(order, [norm_fc_low, norm_fc_high], btype='bandpass', output='sos')

    def __call__(self, sample):
        for ch in sample.shape[0]:
            sample[ch, :] = sosfilt(self.sos, sample[ch, :])
        return sample


class Identity(object):

    def __call__(self, sample):
        return sample


class TimeWindowPostCue(object):

    def __init__(self, fs, t1_factor, t2_factor):
        self.t1 = int(t1_factor * fs)
        self.t2 = int(t2_factor * fs)

    def __call__(self, sample):
        return sample[:, :, self.t1:self.t2]


class ReshapeTensor(object):
    def __call__(self, sample):
        return sample.view(1, sample.shape[0], sample.shape[1])


def load_data_sets(dir_data, data_config, subj_indexes=None, fold=None):
    try:
        cs = data_config['cs']
    except:
        cs = []

    transform      = Compose([ReshapeTensor()])
    if subj_indexes:
        print("Cross-validation with subjects indexes given")
        print("Train: {}\nTest: {}".format(subj_indexes[0], subj_indexes[1]))
        train_set = PhysionetMMMI(datapath=dir_data, num_classes=data_config['num_classes'], subj_indexes=subj_indexes[0], transform=transform, train=True, fold=fold, cs=cs)
        valid_set = PhysionetMMMI(datapath=dir_data, num_classes=data_config['num_classes'], subj_indexes=subj_indexes[1], transform=transform, train=False, fold=fold, cs=cs)
        test_set  = valid_set

    else:

        data_set = PhysionetMMMI(datapath=dir_data, num_classes=data_config['num_classes'], cs=cs)

        if data_config['test_fraction'] != 0:
            # split the dataset into train set and test set, with train set further
            # split to valid set according to valid_fraction
            len_trainvalid = int(len(data_set) * (1.0 - data_config['test_fraction']))
            trainvalid_set, test_set = transform_random_split(data_set, [len_trainvalid, len(data_set) - len_trainvalid], [transform, transform])
            len_train = int(len(trainvalid_set) * (1.0 - data_config['valid_fraction']))
            train_set, valid_set = transform_random_split(trainvalid_set, [len_train, len(trainvalid_set) - len_train], [transform, transform])
        else:
            # split the dataset into train and valid, use valid as test
            len_train            = int(len(data_set) * (1.0 - data_config['valid_fraction']))
            train_set, valid_set = transform_random_split(data_set, [len_train, len(data_set) - len_train], [transform, transform])
            test_set  = valid_set

    return train_set, valid_set, test_set


import pyedflib as edf
import statistics as stats
import random

def get_data(path, long = False, normalization = 0, subjects_list=range(1,110), n_classes=4):
    '''
    Load data samples and return it as one array 

    Parameters:
    -----------
    path:   string
        path to .edf data of 
    normalization   int {0,1}

        normalization per trial 
        0: no normalization; 1: normalized per channel
    long:    bool 
        length of read time window
        True: Trials of length 6s returned; False: Trials of length 3s returned
    subjects_list   list [1, .. , 109] 
        list of subject numbers to be loaded
    n_classes:      int 
        number of classes
        2: L/R, 3: L/R/0, 4
    
    Return: X:  numpy array (n_sub*n_trials, 64, n_samples) 
                EEG data 
            y:  numpy array (n_sub*n_trials, n_samples)
                labels 
    '''
    # Define subjects whose data is not taken, for details see data tester added 106 again to analyze it, deleted from the exluded list
    excluded_subjects = [88,92,100,104]
    # Define subjects whose data is taken, namely from 1 to 109 excluding excluded_subjects
    subjects = [x for x in subjects_list if (x not in excluded_subjects)]
   

    mi_runs = [1, 4, 6, 8, 10, 12, 14]
    # Extract only requested number of classes
    if(n_classes == 3):
        print('Returning 3 Class data')
        mi_runs.remove(6) # feet
        mi_runs.remove(10) # feet
        mi_runs.remove(14) # feet
    elif(n_classes == 2):
        print('Returning 2 Class data')
        mi_runs.remove(6) # feet
        mi_runs.remove(10) # feet
        mi_runs.remove(14) # feet
        mi_runs.remove(1) #rest 
    print(f'Data from runs: {mi_runs}')

    X, y = read_data(subjects = subjects,runs = mi_runs, path=path, long=long)
   
    # do normalization if wanted
    if(normalization == 1):
        #TODO: declare std_dev, mean arrays to return
        for i in range(X.shape[0]):
            for ii in range(X.shape[1]):
                std_dev = stats.stdev(X[i,ii])
                mean = stats.mean(X[i,ii])
                X[i,ii] = (X[i,ii] - mean) / std_dev
        
    return X, y
    

    
def read_data(subjects , runs, path, long=False):
    '''
    Load data samples and return it as one array 

    Parameters:
    -----------
    subjects   list [1, .. , 109] 
        list of subject numbers to be loaded
    path:   string
        path to .edf data of 
    runs    list 
        runs to read from 
    long:    bool 
        length of read time window
        True: Trials of length 6s returned; False: Trials of length 3s returned
    
    
    Return: X:  numpy array (n_sub*n_trials, 64, n_samples) 
                EEG data 
            y:  numpy array (n_sub*n_trials, n_samples)
                labels 
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
    NO_trials = 7
    
    # Define Sample size per Trial 
    if not long:
        n_samples = int(160 * 3) # 3s Trials: 480 samples
    else:
        n_samples = int(160 * 6) # 6s Trials: 960 samples 
    
    # initialize empty arrays to concatanate with itself later
    X = np.empty((0,NO_channels,n_samples))
    y = np.empty(0)
    
    for subject in subjects:

        for run in runs:
            #For each run, a certain number of trials from corresponding classes should be extracted
            counter_0 = 0
            counter_L = 0
            counter_R = 0
            counter_F = 0
            
            # Create file name variable to access edf file
            filename = base_file_name.format(subject,run)
            directory = base_subject_directory.format(subject)
            file_name = os.path.join(path,directory,filename)
            # Read file
            f = edf.EdfReader(file_name)
            # Signal Parameters - measurement frequency
            fs = f.getSampleFrequency(0)
            # Number of eeg channels = number of signals in file
            n_ch = f.signals_in_file
            # Initiate eg.: 64*20000 matrix to hold all datapoints
            sigbufs = np.zeros((n_ch, f.getNSamples()[0]))
            
            for ch in np.arange(n_ch):
                # Fill the matrix with all datapoints from each channel
                sigbufs[ch, :] = f.readSignal(ch)
            
            # Get Label information
            annotations = f.readAnnotations()
            
            # close the file
            f.close()
            
            # Get the specific label information
            labels = annotations[2]
            points = fs*annotations[0]
            
            labels_int = np.empty(0)
            data_step = np.empty((0,NO_channels, n_samples))             
            
            if run in second_set:
                for ii in range(0,np.size(labels)):
                    if(labels[ii] == 'T0' and counter_0 < NO_trials):
                        continue
                        counter_0 += 1
                        labels_int = np.append(labels_int,[2])
                        
                    elif(labels[ii] == 'T2' and counter_F < NO_trials):
                        counter_F += 1
                        labels_int = np.append(labels_int,[3])
                        # change data shape and seperate events
                        data_step = np.vstack((data_step, np.array(sigbufs[:,int(points[ii]):int(points[ii])+n_samples])[None]))        
                
            elif run in first_set:
                for ii in range(0,np.size(labels)):
                    if(labels[ii] == 'T0' and counter_0 < NO_trials):
                        continue
                        counter_0 += 1
                        labels_int = np.append(labels_int, [2])
                        
                    elif(labels[ii] == 'T1' and counter_L < NO_trials):
                        counter_L += 1
                        labels_int = np.append(labels_int, [0])
                        data_step = np.vstack((data_step, np.array(sigbufs[:,int(points[ii]):int(points[ii])+n_samples])[None]))
                        
                    elif(labels[ii] == 'T2' and counter_R < NO_trials):
                        counter_R += 1
                        labels_int = np.append(labels_int, [1])
                        data_step = np.vstack((data_step, np.array(sigbufs[:,int(points[ii]):int(points[ii])+n_samples])[None]))
                        
            elif run in baseline:
                for ii in range(0,20):
                    if(counter_0 < 20):  
                        counter_0 += 1
                        labels_int = np.append(labels_int, [2])
                        data_step = np.vstack((data_step, np.array(sigbufs[:,(ii*n_samples):((ii+1)*n_samples)])[None]))
                # randomly choose resting trials
                np.random.seed(7)
                index = random.randint(0*fs,57*fs)
                labels_int = np.append(labels_int, [2])
                data_step = np.vstack((data_step, np.array(sigbufs[:,(index):(index+n_samples)])[None]))
               
            # concatenate arrays in order to get the whole data in one input array    
            X = np.concatenate((X,data_step))
            y = np.concatenate((y,labels_int))
        
    return X, y

