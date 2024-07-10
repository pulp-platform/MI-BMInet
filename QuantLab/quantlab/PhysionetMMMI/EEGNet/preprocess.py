# Copyright (c) 2019 UniMoRe, Matteo Spallanzani, Tibor Schneider

from os import path

import numpy as np
import scipy.io as sio
from scipy.signal import butter, sosfilt
import numpy as np
import torch as t
from torchvision.transforms import ToTensor, Normalize, Compose

from quantlab.treat.data.split import transform_random_split

from .get_data import get_data


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

    def __init__(self, datapath, num_classes=4, subj_indexes=range(1,110), transform=None, train=None, fold=None):
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

        self.samples = t.Tensor(X).to(dtype=t.float)
        self.labels = t.Tensor(y).to(dtype=t.long)


    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx, :, :]
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
    transform      = Compose([ReshapeTensor()])
    if subj_indexes:
        print("Cross-validation with subjects indexes given")
        print("Train: {}\nTest: {}".format(subj_indexes[0], subj_indexes[1]))
        train_set = PhysionetMMMI(datapath=dir_data, num_classes=data_config['num_classes'], subj_indexes=subj_indexes[0], transform=transform, train=True, fold=fold)
        valid_set = PhysionetMMMI(datapath=dir_data, num_classes=data_config['num_classes'], subj_indexes=subj_indexes[1], transform=transform, train=False, fold=fold)
        test_set  = valid_set

    else:

        data_set = PhysionetMMMI(datapath=dir_data, num_classes=data_config['num_classes'])

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
