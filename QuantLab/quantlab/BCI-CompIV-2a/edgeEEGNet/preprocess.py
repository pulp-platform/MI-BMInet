# Copyright (c) 2019 UniMoRe, Matteo Spallanzani, Tibor Schneider

from os import path

import numpy as np
import scipy.io as sio
from scipy.signal import butter, sosfilt
import numpy as np
import torch as t
from torchvision.transforms import ToTensor, Normalize, Compose

from quantlab.treat.data.split import transform_random_split


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


class BCI_CompIV_2a(t.utils.data.Dataset):

    def __init__(self, root, train, subject, num_classes=4, transform=None, cs=[]):
        self.subject = subject
        self.root = root
        self.train = train
        self.num_classes=num_classes
        self.transform = transform
        self.samples, self.labels = self._load_data()
        if cs:
            self.samples = self.samples[:,cs,:]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx, :, :]
        label = self.labels[idx]
        if self.transform:
            sample = self.transform(sample)
        return sample, label

    def _load_data(self):
        NO_channels = 22
        NO_tests = 6 * 48
        Window_Length = 7 * 250

        class_return = np.zeros(NO_tests, dtype=np.float32)
        data_return = np.zeros((NO_tests, NO_channels, Window_Length), dtype=np.float32)

        n_valid_trials = 0
        if self.train:
            a = sio.loadmat(path.join(self.root, 'A0' + str(self.subject) + 'T.mat'))
        else:
            a = sio.loadmat(path.join(self.root, 'A0' + str(self.subject) + 'E.mat'))
        a_data = a['data']
        for ii in range(0, a_data.size):
            a_data1 = a_data[0, ii]
            a_data2 = [a_data1[0, 0]]
            a_data3 = a_data2[0]
            a_X = a_data3[0]
            a_trial = a_data3[1]
            a_y = a_data3[2]
            a_fs = a_data3[3]
            # a_classes = a_data3[4]
            a_artifacts = a_data3[5]
            # a_gender = a_data3[6]
            # a_age = a_data3[7]

            for trial in range(0, a_trial.size):
                if a_artifacts[trial] == 0:
                    range_a = int(a_trial[trial])
                    range_b = range_a + Window_Length
                    data_return[n_valid_trials, :, :] = np.transpose(a_X[range_a:range_b, :22])
                    class_return[n_valid_trials] = int(a_y[trial])
                    n_valid_trials += 1

        data_return = data_return[0:n_valid_trials, :, :]
        class_return = class_return[0:n_valid_trials]

        class_return = class_return - 1

        if self.num_classes == 2:
            del_feet = np.where(class_return==2)
            class_return = np.delete(class_return, del_feet, axis=0)
            data_return = np.delete(data_return, del_feet, axis=0)
            del_tongue = np.where(class_return==3)
            class_return = np.delete(class_return, del_tongue, axis=0)
            data_return = np.delete(data_return, del_tongue, axis=0)
        if self.num_classes == 3:
            del_tongue = np.where(class_return==3)
            class_return = np.delete(class_return, del_tongue, axis=0)
            data_return = np.delete(data_return, del_tongue, axis=0)

        data_return = t.Tensor(data_return).to(dtype=t.float)
        class_return = t.Tensor(class_return).to(dtype=t.long)

        return data_return, class_return


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


def get_transform(fs, t1_factor, t2_factor, filter_config):
    # make sure that filter_config exists
    if filter_config is None:
        filter_config = {'type': None}
    elif 'type' not in filter_config:
        filter_config['type'] = 'none'

    if filter_config['type'] == 'highpass':
        filter_transform = HighpassFilter(fs, filter_config['fc'], filter_config['order'])
    elif filter_config['type'] == 'bandpass':
        filter_transform = BandpassFilter(fs, filter_config['fc_low'], filter_config['fc_high'],
                                          filter_config['order'])
    else:
        filter_transform = Identity()

    return Compose([filter_transform,
                    ReshapeTensor(),
                    TimeWindowPostCue(fs, t1_factor, t2_factor)])


def load_data_sets(dir_data, data_config, subj_indexes=None, fold=None):
    try:
        cs = data_config['cs']
    except:
        cs = []

    transform      = get_transform(data_config['fs'], data_config['t1_factor'],
                                   data_config['t2_factor'], data_config['filter'])
    trainvalid_set = BCI_CompIV_2a(root=dir_data, train=True, subject=data_config['subject'], num_classes=data_config['num_classes'], cs=cs)
    if data_config.get("use_test_as_valid", False):
        # use the test set as the validation set
        train_set = trainvalid_set
        train_set.transform = transform
        valid_set = BCI_CompIV_2a(root=dir_data, train=False, subject=data_config['subject'], num_classes=data_config['num_classes'], transform=transform, cs=cs)
        test_set  = BCI_CompIV_2a(root=dir_data, train=False, subject=data_config['subject'], num_classes=data_config['num_classes'], transform=transform, cs=cs)
    else:
        # split train set into train and validation set
        len_train            = int(len(trainvalid_set) * (1.0 - data_config['valid_fraction']))
        train_set, valid_set = transform_random_split(trainvalid_set, [len_train, len(trainvalid_set) - len_train],
                                                    [transform, transform])
        test_set             = BCI_CompIV_2a(root=dir_data, train=False, subject=data_config['subject'], num_classes=data_config['num_classes'], transform=transform, cs=cs)

    return train_set, valid_set, test_set
