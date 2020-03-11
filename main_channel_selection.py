#!/usr/bin/env python3

import os
from keras.models import load_model

from channel_selection import CS_Model, channel_selection_eegweights

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

''' Channel selection for global and subject specific using EEGNet weights'''

cs_model = CS_Model()
ss_tl = True

def exclude_subjects(all_subjects=range(1,110), excluded_subjects=[88,92,100,104]):
    subjects = [x for x in all_subjects if (x not in excluded_subjects)]
    return subjects

subjects = exclude_subjects()
sub_idx = 94

if ss_tl:
    sub_str = '{0:03d}'.format(subjects[sub_idx])
    w_ss = 0
    for i in range(0, 4):
        model_ss = load_model(f'SS-TL/64ch/model/subject{sub_str}_fold{i}.h5')
        w_ss += model_ss.layers[3].get_weights()[0] ** 2

    selected_channels = channel_selection_eegweights(w_ss, cs_model.NO_channels, cs_model.NO_selected_channels)
    print(selected_channels)
else:
    w = 0
    for split_ctr in range(0, 5):
        model_global = load_model(f'Global/model/global_class_{cs_model.NO_classes}_ds1_nch64_T3_split_{split_ctr}_v1.h5')
        w += model_global.layers[3].get_weights()[0] ** 2

    selected_channels = channel_selection_eegweights(w, cs_model.NO_channels, cs_model.NO_selected_channels)
    print(selected_channels)

    '''
    [ 3  9 10 29 37 38 43 63]
    [ 3  9 10 11 12 19 29 37 38 42 43 44 48 50 57 63]
    [ 2  3  6  9 10 11 12 16 19 29 37 38 42 43 44 48 50 57 63]
    [ 2  3  6  9 10 11 12 15 16 17 19 29 30 37 38 42 43 44 48 50 55 56 57 63]
    [ 2  3  6  7  8  9 10 11 12 15 16 17 19 20 22 28 29 30 34 35 36 37 38 40 42 43 44 48 50 51 52 54 55 56 57 59 60 63]
    '''
