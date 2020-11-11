import numpy as np
import os
import sys
sys.path.insert(0, "../")
#
#import get_data as get
from tensorflow.keras import utils as np_utils
from keras.callbacks import LearningRateScheduler, ReduceLROnPlateau
from keras.optimizers import Adam
from keras import backend as K
from sklearn.model_selection import KFold

# EEGNet models
import models as models
import modelstest as mt
# Channel reduction, downsampling, time window
from eeg_reduction import eeg_reduction

num_classes = 4
n_ch = 64
n_ds=1
n_samples = 480
kernLength = int(np.ceil(128/n_ds))
poolLength = int(np.ceil(8/n_ds))


model = models.cubeEEGNet(nb_classes = num_classes, Chans=n_ch, Samples=n_samples, regRate=0.25,
                                dropoutRate=0.2, kernLength=kernLength, poolLength=poolLength, numFilters=8, dropoutType='Dropout')

print("\n\ncubeEEGNet")
print(model.summary())


model = models.cubeedgeEEGNetCF1(nb_classes = num_classes, Chans=n_ch, Samples=n_samples, regRate=0.25,
                                dropoutRate=0.2, kernLength=kernLength, poolLength=poolLength, numFilters=8, dropoutType='Dropout')

print("\n\ncubeedgeEEGNet")
print(model.summary())


model = models.edgeEEGNetC(nb_classes = num_classes, Chans=n_ch, Samples=n_samples, regRate=0.25,
                                dropoutRate=0.2, kernLength=kernLength, poolLength=poolLength, numFilters=8, dropoutType='Dropout')

print("\n\nedgeEEGNetC")
print(model.summary())


model = mt.edgeEEGNetCF1_test(nb_classes = num_classes, Chans=n_ch, Samples=n_samples, regRate=0.25,
                                dropoutRate=0.2, kernLength=kernLength, poolLength=poolLength, numFilters=8, dropoutType='Dropout')

print("\n\nedgeEEGNetCF1")
print(model.summary())
