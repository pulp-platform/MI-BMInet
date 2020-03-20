#!/usr/bin/env python3

import os
import numpy as np
from keras.models import load_model

from filters import load_filterbank
from csp import generate_projection

__author__ = "Tianhong Gan"
__email__ = "tianhonggan@outlook.com"

class CS_Model:
	def __init__(self):
		self.data_path 	= '/usr/scratch/xavier/herschmi/EEG_data/physionet/' #data path
		self.channel_selection_method = 2 # 1: w squared sum, 2: csp-rank

		self.fs = 160. # sampling frequency
		self.NO_channels = 64 # number of EEG channels
		self.NO_selected_channels = 16 # number of selected channels
		self.NO_subjects = 105 # number of subjects
		self.NO_csp = 12 # Total number of CSP features per band and timewindow
		self.NO_classes = 4

		self.bw = np.array([26]) # bandwidth of filtered signals
		self.ftype = 'butter' # 'fir', 'butter'
		self.forder= 2 # 4
		self.filter_bank = load_filterbank(self.bw,self.fs,order=self.forder,max_freq=30,ftype = self.ftype) # get filterbank coeffs
		self.NO_bands = self.filter_bank.shape[0]

		time_windows_flt = np.array([
									[0,1],
									[0.5,1.5],
									[1,2],
									[1.5,2.5],
									[2,3],
									[0,2],
									[0.5,2.5],
									[1,3],
									[0,3]])*self.fs # time windows in [s] x fs for using as a feature
		self.time_windows = time_windows_flt.astype(int)
		self.time_windows = self.time_windows[8] # use only largest timewindow
		self.NO_time_windows = int(self.time_windows.size/2)

'''CHANNEL SELECTION USING CSP'''

# V1 using ranking using the squared sum
def channel_selection_squared_sum(w, NO_channels, NO_selected_channels):
    ''' ranking the energy of each channel obtained from the set of 12 spatial filters by the squared sum, select channels with highest energy usage

     Keyword arguments:
     w -- set of 12 spatial filters of size [NO_channels, NO_csp]
     NO_channels - total number of channels
     NO_selected_channels -- number of channels to select

    Return: 'NO_selected_channels' channels with the highest energy usage
    '''
    w_squared_sum = np.zeros((NO_channels,2)) # creating an empty of dimension NO_channels x 2 for channel number and energy

    index = 0 # iterator

    for channel in w:
        w_squared_sum[index][0] = index # set channel number
        for filter in channel:
            w_squared_sum[index][1] += filter**2 # set channel energy
        index+=1

    w_squared_sum_sorted = w_squared_sum[w_squared_sum[:,1].argsort()][::-1] # sort energy in descending order

    if NO_selected_channels <= NO_channels:
        selected_channels = w_squared_sum_sorted[0:NO_selected_channels, 0]
    else:
        selected_channels = w_squared_sum_sorted[0:NO_channels, 0]

    return np.sort(selected_channels)

# V2 using CSP-ranking
def channel_selection_csprank(w, NO_channels, NO_selected_channels, NO_csp):
    ''' ranking and channel selection using the CSP-rank method

    The CSP-rank method first sorts the absolute value of the filter coefficients in each filter respectively,
    then takes the electrode with the next largest coefficient in turn from the 12 spatial filters.

     Keyword arguments:
     w -- set of 12 spatial filters of size [NO_channels, NO_csp]
     NO_channels -- total number of channels
     NO_selected_channels -- number of channels to select
     NO_csp -- number of spatial filters

     Return: 'NO_selected_channels' channels with the highest importance.
    '''

    sort_temp = np.zeros((NO_channels, 2)) # creating an empty array of dimension NO_channels x 2 for channel number and importance
    w_sorted = np.zeros((NO_channels, NO_csp)) # creating an empty array of dimension NO_channels x NO_csp for channel selection, contains channels ranked on energy for each filter descending

    # sorts the absolute value of the filter coefficients in each filter respectively
    filter_index = 0
    channel_index = 0
    w = w.transpose()
    for filter in w:
        channel_index = 0
        for channel in filter:
            sort_temp[channel_index][0] = channel_index
            sort_temp[channel_index][1] = np.sqrt(channel**2)
            channel_index+=1
        w_sorted[:, filter_index] = sort_temp[sort_temp[:,1].argsort()][::-1][:, 0] # sort importance in descending order
        filter_index+=1

    # takes the electrode with the next largest coefficient in turn from the 12 spatial filters
    selected_channels = np.ones(NO_selected_channels) * 404
    selected_index = 0 # index of filter currently being selected
    filter_index = 0 # index of filter being selected from
    w_channel_index = np.zeros(NO_csp) # array to keep track of index in specific filter
    while selected_index < NO_selected_channels:
        if filter_index == NO_csp:
            filter_index = 0
        while w_sorted[int(w_channel_index[filter_index])][filter_index] in selected_channels: # if the channel is already selected, move to next highest channel
            w_channel_index[filter_index] += 1
        selected_channels[selected_index] = w_sorted[int(w_channel_index[filter_index])][filter_index] # selects highest ranked channel from current filter
        selected_index += 1
        filter_index += 1

    return np.sort(selected_channels)

def channel_selection_csp(X_sub, y_sub, NO_csp, filter_bank, time_windows, NO_classes, NO_channels, NO_selected_channels, channel_selection_method):
	w_4d = generate_projection(X_sub, y_sub, NO_csp, filter_bank, time_windows, NO_classes) # obtain filter
	w = w_4d[0][0]
	if channel_selection_method == 1: #V1 using w squared sum
		selected_channels = channel_selection_squared_sum(w, NO_channels, NO_selected_channels).astype(int)
	elif channel_selection_method == 2: # V2 using CSP-ranking
		selected_channels = channel_selection_csprank(w, NO_channels, NO_selected_channels, NO_csp).astype(int)

	return selected_channels

'''CHANNEL SELECTION USING EEGNET WEIGHTS'''

def channel_selection_eegweights(w, NO_channels, NO_selected_channels):
	''' ranking the energy of each channel obtained from the set of filter weights by the squared sum, select channels with highest energy usage

	 Keyword arguments:
	 w -- set of filter weights of size [NO_channels, number of filters, depth, ..]
	 NO_channels - total number of channels
	 NO_selected_channels -- number of channels to select

	Return: 'NO_selected_channels' channels with the highest energy usage
	'''
	w_squared_sum = np.zeros((NO_channels,2)) # creating an empty of dimension NO_channels x 2 for channel number and energy

	index = 0 # iterator

	for channel in w:
		w_squared_sum[index][0] = index # set channel number
		for column in channel:
			for depth in column:
				for i in depth:
					w_squared_sum[index][1] += i**2 # set channel energy
		index+=1

	w_squared_sum_sorted = w_squared_sum[w_squared_sum[:,1].argsort()][::-1] # sort energy in descending order

	if NO_selected_channels <= NO_channels:
		selected_channels = w_squared_sum_sorted[0:NO_selected_channels, 0]
	else:
		selected_channels = w_squared_sum_sorted[0:NO_channels, 0]

	return np.sort(selected_channels).astype(int)

def channel_selection_eegweights_fromglobal(NO_channels, NO_selected_channels, NO_classes):
	w = 0
	for split_ctr in range(0, 5):
		model_global = load_model(f'Global/model/global_class_{NO_classes}_ds1_nch64_T3_split_{split_ctr}_v1.h5')
		w += model_global.layers[3].get_weights()[0] ** 2 # using values from all splits

	selected_channels = channel_selection_eegweights(w, NO_channels, NO_selected_channels)
	print(selected_channels)
	return selected_channels

def channel_selection_eegweights_fromss(subject, NO_channels, NO_selected_channels):
	sub_str = '{0:03d}'.format(subject)
	w_ss = 0
	for i in range(0, 4):
	    model_ss = load_model(f'SS-TL/64ch/model/subject{sub_str}_fold{i}.h5')
	    w_ss += model_ss.layers[3].get_weights()[0] ** 2 # using values from all splits

	selected_channels = channel_selection_eegweights(w_ss, NO_channels, NO_selected_channels)
	print(selected_channels)
	return selected_channels
