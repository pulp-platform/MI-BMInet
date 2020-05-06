#!/usr/bin/env python3

import os
import numpy as np
from keras.models import load_model

__author__ = "Tianhong Gan"
__email__ = "tianhonggan@outlook.com"

'''CHANNEL SELECTION USING EEGNET WEIGHTS'''

def channel_selection_eegweights(w, NO_channels, NO_selected_channels):
	''' ranking the energy of each channel obtained from the set of filter weights by the squared sum, select channels with highest energy usage

	 Keyword arguments:
	 w -- set of filter weights of size [NO_channels, number of filters, depth, ..]
	 NO_channels -- total number of channels
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

def channel_selection_eegweights_fromglobal(NO_channels, NO_selected_channels, NO_classes, split_ctr):
	''' select channels with highest energy usage from eegnet weights - fold specific from global model

	 Keyword arguments:
	 NO_channels -- total number of channels
	 NO_selected_channels -- number of channels to select
	 NO_classes -- number of classes
	 split_ctr -- split number

	Return: 'NO_selected_channels' channels with the highest energy usage in fold specific model
	'''
	model_global = load_model(f'global/model/global_class_{NO_classes}_ds1_nch64_T3_split_{split_ctr}_v1.h5')
	w = model_global.layers[3].get_weights()[0] ** 2
	selected_channels = channel_selection_eegweights(w, NO_channels, NO_selected_channels)

	print(selected_channels)
	return selected_channels

def channel_selection_eegweights_fromss(subject, NO_channels, NO_selected_channels, sub_split_ctr):
	''' select channels with highest energy usage from eegnet weights - fold specific from subject specific model

	 Keyword arguments:
	 subject -- subject number
	 NO_channels -- total number of channels
	 NO_selected_channels -- number of channels to select
	 sub_split_ctr -- split number of SS-TL

	Return: 'NO_selected_channels' channels with the highest energy usage in fold specific model
	'''
	sub_str = '{0:03d}'.format(subject)
	model_ss = load_model(f'ss/64ch/model/subject{sub_str}_fold{sub_split_ctr}.h5')
	w_ss = model_ss.layers[3].get_weights()[0] ** 2
	selected_channels = channel_selection_eegweights(w_ss, NO_channels, NO_selected_channels)

	print(selected_channels)
	return selected_channels

def get_eegweights(w):
	''' ranking the energy of each channel obtained from the set of filter weights by the squared sum, select channels with highest energy usage

	 Keyword arguments:
	 w -- set of filter weights of size [64, 1, 8, 2]

	Return: eeg weights of each channel
	'''
	w_sum = np.zeros(64)

	index = 0 # iterator

	for channel in w:
		for column in channel:
			for depth in column:
				for i in depth:
					w_sum[index] += np.sqrt(i**2) # set channel energy
		index+=1

	return w_sum
