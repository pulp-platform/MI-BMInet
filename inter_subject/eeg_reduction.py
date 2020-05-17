#!/usr/bin/env python3


__author__ = "Michael Hersche, Tianhong Gan"
__email__ = "herschmi@iis.ee.ethz.ch, tianhonggan@outlook.com"

import pdb
import numpy as np
import scipy.signal as scp
from channel_selection import channel_selection_eegweights_fromglobal
from plot_channels import plot_channels

def eeg_reduction_cs(x, split_ctr, results_dir, n_ds = 1, n_ch = 64, T = 3, fs = 160, num_classes = 4):
	'''
	Inputs
	------
	x : np array ()
		input array
	split_ctr : int
		split number
	n_ds: int
		downsampling factor
	n_ch: int
		number of channels
	T: float
		time [s] to classify
	fs: int
		sampling frequency [Hz]
	num_classes : int
		number of classes


	Outputs
	-------
	'''

	if n_ch == 64:
		channels = np.arange(0,64)
	else:
		channels = channel_selection_eegweights_fromglobal(64, n_ch, num_classes, split_ctr, n_ds, T, results_dir)
		plot_channels(channels, num_classes, n_ds, n_ch, T, split_ctr)

	n_s_orig = int(T*fs)
	n_s = int(np.ceil(T*fs/n_ds)) # number of time samples
	n_trial = x.shape[0]

	# channel selection
	if n_ds > 1:
		x = x[:,:,channels]
		y = np.zeros((n_trial, 1, n_ch, n_s))
		for trial in range(n_trial):
			for chan in range(n_ch):
				# downsampling
				#pdb.set_trace()
				y[trial,:,chan] = scp.decimate(x[trial,:,chan,:n_s_orig],n_ds)
	else:
		y = x[:,:,channels]
		y = y[:,:,:,:n_s_orig]

	return y
