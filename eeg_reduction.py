#!/usr/bin/env python3


__author__ = "Michael Hersche"
__email__ = "herschmi@iis.ee.ethz.ch"

import numpy as np
import scipy.signal as scp
import pdb


def eeg_reduction(x, n_ds = 1, n_ch = 64, T = 3, fs = 160):
	'''
	Inputs
	------
	x : np array ()
		input array
	n_ds: int
		downsampling factor
	n_ch: int
		number of channels
	T: float
		time [s] to classify
	fs: int
		sampling frequency [Hz]


	Outputs
	-------
	'''

	# original selected channels
	# if n_ch == 64:
	# 	channels = np.arange(0,n_ch)
	# elif n_ch == 38:
	# 	channels = np.array([0,2,4,6,8,10,12,14,16,18,20,21,22,23,24,26,28,29,31,33,35,37,40,41,42,43,46,48,50,52,54,55,57,59,60,61,62,63])
	# elif n_ch == 27:
	# 	channels = np.array([0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,38,39,40,41,44,45])
	# elif n_ch == 19:
	# 	channels = np.array([8,10,12,21,23,29,31,33,35,37,40,41,46,48,50,52,54,60,62])
	# elif n_ch == 16:
	# 	channels = np.array([2,3,4,5,6,9,10,11,12,13,16,17,18,19,20,34]) #trial
	# elif n_ch == 8:
	# 	channels = np.array([8,10,12,25,27,48,52,57])

	# # multiscale temporal and spectral CSP selected channels
	# if n_ch == 64:
	# 	channels = np.arange(0,n_ch)
	# elif n_ch == 38:
	# 	channels = np.array([0,1,2,3,4,6,7,8,9,10,11,13,14,15,16,17,18,19,20,21,22,26,31,32,33,34,35,37,43,48,49,50,51,53,54,56,57,58])
	# elif n_ch == 24:
	# 	channels = np.array([1,3,4,7,8,9,10,11,16,17,18,19,20,21,26,31,32,33,34,35,48,49,50,56])
	# elif n_ch == 19:
	# 	channels = np.array([1,4,7,8,9,10,11,18,19,20,21,26,31,32,33,35,48,49,50])
	# elif n_ch == 16:
	# 	channels = np.array([1,7,8,9,10,11,19,20,26,31,32,33,35,48,49,50])
	# elif n_ch == 8:
	# 	channels = np.array([8,9,11,20,26,32,33,49])

	# # CSP selected channels
	# if n_ch == 64:
	# 	channels = np.arange(0,n_ch)
	# elif n_ch == 38:
	# 	channels = np.array([0,1,2,3,4,6,7,8,10,11,13,16,17,18,19,20,21,26,28,29,31,32,33,35,36,37,38,48,49,50,51,52,53,54,56,57,59,61])
	# elif n_ch == 24:
	# 	channels = np.array([0,1,3,6,7,10,13,16,17,18,19,20,26,31,32,33,35,37,38,48,49,50,53,54])
	# elif n_ch == 19:
	# 	channels = np.array([0,1,6,7,13,18,19,20,26,31,32,33,35,37,38,48,49,50,54])
	# elif n_ch == 16:
	# 	channels = np.array([0,1,7,13,18,19,20,26,31,32,33,35,37,48,49,50])
	# elif n_ch == 8:
	# 	channels = np.array([7,19,20,31,32,33,35,49])

	# CSP-rank selected channels
	if n_ch == 64:
		channels = np.arange(0,n_ch)
	elif n_ch == 38:
		channels = np.array([0,1,2,3,4,5,6,7,9,10,11,13,15,16,17,18,19,20,25,26,28,31,32,33,34,35,36,37,38,48,49,50,51,53,54,56,57,59])
	elif n_ch == 24:
		channels = np.array([0,1,2,4,6,7,13,16,17,18,19,20,26,31,32,33,34,35,48,49,50,53,56,57])
	elif n_ch == 19:
		channels = np.array([1,2,6,7,13,17,18,19,20,26,31,32,33,34,35,48,49,50,57])
	elif n_ch == 16:
		channels = np.array([2,6,7,13,18,19,20,26,31,32,33,35,48,49,50,57])
	elif n_ch == 8:
		channels = np.array([7,18,20,32,33,35,49,50])


	n_s_orig = int(T*fs)
	n_s = int(np.ceil(T*fs/n_ds)) # number of time samples
	n_trial = x.shape[0]

	# channel selection
	if n_ds >1:
		x = x[:,channels]
		y = np.zeros((n_trial, n_ch,n_s))
		for trial in range(n_trial):
			for chan in range(n_ch):
				# downsampling
				#pdb.set_trace()
				y[trial,chan] = scp.decimate(x[trial,chan,:n_s_orig],n_ds)
	else:
		y = x[:,channels]
		y = y[:,:,:n_s_orig]

	return y
