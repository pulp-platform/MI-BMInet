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
		samlping frequency [Hz]


	Outputs
	-------
	'''


	if n_ch ==64: 
		channels = np.arange(0,n_ch)
	elif n_ch == 38: 
		channels = np.array([0,2,4,6,8,10,12,14,16,18,20,21,22,23,24,26,28,29,31,33,35,37,40,41,42,43,46,48,50,52,54,55,57,59,60,61,62,63])
	elif n_ch == 27: 
		channels = np.array([0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,38,39,40,41,44,45])
	elif n_ch == 19: 
		channels = np.array([8,10,12,21,23,29,31,33,35,37,40,41,46,48,50,52,54,60,62])
	elif n_ch ==8: 
		channels = np.array([8,10,12,25,27,48,52,57])
	elif n_ch == 2: 
		channels = np.array([8,12]) # C3, C4
	elif n_ch == 3: 
		channels = np.array([8,10,12]) # C3, CZ, C4
	elif n_ch == 5:
		channels = np.array([7,8,10,12,13]) # C5, C3, CZ, C4, C6
	elif n_ch == 7:
		channels = np.array([7,8,9,10,11,12,13]) # C5, C3, C1, CZ, C2, C4, C6
	elif n_ch == 9:
		channels = np.array([40,7,8,9,10,11,12,13,41]) # T7,C5, C3, C1, CZ, C2, C4, C6,T8
	elif n_ch == 11:
		channels = np.array([42,40,7,8,9,10,11,12,13,41,43]) # T9,T7,C5,C3,C1,CZ,C2,C4,C6,T8,T10
	# 2 row band rows FC and C
	# elif n_ch ==4: 
	# 	channels = np.array([8,10,1,5]) # C3, C4 // FC3, FC4
	# elif n_ch == 6:
	# 	channels = np.array([8,10,12,1,3,5]) # C3, CZ, C4, // FC3, FCZ, FC4
	# elif n_ch == 10:
	# 	channels = np.array([7,8,10,12,13,0,1,3,5,6]) # C5, C3, CZ, C4, C6 // FC5,FC3, FCZ, FC4,FC6
	# elif n_ch == 14:
	# 	channels = np.array([7,8,9,10,11,12,13,0,1,2,3,4,5,6]) # C5, C3, C1, CZ, C2, C4, C6//FC5, FC3, FC1, FCZ, FC2, FC4, FC6
	# elif n_ch == 18:
	# 	channels = np.array([40,7,8,9,10,11,12,13,41,38,0,1,2,3,4,5,6,39]) # T7,C5, C3, C1, CZ, C2, C4, C6,T8// FT7, FC5, FC3, FC1, FCZ, FC2, FC4, FC6, FT8
	# elif n_ch == 20:
	# 	channels = np.array([42,40,7,8,9,10,11,12,13,41,43,38,0,1,2,3,4,5,6,39])# T9,T7,C5,C3,C1,CZ,C2,C4,C6,T8,T10// FT7, FC5, FC3, FC1, FCZ, FC2, FC4, FC6, FT8


	# 2 row band rows 
	elif n_ch ==4: 
		channels = np.array([8,10,15,19]) # C3, C4 // CP3, CP4
	elif n_ch == 6:
		channels = np.array([8,10,12,15,17,19]) # C3, CZ, C4, // CP3, CPZ, CP4
	elif n_ch == 10:
		channels = np.array([7,8,10,12,13,14,15,17,19,20]) # C5, C3, CZ, C4, C6 // CP5, CP3, CPZ, CP4, CP6
	elif n_ch == 14:
		channels = np.array([7,8,9,10,11,12,13,14,15,16,17,18,19,20]) # C5, C3, C1, CZ, C2, C4, C6// CP5, CP3, CP1, CPZ, CP2, CP4, CP6
	elif n_ch == 18:
		channels = np.array([40,7,8,9,10,11,12,13,41,44,14,15,16,17,18,19,20,45]) # T7,C5, C3, C1, CZ, C2, C4, C6,T8// TP7, CP5, CP3, CP1, CPZ, CP2, CP4, CP6, TP8
	elif n_ch == 20:
		channels = np.array([42,40,7,8,9,10,11,12,13,41,43,44,14,15,16,17,18,19,20,45])# T9,T7,C5,C3,C1,CZ,C2,C4,C6,T8,T10// TP7, CP5, CP3, CP1, CPZ, CP2, CP4, CP6, TP8



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





