import numpy as np
from filters import load_filterbank
from csp import generate_projection
from ranking import channel_selection_squared_sum

class CSP_Model:
	def __init__(self):
		self.data_path 	= '/usr/scratch/xavier/herschmi/EEG_data/physionet/' #data path
		self.channel_selection_method = 1 # 1: w squared sum, 2: csp-rank

		self.fs = 160. # sampling frequency
		self.NO_channels = 64 # number of EEG channels
		self.NO_selected_channels = 8 # number of selected channels
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

def channel_selection_csp(X_sub, y_sub, NO_csp, filter_bank, time_windows, NO_classes, NO_channels, NO_selected_channels):
    w_4d = generate_projection(X_sub, y_sub, NO_csp, filter_bank, time_windows, NO_classes) # obtain filter
    w = w_4d[0][0]
    selected_channels = channel_selection_squared_sum(w, NO_channels, NO_selected_channels).astype(int)

    return selected_channels
