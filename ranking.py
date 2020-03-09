#!/usr/bin/env python3

''' Functions used for ranking and channel selection '''

import numpy as np

__author__ = "Tianhong Gan"
__email__ = "tianhonggan@outlook.com"

def dimension_reduction(w_4d, NO_channels, NO_csp):
    ''' reducing the dimensions of w from 4D to 2D by taking the modulus

    given the 4D input matrix:
       1. for every time window and frequency, there is a 2D array of NO_channels x NO_csp
       2. take every element of every 2D array, square it, and add to sum matrix at the position of that element
       3. take the square root of each element in the final 2D array to obtain modulus of that element

     Keyword arguments:
     w_4D -- four dimensional array of spatial filters of size [NO_timewindows,NO_freqbands,NO_channels,NO_csp]
     NO_channels - total number of channels
     NO_csp -- number of spatial filters

     Return: set of 12 spatial filters obtained from a average of filters obtained from each subject of size [NO_channels, NO_csp]
    '''
    w_2d = np.zeros((NO_channels, NO_csp))

    csp_index = 0
    channel_index = 0

    for window in w_4d:
        for freqband in window:
            for channel in freqband:
                if channel_index == NO_channels:
                    channel_index = 0
                for filter in channel:
                    if csp_index == NO_csp:
                        csp_index = 0
                    w_2d[channel_index][csp_index] += filter**2
                    csp_index+=1
                channel_index+=1

    return np.sqrt(w_2d)

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
    w_sorted = np.zeros((NO_channels, NO_csp)) # creating an empty array of dimension NO_channels x NO_csp for channel selection

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
    selected_index = 0
    filter_index = 0
    w_channel_index = np.zeros(NO_csp)
    while selected_index < NO_selected_channels:
        if filter_index == NO_csp:
            filter_index = 0
        while w_sorted[int(w_channel_index[filter_index])][filter_index] in selected_channels:
            w_channel_index[filter_index] += 1
        selected_channels[selected_index] = w_sorted[int(w_channel_index[filter_index])][filter_index]
        selected_index += 1
        filter_index += 1

    return np.sort(selected_channels)
