import os
import pdb
import numpy as np

from PIL import Image

from keras.models import load_model

__author__ = "Tianhong Gan"
__email__ = "tianhonggan@outlook.com"

def plot_channels(channels, num_classes, n_ds, n_ch, T, split_ctr):
    ''' plot channels selected on 64 channel template

    Keyword arguments:
    channels -- array of selected channels
    num_classes -- number of classes
    n_ds -- downsampling factor
    n_ch -- number of channels selected
    T -- time window length
    split_ctr -- split number
    '''
    os.makedirs(f'plot_channels/plots', exist_ok=True)
    background = Image.open(f'plot_channels/64_channel_sharbrough_bg.png')
    for i in channels:
        channel = i + 1
        img = Image.open(f'plot_channels/channels/{channel}.png')
        background.paste(img, (0, 0), img)
    background.save(f'plot_channels/plots/class{num_classes}_ds{n_ds}_nch{n_ch}_T{T}_split_{split_ctr}.png',"PNG")

def get_eegweights_avg(w):
    ''' get average modulus of EEGNet weights for each channel

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

def plot_heatmap_avg(num_classes,split_ctr,type):
    ''' plot heatmap of 64-channels on 10-10 international system - average across all 16 filters for a specific split

    Keyword arguments:
    num_classes -- number of classes
    split_ctr -- model split
    type -- 'channels' for coloured ring around channels or 'channels_fill' for opaque colour heatmap
    '''
    os.makedirs(f'plot_channels/plots/heatmap', exist_ok=True)

    w = 0
    model_global = load_model(f'global/model/global_class_{num_classes}_ds1_nch64_T3_split_{split_ctr}_v1.h5')
    w_temp = model_global.layers[3].get_weights()[0]
    w = get_eegweights_avg(w_temp)

    background = Image.open(f'plot_channels/64_channel_sharbrough_bg.png')

    mean = np.mean(w)
    sd = np.std(w)

    channel = 1
    for i in w:
        if i > mean+sd:
            img = Image.open(f'plot_channels/{type}/{channel}.png')
            background.paste(img, (0,0), img)
        elif i > mean:
            img = Image.open(f'plot_channels/{type}/{channel}_o.png')
            background.paste(img, (0,0), img)
        elif i > mean-sd:
            img = Image.open(f'plot_channels/{type}/{channel}_y.png')
            background.paste(img, (0,0), img)
        else:
            pass
        channel += 1
    background.save(f'plot_channels/plots/heatmap/class{num_classes}_split{split_ctr}.png',"PNG")

def plot_heatmap(num_classes,split_ctr,type):
    ''' plot heatmap of 64-channels on 10-10 international system - for all 16 filters in a specific split

    Keyword arguments:
    num_classes -- number of classes
    split_ctr -- model split
    type -- 'channels' for coloured ring around channels or 'channels_fill' for opaque colour heatmap
    '''
    os.makedirs(f'plot_channels/plots/heatmap', exist_ok=True)

    model_global = load_model(f'global/model/global_class_{num_classes}_ds1_nch64_T3_split_{split_ctr}_v1.h5')
    w_temp = model_global.layers[3].get_weights()[0] # shape [64, 1, 8, 2]

    w = np.zeros((64,16))

    c = 0 # channel iterator
    f = 0 # filter iterator

    for channel in w_temp:
        for column in channel:
            for depth in column:
                for i in depth:
                    w[c][f] = np.sqrt(i**2) # set channel energy
                    f+=1
        c+=1
        f=0

    mean = np.mean(w)
    sd = np.std(w)

    for filter in range(16):
        background = Image.open(f'plot_channels/64_channel_sharbrough_bg.png')

        channel = 1
        for i in w[:,filter]:
            if i > mean+sd:
                img = Image.open(f'plot_channels/{type}/{channel}.png')
                background.paste(img, (0,0), img)
            elif i > mean:
                img = Image.open(f'plot_channels/{type}/{channel}_o.png')
                background.paste(img, (0,0), img)
            elif i > mean-sd:
                img = Image.open(f'plot_channels/{type}/{channel}_y.png')
                background.paste(img, (0,0), img)
            else:
                pass
            channel += 1
        background.save(f'plot_channels/plots/heatmap/class{num_classes}_split{split_ctr}_filter{filter}.png',"PNG")


# plot_heatmap_avg(4,0,'channels_fill')
# plot_heatmap(4,0,'channels_fill')

''' make white background transparent '''
# for i in range(64):
#     channel = i + 1
#     img = Image.open(f'channels/{channel}.png')
#     img = img.convert("RGBA")
#     datas = img.getdata()
#
#     newData = []
#     for item in datas:
#         if item[0] == 255 and item[1] == 255 and item[2] == 255:
#             newData.append((255, 255, 255, 0))
#         else:
#             newData.append(item)
#
#     img.putdata(newData)
#     img.save(f'plot_channels/{channel}.png', "PNG")
