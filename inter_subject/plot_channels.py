import os
import numpy as np
from PIL import Image
from keras.models import load_model

from channel_selection import get_eegweights

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

def plot_heatmap(num_classes):
    os.makedirs(f'plot_channels/plots/heatmap', exist_ok=True)

    w = 0
    for split_ctr in range(5):
        model_global = load_model(f'global/model/global_class_{num_classes}_ds1_nch64_T3_split_{split_ctr}_v1.h5')
        w_temp = model_global.layers[3].get_weights()[0]
        w += get_eegweights(w_temp)
    w = w / 5

    background = Image.open(f'plot_channels/64_channel_sharbrough_bg.png')

    channel = 1
    for i in w:
        if i > 1.85:
            img = Image.open(f'plot_channels/channels/{channel}.png')
            background.paste(img, (0, 0), img)
        elif i > 1.55:
            img = Image.open(f'plot_channels/channels/{channel}_o.png')
            background.paste(img, (0, 0), img)
        elif i > 1.42:
            img = Image.open(f'plot_channels/channels/{channel}_y.png')
            background.paste(img, (0, 0), img)
        else:
            pass
        channel += 1
    background.save(f'plot_channels/plots/heatmap/class{num_classes}.png',"PNG")

# plot_heatmap(4)

''' make white background transparent '''

# for i in range(64):
#     channel = i + 1
#     img = Image.open(f'plot_channels/channels/{channel}_y.png')
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
#     img.save(f'plot_channels/channels/{channel}_y.png', "PNG")
