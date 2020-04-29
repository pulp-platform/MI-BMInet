import os
from PIL import Image

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
