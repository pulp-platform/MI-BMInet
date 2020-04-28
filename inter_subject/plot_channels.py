import os
from PIL import Image

def plot_channels(channels, num_classes, n_ds, n_ch, T, split_ctr):
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
