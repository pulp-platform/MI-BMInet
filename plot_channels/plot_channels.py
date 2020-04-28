from PIL import Image

channels = [2,4,13,24,35,38,47,60]
no_channels = len(channels)

def plot_channels(channels):
    background = Image.open(f'64_channel_sharbrough_bg.png')
    for i in channels:
        img = Image.open(f'channels/{i}.png')
        background.paste(img, (0, 0), img)
    background.save(f'plots/{no_channels}ch.png',"PNG")

plot_channels(channels)

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
