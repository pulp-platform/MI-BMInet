__author__ = "Tianhong Gan"
__email__ = "tianhonggan@outlook.com"

def get_parameters(kernel_length=128, NO_selected_channels=64, NO_samples=480, pool_length=8, NO_classes=4, NO_subjects=1, NO_frozen_layers=0):
    '''
    get number of parameters (weights) of all four layers, depending on the number of layers that have been frozen, for subject specific models
    '''
    n1 = (8 * kernel_length) + 32
    n2 = (16 * NO_selected_channels) + 64
    n3 = 512 + 64
    n4 = ((((NO_samples / pool_length) / 8) * 16) + 1) * NO_classes

    if NO_frozen_layers == 0:
        NO_parameters = NO_subjects * (n1 + n2 + n3 + n4)

    elif NO_frozen_layers == 1:
        NO_parameters = n1 + (NO_subjects * (n2 + n3 + n4))

    elif NO_frozen_layers == 2:
        NO_parameters = n1 + n2 + (NO_subjects * (n3 + n4))

    elif NO_frozen_layers == 3:
        NO_parameters = n1 + n2 + n3 + (NO_subjects * n4)

    return NO_parameters

def get_featureMapSize(NO_samples=480,NO_selected_channels=64,pool_length=8,NO_classes=4):
    '''
    get feature map size of all four layers
    '''
    n1 = NO_samples * NO_selected_channels * 8
    n2 = (NO_samples / pool_length) * 16
    n3 = ((NO_samples / pool_length) / 8) * 16
    n4 = NO_classes

    size = n1 + n2 + n3 + n4

    return size

def get_sizeInBytes(size, bytes):
    '''
    considering 32-bit floating-point numbers, calculate the estimated flash memory needed for storing the parameters of the model
    and the RAM requirement

    size -- number of parameters
    bytes -- how to return, B, kB, MB

    return -- size in B, kB or MB
    '''
    if bytes == 'mb':
        return (size * 4) / (2**20)
    elif bytes == 'kb':
        return (size * 4) / (2**10)
    else:
        return (size * 4)

# print(get_parameters(NO_frozen_layers=1))
