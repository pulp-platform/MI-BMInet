import numpy as np

__author__ = "Tianhong Gan"
__email__ = "tianhonggan@outlook.com"

# define actual layers
input = np.array([0]) # layer 0
conv_2D = np.array([1,2]) # layer 1
depth_conv_2D = np.array([3,4,5,6,7]) # layer 2
sep_conv_2D = np.array([8,9,10,11,12]) # layer 3
fc = np.array([13,14,15]) # layer 4

def get_layers(layers_to_freeze):
    ''' get sub-layers for each specified layer

    Keyword arguments:
    layers_to_freeze -- array containing number of layers to be frozen

    Return: sub-layers to freeze
    '''
    freeze = np.array([])

    if layers_to_freeze == 1:
        freeze = np.append(freeze, fc)
    if layers_to_freeze == 2:
        freeze = np.append(freeze, sep_conv_2D)
        freeze = np.append(freeze, fc)
    if layers_to_freeze == 3:
        freeze = np.append(freeze, depth_conv_2D)
        freeze = np.append(freeze, sep_conv_2D)
        freeze = np.append(freeze, fc)
    else:
        pass

    return freeze

def freeze_layers(model, layers_to_freeze):
    ''' freeze the specied layers in the model for SS-TL

    Keyword arguments:
    model -- the model to modify
    layers_to_freeze -- array containing number of layers to be frozen
    '''
    freeze = get_layers(layers_to_freeze)
    for sub_layer in freeze:
        model.layers[int(sub_layer)].trainable = False

def print_model(model, layers_to_freeze):
    ''' print the unfrozen layers in the model

    Keyword arguments:
    model -- the model to modify
    layers_to_freeze -- array containing number of layers to be frozen
    '''
    freeze = get_layers(layers_to_freeze)
    unfrozen = np.arange(16)
    unfrozen = [x for x in unfrozen if (x not in freeze)]
    print("---------------------------------------------")
    print("Unfrozen layers:")
    print("---------------------------------------------")
    for sub_layer in unfrozen:
        print(model.layers[sub_layer].name)
        print("---------------------------------------------")

# def get_layers(model, layers_to_freeze):
#     ''' get sub-layers for each specified layer
#
#     Keyword arguments:
#     model -- the model to modify
#     layers_to_freeze -- array containing layers to be frozen
#
#     Return: sub-layers to freeze
#     '''
#     freeze = np.array([])
#     unfrozen = np.arange(16)
#     for layer in layers_to_freeze:
#         if layer == 0:
#             freeze = np.append(freeze, input)
#         if layer == 1:
#             freeze = np.append(freeze, conv_2D)
#         if layer == 2:
#             freeze = np.append(freeze, depth_conv_2D)
#         if layer == 3:
#             freeze = np.append(freeze, sep_conv_2D)
#         if layer == 4:
#             freeze = np.append(freeze, fc)
#     return freeze
#
# def freeze_layers(model, layers_to_freeze):
#     ''' freeze the specied layers in the model for SS-TL
#
#     Keyword arguments:
#     model -- the model to modify
#     layers_to_freeze -- array containing layers to be frozen
#     '''
#     freeze = get_layers(model, layers_to_freeze)
#     for sub_layer in freeze:
#         model.layers[int(sub_layer)].trainable = False
#
# def print_model(model, layers_to_freeze):
#     ''' print the unfrozen layers in the model
#
#     Keyword arguments:
#     model -- the model to modify
#     layers_to_freeze -- array containing layers to be frozen
#     '''
#     freeze = get_layers(model, layers_to_freeze)
#     unfrozen = np.arange(16)
#     unfrozen = [x for x in unfrozen if (x not in freeze)]
#     print("Unfrozen layers:")
#     for sub_layer in unfrozen:
#         print(model.layers[sub_layer].name)
