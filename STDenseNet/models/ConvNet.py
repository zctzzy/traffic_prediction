# -*- coding: utf-8 -*-
"""
/*******************************************
** This is a file created by Chuanting Zhang
** Name: ConvNet
** Date: 2017/9/28
** Email: chuanting.zhang@gmail.com
** BSD license
********************************************/
"""
from keras.models import Sequential
from keras.layers import Merge
from keras.layers.core import Activation
from keras.layers.convolutional import Convolution2D

def seqCNNBaseLayer(conf=(4, 3, 100, 100)):
    # 1 layer CNN for early fusion
    seq_len, n_flow, map_height, map_width = conf
    model = Sequential()
    model.add(Convolution2D(32, 3, 3, input_shape=(n_flow * seq_len, map_height, map_width), border_mode='same'))
    model.add(Activation('relu'))
    return model


def seqCNN_CPT(c_conf=(4, 3, 100, 100), p_conf=(4, 3, 100, 100), t_conf=(4, 3, 100, 100)):
    model = Sequential()
    components = []

    for conf in [c_conf, p_conf, t_conf]:
        if conf is not None:
            components.append(seqCNNBaseLayer(conf))
            nb_flow = conf[1]
    model.add(Merge(components, mode='concat', concat_axis=1))  # concat
    model.add(Convolution2D(32, 3, 3, border_mode='same'))
    model.add(Activation('relu'))

    model.add(Convolution2D(32, 3, 3, border_mode='same'))
    model.add(Activation('relu'))

    model.add(Convolution2D(32, 3, 3, border_mode='same'))
    model.add(Activation('relu'))

    model.add(Convolution2D(16, 3, 3, border_mode='same'))
    model.add(Activation('relu'))

    model.add(Convolution2D(nb_flow, 3, 3, border_mode='same'))
    model.add(Activation('sigmoid'))
    return model