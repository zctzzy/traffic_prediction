# -*- coding: utf-8 -*-
"""
/*******************************************
** This is a file created by Chuanting Zhang
** Name: DenseNet
** Date: 2017/9/28
** Email: chuanting.zhang@gmail.com
** BSD license
********************************************/
"""

from keras.models import Model
from keras.layers.core import Dropout, Activation
from keras.layers.convolutional import Conv2D
from keras.layers import Input, merge
from keras.layers.normalization import BatchNormalization
from keras.regularizers import l2
import keras.backend as K

height, width = 100, 100
dims = (3, 2, height, width)


def conv_factory(x, nb_filter, dropout_rate=None, weight_decay=1e-4, bottleneck=True):
    x = BatchNormalization(mode=0,
                           axis=1,
                           gamma_regularizer=l2(weight_decay),
                           beta_regularizer=l2(weight_decay))(x)
    x = Activation('relu')(x)

    # if bottleneck:
    #     inter_channel = nb_filter * 4  # Obtained from https://github.com/liuzhuang13/DenseNet/blob/master/densenet.lua
    #
    #     x = Conv2D(inter_channel, (1, 1), kernel_initializer='he_normal', padding='same', use_bias=False,
    #                kernel_regularizer=l2(weight_decay))(x)
    #     x = BatchNormalization(axis=1, epsilon=1.1e-5)(x)
    #     x = Activation('relu')(x)

    x = Conv2D(nb_filter, kernel_size=(3, 3),
                      init="he_uniform",
                      border_mode="same",
                      bias=False,
                      W_regularizer=l2(weight_decay))(x)
    if dropout_rate:
        x = Dropout(dropout_rate)(x)

    return x


def denseblock(x, nb_layers, nb_filter,
               dropout_rate=None, weight_decay=1E-4):
    list_feat = [x]

    if K.image_dim_ordering() == "th":
        concat_axis = 1
    elif K.image_dim_ordering() == "tf":
        concat_axis = -1

    for i in range(nb_layers):
        merge_tensor = conv_factory(x, nb_filter, dropout_rate, weight_decay)
        x = merge([merge_tensor, x], mode='concat', concat_axis=concat_axis)
        list_feat.append(x)
        # x = merge(list_feat, mode='concat', concat_axis=concat_axis)

    return x


def densenet(depth, nb_filter, c=dims, p=dims, t=dims,
             dropout_rate=None, weight_decay=1e-4):
    main_inputs = []
    outputs = []

    for conf in [c, p, t]:
        if conf is not None:
            len_seq, nb_flow, h, w = conf

            input = Input(shape=(len_seq*nb_flow, h, w))
            main_inputs.append(input)

            nb_layers = depth - 2

            # initial convolution

            x = Conv2D(nb_filter, kernel_size=(3, 3), padding="same")(input)

            # dense block
            x = denseblock(x, nb_layers, nb_filter,
                           dropout_rate=dropout_rate,
                           weight_decay=weight_decay)

            x = BatchNormalization(axis=1, gamma_regularizer=l2(weight_decay),
                                   beta_regularizer=l2(weight_decay))(x)

            x = Activation('relu')(x)

            x = Conv2D(filters=nb_flow, kernel_size=(3, 3), padding="same")(x)
            outputs.append(x)

    if len(outputs) == 1:
        main_output = outputs[0]
    else:
        # from ..models import iLayer
        from .iLayer import iLayer
        new_outputs = []
        for output in outputs:
            # Non fusion
            # new_outputs.append(output)

            # With fusion
            new_outputs.append(iLayer()(output))

        main_output = merge(new_outputs, mode='sum')

    main_output = Activation('sigmoid')(main_output)
    model = Model(inputs=main_inputs, outputs=main_output)

    return model