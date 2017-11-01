# -*- coding: utf-8 -*-
"""
/*******************************************
** This is a file created by Chuanting Zhang
** Name: ResNet
** Date: 2017/9/28
** Email: chuanting.zhang@gmail.com
** BSD license
********************************************/
"""


from __future__ import print_function

from keras.layers import Input, Activation, merge, Dropout
from keras.layers.merge import add
from keras.layers.convolutional import Convolution2D
from keras.layers.normalization import BatchNormalization
from keras.models import Model
from keras.regularizers import l2


height, width = 100, 100
dims = (3, 2, height, width)


def _shortcut(input, residual):
    return add([input, residual])
    # return merge([input, residual], mode='sum')


def _bn_relu_conv(filters, row, col, subsample=(1, 1), bn=False):
    def f(input):
        if bn:
            input = BatchNormalization(axis=1)(input)

        activation = Activation('relu')(input)
        return Convolution2D(filters, kernel_size=(row, col),
                             subsample=subsample, padding="same")(activation)
    return f


def _res_unit(filters, init_subsample=(1,1)):
    def f(input):
        residual = _bn_relu_conv(filters, 3, 3, init_subsample)(input)
        residual = _bn_relu_conv(filters, 3, 3, init_subsample)(residual)
        return _shortcut(input, residual)
    return f


def resunits(res_unit, nb_filters, repetations=1):
    def f(input):
        for i in range(repetations):
            init_subsample = (1, 1)
            input = res_unit(filters=nb_filters,
                             init_subsample=init_subsample)(input)
        return input
    return f

def resnet(c_conf=dims, p_conf=dims, t_conf=dims, res_unit=12):
    main_inputs = []
    outputs = []
    for conf in [c_conf, p_conf, t_conf]:
        if conf is not None:
            len_seq, nb_flow, h, w = conf
            input = Input(shape=(len_seq*nb_flow, h, w))
            main_inputs.append(input)

            conv1 = Convolution2D(filters=64, kernel_size=(3,3),
                                  padding="same")(input)

            residual_output = resunits(_res_unit, nb_filters=64,
                                       repetations=res_unit)(conv1)

            activation = Activation('relu')(residual_output)

            conv2 = Convolution2D(filters=nb_flow, kernel_size=(3,3), padding="same")(activation)

            outputs.append(conv2)

    if len(outputs) == 1:
        main_output = outputs[0]
    else:
        from iLayer import iLayer
        new_outputs = []
        for output in outputs:
            new_outputs.append(iLayer()(output))
        # main_output = merge(new_outputs, mode="sum")
        main_output = add(new_outputs)

    main_output = Activation('sigmoid')(main_output)
    model = Model(inputs=main_inputs, outputs=main_output)

    return model



if __name__ == '__main__':
    model = resnet(res_unit=4)
    model.summary()