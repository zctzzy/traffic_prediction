# -*- coding: utf-8 -*-
"""
/*******************************************
** This is a file created by Chuanting Zhang
** Name: iLayer
** Date: 2017/9/28
** Email: chuanting.zhang@gmail.com
** BSD license
********************************************/
"""

from keras import backend as K
from keras.engine.topology import Layer
import numpy as np


class iLayer(Layer):
    def __init__(self, **kwargs):
        super(iLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        initial_weight_value = np.random.random(input_shape[1:])
        self.W = K.variable(initial_weight_value)
        self.trainable_weights = [self.W]

    def call(self, x, mask=None):
        return x * self.W

    def get_output_shape_for(self, input_shape):
        return input_shape