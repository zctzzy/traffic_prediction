# -*- coding: utf-8 -*-
"""
/*******************************************
** This is a file created by Chuanting Zhang
** Name: DenseNet
** Date: 1/15/18
** Email: chuanting.zhang@gmail.com
** BSD license
********************************************/
"""
import torch.nn as nn
import torch
import torch.nn.functional as F
from collections import OrderedDict


class _DenseLayer(nn.Sequential):
    def __init__(self, num_input_features, growth_rate, bn_size, drop_rate):
        super(_DenseLayer, self).__init__()
        self.add_module('norm.1', nn.BatchNorm2d(num_input_features))
        self.add_module('relu.1', nn.ReLU(inplace=True))
        self.add_module('conv.1', nn.Conv2d(num_input_features, bn_size * growth_rate,
                                            kernel_size=1, stride=1, bias=False))
        self.add_module('norm.2', nn.BatchNorm2d(bn_size * growth_rate))
        self.add_module('relu.2', nn.ReLU(inplace=True))
        self.add_module('conv.2', nn.Conv2d(bn_size * growth_rate, growth_rate,
                                            kernel_size=3, stride=1, padding=1,
                                            bias=False))
        self.drop_rate = drop_rate

    def forward(self, input):
        new_features = super(_DenseLayer, self).forward(input)
        if self.drop_rate > 0:
            new_features = F.dropout(new_features, p=self.drop_rate,
                                     training=self.training)
        return torch.cat([input, new_features], 1)


class _DenseBlock(nn.Sequential):
    def __init__(self, num_layers, num_input_features, bn_size, growth_rate, drop_rate):
        super(_DenseBlock, self).__init__()
        for i in range(num_layers):
            layer = _DenseLayer(num_input_features + i * growth_rate, growth_rate,
                                bn_size, drop_rate)
            self.add_module('denselayer%d' % (i + 1), layer)


class iLayer(nn.Module):
    def __init__(self):
        super(iLayer, self).__init__()
        self.w = nn.Parameter(torch.randn(1))

    def forward(self, x):
        w = self.w.expand_as(x)
        return x * w


class DenseNet(nn.Module):
    def __init__(self, growth_rate=12, num_init_features=32, bn_size=4,
                 drop_rate=0.2, nb_flows=1):
        super(DenseNet, self).__init__()
        self.features = nn.Sequential(OrderedDict([
            ('conv0', nn.Conv2d(6, num_init_features, kernel_size=3, padding=1, bias=False)),
        ]))
        # Dense Block
        num_features = num_init_features
        num_layers = 6
        block = _DenseBlock(num_layers=num_layers, num_input_features=num_features,
                            bn_size=bn_size, growth_rate=growth_rate, drop_rate=drop_rate)
        self.features.add_module('denseblock', block)
        num_features = num_features + num_layers * growth_rate

        # Final batch norm
        self.features.add_module('norm.last', nn.BatchNorm2d(num_features))

        # change from batch norm to relu
        self.features.add_module('relu.last', nn.ReLU(inplace=True))
        self.features.add_module('conv.last', nn.Conv2d(num_features, nb_flows, kernel_size=1, padding=0, bias=False))

        self.iLayer = iLayer()


        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal(m.weight.data)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()

    def forward(self, inputs):
        out = [self.features(in_var) for in_var in inputs]
        out = [0.0+self.iLayer(f) for f in out]
        out = F.sigmoid(out[0])
        return out

