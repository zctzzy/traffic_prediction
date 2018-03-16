# -*- coding: utf-8 -*-
"""
/*******************************************
** This is a file created by Chuanting Zhang
** Name: linear_model
** Date: 1/28/18
** Email: chuanting.zhang@gmail.com
** BSD license
********************************************/
"""

import os
import sys
import argparse
import numpy as np
from datetime import datetime
from sklearn import metrics
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader
from torch.autograd import Variable
from torch.utils.data.sampler import SubsetRandomSampler
sys.path.append('../../')
from STDenseNet.dataloader.milano_crop import load_data
from STDenseNet.models.DenseNet_v2 import DenseNet

torch.manual_seed(22)

parse = argparse.ArgumentParser()
parse.add_argument('-height', type=int, default=100)
parse.add_argument('-width', type=int, default=100)
parse.add_argument('-traffic', type=str, default='sms')
parse.add_argument('-close_size', type=int, default=3)
parse.add_argument('-period_size', type=int, default=3)
parse.add_argument('-trend_size', type=int, default=0)
parse.add_argument('-test_size', type=int, default=24*7)
parse.add_argument('-nb_flow', type=int, default=2)
parse.add_argument('-crop', dest='crop', action='store_true')
parse.add_argument('-no-crop', dest='crop', action='store_false')
parse.set_defaults(crop=False)
parse.add_argument('-train', dest='train', action='store_true')
parse.add_argument('-no-train', dest='train', action='store_false')
parse.set_defaults(training=False)
parse.add_argument('-rows', nargs='+', type=int, default=[40, 60])
parse.add_argument('-cols', nargs='+', type=int, default=[40, 60])
parse.add_argument('-loss', type=str, default='l2', help='l1 | l2')
parse.add_argument('-lr', type=float, default=0.0005)
parse.add_argument('-batch_size', type=int, default=32, help='batch size')
parse.add_argument('-epoch_size', type=int, default=500, help='epochs')
parse.add_argument('-test_row', type=int, default=10, help='test row')
parse.add_argument('-test_col', type=int, default=10, help='test col')

parse.add_argument('-save_dir', type=str, default='results')

opt = parse.parse_args()
print(opt)
opt.save_dir = '{}/{}'.format(opt.save_dir, opt.traffic)
opt.model_filename = '{}/model={}-loss={}-lr={}-close={}-period=' \
                     '{}-trend={}'.format(opt.save_dir,
                                          'densenet',
                                          opt.loss, opt.lr, opt.close_size,
                                          opt.period_size, opt.trend_size)
print('Saving to ' + opt.model_filename)



def train_valid_split(dataloader, test_size=0.2, shuffle=True, random_seed=0):
    length = len(dataloader)
    indices = list(range(0, length))

    if shuffle:
        np.random.seed(random_seed)
        np.random.shuffle(indices)

    if type(test_size) is float:
        split = int(np.floor(test_size * length))
    elif type(test_size) is int:
        split = test_size
    else:
        raise ValueError('%s should be an int or float'.format(str))
    return indices[split:], indices[:split]


if __name__ == '__main__':
    path = '/home/dl/ct/data/all_data_ct.h5'
    x_train, y_train, x_test, y_test, mmn = load_data(path, opt.traffic, opt.close_size, opt.period_size,
                                                      opt.trend_size,
                                                      opt.test_size, opt.nb_flow, opt.height, opt.width, opt.crop,
                                                      opt.rows, opt.cols)
    x_train.append(y_train)
    x_test.append(y_test)
    train_data = zip(*x_train)
    test_data = zip(*x_test)
    print(len(train_data), len(test_data))

    # split the training data into train and validation
    train_idx, valid_idx = train_valid_split(train_data, 0.1)
    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)

    train_loader = DataLoader(train_data, batch_size=opt.batch_size, sampler=train_sampler,
                              num_workers=8, pin_memory=True)
    valid_loader = DataLoader(train_data, batch_size=opt.batch_size, sampler=valid_sampler,
                              num_workers=2, pin_memory=True)

    test_loader = DataLoader(test_data, batch_size=opt.batch_size, shuffle=False)
