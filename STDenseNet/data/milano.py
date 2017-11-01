# -*- coding: utf-8 -*-
"""
/*******************************************
** This is a file created by Chuanting Zhang
** Name: milano
** Date: 2017/9/28
** Email: chuanting.zhang@gmail.com
** BSD license
********************************************/
"""
import numpy as np
import h5py
import pickle
from STDenseNet.data.STMatrix import STMatrix

height, width = 100, 100

class MinMaxNormalization(object):
    '''MinMax Normalization --> [0, 1]
       x = (x - min) / (max - min).
    '''

    def __init__(self):
        pass

    def fit(self, X):
        self._min = X.min()
        self._max = X.max()
        print("min:", self._min, "max:", self._max)

    def transform(self, X):
        X = 1. * (X - self._min) / (self._max - self._min)
        return X

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)

    def inverse_transform(self, X):
        X = 1. * X * (self._max - self._min) + self._min
        return X


class MinMaxNormalization1(object):
    '''MinMax Normalization --> [-1, 1]
       x = (x - min) / (max - min).
       x = x * 2 - 1
    '''

    def __init__(self):
        pass

    def fit(self, X):
        self._min = X.min()
        self._max = X.max()
        print("min:", self._min, "max:", self._max)

    def transform(self, X):
        X = 1. * (X - self._min) / (self._max - self._min)
        X = X * 2. - 1.
        return X

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)

    def inverse_transform(self, X):
        X = (X + 1.) / 2.
        X = 1. * X * (self._max - self._min) + self._min
        return X


def load_data(path, T = 24, nb_flows=2, len_closeness=None,
              len_period=None, len_trend=None, len_test=None,
              height=100, width=100):
    f = h5py.File(path, 'r')
    data = f['data'].value
    index = f['idx'].value

    samples, dimension, flows = data.shape
    tmp_data = []
    for s in range(samples):
        tmp_data.append(np.transpose(data[s]))
    data = np.array(tmp_data).reshape((samples, flows, height, width))
    # data = data.reshape((samples, height, width, flows))

    # data = data[:, :, :, nb_flows]
    data = data[:, 2:4]

    data_all = [data]
    index_all = [index]

    # Normalization
    mms = MinMaxNormalization()

    data_train = data[:-len_test]
    mms.fit(data_train)
    print("Train data shape:", data_train.shape)

    data_all_mms = []
    for d in data_all:
        data_all_mms.append(mms.transform(d))

    fpkl = open('preprocessing.pkl', 'wb')
    for obj in [mms]:
        pickle.dump(obj, fpkl)
    fpkl.close()

    xc, xp, xt = [], [], []
    y = []
    timestamps_y = []


    for data, index in zip(data_all_mms, index_all):
        st = STMatrix(data, index, T)
        _xc, _xp, _xt, _y, _timestamps_y = st.create_dataset(
            len_closeness=len_closeness, len_trend=len_trend,
            TrendInterval=7, len_period=len_period, PeriodInterval=1
        )

        xc.append(_xc)
        xp.append(_xp)
        xt.append(_xt)
        y.append(_y)
        timestamps_y += _timestamps_y


    xc = np.vstack(xc)
    xp = np.vstack(xp)
    xt = np.vstack(xt)
    y = np.vstack(y)

    xc_train, xp_train, xt_train, y_train = xc[:-len_test], xp[:-len_test], xt[:-len_test], y[:-len_test]
    xc_test, xp_test, xt_test, y_test = xc[-len_test:], xp[-len_test:], xt[-len_test:], y[-len_test:]
    timestamps_train, timestamps_test = timestamps_y[:-len_test], timestamps_y[-len_test:]

    x_train = []
    x_test = []

    for l, x_ in zip([len_closeness, len_period, len_trend], [xc_train, xp_train, xt_train]):
        if l > 0:
            x_train.append(x_)

    for l, x_ in zip([len_closeness, len_period, len_trend], [xc_test, xp_test, xt_test]):
        if l > 0:
            x_test.append(x_)

    return x_train, y_train, x_test, y_test, mms, timestamps_train, timestamps_test