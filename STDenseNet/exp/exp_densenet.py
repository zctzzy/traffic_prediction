# -*- coding: utf-8 -*-
"""
/*******************************************
** This is a file created by Chuanting Zhang
** Name: exp_densenet
** Date: 2017/9/28
** Email: chuanting.zhang@gmail.com
** BSD license
********************************************/
"""

import argparse
import numpy as np
import os
from STDenseNet.data.milano import load_data
from STDenseNet.models.ConvNet import seqCNN_CPT
from STDenseNet.models.ResNet import resnet
from STDenseNet.models.DenseNet import densenet
from STDenseNet.models import metrics, iLayer
from keras.optimizers import SGD, Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint, LearningRateScheduler
import pickle
import h5py
from sklearn.metrics import mean_squared_error, mean_absolute_error
from keras import models
np.random.seed(2018)


epochs = 100
cont_epochs = 50

h, w = 100, 100
nb_cells = 10000
m_factors = np.sqrt(h*w / nb_cells)  # 1.32055

path = '../../all_data_ct.h5'
# path = path = '/home/wmct/ct/sms-in-out.hdf5'


def run_trentino(slots, depth, closeness_size, period_size,
                 trend_size, days_test, height, width,
                 nb_flows, nb_net_units, lr, batch_size):

    len_test = days_test * slots
    x_train, y_train, x_test, y_test, mmn, timestamp_train, \
    timestamp_test = load_data(path, slots,
                               nb_flows,
                               closeness_size,
                               period_size,
                               trend_size,
                               len_test, height, width)

    path_model = 'results'
    if os.path.isdir(path_model) is False:
        os.makedirs(path_model)
    c_conf = (closeness_size, nb_flows, height, width) if closeness_size > 0 else None
    p_conf = (period_size, nb_flows, height, width) if period_size > 0 else None
    t_conf = (trend_size, nb_flows, height, width) if trend_size > 0 else None
    print("Building model...")

    # model = seqCNN_CPT(depth, nb_net_units, 4, 4, c_conf, p_conf, t_conf, 0.2)
    # model = seqCNN_CPT(c_conf, p_conf, t_conf)
    # model = resnet(c_conf, p_conf, t_conf, 3)
    model = densenet(depth, 32, c_conf, p_conf, t_conf, 0.2)

    adam = Adam(lr=lr, decay=lr/float(epochs))
    # sgd = SGD(lr=lr, momentum=0.9)
    model.compile(loss='mse', optimizer=adam, metrics=[metrics.rmse])
    model.summary()

    hyper_parameters = 'c{}.p{}.t{}.blocks{}.lr{}.test{}'.format(
        closeness_size, period_size, trend_size, nb_net_units, lr, days_test
    )

    file_name = os.path.join('results', '{}.best.h5'.format(hyper_parameters))

    early_stopping = EarlyStopping(monitor='val_rmse', patience=3, mode='min')
    model_checkpoint = ModelCheckpoint(file_name, monitor='val_rmse',
                                       verbose=0, save_best_only=True, mode='min')

    print("*" * 20)
    print("Training model...")
    history = model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs,
                        validation_split=0.1,
                        callbacks=[early_stopping, model_checkpoint],
                        verbose=1)

    model.save_weights(os.path.join('results', '{}.h5'.format(hyper_parameters)),
                       overwrite=True)
    pickle.dump((history.history), open(os.path.join(
        'results', '{}.history.pkl'.format(hyper_parameters)), 'wb')
                )

    print("*" * 20)
    print("Evaluating using the best model...")
    model.load_weights(file_name)
    score = model.evaluate(x_train, y_train, batch_size=batch_size, verbose=0)
    print("Train score: %.6f rmse (norm): %.6f rmse (real): %.6f" % (
        score[0], score[1], score[1] * (mmn._max - mmn._min) * m_factors
    ))

    score = model.evaluate(x_test, y_test, batch_size=batch_size, verbose=0)
    print("Test score: %.6f rmse (norm): %.6f rmse (real): %.6f" % (
        score[0], score[1], score[1] * (mmn._max - mmn._min) * m_factors
    ))

    print('=' * 10)
    print("training model (cont)...")
    fname_param = os.path.join(
        'results', '{}.cont.best.h5'.format(hyper_parameters))
    model_checkpoint = ModelCheckpoint(
        fname_param, monitor='val_rmse', verbose=0, save_best_only=True, mode='min')
    history = model.fit(x_train, y_train, epochs=cont_epochs, verbose=1, batch_size=batch_size,
                        callbacks=[early_stopping, model_checkpoint], validation_data=(x_test, y_test))
    pickle.dump((history.history), open(os.path.join(
        'results', '{}.cont.history.pkl'.format(hyper_parameters)), 'wb'))
    model.save_weights(os.path.join(
        'results', '{}.cont.h5'.format(hyper_parameters)), overwrite=True)

    print('=' * 10)
    print('evaluating using the final model')
    score = model.evaluate(x_train, y_train, batch_size=batch_size, verbose=0)
    print('Train score: %.6f rmse (norm): %.6f rmse (real): %.6f' %
          (score[0], score[1], score[1] * (mmn._max - mmn._min) * m_factors))

    score = model.evaluate(
        x_test, y_test, batch_size=batch_size, verbose=0)
    print('Test score: %.6f rmse (norm): %.6f rmse (real): %.6f' %
          (score[0], score[1], score[1] * (mmn._max - mmn._min) * m_factors))

    model.load_weights(fname_param)

    y_pred = model.predict(x_test, batch_size=batch_size, verbose=1)
    root_mse = mean_squared_error(y_test.ravel(), y_pred.ravel()) ** 0.5
    root_mae = mean_absolute_error(y_test.ravel(), y_pred.ravel())
    print root_mse, root_mae
    print('Test score (sklearn): %.6f rmse' % (root_mse * mmn._max))
    print('Test score (sklearn): %.6f mae' % (root_mae * mmn._max))

    wf = h5py.File('DenseNet_Prediction_CALL.h5', 'w')
    wf.create_dataset('pred', data=y_pred)
    wf.close()



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run trentino exp...')
    parser.add_argument('--batch_size', default=32, type=int,
                        help='Batch size')

    parser.add_argument('--depth', default=5, type=int,
                        help='depth')

    parser.add_argument('--closeness_size', default=3, type=int,
                        help='Closeness')

    parser.add_argument('--period_size', default=3, type=int,
                        help='Period')

    parser.add_argument('--trend_size', default=0, type=int,
                        help='Trend')

    parser.add_argument('--nb_net_units', default=1, type=int,
                        help='number of units in a block(residual net or dense net)')

    parser.add_argument('--days_test', default=7, type=int,
                        help='days to test performance')

    parser.add_argument('--slots', default=24, type=int,
                        help='number of time intervals in one day')

    parser.add_argument('--height', default=100, type=int,
                        help='image height')

    parser.add_argument('--width', default=100, type=int,
                        help='image width')

    parser.add_argument('--nb_flows', default=2, type=int,
                        help='number of flows')

    parser.add_argument('--lr', default=0.01, type=float,
                        help='learning rate')


    args = parser.parse_args()
    print("Parameters:")
    for key, value in args._get_kwargs():
        print(key, value)

    run_trentino(args.slots,
                 args.depth,
                 args.closeness_size,
                 args.period_size,
                 args.trend_size,
                 args.days_test,
                 args.height,
                 args.width,
                 args.nb_flows,
                 args.nb_net_units,
                 args.lr,
                 args.batch_size)