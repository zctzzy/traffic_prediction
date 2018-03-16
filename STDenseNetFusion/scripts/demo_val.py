# -*- coding: utf-8 -*-
"""
/*******************************************
** This is a file created by Chuanting Zhang
** Name: demo_val
** Date: 1/16/18
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
import pandas as pd
import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader
from torch.autograd import Variable
from torch.utils.data.sampler import SubsetRandomSampler
sys.path.append('../../')
from STDenseNetFusion.dataloader.milano_crop import load_data
from STDenseNetFusion.models.DenseNet_v2 import DenseNet

torch.manual_seed(22)

parse = argparse.ArgumentParser()
parse.add_argument('-height', type=int, default=100)
parse.add_argument('-width', type=int, default=100)
parse.add_argument('-traffic', type=str, default='call')
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
parse.set_defaults(train=True)
parse.add_argument('-rows', nargs='+', type=int, default=[40, 60])
parse.add_argument('-cols', nargs='+', type=int, default=[40, 60])
parse.add_argument('-loss', type=str, default='l2', help='l1 | l2')
parse.add_argument('-lr', type=float, default=0.01)
parse.add_argument('-batch_size', type=int, default=32, help='batch size')
parse.add_argument('-epoch_size', type=int, default=100, help='epochs')
parse.add_argument('-test_row', type=int, default=51, help='test row')
parse.add_argument('-test_col', type=int, default=60, help='test col')

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


def log(fname, s):
    if not os.path.isdir(os.path.dirname(fname)):
        os.system("mkdir -p " + os.path.dirname(fname))
    f = open(fname, 'a')
    f.write(str(datetime.now()) + ': ' + s + '\n')
    f.close()


def set_lr(optimizer, epoch, n_epochs, lr):
    lr = lr
    if float(epoch) / n_epochs > 0.75:
        lr = lr * 0.01
    if float(epoch) / n_epochs > 0.5:
        lr = lr * 0.1

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    return lr


def train_epoch(data_type='train'):
    total_loss = 0
    if data_type == 'train':
        model.train()
        data = train_loader
    if data_type == 'valid':
        model.eval()
        data = valid_loader

    if (opt.period_size > 0) & (opt.close_size > 0) & (opt.trend_size > 0):
        for idx, (c, p, t, target) in enumerate(data):
            optimizer.zero_grad()
            model.zero_grad()
            input_var = [Variable(_.float()).cuda(async=True) for _ in [c, p, t]]
            target_var = Variable(target.float(), requires_grad=False).cuda(async=True)

            pred = model(input_var)
            loss = criterion(pred, target_var)
            total_loss += loss.data[0]
            loss.backward()
            optimizer.step()
    elif (opt.close_size > 0) & (opt.period_size > 0):
        for idx, (c, p, target) in enumerate(data):
            optimizer.zero_grad()
            model.zero_grad()
            input_var = [Variable(_.float()).cuda(async=True) for _ in [c, p]]
            target_var = Variable(target.float(), requires_grad=False).cuda(async=True)

            pred = model(input_var)
            loss = criterion(pred, target_var)
            total_loss += loss.data[0]
            loss.backward()
            optimizer.step()
    elif opt.close_size > 0:
        for idx, (c, target) in enumerate(data):
            optimizer.zero_grad()
            model.zero_grad()
            x = Variable(c.float()).cuda(async=True)
            y = Variable(target.float(), requires_grad=False).cuda(async=True)

            pred = model(x)
            loss = criterion(pred, y)
            total_loss += loss.data[0]
            loss.backward()
            optimizer.step()

    return total_loss


def train():
    os.system("mkdir -p " + opt.save_dir)
    best_valid_loss = 1.0
    train_loss, valid_loss = [], []
    for i in range(opt.epoch_size):
        lr = set_lr(optimizer, i, opt.epoch_size, opt.lr)
        train_loss.append(train_epoch('train'))
        valid_loss.append(train_epoch('valid'))

        if valid_loss[-1] < best_valid_loss:
            best_valid_loss = valid_loss[-1]

            torch.save({'epoch': i, 'model': model, 'train_loss': train_loss,
                        'valid_loss': valid_loss}, opt.model_filename + '.model')
            torch.save(optimizer, opt.model_filename + '.optim')

        log_string = ('iter: [{:d}/{:d}], train_loss: {:0.6f}, valid_loss: {:0.6f}, '
                      'best_valid_loss: {:0.6f}, lr: {:0.5f}').format((i + 1), opt.epoch_size,
                                                                      train_loss[-1],
                                                                      valid_loss[-1],
                                                                      best_valid_loss,
                                                                      opt.lr)
        print(log_string)
        log(opt.model_filename + '.log', log_string)
    loss_data = pd.DataFrame(train_loss)
    loss_data.to_csv('/home/dl/ct/val_result_{}_{}_loss_100.csv'.format(str(opt.test_row),
                                                                    str(opt.test_col)), index=False)


def predict(test_type='train'):
    predictions = []
    ground_truth = []
    loss = []
    best_model = torch.load(opt.model_filename + '.model').get('model')
    # best_model = model

    if test_type == 'train':
        data = train_loader
    elif test_type == 'test':
        data = test_loader
    elif test_type == 'valid':
        data = valid_loader

    if (opt.period_size > 0) & (opt.close_size > 0) & (opt.trend_size > 0):
        for idx, (c, p, t, target) in enumerate(data):
            input_var = [Variable(_.float()).cuda(async=True) for _ in [c, p, t]]
            target_var = Variable(target.float(), requires_grad=False).cuda(async=True)
            pred = best_model(input_var)
            predictions.append(pred.data.cpu().numpy())
            ground_truth.append(target.numpy())
            loss.append(criterion(pred, target_var).data[0])
    elif (opt.close_size > 0) & (opt.period_size > 0):
        for idx, (c, p, target) in enumerate(data):
            input_var = [Variable(_.float()).cuda(async=True) for _ in [c, p]]
            target_var = Variable(target.float(), requires_grad=False).cuda(async=True)
            pred = best_model(input_var)
            predictions.append(pred.data.cpu().numpy())
            ground_truth.append(target.numpy())
            loss.append(criterion(pred, target_var).data[0])
    elif opt.close_size > 0:
        for idx, (c, target) in enumerate(data):
            input_var = Variable(c.float()).cuda(async=True)
            target_var = Variable(target.float(), requires_grad=False).cuda(async=True)
            pred = best_model(input_var)
            predictions.append(pred.data.cpu().numpy())
            ground_truth.append(target.numpy())
            loss.append(criterion(pred, target_var).data[0])

    final_predict = np.concatenate(predictions)
    ground_truth = np.concatenate(ground_truth)
    print(final_predict.shape, ground_truth.shape)

    factor = mmn.max - mmn.min
    sklearn_rmse = metrics.mean_squared_error(final_predict.ravel()*factor,
                                              ground_truth.ravel()*factor)
    print(test_type + ' Real RMSE:{:0.5f}'.format(
        sklearn_rmse ** 0.5
    ))

    a = mmn.inverse_transform(ground_truth)
    b = mmn.inverse_transform(final_predict)
    print('RMSE:{:0.5f}'.format(metrics.mean_squared_error(a.ravel(), b.ravel()) ** 0.5))

    if opt.test_row & opt.test_col:
        row, col = opt.test_row, opt.test_col
    else:
        row_length, col_length = ground_truth.shape[-2:]
        row, col = int(row_length/2), int(col_length/2)
    plt.figure()
    plt.plot(final_predict[:, 0, row, col] * (mmn.max - mmn.min), 'r-', label='Predicted')
    plt.plot(ground_truth[:, 0, row, col] * (mmn.max - mmn.min), 'k-', label='Ground Truth')
    plt.legend(loc='upper right')


    plot_data_in = pd.DataFrame({'predicted':final_predict[:, 0, row, col] * (mmn.max - mmn.min),
                              'groudtruth':ground_truth[:, 0, row, col] * (mmn.max - mmn.min)})
    plot_data_in.to_csv('/home/dl/ct/val_result_{}_{}_in.csv'.format(str(opt.test_row),
                                                           str(opt.test_col)), index=False)

    plot_data_out = pd.DataFrame({'predicted': final_predict[:, 1, row, col] * (mmn.max - mmn.min),
                                 'groudtruth': ground_truth[:, 1, row, col] * (mmn.max - mmn.min)})
    plot_data_out.to_csv('/home/dl/ct/val_result_{}_{}_out.csv'.format(str(opt.test_row),
                                                                  str(opt.test_col)), index=False)

    # train_loss_data = torch.load(opt.model_filename + '.model').get('train_loss')
    # valid_loss_data = torch.load(opt.model_filename + '.model').get('valid_loss')
    # plt.figure()
    # plt.plot(train_loss_data)
    # plot_data_loss = pd.DataFrame(train_loss_data)
    # plot_data_loss.to_csv('/home/dl/ct/val_result_{}_{}_loss.csv'.format(str(opt.test_row),
    #                                                                    str(opt.test_col)), index=False)
    # plt.show()


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

    model = DenseNet(nb_flows=opt.nb_flow, drop_rate=0.2).cuda()
    optimizer = optim.Adam(model.parameters(), opt.lr)
    # optimizer = optim.SGD(model.parameters(), lr=opt.lr, momentum=0.9)
    print(model)

    if not os.path.exists(opt.save_dir):
        os.makedirs(opt.save_dir)
    if not os.path.isdir(opt.save_dir):
        raise Exception('%s is not a dir' % opt.save_dir)

    if opt.loss == 'l1':
        criterion = nn.L1Loss().cuda()
    elif opt.loss == 'l2':
        criterion = nn.MSELoss().cuda()

    print('Training...')
    log(opt.model_filename + '.log', '[training]')
    if opt.train:
        train()
    predict('train')
#    predict('valid')
    predict('test')
    plt.figure()
    plt.plot(torch.load(opt.model_filename + '.model').get('train_loss')[1:-1], 'rs-')
    loss_data = pd.DataFrame(torch.load(opt.model_filename + '.model').get('train_loss')[1:-1])
    loss_data.to_csv('/home/dl/ct/val_result_{}_{}_loss.csv'.format(str(opt.test_row),
                                                                    str(opt.test_col)), index=False)
    # plt.figure()
    # plt.plot(torch.load(opt.model_filename + '.model').get('valid_loss')[1:-1], 'k-')

    plt.show()
