# -*- coding: utf-8 -*-
"""
/*******************************************
** This is a file created by Chuanting Zhang
** Name: __init__.py
** Date: 2017/9/28
** Email: chuanting.zhang@gmail.com
** BSD license
********************************************/
"""

from __future__ import print_function
import h5py
import pandas as pd
from datetime import datetime
import time

def load_stdata(fname):
    f = h5py.File(fname, 'r')
    data = f['data'].value
    timestamps = f['date'].value
    f.close()
    return data, timestamps


def string2timestamp(strings, T=24):
    timestamps = []

    time_per_slot = 24.0 / T
    num_per_T = T // 24
    for t in strings:
        # year, month, day, slot = int(t[:4]), int(t[4:6]), int(t[6:8]), int(t[8:])-1
        year, month, day, hour, minute = int(t[:4]), \
                                         int(t[5:7]), \
                                         int(t[8:10]), \
                                         int(t[11:13]), \
                                         int(t[14:16])
        timestamps.append(
            pd.Timestamp(datetime(
                year,
                month,
                day,
                hour=hour,
                minute=minute)))

    return timestamps
#
#
# def stat(fname):
#     def get_nb_timeslot(f):
#         s = f['date'][0]
#         e = f['date'][-1]
#         year, month, day = map(int, [s[:4], s[4:6], s[6:8]])
#         ts = time.strptime("%04i-%02i-%02i" % (year, month, day), "%Y-%m-%d")
#         year, month, day = map(int, [e[:4], e[4:6], e[6:8]])
#         te = time.strptime("%04i-%02i-%02i" % (year, month, day), "%Y-%m-%d")
#         nb_timeslot = (time.mktime(te) - time.mktime(ts)) / (0.5 * 3600) + 48
#         ts_str, te_str = time.strftime("%Y-%m-%d", ts), time.strftime("%Y-%m-%d", te)
#         return nb_timeslot, ts_str, te_str
#
#     with h5py.File(fname) as f:
#         nb_timeslot, ts_str, te_str = get_nb_timeslot(f)
#         nb_day = int(nb_timeslot / 48)
#         mmax = f['data'].value.max()
#         mmin = f['data'].value.min()
#         stat = '=' * 5 + 'stat' + '=' * 5 + '\n' + \
#                'data shape: %s\n' % str(f['data'].shape) + \
#                '# of days: %i, from %s to %s\n' % (nb_day, ts_str, te_str) + \
#                '# of timeslots: %i\n' % int(nb_timeslot) + \
#                '# of timeslots (available): %i\n' % f['date'].shape[0] + \
#                'missing ratio of timeslots: %.1f%%\n' % ((1. - float(f['date'].shape[0] / nb_timeslot)) * 100) + \
#                'max: %.3f, min: %.3f\n' % (mmax, mmin) + \
#                '=' * 5 + 'stat' + '=' * 5
#         print(stat)