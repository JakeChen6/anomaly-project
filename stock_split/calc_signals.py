#!/usr/bin/env python
# coding: utf-8

#%%


import os
import time
import pickle as pk
from concurrent import futures

import numpy as np
import pandas as pd

DIR = '/Users/zhishe/myProjects/anomaly'


#%%

# read data

MSF = pd.read_hdf(DIR + '/data/msf.h5', key='msf')

with open(DIR + '/data/common_stock_permno.pkl', 'rb') as f:
    COMMON_STOCK_PERMNO = pk.load(f)

DATE_RANGE = MSF.DATE.unique()
DATE_RANGE.sort()


#%%


# anomaly setting

NAME = 'stock_split'

START = 1976
END = 2018
LOOK_BACK = [1]
HOLDING = [12, 24, 36]

SPLIT_THRESHOLD = 1.25
REVERSE_SPLIT_THRESHOLD = 1.


#%%


# constraints

"""
1976 - 1991
NYSE, AMEX, NASDAQ
Common stocks
Exclude if price < $5
"""
START = DATE_RANGE[DATE_RANGE >= np.datetime64(f'{START}-01-01')][0]
END = DATE_RANGE[DATE_RANGE <= np.datetime64(f'{END}-12-31')][-1]

EXCH_CODE = [1, 2, 3]  # NYSE, AMEX, NASDAQ
COMMON_STOCK_CD = [10, 11]  # only common stocks
PRICE_LIMIT = 5. # Exclude if price < $5


#%%


# signal calculation algorithm

# an indicator equal to one, if a CRSP firm has a share adjustment factor that
# is greater than 1.25, it is designated as split for 12 months or until another
# split or reverse split occurs. All other observations have a value of zero.


def calc_indicator(month, lb):
    """
    Use information before 'month' (including 'month') to calculate the signals.

    month : np.datetime64
    lb : int
    """
    last_month = DATE_RANGE[DATE_RANGE < month][-1]  # last month

    data = MSF[MSF.DATE == month]
    # apply constraints
    data = data[data.HEXCD.isin(EXCH_CODE)]  # exchange constraint
    data = data[data.PRC.abs() >= PRICE_LIMIT]  # price constraint

    common_stock_permno = COMMON_STOCK_PERMNO[month]  # PERMNO of common stocks
    data = data[data.PERMNO.isin(common_stock_permno)]  # common stock constraint

    # get data in the past 2 months for the eligible stocks
    eligible = data.PERMNO.values
    data = MSF[
        (MSF.DATE >= last_month) &
        (MSF.DATE <= month) &
        (MSF.PERMNO.isin(eligible))
        ]

    # calculate signals
    data = data.sort_values('DATE')
    ratios = data.groupby('PERMNO').apply(lambda df: df.iloc[0].CFACSHR / df.iloc[-1].CFACSHR)
    ratios.dropna(inplace=True)

    split = ratios[ratios > SPLIT_THRESHOLD].index
    reverse_split = ratios[ratios < REVERSE_SPLIT_THRESHOLD].index

    ratios.loc[split] = 1
    ratios.loc[reverse_split] = -1
    ratios.loc[ratios.index.difference(split.union(reverse_split))] = 0

    ratios.name = 'INDICATOR'

    return ratios


def calc_signals(args):
    sub_range, lb = args
    signals = {m: calc_indicator(m, lb) for m in sub_range}
    return signals


#%%


# distribute computations to multiple CPUs

CPU_COUNT = 8

collector = {}

for lb in LOOK_BACK:

    # on this date we calculate the first set of signals
    first_date = DATE_RANGE[DATE_RANGE < START][-1]
    # on this date we calculate the last set of signals
    last_date = DATE_RANGE[DATE_RANGE < END][-1]
    # calculate signals for every month in this range
    date_range = DATE_RANGE[
        (DATE_RANGE >= first_date) &
        (DATE_RANGE <= last_date)
        ]

    # split the workload
    size = len(date_range) // CPU_COUNT
    sub_ranges = []
    for i in range(CPU_COUNT):
        if i != CPU_COUNT-1:
            sub_ranges.append(date_range[size*i:size*(i+1)])
        else:
            sub_ranges.append(date_range[size*i:])

    with futures.ProcessPoolExecutor(max_workers=CPU_COUNT) as ex:
        ts = time.time()
        res = ex.map(calc_signals, zip(sub_ranges, [lb] * CPU_COUNT))
        for signals in res:
            collector.setdefault(lb, {}).update(signals)
        te = time.time()
        print('{:.2f}s'.format(te - ts))


#%%


# save results to local

path = DIR + f'/anomaly-project/{NAME}/signals'
if not os.path.exists(path):
    os.mkdir(path)

for lb in LOOK_BACK:
    table = pd.DataFrame()
    signals = collector[lb]
    # consolidate each month's signals into a single table
    for k, v in signals.items():
        df = pd.DataFrame(v)
        df.reset_index(inplace=True)
        df['DATE'] = k
        table = pd.concat([table, df], ignore_index=True)

    table = table.reindex(columns=['DATE', 'PERMNO', 'INDICATOR'])
    table.sort_values(by='DATE', inplace=True)
    table.to_csv(path + f'/{lb}.csv')
    print(f'{lb} done.')

