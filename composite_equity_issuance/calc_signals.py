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

NAME = 'composite_equity_issuance'

START = 1968
END = 2018
LOOK_BACK = [12, 12*5]
HOLDING = [12]


#%%


# constraints

"""
July 1968 - December 2003
NYSE, AMEX, NASDAQ
Common stocks
Exclude if price < $5
"""
START = DATE_RANGE[DATE_RANGE >= np.datetime64(f'{START}-07-01')][0]
END = DATE_RANGE[DATE_RANGE <= np.datetime64(f'{END}-12-31')][-1]

EXCH_CODE = [1, 2, 3]  # NYSE, AMEX, NASDAQ
COMMON_STOCK_CD = [10, 11]  # only common stocks
PRICE_LIMIT = 5. # Exclude if price < $5


#%%


# signal calculation algorithm

# composite equity issuance: annual growth in market equity minus the stock return,
# computed by subtracting the 12-month cumulative stock return from the 12-month
# growth in market capitalization.

# nominal or log

def calc_composite_equity_issuance(month, lb):
    """
    Use information before 'month' (including 'month') to calculate the signals.

    month : np.datetime64
    lb : int
    """
    # past lb months
    start, end = DATE_RANGE[DATE_RANGE <= month][[-lb, -1]]

    data = MSF[MSF.DATE == end]
    # apply constraints
    data = data[data.HEXCD.isin(EXCH_CODE)]  # exchange constraint
    data = data[data.PRC.abs() >= PRICE_LIMIT]  # price constraint

    common_stock_permno = COMMON_STOCK_PERMNO[end]  # PERMNO of common stocks
    data = data[data.PERMNO.isin(common_stock_permno)]  # common stock constraint

    # get data in the past lb months for the eligible stocks
    eligible = data.PERMNO.values
    data = MSF[
        (MSF.DATE >= start) &
        (MSF.DATE <= end) &
        (MSF.PERMNO.isin(eligible))
        ]

    # calculate signals
    data = data.sort_values('DATE')
    data['mktcap'] = data.PRC.abs() * data.SHROUT  # market cap
    com_eqy_iss = data.groupby('PERMNO').apply(
        lambda df: np.log(df.iloc[-1].mktcap / df.iloc[0].mktcap) -
                   np.log((df.RET+1).prod(min_count=1))
                   )
    com_eqy_iss.dropna(inplace=True)

    com_eqy_iss.name = 'COM'

    return com_eqy_iss


def calc_signals(args):
    sub_range, lb = args
    # calculate signals at the beginning of every July
    signals = {m: calc_composite_equity_issuance(m, lb) for m in sub_range if pd.Timestamp(m).month == 6}
    return signals


#%%


# distribute computations to multiple CPUs

CPU_COUNT = 8

collector = {}

for lb in LOOK_BACK:
    hd = max(HOLDING)
    print(f'\nCalculating ({lb}, {hd}) strategy...', end='\t')

    # on this date we calculate the first set of signals
    first_date = DATE_RANGE[DATE_RANGE < START][-hd]
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

    table = table.reindex(columns=['DATE', 'PERMNO', 'COM'])
    table.sort_values(by='DATE', inplace=True)
    table.to_csv(path + f'/{lb}.csv')
    print(f'{lb} done.')

