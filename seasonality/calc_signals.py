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

NAME = 'seasonality'

START = 1965
END = 2018
LOOK_BACK = [12*20]  # 20 years
HOLDING = [1]


#%%


# constraints

"""
1965 - 2002
NYSE, AMEX
Common stocks
Exclude if price < $5
At least one year of data to be included in the sample
"""
START = DATE_RANGE[DATE_RANGE >= np.datetime64(f'{START}-01-01')][0]
END = DATE_RANGE[DATE_RANGE <= np.datetime64(f'{END}-12-31')][-1]

EXCH_CODE = [1, 2]  # NYSE, AMEX
COMMON_STOCK_CD = [10, 11]  # only common stocks
PRICE_LIMIT = 5. # Exclude if price < $5
HISTORY_LIMIT = 12  # At least one year of data to be included in the sample


#%%

# signal calculation algorithm

# average monthly return in the same month over the last 20 years.


def calc_avg_monthly_return(month, lb):
    """
    Use information before 'month' (including 'month') to calculate the signals.

    month : np.datetime64
    lb : int
    """
    next_month = DATE_RANGE[DATE_RANGE > month][0]  # next month
    start = DATE_RANGE[DATE_RANGE < next_month][-lb]  # 20 years ago
    cycle = DATE_RANGE[  # the same months in the last 20 years
        (DATE_RANGE >= start) &
        (DATE_RANGE < next_month)
        ][::12]

    data = MSF[MSF.DATE == month]
    # apply constraints
    data = data[data.HEXCD.isin(EXCH_CODE)]  # exchange constraint
    data = data[data.PRC.abs() >= PRICE_LIMIT]  # price constraint

    common_stock_permno = COMMON_STOCK_PERMNO[month]  # PERMNO of common stocks
    data = data[data.PERMNO.isin(common_stock_permno)]  # common stock constraint

    start = DATE_RANGE[DATE_RANGE < next_month][-HISTORY_LIMIT]
    past_data = MSF[
        (MSF.PERMNO.isin(data.PERMNO.values)) &
        (MSF.DATE >= start) &
        (MSF.DATE <= month)
        ]
    ret_data_months = past_data.groupby('PERMNO').RET.count()  # missing values not included in count
    eligible = ret_data_months[ret_data_months == HISTORY_LIMIT].index  # history constraint

    # get data in the same months in the last 20 years for the eligible stocks
    data = MSF[
        (MSF.PERMNO.isin(eligible)) &
        (MSF.DATE.isin(cycle))
        ]
    
    # calculate signals
    avg_monthly_rets = data.groupby('PERMNO').RET.mean()
    avg_monthly_rets.dropna(inplace=True)

    return avg_monthly_rets


def calc_signals(args):
    sub_range, lb = args
    signals = {m: calc_avg_monthly_return(m, lb) for m in sub_range}
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

    table = table.reindex(columns=['DATE', 'PERMNO', 'RET'])
    table.sort_values(by='DATE', inplace=True)
    table.to_csv(path + f'/{lb}.csv')
    print(f'{lb} done.')

