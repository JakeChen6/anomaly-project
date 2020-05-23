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


# anomaly setting

NAME = 'firm_age_momentum'

START = 1983
END = 2018
LOOK_BACK = [11]
HOLDING = [1]


#%%


# constraints

"""
1983 - 2001
NYSE, AMEX, NASDAQ
Common stocks
Exclude if price < $5
Exclude firms with less than 12 months of past return data
"""

EXCH_CODE = [1, 2, 3]  # NYSE, AMEX, NASDAQ
COMMON_STOCK_CD = [10, 11]  # only common stocks
PRICE_LIMIT = 5. # Exclude if price < $5
HISTORY_REQUIREMENT = 12  # exclude firms with less than 12 months of past return data


#%%


# read data

MSF = pd.read_hdf(DIR + '/data/msf.h5', key='msf')

with open(DIR + '/data/common_stock_permno.pkl', 'rb') as f:
    COMMON_STOCK_PERMNO = pk.load(f)

DATE_RANGE = MSF.DATE.unique()
DATE_RANGE.sort()


#%%


# signal calculation algorithm

# returns from month t-11 to t-1

CUT = 5

def calc_cum_ret(m, lb):
    """
    m: np.datetime64
    lb: int
    """
    # past lb months
    start, end = DATE_RANGE[DATE_RANGE < m][[-lb, -1]]

    data = MSF[MSF.DATE == end]  # data of month m-1
    # apply constraints
    data = data[data.HEXCD.isin(EXCH_CODE)]  # exchange constraint
    data = data[data.PRC.abs() >= PRICE_LIMIT]  # price constraint

    common_stock_permno = COMMON_STOCK_PERMNO[end]  # PERMNO of common stocks according to information on 'end'
    data = data[data.PERMNO.isin(common_stock_permno)]  # common stock constraint

    past_data = MSF[(MSF.PERMNO.isin(data.PERMNO.values)) & (MSF.DATE <= end)]
    ret_data_months = past_data.groupby('PERMNO').RET.count()  # compute months of return data, excluding missing values
    ret_data_months = ret_data_months[ret_data_months >= HISTORY_REQUIREMENT]  # history constraint

    # focus on "high uncertainty" stocks
    cut = pd.qcut(ret_data_months.rank(method='first'), CUT, labels=False)
    eligible = cut[cut == 0].index  # stocks with the least firm age

    # get data in the past lb months for the eligible stocks
    data = MSF[(MSF.DATE >= start) & (MSF.DATE <= end) & (MSF.PERMNO.isin(eligible))]

    # cumulative returns in the past lb months
    data = data.set_index('PERMNO')
    cum_rets = (data.RET + 1).groupby(level=0).prod(min_count=1)
    cum_rets.dropna(inplace=True)

    return cum_rets


def calc_signals(args):
    sub_range, lb = args
    signals = {m: calc_cum_ret(m, lb) for m in sub_range}
    return signals


#%%


# distribute computations to multiple CPUs

CPU_COUNT = 8

start = DATE_RANGE[DATE_RANGE >= np.datetime64(f'{START}-01-01')][0]
end = DATE_RANGE[DATE_RANGE <= np.datetime64(f'{END}-12-31')][-1]

collector = {}

for lb in LOOK_BACK:
    hd = max(HOLDING)
    print(f'\nCalculating ({lb}, {hd}) strategy...', end='\t')

    # on this date we calculate the first set of signals
    first_date = DATE_RANGE[DATE_RANGE <= start][-hd]
    # calculate signals for every month in this range
    date_range = DATE_RANGE[(DATE_RANGE >= first_date) & (DATE_RANGE <= end)]
    
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

