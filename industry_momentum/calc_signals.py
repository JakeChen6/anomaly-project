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

NAME = 'industry_momentum'

START = 1963
END = 2018
LOOK_BACK = [6]
HOLDING = [6]

# SIC codes
SIC = {'Mining': list(range(10, 15)),
       'Food': 20,
       'Apparel': [22, 23],
       'Paper': 26,
       'Chemical': 28,
       'Petroleum': 29,
       'Construction': 32,
       'Prim. Metals': 33,
       'Fab. Metals': 34,
       'Machinery': 35,
       'Electrical Eq.': 36,
       'Transport Eq.': 37,
       'Manufacturing': [38, 39],
       'Railroads': 40,
       'Other Transport': list(range(41, 48)),
       'Utilities': 49,
       'Dept. Stores': 53,
       'Retail': list(range(50, 53)) + list(range(54, 60)),
       'Financial': list(range(60, 70)),
       'Other': 'other'
       }

# ordered industry names
INDUSTRY = [
    'Mining',
    'Food',
    'Apparel',
    'Paper',
    'Chemical',
    'Petroleum',
    'Construction',
    'Prim. Metals',
    'Fab. Metals',
    'Machinery',
    'Electrical Eq.',
    'Transport Eq.',
    'Manufacturing',
    'Railroads',
    'Other Transport',
    'Utilities',
    'Dept. Stores',
    'Retail',
    'Financial',
    'Other'
    ]


#%%


# constraints

"""
1963 - 1995
NYSE, AMEX, NASDAQ
Common stocks
Exclude if price < $5
"""

EXCH_CODE = [1, 2, 3]  # NYSE, AMEX, NASDAQ
COMMON_STOCK_CD = [10, 11]  # only common stocks
PRICE_LIMIT = 5. # Exclude if price < $5


#%%


# read data

MSF = pd.read_hdf(DIR + '/data/msf.h5', key='msf')

with open(DIR + '/data/common_stock_permno.pkl', 'rb') as f:
    COMMON_STOCK_PERMNO = pk.load(f)

# transform HSICCD to two-digit codes
MSF['HSICCD'] //= 10  # three-digit -> two-digit, four-digit -> three-digit
index = MSF[MSF['HSICCD'] >= 100].index
MSF.loc[index, 'HSICCD'] //= 10  # three-digit -> two-digit

DATE_RANGE = MSF.DATE.unique()
DATE_RANGE.sort()


#%%


# signal calculation algorithm

# value-weighted return, over the past lb months, of the industry that the stock belongs to.

def helper_get_weights(df):
    df = df.copy()
    df['mktcap'] = df.PRC.abs() * df.SHROUT
    df.sort_values('DATE', inplace=True)

    weights = df.groupby(level=0).mktcap.apply(
        lambda s: s.dropna().iloc[-1] if not s.dropna().empty else np.nan)

    weights /= weights.sum()

    return weights


def calc_past_ind_rets(m, lb):
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

    # get data in the past lb months for the eligible stocks
    eligible_permno = data.PERMNO.values
    data = MSF[(MSF.DATE >= start) & (MSF.DATE <= end) & (MSF.PERMNO.isin(eligible_permno))]

    # calculate each industry's value-weighted return
    ind_rets = {}

    for ind in INDUSTRY:
        if ind != 'Other':
            siccd = SIC[ind]
            if isinstance(siccd, list):
                rows = data[data.HSICCD.isin(siccd)]
            else:
                rows = data[data.HSICCD == siccd]
            rows = rows.copy()
            data = data.drop(rows.index)  # what are left finally are stocks in industry 'Other'
        else:
            rows = data

        # cumulative return in the past lb months
        rows = rows.set_index('PERMNO')
        cum_rets = (rows.RET + 1).groupby(level=0).prod(min_count=1)
        cum_rets.dropna(inplace=True)

        # value weight this industry
        rows = rows.loc[cum_rets.index]
        weights = helper_get_weights(rows)
        val_weighted_ret = (cum_rets * weights).sum()
        ind_rets[ind] = {s: val_weighted_ret for s in cum_rets.index}

    # save in a pandas.Series
    d = {k: v for val in ind_rets.values() for k, v in val.items()}
    s = pd.Series(d)
    s.name = 'RET'
    s.index.name = 'PERMNO'
    
    return s


def calc_signals(args):
    sub_range, lb = args
    signals = {m: calc_past_ind_rets(m, lb) for m in sub_range}
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

