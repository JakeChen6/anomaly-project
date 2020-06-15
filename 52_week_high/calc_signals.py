#!/usr/bin/env python
# coding: utf-8

#%%


import os
import time
import pickle as pk

import numpy as np
import pandas as pd
import vaex

DIR = '/Users/zhishe/myProjects/anomaly'


#%%


# read data

DSF = vaex.open(DIR + '/data/Stock/h5/dsf.h5')

with open(DIR + '/data/common_stock_permno.pkl', 'rb') as f:
    COMMON_STOCK_PERMNO = pk.load(f)

MSF = pd.read_hdf(DIR + '/data/msf.h5', key='msf')
DATE_RANGE = MSF.DATE.unique()
DATE_RANGE.sort()


#%%


# anomaly setting

NAME = '52_week_high'

START = 1963
END = 2018
LOOK_BACK = [12]
HOLDING = [6]


#%%


# constraints

"""
July 1963 - December 2001
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

# ratio of current price to the highest price during the past 12 months

def calc_ratio(month, lb):
    """
    Use information before 'month' (including 'month') to calculate the signals.

    month: np.datetime64
    lb: int
    """
    ts = time.time()

    # start of the rolling window
    start = DATE_RANGE[DATE_RANGE <= month][-lb]
    start = pd.tseries.offsets.MonthBegin().rollback(start).to_datetime64()

    data = MSF[MSF.DATE == month]
    # apply constraints
    data = data[data.HEXCD.isin(EXCH_CODE)]  # exchange constraint
    data = data[data.PRC.abs() >= PRICE_LIMIT]  # price constraint

    common_stock_permno = COMMON_STOCK_PERMNO[month]  # PERMNO of common stocks according to information on 'end'
    data = data[data.PERMNO.isin(common_stock_permno)]  # common stock constraint

    # get data in the past for the eligible stocks
    eligible = data.PERMNO.values
    dsf_data = DSF[(DSF.PERMNO.isin(eligible)) &
                   (DSF.DATE >= start) &
                   (DSF.DATE <= month)].to_pandas_df()

    # calculate 52 week high for all the stocks
    dsf_data['ASKHI'] = dsf_data.ASKHI.abs()
    _52_week_high = dsf_data.groupby('PERMNO').ASKHI.max()
    _52_week_high.dropna(inplace=True)  # drop nan
    _52_week_high = _52_week_high[_52_week_high != 0]  # exclude 0

    # ratio of current price to the highest price during the past 12 months
    data = data.set_index('PERMNO')
    data.dropna(subset=['PRC'], inplace=True)
    ratios = data.PRC.abs() / _52_week_high.loc[data.index]  # current price scaled by the highest price
    ratios.dropna(inplace=True)
    ratios.name = 'RATIO'

    te = time.time()
    print('{}, {:.2f}s'.format(month, te - ts))

    return ratios


def calc_signals(args):
    sub_range, lb = args
    signals = {m: calc_ratio(m, lb) for m in sub_range}
    return signals


#%%


# single process

collector = {}

for lb in LOOK_BACK:
    hd = max(HOLDING)
    print(f'\nCalculating ({lb}, {hd}) strategy...', end='\t')

    # on this date we calculate the first set of signals
    first_date = DATE_RANGE[DATE_RANGE < START][-hd]
    # on this date we calculate the last set of signals
    last_date = DATE_RANGE[DATE_RANGE < END][-1]
    # calculate signals for every month in this range
    date_range = DATE_RANGE[(DATE_RANGE >= first_date) &
                            (DATE_RANGE <= last_date)]

    signals = calc_signals((date_range, lb))
    collector.setdefault(lb, {}).update(signals)


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

    table = table.reindex(columns=['DATE', 'PERMNO', 'RATIO'])
    table.sort_values(by='DATE', inplace=True)
    table.to_csv(path + f'/{lb}.csv')
    print(f'{lb} done.')

