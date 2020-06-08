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

NAME = 'continuing_overreaction'

START = 1965
END = 2018
LOOK_BACK = [12]
HOLDING = [3, 6, 9, 12]


#%%


# constraints

"""
1965 - 2009
NYSE, AMEX, NASDAQ / NYSE, AMEX
Common stocks
Exclude if after dropna, length < 12
Exclude if price < $1 / $5
"""
START = DATE_RANGE[DATE_RANGE >= np.datetime64(f'{START}-01-01')][0]
END = DATE_RANGE[DATE_RANGE <= np.datetime64(f'{END}-12-31')][-1]

EXCH_CODE = [1, 2, 3]  # NYSE, AMEX, NASDAQ
COMMON_STOCK_CD = [10, 11]  # only common stocks
HISTORY_LIMIT = 12  # Exclude if after dropna, length < 12
PRICE_LIMIT = 1. # Exclude if price < $1
MONTH_GAP = 1  # impose an one-month gap between the formation period and the holding period


#%%


# signal calculation algorithm

# normalized weighted sum of the signed monthly trading volumes during the past
# 12 months

DOLLAR_VOL = {}  # to store the calculated dollar volumes
#with open(DIR + f'/anomaly-project/{NAME}/dollar_volmes.pkl', 'rb') as f:
#    DOLLAR_VOL = pk.load(f)

MONTHBEGIN = pd.tseries.offsets.MonthBegin()
WEIGHTS = np.arange(1, 12+1)  # the more recent, the more weight assigned


def helper_to_sign(x):
    if x == 0:
        return 0
    elif x > 0:
        return 1
    else:
        return -1


def calc_co_measure(month, lb):
    """
    Use information before 'month' (including 'month') to calculate the signals.

    month: np.datetime64
    lb: int
    """
    ts = time.time()

    # past lb months
    start, end = DATE_RANGE[DATE_RANGE <= month][[-lb, -1]]

    data = MSF[MSF.DATE == end]  # get last available data to apply constraints
    # apply constraints
    data = data[data.HEXCD.isin(EXCH_CODE)]  # exchange constraint
    data = data[data.PRC.abs() >= PRICE_LIMIT]  # price constraint (prices adjusted)

    common_stock_permno = COMMON_STOCK_PERMNO[end]  # PERMNO of common stocks according to information on 'end'
    data = data[data.PERMNO.isin(common_stock_permno)]  # common stock constraint

    past_data = MSF[  # past 12 months' data
        (MSF.DATE >= start) &
        (MSF.DATE <= end) &
        (MSF.PERMNO.isin(data.PERMNO.values))
        ]
    history = past_data.groupby('PERMNO').apply(
        lambda df: df.dropna(subset=['RET', 'VOL', 'PRC']).shape[0])
    eligible = history[history == 12].index  # history constraint

    # calculate normalized weighted sum of the signed monthly trading volumes

    # first, calculate the dolalr volumes for the past 12 months.
    past_12_months = DATE_RANGE[DATE_RANGE <= month][-lb:]

    for month in past_12_months:
        if month not in DOLLAR_VOL:
            month_begin = MONTHBEGIN.rollback(month).to_datetime64()
            # get data in month m-i for ALL stocks because the calculated result will be reused
            data = DSF[
                (DSF.DATE >= month_begin) &
                (DSF.DATE <= month)
                ]
            data = data[['PERMNO', 'VOL', 'PRC']].to_pandas_df()
            dollar_vol = data.groupby('PERMNO').apply(lambda df: (df.PRC.abs()*df.VOL).sum())
            DOLLAR_VOL[month] = dollar_vol

    # then, calculate the normalized weighted sum of the signed volumes for the eligible stocks
    past_data = past_data[past_data.PERMNO.isin(eligible)]
    past_data = past_data.set_index('PERMNO')
    past_data.sort_values('DATE', inplace=True)

    co_measure = {}
    for p in eligible:
        dollar_vol = [DOLLAR_VOL[month].loc[p] for month in past_12_months]
        rets = past_data.loc[p].RET.values
        signs = list(map(helper_to_sign, rets))
        signed_vol = np.multiply(dollar_vol, signs)

        co = np.dot(WEIGHTS, signed_vol) / np.mean(dollar_vol)
        co_measure[p] = co

    # save in a pd.Series
    co_measure = pd.Series(co_measure)
    co_measure.dropna(inplace=True)
    co_measure.name = 'CO'
    co_measure.index.name = 'PERMNO'

    te = time.time()
    print('{}, {:.2f}s'.format(month, te - ts))

    return co_measure


def calc_signals(args):
    sub_range, lb = args
    signals = {m: calc_co_measure(m, lb) for m in sub_range}
    return signals


#%%

# single process

collector = {}

for lb in LOOK_BACK:
    hd = max(HOLDING)
    print(f'\nCalculating ({lb}, {hd}) strategy...', end='\t')

    # on this date we calculate the first set of signals
    first_date = DATE_RANGE[DATE_RANGE < START][-hd-MONTH_GAP]  # an one-month gap imposed
    # on this date we calculate the last set of signals
    last_date = DATE_RANGE[DATE_RANGE < END][-1]
    # calculate signals for every month in this range
    date_range = DATE_RANGE[
        (DATE_RANGE >= first_date) &
        (DATE_RANGE <= last_date)
        ]

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

    table = table.reindex(columns=['DATE', 'PERMNO', 'CO'])
    table.sort_values(by='DATE', inplace=True)
    table.to_csv(path + f'/{lb}.csv')
    print(f'{lb} done.')

