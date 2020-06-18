#!/usr/bin/env python
# coding: utf-8

#%%


import os
import time
import pickle as pk
from concurrent import futures

import numpy as np
import pandas as pd
import vaex
import statsmodels.api as sm

DIR = '/Users/zhishe/myProjects/anomaly'


#%%


# read data

DSF = vaex.open(DIR + '/data/Stock/h5/dsf.h5')

with open(DIR + '/data/common_stock_permno.pkl', 'rb') as f:
    COMMON_STOCK_PERMNO = pk.load(f)

MSF = pd.read_hdf(DIR + '/data/msf.h5', key='msf')
DATE_RANGE = MSF.DATE.unique()
DATE_RANGE.sort()

# Fama-French Factors - to get the rf
FF = pd.read_csv(DIR + '/data/factors_daily.csv', index_col=0, parse_dates=True)

DSI = pd.read_csv(DIR + '/data/dsi.csv', index_col=0, parse_dates=True)
DSI = DSI.reindex(index=FF.index).dropna(subset=['ewretd'])
DSI['ewmktrf'] = DSI.ewretd - FF.loc[DSI.index, 'rf']  # EW market excess returns

#%%


# anomaly setting

NAME = 'short_term_smoothed_beta'

START = 1964
END = 2018
LOOK_BACK = [12]
HOLDING = [1]

PRIOR_WEEKS = 52  # the past period over which data is used to estimate the smoothed market beta
LAG = 4  # this determines how many lags of market return in the regression


#%%


# constraints

"""
July 1964 - December 2001
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

# smoothed market beta (estimated over the prior 52 weeks): the average of the
# slope coefficients estimated from a regression of the excess return on the
# contemporaneous and four lags of the CRSP equal-weighted market excess return.

# independent variables in the regression: contemporaneous and lagged
# CRSP value-weighted market excess returns.
MKTRF = DSI[['ewmktrf']].copy()
MKTRF['ewmktrf'] += 1
MKTRF.index = MKTRF.index.shift(-2, 'D')
MKTRF.to_period('W', copy=False)  # to calculate weekly returns
MKTRF = MKTRF.groupby(level=0).ewmktrf.prod(min_count=1) - 1

# shift mktrf to get lagged values
MKTRF = pd.DataFrame(MKTRF)
for i in range(1, LAG+1):
    MKTRF[str(i)] = MKTRF.ewmktrf.shift(i)
# add the column of ones to the inputs to take into account the intercept
MKTRF = sm.add_constant(MKTRF)

# will be used many times, so defined once here.
INDEX = ['ewmktrf'] + [str(i) for i in range(1, LAG+1)]

CPU_COUNT = 8


def calc_smoothed_beta(month, lb):
    """
    Use information before 'month' (including 'month') to calculate the signals.

    month: np.datetime64
    lb: int
    """
    ts = time.time()

    data = MSF[MSF.DATE == month]
    # apply constraints
    data = data[data.HEXCD.isin(EXCH_CODE)]  # exchange constraint
    data = data[data.PRC.abs() >= PRICE_LIMIT]  # price constraint

    common_stock_permno = COMMON_STOCK_PERMNO[month]  # PERMNO of common stocks according to information on 'end'
    data = data[data.PERMNO.isin(common_stock_permno)]  # common stock constraint

    # last Wednesday
    month_isoweekday = pd.Timestamp(month).isoweekday()
    if month_isoweekday == 3:
        last_wednesday = pd.Timestamp(month)
    elif month_isoweekday > 3:
        last_wednesday = pd.Timestamp(month) - pd.Timedelta(days=month_isoweekday-3)
    else:
        last_wednesday = pd.Timestamp(month) - pd.Timedelta(days=month_isoweekday-3+7)

    # first Wednesday
    first_wednesday = last_wednesday - pd.Timedelta(days=7*PRIOR_WEEKS)

    # convert dtype
    first_wednesday = first_wednesday.to_datetime64()
    last_wednesday = last_wednesday.to_datetime64()

    # get data in the past 52 weeks for the eligible stocks
    eligible = data.PERMNO.values
    data = DSF[(DSF.DATE >= first_wednesday) &
               (DSF.DATE < last_wednesday) &  # last Wednesday not included
               (DSF.PERMNO.isin(eligible))]

    # calculate smoothed market beta below

    # calculate weekly returns as the compounded daily returns
    data = data[['PERMNO', 'DATE', 'RET']].to_pandas_df()
    data.set_index('DATE', inplace=True)
    data['RET'] -= FF.loc[data.index, 'rf']  # returns become excess returns
    data['RET'] += 1  # to calculate cumulative returns
    data.index = data.index.shift(-2, 'D')  # in order to apply 'to_period' below
    data.to_period('W', copy=False)
    data.reset_index(inplace=True)
    weekly_rets = data.groupby(['PERMNO', 'DATE']).RET.prod(min_count=1) - 1

    # regress to get slope coefficients
    workloads = weekly_rets.index.levels[0]
    s = len(workloads) // CPU_COUNT  # load size
    to_distribute = [workloads[s*i:s*(i+1)] if i != CPU_COUNT-1 else workloads[s*i:]
                     for i in range(CPU_COUNT)]

    beta = {}
    with futures.ProcessPoolExecutor(max_workers=CPU_COUNT) as ex:
        wait_for = []
        for elm in to_distribute:
            f = ex.submit(helper_regress, weekly_rets, elm)
            wait_for.append(f)
        for f in futures.as_completed(wait_for):
            beta.update(f.result())

    beta = pd.Series(beta)
    beta.name = 'BETA'
    beta.index.name = 'PERMNO'
    beta.dropna(inplace=True)

    te = time.time()
    print('{}, {:.2f}s'.format(month, te - ts))

    return beta


def helper_regress(weekly_rets, permnos):
    res = {}
    for p in permnos:
        y = weekly_rets.loc[p].dropna()  # dependent variable, stock's return
        x = MKTRF.loc[y.index]
        if (not y.size) or (not x.dropna().size):  # 0 size after dropping nan
                continue
        # create a model and fit it
        # firm-week observations are excluded when weekly returns are missing
        model = sm.OLS(y, x, missing='drop')
        results = model.fit()
        res[p] = results.params.loc[INDEX].mean()
    return res


def calc_signals(args):
    sub_range, lb = args
    signals = {m: calc_smoothed_beta(m, lb) for m in sub_range}
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

    table = table.reindex(columns=['DATE', 'PERMNO', 'BETA'])
    table.sort_values(by='DATE', inplace=True)
    table.to_csv(path + f'/{lb}.csv')
    print(f'{lb} done.')

