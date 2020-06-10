#!/usr/bin/env python
# coding: utf-8

#%%


import os
import time
import pickle as pk

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


#%%


# anomaly setting

NAME = 'long_term_smoothed_beta'

START = 1964
END = 2018
LOOK_BACK = [36]
HOLDING = [1]

PRIOR_WEEKS = 156  # the past period over which data is used to estimate the smoothed market beta
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

# will be used many times, so defined once here.
INDEX = ['RET'] + [str(i) for i in range(1, LAG+1)]


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
    data.index = data.index.shift(-2, 'D')  # so that we can apply 'to_period' below
    data.to_period('W', copy=False)
    data.reset_index(inplace=True)
    data['RET'] += 1  # to calculate cumulative returns
    weekly_rets = data.groupby(['DATE', 'PERMNO']).RET.prod(min_count=1)

    # independent variables: contemporaneous and lagged CRSP equal-weighted market return
    mkt_rets = weekly_rets.mean(level='DATE', skipna=True)
    mkt_rets = pd.DataFrame(mkt_rets)
    for i in range(1, LAG+1):  # shift mkt_rets to get lagged values
        mkt_rets[str(i)] = mkt_rets.RET.shift(i)
    # add the column of ones to the inputs to take into account the intercept
    x = sm.add_constant(mkt_rets)

    # regress to get slope coefficients
    weekly_rets = weekly_rets.reorder_levels(['PERMNO', 'DATE'])
    beta = {}
    for p in weekly_rets.index.levels[0]:
        y = weekly_rets.loc[p].dropna()  # dependent variable, stock's return
        x_ = x.loc[y.index]
        if (not y.size) or (not x_.dropna().size):  # 0 size after dropping nan
            continue
        # create a model and fit it
        # firm-week observations are excluded when weekly returns are missing
        model = sm.OLS(y, x_, missing='drop')
        results = model.fit()
        beta[p] = results.params.loc[INDEX].mean()

    beta = pd.Series(beta)
    beta.name = 'BETA'
    beta.index.name = 'PERMNO'
    beta.dropna(inplace=True)

    te = time.time()
    print('{}, {:.2f}s'.format(month, te - ts))

    return beta


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

