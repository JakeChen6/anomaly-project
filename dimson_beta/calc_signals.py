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

# Fama-French Factors
FF = pd.read_csv(DIR + '/data/factors_daily.csv', index_col=0, parse_dates=True)


#%%


# anomaly setting

NAME = 'dimson_beta'

START = 1967
END = 2018
LOOK_BACK = [1]
HOLDING = [1, 6, 12]


#%%


# constraints

"""
1967 - 2016
NYSE, AMEX, NASDAQ
Common stocks
Exclude if price < $5
at least 15 daily returns in the month
"""
START = DATE_RANGE[DATE_RANGE >= np.datetime64(f'{START}-01-01')][0]
END = DATE_RANGE[DATE_RANGE <= np.datetime64(f'{END}-12-31')][-1]

EXCH_CODE = [1, 2, 3]  # NYSE, AMEX, NASDAQ
COMMON_STOCK_CD = [10, 11]  # only common stocks
PRICE_LIMIT = 5. # Exclude if price < $5
HISTORY_LIMIT = 15  # at least 15 daily returns in the month


#%%


# signal calculation algorithm

# beta: beta1 + beta2 + beta3

# independent variables in the regression: contemporaneous as well as lead and
# lag market excess returns (CRSP value-weighted).
MKTRF = FF.drop('rf', axis=1)
MKTRF['-1'] = MKTRF['mktrf'].shift(1)  # lag value
MKTRF['1'] = MKTRF['mktrf'].shift(-1)  # lead value
MKTRF.dropna(inplace=True)

# add the column of ones to the inputs to take into account the intercept
MKTRF = sm.add_constant(MKTRF)

# will be used many times, so defined once here.
MONTHBEGIN = pd.tseries.offsets.MonthBegin()

CPU_COUNT = 6


def calc_beta(month, lb):
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

    # get data in the past 1 month and apply history constraint
    month_begin = MONTHBEGIN.rollback(month).to_datetime64()
    data = DSF[(DSF.PERMNO.isin(data.PERMNO.values)) &
               (DSF.DATE >= month_begin) &
               (DSF.DATE <= month)].to_pandas_df()
    rets_count = data.groupby('PERMNO').RET.count()  # count non-missing returns
    eligible = rets_count[rets_count >= HISTORY_LIMIT].index  # history constraint

    # keep eligible stocks and calculate their signals
    data = data[data.PERMNO.isin(eligible)][['PERMNO', 'DATE', 'RET']]

    # below I distribute computations to multiple CPUs to reduce run-time
    workloads = data.PERMNO.unique()
    s = len(workloads) // CPU_COUNT  # load size
    to_distribute = [workloads[s*i:s*(i+1)] if i != CPU_COUNT-1 else workloads[s*i:]
                     for i in range(CPU_COUNT)]

    beta = {}
    with futures.ProcessPoolExecutor(max_workers=CPU_COUNT) as ex:
        wait_for = []
        for elm in to_distribute:
            f = ex.submit(helper_calc_beta, data, elm)  # schedule the future to run
            wait_for.append(f)
        for f in futures.as_completed(wait_for):  # collect the results returned from processes
            beta.update(f.result())

    # save into a pd.Series
    beta = pd.Series(beta)
    beta.dropna(inplace=True)
    beta.name = 'BETA'
    beta.index.name = 'PERMNO'

    te = time.time()
    print('{}, {:.2f}s'.format(month, te - ts))

    return beta


def helper_calc_beta(data, permnos):
    res = {}
    for p in permnos:
        rows = data[data.PERMNO == p].set_index('DATE')
        y = rows.RET.dropna()
        y -= FF.loc[y.index, 'rf']  # dependent variable, stock's excess return
        x = MKTRF.loc[y.index]
        if (not y.size) or (not x.dropna().size):  # 0 size after dropping nan
            continue
        # create a model and fit it
        model = sm.OLS(y, x, missing='drop')
        results = model.fit()
        res[p] = results.params.loc[['mktrf', '-1', '1']].sum()
    return res


def calc_signals(args):
    sub_range, lb = args
    signals = {m: calc_beta(m, lb) for m in sub_range}
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

