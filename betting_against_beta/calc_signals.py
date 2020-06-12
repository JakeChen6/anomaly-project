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
from scipy.stats import pearsonr

DIR = '/Users/zhishe/myProjects/anomaly'


#%%


# read data

DSF = vaex.open(DIR + '/data/Stock/h5/dsf.h5')

with open(DIR + '/data/common_stock_permno.pkl', 'rb') as f:
    COMMON_STOCK_PERMNO = pk.load(f)

MSF = pd.read_hdf(DIR + '/data/msf.h5', key='msf')
DATE_RANGE = MSF.DATE.unique()
DATE_RANGE.sort()

DSI = pd.read_csv(DIR + '/data/dsi.csv', index_col=0, parse_dates=True)


#%%


# anomaly setting

NAME = 'betting_against_beta'

START = 1932
END = 2018
LOOK_BACK = [60]
HOLDING = [1, 6, 12]


#%%


# constraints

"""
July 1926 - March 2012
NYSE, AMEX, NASDAQ
Common stocks
Exclude if price < $5
at least 120 daily returns in the one-year period
at least 750 daily returns in the five-year period
"""
START = DATE_RANGE[DATE_RANGE >= np.datetime64(f'{START}-01-01')][0]
END = DATE_RANGE[DATE_RANGE <= np.datetime64(f'{END}-12-31')][-1]

EXCH_CODE = [1, 2, 3]  # NYSE, AMEX, NASDAQ
COMMON_STOCK_CD = [10, 11]  # only common stocks
PRICE_LIMIT = 5. # Exclude if price < $5
HISTORY_LIMIT_SIGMA = 120  # at least 120 daily returns in the one-year period
HISTORY_LIMIT_ROU = 750  # at least 750 daily returns in the five-year period


#%%


# signal calculation algorithm

# beta: rou * firm sigma / market sigma, sigmas are calculated as the standard
# deviation of daily log returns over a 1-year rolling window (with at least
# 120 daily returns); correlation is calculated using overlapping 3-day log
# returns over a 5-year rolling window (with at least 750 daily returns).

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

    # one year ago
    one_year_ago = DATE_RANGE[DATE_RANGE <= month][-12]
    one_year_ago = MONTHBEGIN.rollback(one_year_ago).to_datetime64()
    # five years ago
    five_years_ago = DATE_RANGE[DATE_RANGE <= month][-12*5]
    five_years_ago = MONTHBEGIN.rollback(five_years_ago).to_datetime64()

    # apply history constraints
    past_data_5y = DSF[(DSF.PERMNO.isin(data.PERMNO.values)) &  # data in the past 5 year
                       (DSF.DATE >= five_years_ago) &
                       (DSF.DATE <= month)].to_pandas_df()
    rets_count = past_data_5y.groupby('PERMNO').RET.count()  # count non-missing returns
    eligible = rets_count[rets_count >= HISTORY_LIMIT_ROU].index  # history constraint 1

    past_data_1y = past_data_5y[(past_data_5y.PERMNO.isin(eligible)) &  # data in the past 1 year
                                (past_data_5y.DATE >= one_year_ago)]
    rets_count = past_data_1y.groupby('PERMNO').RET.count()  # count non-missing returns
    eligible = rets_count[rets_count >= HISTORY_LIMIT_SIGMA].index  # history constraint 2

    # get data in the past 5 years for the eligible stocks
    past_data_5y = past_data_5y[past_data_5y.PERMNO.isin(eligible)]
    # calculate rou
    rous = {}
    mktret = DSI[(DSI.index >= five_years_ago) &
                 (DSI.index <= month)][['vwretd']].copy()
    mktret['1'] = mktret.vwretd.shift(-1)
    mktret['2'] = mktret.vwretd.shift(-2)
    mktret.dropna(inplace=True)
    mktret = np.log(mktret + 1).sum(axis=1)  # overlapping 3-day log returns

    past_data_5y = past_data_5y[['PERMNO', 'DATE', 'RET']]
    workloads = past_data_5y.PERMNO.unique()
    s = len(workloads) // CPU_COUNT  # load size
    to_distribute = [workloads[s*i:s*(i+1)] if i != CPU_COUNT-1 else workloads[s*i:]
                     for i in range(CPU_COUNT)]

    rous = {}
    with futures.ProcessPoolExecutor(max_workers=CPU_COUNT) as ex:
        wait_for = []
        for elm in to_distribute:
            f = ex.submit(helper_calc_corr, past_data_5y, mktret, elm)
            wait_for.append(f)
        for f in futures.as_completed(wait_for):
            rous.update(f.result())

    # get data in the past 1 year for the eligible stocks
    past_data_1y = past_data_1y[past_data_1y.PERMNO.isin(eligible)].copy()
    # calculate sigma
    past_data_1y['RET'] = np.log(past_data_1y.RET + 1)  # daily log returns
    firm_sigma = past_data_1y.groupby('PERMNO').RET.std()
    mktret = DSI[(DSI.index >= one_year_ago) &
                 (DSI.index <= month)]['vwretd']
    mkt_sigma = np.log(mktret + 1).std()

    # calculate beta
    beta = pd.Series(rous) * firm_sigma / mkt_sigma
    beta.dropna(inplace=True)
    beta.name = 'BETA'
    beta.index.name = 'PERMNO'

    te = time.time()
    print('{}, {:.2f}s'.format(month, te - ts))

    return beta


def helper_calc_corr(past_data_5y, mktret, permnos):
    res = {}
    for p in permnos:
        rows = past_data_5y[past_data_5y.PERMNO == p][['DATE', 'RET']].set_index('DATE')
        rows['1'] = rows.RET.shift(-1)
        rows['2'] = rows.RET.shift(-2)
        rows.dropna(inplace=True)
        rets = np.log(rows + 1).sum(axis=1)  # overlapping 3-day log returns
        res[p] = pearsonr(rets, mktret.loc[rets.index])[0]
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

