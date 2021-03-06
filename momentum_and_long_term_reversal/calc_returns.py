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

NAME = 'momentum_and_long_term_reversal'

START = 1967
END = 2018
LOOK_BACK = [3, 6, 9, 12]
HOLDING = [3, 6, 9, 12]


#%%


# read data

MSF = pd.read_hdf(DIR + '/data/msf.h5', key='msf')

DATE_RANGE = MSF.DATE.unique()
DATE_RANGE.sort()


#%%


START = DATE_RANGE[DATE_RANGE >= np.datetime64(f'{START}-01-01')][0]


#%%


# return calculation algorithm

# given signals, calculate portfolio series;
# given portfolio series, calculate monthly return series;
# given monthly return series, display plots and statistics.

CUT = 10

SWLL = True  # short-term winner & long-term loser

def calc_portfolios(lb):
    """
    Given signals from a CSV file, calculate a winner portfolio series
    and a loser portfolio series.
    
    lb: int
    """
    short_term_rets = pd.read_csv(DIR + f'/anomaly-project/momentum/signals/{lb}.csv')
    long_term_rets = pd.read_csv(DIR + f'/anomaly-project/long_term_reversal/signals/60.13.csv')

    # str to datetime
    short_term_rets['DATE'] = pd.to_datetime(short_term_rets['DATE'])
    long_term_rets['DATE'] = pd.to_datetime(long_term_rets['DATE'])

    winners = {}
    losers = {}

    # in each month we have a new winner portfolio and a new loser portfolio
    # computed based on signal ranking.
    for month in long_term_rets.DATE.unique():
        # first, sort by past short-term returns
        rows = short_term_rets[(short_term_rets.DATE == month)]
        rows = rows.set_index('PERMNO')
        cut = pd.qcut(rows.RET.rank(method='first'), CUT, labels=False)  # cut based on signal ranking
        w = cut[cut == CUT-1].index.tolist()
        l = cut[cut == 0].index.tolist()

        # then, dependent sort based on past long-term returns
        rows = long_term_rets[(long_term_rets.DATE == month) & (long_term_rets.PERMNO.isin(w))]
        rows = rows.set_index('PERMNO')
        cut = pd.qcut(rows.RET.rank(method='first'), 2, labels=False)
        k = 0 if SWLL else 1
        winners[month] = cut[cut == k].index.tolist()

        rows = long_term_rets[(long_term_rets.DATE == month) & (long_term_rets.PERMNO.isin(l))]
        rows = rows.set_index('PERMNO')
        cut = pd.qcut(rows.RET.rank(method='first'), 2, labels=False)
        k = 1 if SWLL else 0
        losers[month] = cut[cut == k].index.tolist()

    return winners, losers


def calc_port_ret(portfolio, data):
    """
    Calculate the portfolio's return in the month.
    
    portfolio: list
        a list of PERMNO representing the equally weighted portfolio
    """
    permnos = set(portfolio) & set(data.index)
    ret = data.loc[permnos, 'RET'].sum() / len(portfolio)

    return ret


def calc_monthly_rets(*args):
    """
    Given winners and losers, calculate their monthly return series
    and the monthly return series of the corresponding long-short portfolio.
    
    portfolios: tuple
        (winners, losers)
    hd: int
    """
    portfolios, hd = args
    winners, losers = portfolios
    
    monthly_rets = {t: {} for t in ['w', 'l', 'ls']}

    months = sorted(winners.keys())
    for i, current_month in enumerate(months):
        if current_month < START:
            continue
        # in each month, the monthly return is calculated as the
        # equally weighted average of the returns from the hd separate portfolios.
        current_month_rets = {t: [] for t in ['w', 'l', 'ls']}

        data = MSF[(MSF.DATE == current_month) & (MSF.RET.notna())]
        data = data.set_index('PERMNO')
        for n in range(hd):
            m = months[i-n]
            w = winners[m]  # the winner portfolio
            l = losers[m]  # the loser portfolio
            wret = calc_port_ret(w, data)  # winner's return in the current month
            lret = calc_port_ret(l, data)  # loser's return in the current month
            current_month_rets['w'].append(wret)
            current_month_rets['l'].append(lret)
            current_month_rets['ls'].append(wret - lret)  # long winner, short loser
        
        # the equally weighted average
        for t in ['w', 'l', 'ls']:
            monthly_rets[t][current_month] = np.mean(current_month_rets[t])

    return monthly_rets


#%%


def helper_timeit(func, *args):
    ts = time.time()
    res = func(*args)
    te = time.time()
    return res, te - ts


#%%


# calculate portfolios

if not os.path.exists(DIR + f'/anomaly-project/{NAME}/returns'):
    os.mkdir(DIR + f'/anomaly-project/{NAME}/returns')

collector_portfolios = {}

with futures.ProcessPoolExecutor(max_workers=4) as ex:
    wait_for = []
    for lb in LOOK_BACK:
        print(f'Calculating look back {lb} portfolios...')
        f = ex.submit(helper_timeit, calc_portfolios, lb)  # schedule the future to run
        f.arg = lb
        wait_for.append(f)
    
    # collect results returned from processes
    for f in futures.as_completed(wait_for):
        lb = f.arg
        res, t = f.result()
        collector_portfolios[lb] = res
        print('look back {} done, {:.2f}s.'.format(lb, t))

# save to local
with open(DIR + f'/anomaly-project/{NAME}/returns/portfolios.pkl', 'wb') as f:
    pk.dump(collector_portfolios, f)


#%%


# calculate monthly returns

collector_monthly_rets = {}

with futures.ProcessPoolExecutor(max_workers=4) as ex:
    wait_for = []
    for lb in LOOK_BACK:
        for hd in HOLDING:
            print(f'Calculating {lb}-{hd} monthly returns...')
            winners, losers = collector_portfolios[lb]
            f = ex.submit(helper_timeit, calc_monthly_rets, (winners, losers), hd)  # schedule the future to run
            f.arg = (lb, hd)
            wait_for.append(f)

    # collect results returned from processes
    for f in futures.as_completed(wait_for):
        key = f.arg
        res, t = f.result()
        collector_monthly_rets[key] = res
        print('{}-{} done, {:.2f}s.'.format(*key, t))

# save to local
with open(DIR + f'/anomaly-project/{NAME}/returns/monthly_returns.pkl', 'wb') as f:
    pk.dump(collector_monthly_rets, f)

