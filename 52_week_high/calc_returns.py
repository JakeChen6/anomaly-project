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

DATE_RANGE = MSF.DATE.unique()
DATE_RANGE.sort()


#%%

# anomaly setting

NAME = '52_week_high'

START = 1963
END = 2018
LOOK_BACK = [12]
HOLDING = [6]


START = DATE_RANGE[DATE_RANGE >= np.datetime64(f'{START}-07-01')][0]


#%%


# return calculation algorithm

# given signals, calculate portfolio series;
# given portfolio series, calculate monthly return series;
# given monthly return series, display plots and statistics.

PERCENTAGE = 0.3


def calc_portfolios(lb):
    """
    Given signals from a CSV file, calculate a winner portfolio series
    and a loser portfolio series.
    
    lb: int
    """
    signals = pd.read_csv(DIR + f'/anomaly-project/{NAME}/signals/{lb}.csv')
    signals['DATE'] = pd.to_datetime(signals['DATE'])  # str to datetime

    winners = {}
    losers = {}
    
    # in each month we have a new winner portfolio and a new loser portfolio
    # computed based on signal ranking.
    for month in signals.DATE.unique():
        rows = signals[signals.DATE == month]
        rows = rows.set_index('PERMNO')
        ratios = rows.RATIO.sort_values()
        num = round(ratios.size * PERCENTAGE)  # not quantiles, but the top or bottom x%
        winner_threshold = ratios.iloc[-num]  # threshold of bottom portfolio 
        loser_threshold = ratios.iloc[num-1]  # threshold of top portfolio
        winners[month] = ratios[ratios >= winner_threshold].index.tolist()
        losers[month] = ratios[ratios <= loser_threshold].index.tolist()

    return winners, losers


def calc_port_ret(portfolio, data):
    """
    Calculate the portfolio's return in the month.
    
    portfolio: list
        a list of PERMNO representing the equally weighted portfolio
    """
    if not portfolio:
        return 0

    permnos = data.index.intersection(portfolio)
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
    
    monthly_rets = {
        'w': {},
        'l': {},
        'ls': {}
        }

    sorted_months = np.array(sorted(winners.keys()))

    for current_month in DATE_RANGE:
        if current_month < START:
            continue

        data = MSF[(MSF.DATE == current_month) & (MSF.RET.notna())]
        data = data.set_index('PERMNO')

        # the monthly return in each month is calculated as the
        # equally weighted average of the returns from the hd separate portfolios.
        current_month_rets = {
            'w': 0,
            'l': 0,
            'ls': 0
            }

        for previous_month in sorted_months[sorted_months < current_month][-hd:]:
            w = winners[previous_month]  # the winner portfolio
            l = losers[previous_month]  # the loser portfolio
            wret = calc_port_ret(w, data)  # winner's return in the current month
            lret = calc_port_ret(l, data)  # loser's return in the current month
            current_month_rets['w'] += wret
            current_month_rets['l'] += lret
            current_month_rets['ls'] += wret - lret  # long winner, short loser

        # the equally weighted average
        monthly_rets['w'][current_month] = current_month_rets['w'] / hd
        monthly_rets['l'][current_month] = current_month_rets['l'] / hd
        monthly_rets['ls'][current_month] = current_month_rets['ls'] / hd

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

