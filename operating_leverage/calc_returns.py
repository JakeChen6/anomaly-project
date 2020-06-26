#!/usr/bin/env python
# coding: utf-8

#%%


import pickle as pk

import numpy as np
import pandas as pd
from scipy import stats

DIR = '/Users/zhishe/myProjects/anomaly'

NAME = 'operating_leverage'


#%%


# read data

# CRSP MSF
MSF = pd.read_hdf(DIR + '/data/msf.h5', key='msf')

DATE_RANGE = MSF.DATE.unique()
DATE_RANGE.sort()

# PERMNOs of common stocks
with open(DIR + '/data/common_stock_permno.pkl', 'rb') as f:
    COMMON_STOCK_PERMNO = pk.load(f)


#%%


# constraints

"""
NYSE, AMEX, NASDAQ
Common stocks
Exclude if price < $5
"""

EXCH_CODE = [1, 2, 3]  # NYSE, AMEX, NASDAQ
COMMON_STOCK_CD = [10, 11]  # only common stocks
PRICE_LIMIT = 5. # Exclude if price < $5


#START = DATE_RANGE[DATE_RANGE >= np.datetime64('1987-07-01')][0]


#%%


# return calculation algorithm

# given signals, calculate portfolio series;
# given portfolio series, calculate monthly return series;
# given monthly return series, display plots and statistics.

"""
At the beginning of each month, we sort stocks into deciles based on
operating leverage (Novy-Marx 2011) for the most recent reporting year.
OL is the cost of goods sold (Compustat annual item COGS) plus selling,
general, and administrative expense (item XSGA), all divided by
total assets (item AT). The most recent reporting year is the closest one that
ends (according to item DATADATE) at least four months before the beginning
of the current month.
"""

def calc_portfolios():
    """
    Given signals from a CSV file, calculate a winner portfolio series
    and a loser portfolio series.
    
    """
    signals = pd.read_csv(DIR + f'/anomaly-project/{NAME}/signals.csv', index_col=0)
    signals['datadate'] = pd.to_datetime(signals['datadate'])

    winners = {}
    losers = {}

    # At the end of June of year t, we sort stocks into deciles based on Ol for the
    # fiscal year ending in calendar year t − 1.

    for i, date in enumerate(DATE_RANGE):
        if i < 4:
            continue
        four_months_ago = DATE_RANGE[i-4]  # four months before
        rows = signals[signals['datadate'] <= four_months_ago]
        if rows.empty:
            continue
        
        rows = rows.drop_duplicates(subset=['lpermno'], keep='last')  # the most recent reporting year
        rows = rows[rows['fyear'] >= pd.Timestamp(four_months_ago).year-1]  # keep recent signals

        last_month = DATE_RANGE[i-1]
        msf_data = MSF[MSF['DATE'] == last_month]
        # exchange constraint
        msf_data = msf_data[msf_data.HEXCD.isin(EXCH_CODE)]
        # price constraint
        msf_data = msf_data[msf_data.PRC.abs() >= PRICE_LIMIT]
        # common stocks
        common_stock_permno = COMMON_STOCK_PERMNO[last_month]  # PERMNO of common stocks
        msf_data = msf_data[msf_data.PERMNO.isin(common_stock_permno)]

        rows = rows[rows['lpermno'].isin(msf_data.PERMNO.values)].set_index('lpermno')
        ol = rows['signal-ol'].sort_values()

        # drop outliers
        ol = ol[np.abs(stats.zscore(ol)) < 3]

        num_in_decile = ol.size // 10  # number of stocks in a decile
        winner_threshold = ol.iloc[-num_in_decile]  # threshold of bottom decile
        loser_threshold = ol.iloc[num_in_decile-1]  # threshold of top decile
        winners[last_month] = ol[ol >= winner_threshold].index.tolist()
        losers[last_month] = ol[ol <= loser_threshold].index.tolist()

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


def calc_monthly_rets(portfolios):
    """
    Given winners and losers, calculate their monthly return series
    and the monthly return series of the corresponding long-short portfolio.

    portfolios: tuple
        (winners, losers)
    """
    winners, losers = portfolios

    monthly_rets = {
        'w': {},
        'l': {},
        'ls': {}
        }

    sorted_months = np.array(sorted(winners.keys()))

    for i, current_month in enumerate(DATE_RANGE):
        if i == 0:
            continue
        last_month = DATE_RANGE[i-1]
        if last_month not in sorted_months:
            continue

        data = MSF[(MSF.DATE == current_month) & (MSF.RET.notna())]
        data = data.set_index('PERMNO')

        w = winners[last_month]  # the winner portfolio
        l = losers[last_month]  # the loser portfolio
        wret = calc_port_ret(w, data)  # winner's return in the current month
        lret = calc_port_ret(l, data)  # loser's return in the current month
        monthly_rets['w'][current_month] = wret
        monthly_rets['l'][current_month] = lret
        monthly_rets['ls'][current_month] = wret - lret  # long winner, short loser

    return monthly_rets


#%%


# calculate portfolios
portfolios = calc_portfolios()

# calculate monthly returns
monthly_rets = calc_monthly_rets(portfolios)

# save to local
with open(DIR + f'/anomaly-project/{NAME}/portfolios.pkl', 'wb') as f:
    pk.dump(portfolios, f)

with open(DIR + f'/anomaly-project/{NAME}/monthly_returns.pkl', 'wb') as f:
    pk.dump(monthly_rets, f)
