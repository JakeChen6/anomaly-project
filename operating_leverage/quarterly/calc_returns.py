#!/usr/bin/env python
# coding: utf-8

#%%


import pickle as pk

import numpy as np
import pandas as pd

DIR = '/Users/zhishe/myProjects/anomaly'

NAME = 'operating_leverage'

EXCH_CODE = [1, 2, 3]  # NYSE, AMEX, NASDAQ
COMMON_STOCK_CD = [10, 11]  # only common stocks
PRICE_LIMIT = 5  # Exclude if price < $5

HOLDING = [1, 6, 12]


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


# return calculation algorithm

# given signals, calculate portfolio series;
# given portfolio series, calculate monthly return series;
# given monthly return series, display plots and statistics.

START = 1973
START = DATE_RANGE[DATE_RANGE >= np.datetime64(f'{START}-01-01')][0]


def calc_portfolios():
    """
    Given signals from a CSV file, calculate a winner portfolio series
    and a loser portfolio series.
    
    """
    signals = pd.read_csv(DIR + f'/anomaly-project/{NAME}/quarterly/signals.csv', index_col=0)
    signals['datadate'] = pd.to_datetime(signals['datadate'])
    signals.sort_values('datadate', inplace=True)

    winners = {}
    losers = {}

    for i, date in enumerate(DATE_RANGE):
        if i < 4:
            continue

        four_months_before = DATE_RANGE[i-4]
        df = signals[signals['datadate'] <= four_months_before]
        if df.empty:
            continue

        df = df.drop_duplicates(subset=['lpermno'], keep='last')
        df = df[df['fyearq'] >= pd.Timestamp(four_months_before).year-1]
        df = df[df['atq'] >= 5]  # filter: total asset no less than 5 million later
        df = df[df['cogsq'] > 0]
        df = df[df['xsgaq'] > 0]
        if df.empty:
            continue

        last_month = DATE_RANGE[i-1]

        # apply constraints
        msf_data = MSF[MSF['DATE'] == last_month]
        msf_data = msf_data[msf_data.HEXCD.isin(EXCH_CODE)]  # exchange constraint
        msf_data = msf_data[msf_data.PRC.abs() >= PRICE_LIMIT]  # price constraint
        common_stock_permno = COMMON_STOCK_PERMNO[last_month]  # common-stock constraint
        msf_data = msf_data[msf_data.PERMNO.isin(common_stock_permno)]

        df = df[df['lpermno'].isin(msf_data.PERMNO.values)]
        df = df.set_index('lpermno')
        ol = df['ol'].sort_values()

        # drop outliers?

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


def calc_monthly_rets(portfolios, hd):
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


# calculate portfolios
portfolios = calc_portfolios()

# calculate monthly returns
collector_monthly_rets = {hd: calc_monthly_rets(portfolios, hd) for hd in HOLDING}

# save to local
with open(DIR + f'/anomaly-project/{NAME}/quarterly/portfolios.pkl', 'wb') as f:
    pk.dump(portfolios, f)

with open(DIR + f'/anomaly-project/{NAME}/quarterly/monthly_returns.pkl', 'wb') as f:
    pk.dump(collector_monthly_rets, f)
