#!/usr/bin/env python
# coding: utf-8

#%%


import pickle as pk

import numpy as np
import pandas as pd
from scipy import stats

DIR = '/Users/zhishe/myProjects/anomaly'

NAME = 'asset_growth'

EXCH_CODE = [1, 2, 3]  # NYSE, AMEX, NASDAQ
COMMON_STOCK_CD = [10, 11]  # only common stocks
PRICE_LIMIT = 5  # Exclude if price < $5


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


def calc_portfolios():
    """
    Given signals from a CSV file, calculate a winner portfolio series
    and a loser portfolio series.
    
    """
    signals = pd.read_csv(DIR + f'/anomaly-project/{NAME}/signals.csv', index_col=0)
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
        df = df[df['fyear'] >= pd.Timestamp(four_months_before).year-1]
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
        growth = df['growth'].sort_values()

        # drop outliers?

        num_in_decile = growth.size // 10  # number of stocks in a decile
        winner_threshold = growth.iloc[-num_in_decile]  # threshold of bottom decile
        loser_threshold = growth.iloc[num_in_decile-1]  # threshold of top decile
        winners[last_month] = growth[growth >= winner_threshold].index.tolist()
        losers[last_month] = growth[growth <= loser_threshold].index.tolist()

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

    for current_month in DATE_RANGE:
        if current_month <= sorted_months[0]:
            continue

        data = MSF[(MSF.DATE == current_month) & (MSF.RET.notna())]
        data = data.set_index('PERMNO')

        previous_month = sorted_months[sorted_months < current_month][-1]
        w = winners[previous_month]  # the winner portfolio
        l = losers[previous_month]  # the loser portfolio
        wret = calc_port_ret(w, data)  # winner's return in the current month
        lret = calc_port_ret(l, data)  # loser's return in the current month
        monthly_rets['w'][current_month] = wret
        monthly_rets['l'][current_month] = lret
        monthly_rets['ls'][current_month] = lret - wret  # long loser, short winner

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
