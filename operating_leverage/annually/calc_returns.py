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

JUNE = '06'

END = 2018

def calc_portfolios():
    """
    Given signals from a CSV file, calculate a winner portfolio series
    and a loser portfolio series.
    
    """
    signals = pd.read_csv(DIR + f'/anomaly-project/{NAME}/annually/signals.csv', index_col=0)
    signals['datadate'] = pd.to_datetime(signals['datadate'])

    winners = {}
    losers = {}

    for fyear in signals['fyear'].unique():
        if fyear == END:
            break

        df = signals[signals['fyear'] == fyear]
        df = df[df['at'] >= 5]  # filter: total asset no less than 5 million later
        df = df[df['cogs'] > 0]
        df = df[df['xsga'] > 0]

        current_year = int(fyear + 1)
        month = DATE_RANGE[DATE_RANGE >= np.datetime64(f'{current_year}-{JUNE}-01')][0]

        # apply constraints
        msf_data = MSF[MSF['DATE'] == month]
        msf_data = msf_data[msf_data.HEXCD.isin(EXCH_CODE)]  # exchange constraint
        msf_data = msf_data[msf_data.PRC.abs() >= PRICE_LIMIT]  # price constraint
        common_stock_permno = COMMON_STOCK_PERMNO[month]  # common-stock constraint
        msf_data = msf_data[msf_data.PERMNO.isin(common_stock_permno)]

        df = df[df['lpermno'].isin(msf_data.PERMNO.values)]
        df = df.set_index('lpermno')
        ol = df['ol'].sort_values()

        # drop outliers?

        num_in_decile = ol.size // 10  # number of stocks in a decile
        winner_threshold = ol.iloc[-num_in_decile]  # threshold of bottom decile
        loser_threshold = ol.iloc[num_in_decile-1]  # threshold of top decile
        winners[month] = ol[ol >= winner_threshold].index.tolist()
        losers[month] = ol[ol <= loser_threshold].index.tolist()

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
        monthly_rets['ls'][current_month] = wret - lret  # long winner, short loser

    return monthly_rets


#%%


# calculate portfolios
portfolios = calc_portfolios()

# calculate monthly returns
monthly_rets = calc_monthly_rets(portfolios)

# save to local
with open(DIR + f'/anomaly-project/{NAME}/annually/portfolios.pkl', 'wb') as f:
    pk.dump(portfolios, f)

with open(DIR + f'/anomaly-project/{NAME}/annually/monthly_returns.pkl', 'wb') as f:
    pk.dump(monthly_rets, f)
