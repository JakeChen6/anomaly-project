import os
import pickle as pk

import pandas as pd

ENV_PATH = '/Users/zhishe/myProjects/anomaly'

NAME = 'long_term_reversal'


#%%

# Read data

# monthly stock data; we only need return data
MSF = pd.read_hdf(ENV_PATH + '/data/msf.h5', key='msf')

# range of dates
DATE_RANGE = MSF['DATE'].unique()
DATE_RANGE.sort()


#%%

# Basically we get the final results in a "layer by layer" fashion.

# Briefly,
# given signals, calculate portfolio series;
# given portfolio series, calculate monthly return series;
# given monthly return series, display plots and statistics.


# we can try decile portfolio, or top (bottom) 35/50.

def calc_portfolios(lb):
    """
    Given signals (a CSV file), calculate a winner portfolio series and a loser
    portfolio series.

    Parameters
    ----------
    lb : int
        look back period

    Returns
    -------
    winner portfolio series : dict
    loser portfolio series : dict

    """
    signals = pd.read_csv(ENV_PATH + f'/results/{NAME}/signals/{lb}.csv')  # read signals from CSV
    signals['DATE'] = pd.to_datetime(signals['DATE'])  # str -> datetime

    winners = {}
    losers = {}

    for m in signals['DATE'].unique():
        rows = signals[signals['DATE'] == m]  # select data on that date
        rows = rows.set_index('PERMNO')  # set 'PERMNO' as index
        cum_excess_rets = rows['SIGNAL']  # the 'SIGNAL' column
        deciles = pd.qcut(cum_excess_rets.rank(method='first'), 10, labels=False)  # cut into deciles based on signal ranking
        winners[m] = deciles[deciles == 9].index.tolist()  # winner decile
        losers[m] = deciles[deciles == 0].index.tolist()  # loser decile

    return winners, losers


def helper_calc_port_ret_in_month(portfolio, data):
    """
    Given a portfolio composition and a month, calculate the portfolio's return
    in the month.

    Parameters
    ----------
    portfolio : list
        a list of "PERMNO" representing the equally weighted portfolio
    month : np.datetime64
        a particular month

    Returns
    -------
    the equally weighted portfolio's return in the month : float

    """
    permnos = set(portfolio) & set(data.index)  # permnos in both portfolio and data.index
    ret = data.loc[permnos, 'RET'].sum() / len(portfolio)  # return of the equally weighted portfolio

    return ret


def calc_monthly_rets(*args):
    """
    Given a winner portfolio series and a loser portfolio series, calculate
    the monthly return series of the winner, loser, and winner-loser portfolio
    series, respectively.

    Parameters
    ----------
    portfolios : tuple
        holding the winner portfolio series and the loser portfolio series
    hd : int
        holding period

    Returns
    -------
    monthly return series : dict
        monthly return series of the winner, loser, and winner-loser portfolio
        series

    """
    portfolios, hd = args
    winners, losers = portfolios

    monthly_rets = {'w': {},  # winner
                    'l': {},  # loser
                    'ls': {}}  # long-short

    months = sorted(winners.keys())  # all the months; winners and losers have the same keys
    for m in months:
        # portfolios are held for 3 years
        w = winners[m]  # the winner portfolio
        l = losers[m]  # the loser portfolio
        three_years = DATE_RANGE[DATE_RANGE >= m][:36]

        for current_month in three_years:
            data = MSF[(MSF['DATE'] == current_month) & (MSF['RET'].notna())]
            data = data.set_index('PERMNO')  # set "PERMNO" as index
            wret = helper_calc_port_ret_in_month(w, data)  # w's return in the currrent month
            lret = helper_calc_port_ret_in_month(l, data)  # l's return in the currrent month
            monthly_rets['w'][current_month] = wret
            monthly_rets['l'][current_month] = lret
            monthly_rets['ls'][current_month] = lret - wret  # long loser, short winner

    return monthly_rets


#%%

# look back periods & holding periods
LOOK_BACK = [36]
HOLDING   = [36]


#%%

# Calculate portfolios

if not os.path.exists(ENV_PATH + f'/results/{NAME}/plots'):
    os.mkdir(ENV_PATH + f'/results/{NAME}/plots')

lb = LOOK_BACK[0]
collector = dict()  # to collect the results returned from multiple processes
collector[lb] = calc_portfolios(lb)

# save the collector to local
with open(ENV_PATH + f'/results/{NAME}/plots/portfolios.pkl', 'wb') as f:
    pk.dump(collector, f)


#%%

# Calculate monthly returns

# need collector_portfolios as input
with open(ENV_PATH + f'/results/{NAME}/plots/portfolios.pkl', 'rb') as f:
    collector_portfolios = pk.load(f)

hd = HOLDING[0]
collector = dict()  # to collect the results returned from multiple processes
winners, losers = collector_portfolios[lb]
collector[(lb, hd)] = calc_monthly_rets((winners, losers), hd)

# save the collector to local        
with open(ENV_PATH + f'/results/{NAME}/plots/monthly_returns.pkl', 'wb') as f:
    pk.dump(collector, f)
