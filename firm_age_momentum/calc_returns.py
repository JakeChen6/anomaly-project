import os
import time
import pickle as pk
from concurrent import futures

import pandas as pd

ENV_PATH = '/Users/zhishe/myProjects/anomaly'

NAME = 'firm_age_momentum'


#%%

# Read data

# monthly stock data; we only need return data
MSF = pd.read_hdf(ENV_PATH + '/data/msf.h5', key='msf')


#%%

# Basically we get the final results in a "layer by layer" fashion.

# Briefly,
# given signals, calculate portfolio series;
# given portfolio series, calculate monthly return series;
# given monthly return series, display plots and statistics.


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

    # in each month we have a new winner portfolio and a new loser portfolio
    # computed based on signal ranking.
    for m in signals['DATE'].unique():
        rows = signals[signals['DATE'] == m]  # select data on that date
        rows = rows.set_index('PERMNO')  # set 'PERMNO' as index
        cum_rets = rows['SIGNAL']  # the 'SIGNAL' column
        quintiles = pd.qcut(cum_rets.rank(method='first'), 5, labels=False)  # cut into quintiles based on signal ranking
        winners[m] = quintiles[quintiles == 4].index.tolist()  # winner quintile
        losers[m] = quintiles[quintiles == 0].index.tolist()  # loser quintile

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

    monthly_rets = {'w': {},
                    'l': {},
                    'ls': {}}

    months = sorted(winners.keys())  # all the months; winners and losers have the same keys
    for current_month in months:
        data = MSF[(MSF['DATE'] == current_month) & (MSF['RET'].notna())]
        data = data.set_index('PERMNO')
        w = winners[current_month]  # the winner portfolio
        l = losers[current_month]  # the loser portfolio
        wret = helper_calc_port_ret_in_month(w, data)  # w's return in the currrent month
        lret = helper_calc_port_ret_in_month(l, data)  # l's return in the currrent month
        monthly_rets['w'][current_month] = wret
        monthly_rets['l'][current_month] = lret
        monthly_rets['ls'][current_month] = wret - lret  # long winner, short loser

    return monthly_rets


#%%

# look back periods & holding periods
LOOK_BACK = [11]
HOLDING   = [1]


def helper_timeit(fn, *args, **kwargs):
    """
    To get fn's execution time

    Parameters
    ----------
    fn : function
        the function to time

    Returns
    -------
    fn(*args, **kwargs), fn's execution time

    """
    start = time.time()
    result = fn(*args, **kwargs)
    return result, time.time() - start


#%%

# Calculate portfolios
# We distribute the workload to multiple processes in order to get a speed-up.

if not os.path.exists(ENV_PATH + f'/results/{NAME}/plots'):
    os.mkdir(ENV_PATH + f'/results/{NAME}/plots')

with futures.ProcessPoolExecutor(max_workers=4) as ex:
    collector = dict()  # to collect the results returned from multiple processes
    wait_for = list()  # to record tasks assigned to processes

    for lb in LOOK_BACK:
        print(f'Calculating look back {lb} portfolios...')
        f = ex.submit(helper_timeit, calc_portfolios, lb)  # schedule the future to run
        f.arg = lb
        wait_for.append(f)  # record the scheduled future

    # collect the results returned from multiple processes
    for f in futures.as_completed(wait_for):
        lb = f.arg
        res, exec_time = f.result()
        collector[lb] = res
        print('look back {} done, {:.2f}s.'.format(lb, exec_time))

# save the collector to local
with open(ENV_PATH + f'/results/{NAME}/plots/portfolios.pkl', 'wb') as f:
    pk.dump(collector, f)


#%%

# Calculate monthly returns
# We distribute the workload to multiple processes in order to get a speed-up.

# need collector_portfolios as input
with open(ENV_PATH + f'/results/{NAME}/plots/portfolios.pkl', 'rb') as f:
    collector_portfolios = pk.load(f)

with futures.ProcessPoolExecutor(max_workers=4) as ex:
    collector = dict()  # to collect the results returned from multiple processes
    wait_for = list()  # to record tasks assigned to processes

    for lb in LOOK_BACK:
        for hd in HOLDING:
            print(f'Calculating {lb}-{hd} monthly returns...')
            winners, losers = collector_portfolios[lb]
            f = ex.submit(helper_timeit, calc_monthly_rets, (winners, losers), hd)
            f.arg = (lb, hd)
            wait_for.append(f)  # record the scheduled future

    # collect the results returned from multiple processes
    for f in futures.as_completed(wait_for):
        key = f.arg
        res, exec_time = f.result()
        collector[key] = res
        print('{}-{} done, {:.2f}s.'.format(*key, exec_time))

# save the collector to local        
with open(ENV_PATH + f'/results/{NAME}/plots/monthly_returns.pkl', 'wb') as f:
    pk.dump(collector, f)
