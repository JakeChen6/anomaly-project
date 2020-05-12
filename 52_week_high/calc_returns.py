import time
import pickle as pk
from concurrent import futures

import psutil
import numpy as np
import pandas as pd

ENV_PATH = '/Users/zhishe/myProjects/anomaly'

ANOMALY_PATH = ENV_PATH + '/anomalies/3-52-Week High'


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

PERCENTAGE = 0.3


def calc_portfolios(*args):
    """
    Given signals (a CSV file), calculate a winner portfolio series and a loser
    portfolio series.

    Parameters
    ----------
    lb : int
        look back period
    hd : int
        holding period

    Returns
    -------
    winner portfolio series : dict
    loser portfolio series : dict

    """
    lb, hd = args
    signals = pd.read_csv(ANOMALY_PATH + f'/signals/{lb}-{hd}.csv')  # read signals from CSV
    signals['DATE'] = pd.to_datetime(signals['DATE'])  # str -> datetime

    winners = {}
    losers = {}

    # in each month we have a new winner portfolio and a new loser portfolio
    # computed based on signal ranking, the new portfolio is held along with
    # the other (hd - 1) portfolios carried over from the previous months.
    for m in signals['DATE'].unique():
        rows = signals[signals['DATE'] == m]  # select data on that date
        rows = rows.set_index('PERMNO')  # set 'PERMNO' as index
        cum_rets = rows['SIGNAL']  # the 'SIGNAL' column
        cum_rets = cum_rets.sort_values()
        num = int(cum_rets.shape[0] * PERCENTAGE)
        winners[m] = cum_rets.iloc[-num:].index.tolist()
        losers[m] = cum_rets.iloc[:num].index.tolist()

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
                    'wl': {}}

    months = sorted(winners.keys())  # all the months; winners and losers have the same keys
    for i, current_month in enumerate(months):
        if i < hd-1:  # still in "warm-up" period
            continue
        # in each month, the return is calculated as the equally weighted average
        # of the returns from the hd separate portfolios.
        current_month_rets = {'w': [],
                              'l': [],
                              'wl': []}
        data = MSF[(MSF['DATE'] == current_month) & (MSF['RET'].notna())]  # select data; "RET" not NaN
        data = data.set_index('PERMNO')  # set "PERMNO" as index
        for n in range(hd):
            m = months[i-n]
            w = winners[m]  # the winner portfolio
            l = losers[m]  # the loser portfolio
            wret = helper_calc_port_ret_in_month(w, data)  # w's return in the currrent month
            lret = helper_calc_port_ret_in_month(l, data)  # l's return in the currrent month
            wlret = wret - lret  # (w-l)'s return in the current month
            current_month_rets['w'].append(wret)
            current_month_rets['l'].append(lret)
            current_month_rets['wl'].append(wlret)

        # the equally weighted average
        monthly_rets['w'][current_month] = np.mean(current_month_rets['w'])
        monthly_rets['l'][current_month] = np.mean(current_month_rets['l'])
        monthly_rets['wl'][current_month] = np.mean(current_month_rets['wl'])

    return monthly_rets


#%%

# look back periods & holding periods
LOOK_BACK = [6]
HOLDING   = [6]

CPU_COUNT = psutil.cpu_count(logical=False)


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

with futures.ProcessPoolExecutor(max_workers=CPU_COUNT) as ex:
    collector = dict()  # to collect the results returned from multiple processes
    wait_for = list()  # to record tasks assigned to processes

    for lb in LOOK_BACK:
        for hd in HOLDING:
            print('Calculating {}-{} portfolios...'.format(lb, hd))
            f = ex.submit(helper_timeit, calc_portfolios, lb, hd)  # schedule the future to run
            f.arg = (lb, hd)
            wait_for.append(f)  # record the scheduled future

    # collect the results returned from multiple processes
    for f in futures.as_completed(wait_for):
        key = f.arg
        res, exec_time = f.result()
        collector[key] = res
        print('{}-{} done, {:.2f}s.'.format(*key, exec_time))

# save the collector to local
with open(ANOMALY_PATH + '/results/portfolios.pkl', 'wb') as f:
    pk.dump(collector, f)


#%%

# Calculate monthly returns
# We distribute the workload to multiple processes in order to get a speed-up.

# need collector_portfolios as input
with open(ANOMALY_PATH + '/results/portfolios.pkl', 'rb') as f:
    collector_portfolios = pk.load(f)

with futures.ProcessPoolExecutor(max_workers=4) as ex:
    collector = dict()  # to collect the results returned from multiple processes
    wait_for = list()  # to record tasks assigned to processes

    for lb in LOOK_BACK:
        for hd in HOLDING:
            print('Calculating {}-{} monthly returns...'.format(lb, hd))
            winners, losers = collector_portfolios[(lb, hd)]
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
with open(ANOMALY_PATH + '/results/monthly_returns.pkl', 'wb') as f:
    pk.dump(collector, f)
