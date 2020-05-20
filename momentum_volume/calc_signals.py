#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
import time
import pickle as pk

import numpy as np
import pandas as pd
import vaex

DIR = '/Users/zhishe/myProjects/anomaly'


# In[ ]:


# anomaly setting

NAME = 'momentum_volume'

START = 1965
END = 2018
LOOK_BACK = [6]
HOLDING = [6]


# In[ ]:


# constraints

"""
1965 - 1995
NYSE, AMEX
At least two years of data prior to the formation date (in CRSP for at least 2 years)
Common stocks
Not less than a dollar
"""

EXCH_CODE = [1, 2]  # NYSE or AMEX
HISTORY_LIMIT = 2  # in CRSP for at least 2 years
COMMON_STOCK_CD = [10, 11]  # only common stocks
PRICE_LIMIT = 1. # not less than a dollar


# In[ ]:


# read data

DSF = vaex.open(DIR + '/data/Stock/h5/dsf.h5')

with open(DIR + '/data/common_stock_permno.pkl', 'rb') as f:
    COMMON_STOCK_PERMNO = pk.load(f)

MSF = pd.read_hdf(DIR + '/data/msf.h5', key='msf')
DATE_RANGE = MSF.DATE.unique()
DATE_RANGE.sort()


# In[ ]:


# signal calculation algorithm

# average daily turnover in percentage during the portfolio formation period
# the daily turnover is defined as the ratio of VOL to SHROUT

def calc_avg_daily_turnover(m, lb):
    """
    m: np.datetime64
    lb: int
    """
    ts = time.time()

    # past lb months
    start, end = DATE_RANGE[DATE_RANGE < m][[-lb, -1]]

    data = DSF[DSF.DATE == end]  # data on the last day prior to the formation date
    # apply constraints
    data = data[data.HEXCD.isin(EXCH_CODE)]  # exchange constraint
    data = data[data.PRC.abs() >= PRICE_LIMIT]  # price constraint

    common_stock_permno = COMMON_STOCK_PERMNO[end]  # PERMNO of common stocks according to information on 'end'
    data = data[data.PERMNO.isin(common_stock_permno)]  # common stock constraint

    past_data = DSF[(DSF.PERMNO.isin(data.PERMNO.values)) & (DSF.DATE <= end)]
    past_data = past_data[['PERMNO', 'DATE']].to_pandas_df()
    grouped = past_data.groupby('PERMNO')
    history = grouped.DATE.max() - grouped.DATE.min()  # pandas.Series
    history = history[history / np.timedelta64(1, 'D') >= (365 * HISTORY_LIMIT)]
    data = data[data.PERMNO.isin(history.index)]  # history constraint

    # get data in the past lb months for the eligible stocks
    eligible_permno = data.PERMNO.values
    data = DSF[(DSF.DATE >= start) & (DSF.DATE <= end) & (DSF.PERMNO.isin(eligible_permno))]
    data = data[['PERMNO', 'VOL', 'SHROUT']].to_pandas_df()
    avg_daily_turnover = data.groupby('PERMNO').apply(lambda df: (df['VOL'] / df['SHROUT']).mean() * 100)  # in percentage
    avg_daily_turnover.dropna(inplace=True)

    avg_daily_turnover.name = 'avg_daily_turnover'

    te = time.time()
    print('{}, {:.2f}s'.format(m, te - ts))
    
    return avg_daily_turnover


def calc_signals(args):
    sub_range, lb = args
    signals = {m: calc_avg_daily_turnover(m, lb) for m in sub_range}
    return signals


# In[ ]:


# single process

start = DATE_RANGE[DATE_RANGE >= np.datetime64(f'{START}-01-01')][0]
end = DATE_RANGE[DATE_RANGE <= np.datetime64(f'{END}-12-31')][-1]

collector = {}

for lb in LOOK_BACK:
    hd = max(HOLDING)
    print(f'\nCalculating ({lb}, {hd}) strategy...', end='\t')

    # on this date we calculate the first set of signals
    first_date = DATE_RANGE[DATE_RANGE <= start][-hd]
    # calculate signals for every month in this range
    date_range = DATE_RANGE[(DATE_RANGE >= first_date) & (DATE_RANGE <= end)]

    signals = calc_signals((date_range, lb))
    collector.setdefault(lb, {}).update(signals)


# In[ ]:


# save results to local

if not os.path.exists(DIR + f'/results/{NAME}'):
    os.mkdir(DIR + f'/results/{NAME}')
    os.mkdir(DIR + f'/results/{NAME}/signals')

for lb in LOOK_BACK:
    table = pd.DataFrame()
    signals = collector[lb]
    # consolidate each month's signals into a single table
    for k, v in signals.items():
        df = pd.DataFrame(v)
        df.reset_index(inplace=True)
        df['DATE'] = k
        table = pd.concat([table, df], ignore_index=True)

    table = table.reindex(columns=['DATE', 'PERMNO', 'avg_daily_turnover'])
    table.sort_values(by='DATE', inplace=True)
    table.to_csv(DIR + f'/results/{NAME}/signals/{lb}.csv')
    print(f'{lb} done.')

