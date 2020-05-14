import os
import time
import datetime as dt
import pickle as pk
from functools import reduce
from concurrent import futures

import psutil
import numpy as np
import pandas as pd

ENV_PATH = '/Users/zhishe/myProjects/anomaly'

NAME = 'firm_age_momentum'


#%%

# Filter

EXCH_CODE = [1, 2, 3]  # listed on NYSE, AMEX, or NASDAQ

PRICE_LIMIT = 5  # exclude stocks with a price below $5 at the portfolio formation date

HISTORY_REQUIREMENT = 12  # exclude firms with less than 12 months of past return data

#COMMON_STOCKS = [10, 11]  # exclude non-common stocks


#%%

# Read data

# monthly stock data
MSF = pd.read_hdf(ENV_PATH + '/data/msf.h5', key='msf')

# common stocks - PERMNOs
with open(ENV_PATH + '/data/common_stock_permno.pkl', 'rb') as f:
    COMMON_STOCK_PERMNO = pk.load(f)

# range of dates
DATE_RANGE = MSF['DATE'].unique()
DATE_RANGE.sort()


#%%

# Apply filter

# listed on NYSE, AMEX, or NASDAQ
conds = (MSF['HEXCD'] == c for c in EXCH_CODE)
cond = reduce(lambda x, y: x | y, conds)
MSF = MSF[cond].copy()


#%%

# Define signal calculation process

# signal: returns from month t-11 to t-1, 


def calc_past_cum_rets(m):
    """
    Calculate returns from month t-11 to t-1.

    Parameters
    ----------
    m : np.datetime64
        current month

    Returns
    -------
    pd.Series

    """
    # get data in the past 11 months
    start, end = DATE_RANGE[DATE_RANGE < m][[-11, -1]]
    data = MSF[
        (MSF['DATE'] >= start) &
        (MSF['DATE'] <= end)
        ]

    # set PERMNO as index
    data = data.set_index('PERMNO')

    # exclude stocks with a price < $5 at the portfolio formation date
    below_5 = data[
        (data['DATE'] == end) &
        (data['PRC'].abs() < PRICE_LIMIT)
        ]
    data.drop(below_5.index, inplace=True)

    # exclude non-common stocks
    common_stock_permno = COMMON_STOCK_PERMNO[end]
    permno_non_common = set(data.index) - set(common_stock_permno)
    data.drop(permno_non_common, inplace=True)

    # exclude firms with less than 12 months of past return data
    past_data = MSF[MSF['DATE'] <= end]
    past_data = past_data.set_index('PERMNO')  # set PERMNO as index
    ret_data_months = past_data.groupby(past_data.index).count()['RET']  # count months with non-NaN return
    less_than_12_months = ret_data_months[ret_data_months < HISTORY_REQUIREMENT]
    less_than_12_months = less_than_12_months.index.intersection(data.index).unique()
    data.drop(less_than_12_months, inplace=True)

    # focus on "high uncertainty" stocks
    firm_ages = past_data.groupby(past_data.index).count()['DATE']
    firm_ages = firm_ages.loc[data.index.unique()]
    quintiles = pd.qcut(firm_ages.rank(method='first'), 5, labels=False)  # cut into deciles based on firm age
    uncertain = quintiles[quintiles == 0]  # stocks with the least firm age
    data = data.loc[uncertain.index]

    # cumulative returns in the past 11 months
    cum_rets = (data['RET'] + 1).groupby(level=0).prod(min_count=1)
    cum_rets.dropna(inplace=True)  # drop NaN

    return cum_rets


def get_signals(args):
    """
    pno:            process identifier
    subrange:       subset of the date_range
    lb:             look back period
    """
    subrange, lb = args

    signals = {m: calc_past_cum_rets(m) for m in subrange}

    return signals


#%%

CPU_COUNT = psutil.cpu_count(logical=False)

COLLECTOR = {}  # the container where we store the computation results

START = 1983
END   = 2018

# look back periods & holding periods
LOOK_BACK = [11]
HOLDING   = [1]

START = DATE_RANGE[DATE_RANGE >= np.datetime64(dt.date(START, 1, 1))][0]
END = DATE_RANGE[DATE_RANGE <= np.datetime64(dt.date(END, 12, 31))][-1]


for lb in LOOK_BACK:
    hd = max(HOLDING)
    print('\nCalculating (%s, %s) strategy...' % (lb, hd), end='\t')

    # on this date we calculate the first set of signals
    first_date = DATE_RANGE[DATE_RANGE <= START][-hd]
    # calculate signals for every month in this range
    date_range = DATE_RANGE[(DATE_RANGE >= first_date) & (DATE_RANGE <= END)]

    # we will split the task and distribute the workloads to multiple processes
    size = len(date_range) // CPU_COUNT  # the size of each workload
    chunks = []
    for i in range(CPU_COUNT):
        if i != CPU_COUNT-1:
            chunks.append(date_range[size*i:size*(i+1)])
        else:
            chunks.append(date_range[size*i:])

    with futures.ProcessPoolExecutor(max_workers=CPU_COUNT) as ex:
        start_time = time.time()
        res = ex.map(get_signals, zip(chunks, [lb] * CPU_COUNT))
        for signals in res:
            COLLECTOR.setdefault(lb, {}).update(signals)
        print('{:.2f} s.'.format(time.time() - start_time))


#%%

# Output

if not os.path.exists(ENV_PATH + f'/results/{NAME}'):
    os.mkdir(ENV_PATH + f'/results/{NAME}')
    os.mkdir(ENV_PATH + f'/results/{NAME}/signals')

for lb in LOOK_BACK:
    table = pd.DataFrame()
    signals = COLLECTOR[lb]
    # consolidate each month's signals into a single table
    for k, v in signals.items():
        df = pd.DataFrame(v)
        df.reset_index(inplace=True)
        df['DATE'] = k
        table = pd.concat([table, df], ignore_index=True)

    table.rename(columns={'RET': 'SIGNAL'}, inplace=True)
    table = table.reindex(columns=['DATE', 'PERMNO', 'SIGNAL'])
    table.sort_values(by='DATE', inplace=True)
    table.to_csv(ENV_PATH + f'/results/{NAME}/signals/{lb}.csv')
    print('%s done.' % lb)
