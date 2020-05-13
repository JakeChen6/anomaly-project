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

NAME = 'short_term_reversal'


#%%

# Filter

# common stocks only
#COMMON_STOCKS = [10, 11]

# exclude penny stocks
PRICE_LIMIT = 5

# listed on NYSE, AMEX, or NASDAQ
EXCH_CODE = [1, 2, 3]


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

# exclude penny stocks
MSF = MSF[MSF['PRC'].abs() >= PRICE_LIMIT].copy()

# listed on NYSE, AMEX, or NASDAQ
conds = (MSF['HEXCD'] == c for c in EXCH_CODE)
cond = reduce(lambda x, y: x | y, conds)
MSF = MSF[cond].copy()


#%%

# Define signal calculation process

# signal: one-month lagged returns


def get_lagged_returns(m):
    """
    Get the one-month lagged returns for all the stocks.

    Parameters
    ----------
    m : np.datetime64
        current month

    Returns
    -------
    pd.Series

    """
    # select data
    prev_month = DATE_RANGE[DATE_RANGE < m][-1]  # previous month
    data = MSF[MSF['DATE'] == prev_month]

    # only include common stocks
    common_stock_permno = COMMON_STOCK_PERMNO[prev_month]

    data = data.set_index('PERMNO')
    index = set(data.index) & set(common_stock_permno)
    data = data.loc[index]  # only keep common stocks

    return data['RET'].dropna()


def get_signals(args):
    """
    pno:            process identifier
    subrange:       subset of the date_range
    lb:             look back period
    """
    subrange, lb = args

    signals = {m: get_lagged_returns(m) for m in subrange}

    return signals


#%%

CPU_COUNT = psutil.cpu_count(logical=False)

COLLECTOR = {}  # the container where we store the computation results

START = 1934
END   = 2018

# look back periods & holding periods
LOOK_BACK = [1]
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
