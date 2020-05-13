import os
import time
import datetime as dt
import pickle as pk
from concurrent import futures

import psutil
import numpy as np
import pandas as pd

ENV_PATH = '/Users/zhishe/myProjects/anomaly'

NAME = 'long_term_reversal'


#%%

# Filter

# common stocks only
#COMMON_STOCKS = [10, 11]

# exclude penny stocks
PRICE_LIMIT = 5

# NYSE only
EXCH_CODE = [1]


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

# NYSE only
MSF = MSF[MSF['HEXCD'] == EXCH_CODE[0]].copy()


#%%

# Define signal calculation process

# signal: cumulative excess returns for the prior 36 months


def get_prior_cum_excess_rets(m):
    """
    Compute the cumulative excess returns for the prior 36 months

    Parameters
    ----------
    m : np.datetime64
        current month

    Returns
    -------
    pd.Series

    """
    data = MSF[MSF['DATE'] < m]
    prev_month = DATE_RANGE[DATE_RANGE < m][-1]

    # only include common stocks
    common_stock_permno = COMMON_STOCK_PERMNO[prev_month]

    data = data.set_index('PERMNO')
    index = set(data.index) & set(common_stock_permno)
    data = data.loc[index]  # only keep common stocks

    # to be included in sample, stocks need to have at least 85 months of return data.
    # since the current month 'm' is included in this 85, what we actually need is
    # at least 84 months of return data.
    data = data.dropna(subset=['RET'])  # drop NaN
    grouped = data.groupby(data.index)
    data = grouped.filter(lambda x: x['DATE'].count() >= 84)

    # now we are ready to calculate the signals
    mkt_idx = dict()
    cum_excess_rets = dict()
    for p in data.index.unique():
        subdata = data.loc[p].sort_values('DATE').iloc[-36:]  # prior 36 months' data
        subdata = subdata.set_index('DATE')
        # prepare the market index values first
        for idx in subdata.index:
            if idx not in mkt_idx:
                mkt_idx[idx] = data[data['DATE'] == idx]['RET'].mean()
        # calculate the cumulative excess returns
        cum_excess_rets[p] = sum(row['RET'] - mkt_idx[idx] for idx, row in subdata.iterrows())

    # save in a pd.Series
    series = pd.Series(cum_excess_rets)
    series.name = 'RET'
    series.index.name = 'PERMNO'
    
    return series


def get_signals(args):
    """
    pno:            process identifier
    subrange:       subset of the date_range
    lb:             look back period
    """
    subrange, lb = args

    signals = {m: get_prior_cum_excess_rets(m) for m in subrange}

    return signals


#%%

CPU_COUNT = psutil.cpu_count(logical=False)

COLLECTOR = {}  # the container where we store the computation results

# look back periods & holding periods
LOOK_BACK = [36]
HOLDING   = [36]

START = 1933
END   = 2014

START = DATE_RANGE[DATE_RANGE >= np.datetime64(dt.date(START, 1, 1))][0]
END = DATE_RANGE[DATE_RANGE >= np.datetime64(dt.date(END, 1, 1))][0]

date_range = DATE_RANGE[(DATE_RANGE >= START) & (DATE_RANGE <= END)][::36]  # every 3 years

# distribute the workloads to multiple processes
size = len(date_range) // CPU_COUNT  # the size of each workload
chunks = []
for i in range(CPU_COUNT):
    if i != CPU_COUNT-1:
        chunks.append(date_range[size*i:size*(i+1)])
    else:
        chunks.append(date_range[size*i:])

lb = LOOK_BACK[0]
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
