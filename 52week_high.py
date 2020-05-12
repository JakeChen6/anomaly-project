import time
import datetime as dt
from functools import reduce
from concurrent import futures

import psutil
import numpy as np
import pandas as pd

ENV_PATH = '/Users/zhishe/myProjects/anomaly'

OUTPUT_PATH = ENV_PATH + '/anomalies/3-52-Week High/signals'


#%%

# Filter

# common stocks only
COMMON_STOCKS = [10, 11]

# exclude penny stocks
PRICE_LIMIT = 5

# listed on NYSE, AMEX, or NASDAQ
EXCH_CODE = [1, 2, 3]


#%%

# Read data

# monthly stock data
MSF = pd.read_hdf(ENV_PATH + '/data/msf.h5', key='msf')
# monthly stock event
MSEALL = pd.read_hdf(ENV_PATH + '/data/mseall.h5', key='mseall')

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

# common stocks only
conds = (MSEALL['SHRCD'] == c for c in COMMON_STOCKS)
cond = reduce(lambda x, y: x | y, conds)
MSEALL = MSEALL[cond].copy()


#%%

# Define signal calculation process

# signal: ratio of current price to the highest price during the past 12 months.

MSF['PRC'] = MSF['PRC'].abs()
MSF['ASKHI'] = MSF['ASKHI'].abs()


def calc_ratios(m):
    """
    Calculate for each stock its ratio of the current price to the highest price
    during the past 12 months.

    Parameters
    ----------
    m : np.datetime64
        current month

    Returns
    -------
    pd.Series

    """
    # SELECT data
    start, end = DATE_RANGE[DATE_RANGE < m][[-12, -1]]  # data of past 12 months
    data = MSF[
        (MSF['DATE'] >= start) &
        (MSF['DATE'] <= end)
    ]

    # filter out non-common stocks
    mseall = MSEALL[
        (MSEALL['DATE'] >= start) &
        (MSEALL['DATE'] <= end)
    ]    

    data = data.set_index('PERMNO')
    index = set(data.index) & set(mseall['PERMNO'].values)
    data = data.loc[index]  # only keep common stocks
    
    # calculate each stock's ratio
    ratios = {}
    for p in data.index.unique():
        subdata = data[data.index == p]
        if subdata.shape[0] < 10:  # less than 10 months' data available
            continue
        subdata = subdata.set_index('DATE').sort_index()
        if subdata.index[-1] != end:  # no data on month m-1
            continue
        highest = subdata['ASKHI'].max()
        r = subdata.iloc[-1]['PRC'] / highest
        ratios[p] = r

    # save in a pandas.Series
    series = pd.Series(ratios)
    series.name = 'RATIO'
    series.index.name = 'PERMNO'

    return series


def get_signals(args):
    """
    pno:            process identifier
    subrange:       subset of the date_range
    lb:             look back period
    """
    subrange, lb = args

    signals = {m: calc_ratios(m) for m in subrange}

    return signals


#%%

CPU_COUNT = psutil.cpu_count(logical=False)

COLLECTOR = {}  # the container where we store the computation results

START = 1963
END   = 2018

# look back periods & holding periods
LOOK_BACK = [6]
HOLDING   = [6]

START = DATE_RANGE[DATE_RANGE >= np.datetime64(dt.date(START, 1, 1))][0]
END = DATE_RANGE[DATE_RANGE <= np.datetime64(dt.date(END, 12, 31))][-1]


for lb in LOOK_BACK:
    for hd in HOLDING:
        print('\nCalculating (%s, %s) strategy...' % (lb, hd), end='\t')

        # split the task into several subtasks.
        first_date = DATE_RANGE[DATE_RANGE <= START][-hd]  # on this date we calc the first set of portfolios
        date_range = DATE_RANGE[(DATE_RANGE >= first_date) & (DATE_RANGE <= END)]  # each process gets a subset of this array
        size = len(date_range) // CPU_COUNT  # size of subset

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
                COLLECTOR.setdefault((lb, hd), {}).update(signals)
            print('{:.2f} s.'.format(time.time() - start_time))


#%%
# Output

for lb in LOOK_BACK:
    for hd in HOLDING:
        table = pd.DataFrame()
        signals = COLLECTOR[(lb, hd)]
        # consolidate each month's signals into a single table
        for k, v in signals.items():
            df = pd.DataFrame(v)
            df.reset_index(inplace=True)
            df['DATE'] = k
            table = pd.concat([table, df], ignore_index=True)

        table.rename(columns={'RATIO': 'SIGNAL'}, inplace=True)
        table = table.reindex(columns=['DATE', 'PERMNO', 'SIGNAL'])
        table.sort_values(by='DATE', inplace=True)
        table.to_csv(OUTPUT_PATH + '/{}-{}.csv'.format(lb, hd))
        print('%s-%s done.' % (lb, hd))
