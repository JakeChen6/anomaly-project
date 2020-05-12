import time
import datetime as dt
from functools import reduce
from concurrent import futures

import psutil
import numpy as np
import pandas as pd

ENV_PATH = '/Users/zhishe/myProjects/anomaly'

OUTPUT_PATH = ENV_PATH + '/anomalies/1-Momentum/signals'


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

def calc_past_cum_rets(m, lb):
    """
    m:      current month
    lb:     look back period
    """
    # select data
    start, end = DATE_RANGE[DATE_RANGE < m][[-lb, -1]]
    data = MSF[
        (MSF['DATE'] >= start) &
        (MSF['DATE'] <= end)
    ]

    # exclude non-common stocks
    mseall = MSEALL[
        (MSEALL['DATE'] >= start) &
        (MSEALL['DATE'] <= end)
    ]

    data = data.set_index('PERMNO')
    index = set(data.index) & set(mseall['PERMNO'].values)
    data = data.loc[index].copy()  # only keep common stocks

    # cumulative return in the past lb months
    cum_rets = (data['RET'] + 1).groupby(level=0).prod(min_count=1)
    cum_rets.dropna(inplace=True)  # drop NaN

    return cum_rets


def get_signals(args):
    """
    subrange:       subset of the date_range
    lb:             look back period
    """
    subrange, lb = args

    # signal: the cumulative return over the past lb months.
    signals = {m: calc_past_cum_rets(m, lb) for m in subrange}

    return signals


#%%

# Distribute workloads to multiple CPUs

CPU_COUNT = psutil.cpu_count(logical=False)

COLLECTOR = {}  # the container where we store the computation results

START = 1965
END   = 2018

# look back periods & holding periods
LOOK_BACK = [3, 6, 9, 12]
HOLDING   = [3, 6, 9, 12]

# we only specify starting and ending years above, here we infer the starting
# and ending months.
START = DATE_RANGE[DATE_RANGE >= np.datetime64(dt.date(START, 1, 1))][0]
END = DATE_RANGE[DATE_RANGE <= np.datetime64(dt.date(END, 12, 31))][-1]


for lb in LOOK_BACK:
    for hd in HOLDING:
        print('\nCalculating (%s, %s) strategy...' % (lb, hd), end='\t')

        # define the total workload and then split it
        first_date = DATE_RANGE[DATE_RANGE <= START][-hd]  # on this date we calc the first set of portfolios
        date_range = DATE_RANGE[(DATE_RANGE >= first_date) & (DATE_RANGE <= END)]  # the total workload
        size = len(date_range) // CPU_COUNT  # each process gets a subset, of this size, of the total workload

        chunks = []  # container where we store all the subsets of the total workload.
        for i in range(CPU_COUNT):
            if i != CPU_COUNT-1:
                chunks.append(date_range[size*i:size*(i+1)])
            else:
                chunks.append(date_range[size*i:])

        # distribute the workloads to the processes managed by a ProcessPoolExecutor
        with futures.ProcessPoolExecutor(max_workers=CPU_COUNT) as ex:
            start_time = time.time()
            res = ex.map(get_signals, zip(chunks, [lb] * CPU_COUNT))
            # collect the returned results and save them into 'COLLECTOR'
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

        table.rename(columns={'RET': 'SIGNAL'}, inplace=True)
        table = table.reindex(columns=['DATE', 'PERMNO', 'SIGNAL'])  # reorder the columns
        table.sort_values(by='DATE', inplace=True)  # sort by 'DATE'
        table.to_csv(OUTPUT_PATH + '/{}-{}.csv'.format(lb, hd))  # write to a CSV
        print('%s-%s done.' % (lb, hd))